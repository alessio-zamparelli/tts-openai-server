"""Wyoming TTS server (TCP JSONL + PCM).

Implements the ``tts`` domain of the Wyoming protocol so Home Assistant and
Rhasspy clients can use this server alongside the OpenAI-compatible HTTP API.
The handler shares the single :class:`TTSEngine`, keeping generation
serialization, the voice-state LRU and idle eviction identical across
transports.

Protocol summary (see https://github.com/rhasspy/rhasspy3/blob/master/docs/wyoming.md):

    client -> describe            client <- info
    client -> synthesize          client <- audio-start / audio-chunk* / audio-stop
    client -> synthesize-start    (streaming text)
    client -> synthesize-chunk*
    client -> synthesize-stop     client <- audio-start / chunks / audio-stop,
                                             then synthesize-stopped

Errors are reported as ``error`` events; the connection stays open so clients
can retry (HA raises on the event itself).
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from sentence_stream import SentenceBoundaryDetector
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import (
    Attribution,
    Describe,
    Info,
    TtsProgram,
    TtsVoice,
    TtsVoiceSpeaker,
)
from wyoming.ping import Ping, Pong
from wyoming.server import AsyncEventHandler, AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeTextFormat,
    SynthesizeVoice,
)

from . import __version__
from .config import Config
from .engine import RateLimited, TTSEngine
from .voices import (
    KYUTAI_CATALOG,
    VOICE_LANGUAGE,
    piper_language_tag,
    pocket_language_tag,
)

logger = logging.getLogger(__name__)

WIDTH = 2  # bytes per sample (s16le)
CHANNELS = 1  # mono
ATTRIBUTION = Attribution(name="tts-openai", url="")
PROGRAM_DESCRIPTION = "Local neural text to speech"

EngineGetter = Callable[[], "TTSEngine | None"]


# -- info --------------------------------------------------------------------


def _voice_names(engine: TTSEngine, config: Config) -> list[str]:
    """Concrete voice names advertised to clients (aliases are accepted too)."""
    names: list[str] = []

    def add(name: str | None) -> None:
        if name and name not in names:
            names.append(name)

    add(engine.default_voice)
    for name in engine.installed_voices():
        add(name)
    if engine.backend != "piper":
        for name in sorted(KYUTAI_CATALOG):
            add(name)
    if engine.registry is not None:
        for name in engine.registry.names():
            add(name)
    return sorted(names)


def _voice_info(engine: TTSEngine, config: Config, name: str) -> TtsVoice:
    """Build one ``TtsVoice`` from backend metadata with sensible fallbacks."""
    meta: dict[str, object] = {}
    if engine.backend == "piper":
        try:
            meta = engine.voice_metadata(name)
        except (OSError, ValueError, TypeError):
            meta = {}

    languages = [str(lang) for lang in (meta.get("languages") or [])]
    if not languages:
        if engine.backend == "piper":
            languages = [piper_language_tag(name)]
        else:
            language = None
            if engine.registry is not None:
                custom = engine.registry.get(name)
                language = custom.language if custom is not None else None
            target = config.voice_map.get(name, name)
            languages = [
                language
                or VOICE_LANGUAGE.get(target)
                or pocket_language_tag(config.language)
            ]

    speakers = [str(speaker) for speaker in (meta.get("speakers") or [])]
    return TtsVoice(
        name=name,
        description=str(meta.get("description") or name),
        version=None,
        attribution=ATTRIBUTION,
        installed=True,
        languages=languages,
        speakers=[TtsVoiceSpeaker(name=speaker) for speaker in speakers] or None,
    )


def build_info(engine: TTSEngine, config: Config) -> Info:
    """Describe this TTS program and its voices (rebuilt on every ``describe``)."""
    return Info(
        tts=[
            TtsProgram(
                name=engine.model_name,
                description=PROGRAM_DESCRIPTION,
                attribution=ATTRIBUTION,
                installed=True,
                voices=[_voice_info(engine, config, name) for name in _voice_names(engine, config)],
                version=__version__,
                supports_synthesize_streaming=True,
            )
        ]
    )


# -- handler -----------------------------------------------------------------


@dataclass
class _StreamState:
    """Per-connection state for the text-streaming extension."""

    voice: str
    speaker: str | None
    detector: SentenceBoundaryDetector = field(default_factory=SentenceBoundaryDetector)
    rate: int | None = None
    received_chunks: bool = False


class TTSEventHandler(AsyncEventHandler):
    """Serve one Wyoming client connection."""

    def __init__(
        self,
        engine_getter: EngineGetter,
        config: Config,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        super().__init__(reader, writer)
        self._engine_getter = engine_getter
        self._config = config
        self._stream: _StreamState | None = None

    # -- event dispatch ------------------------------------------------------

    async def handle_event(self, event: Event) -> bool:
        if Ping.is_type(event.type):
            await self.write_event(Pong.from_event(event).event())
            return True

        if Describe.is_type(event.type):
            engine = await self._wait_for_engine()
            if engine is None:
                return await self._error(
                    "model-loading", "Model is still loading; retry shortly."
                )
            await self.write_event(build_info(engine, self._config).event())
            return True

        if SynthesizeStart.is_type(event.type):
            engine = await self._wait_for_engine()
            if engine is None:
                return await self._error(
                    "model-loading", "Model is still loading; retry shortly."
                )
            start = SynthesizeStart.from_event(event)
            self._stream = _StreamState(
                voice=_select_voice(start.voice, engine),
                speaker=start.voice.speaker if start.voice is not None else None,
            )
            return True

        if SynthesizeChunk.is_type(event.type):
            engine = await self._wait_for_engine()
            if engine is None:
                return await self._error(
                    "model-loading", "Model is still loading; retry shortly."
                )
            return await self._handle_chunk(engine, SynthesizeChunk.from_event(event))

        if SynthesizeStop.is_type(event.type):
            engine = await self._wait_for_engine()
            if engine is None:
                return await self._error(
                    "model-loading", "Model is still loading; retry shortly."
                )
            return await self._handle_stop(engine)

        if Synthesize.is_type(event.type):
            engine = await self._wait_for_engine()
            if engine is None:
                return await self._error(
                    "model-loading", "Model is still loading; retry shortly."
                )
            synth = Synthesize.from_event(event)
            if self._stream is not None:
                # Compatibility event sent alongside the text stream. Ignore it
                # when chunks already produced audio; otherwise use it so a
                # client that only sends start/synthesize/stop still works.
                if self._stream.received_chunks:
                    return True
                return await self._handle_compat_synthesize(engine, synth)
            return await self._handle_synthesize(engine, synth)

        # Unknown/unsupported events are dropped per the protocol spec.
        return True

    # -- plain synthesize ----------------------------------------------------

    async def _handle_synthesize(self, engine: TTSEngine, synth: Synthesize) -> bool:
        text_format = synth.text_format
        if text_format not in (None, "text", SynthesizeTextFormat.TEXT):
            return await self._error(
                "unsupported-text-format",
                f"Unsupported text_format {text_format!r}; only plain text is supported.",
            )
        text = synth.text.strip()
        if not text:
            return await self._error("invalid-text", "text must be non-empty")

        voice = _select_voice(synth.voice, engine)
        speaker = synth.voice.speaker if synth.voice is not None else None
        if speaker is not None and not engine.supports_speaker:
            return await self._error(
                "unsupported-option",
                f"Speaker selection is not supported by the {engine.model_name} backend.",
            )

        sentences = _split_text(text)
        if not sentences:
            return await self._error("invalid-text", "text must be non-empty")

        try:
            engine.check_capacity()
            rate = await asyncio.to_thread(engine.sample_rate_for, voice)
        except ValueError as exc:
            return await self._error("unknown-voice", str(exc))
        except RateLimited as exc:
            return await self._error("rate-limited", str(exc))

        await self.write_event(
            AudioStart(rate=rate, width=WIDTH, channels=CHANNELS).event()
        )
        try:
            for index, sentence in enumerate(sentences):
                await self._write_sentence(
                    engine,
                    sentence,
                    voice,
                    rate,
                    speaker,
                    add_silence=index < len(sentences) - 1,
                )
        except ValueError as exc:
            return await self._error("unknown-voice", str(exc))
        except RateLimited as exc:
            return await self._error("rate-limited", str(exc))
        except (ConnectionError, asyncio.CancelledError):
            return False  # client went away; nothing left to report to
        except Exception as exc:  # pragma: no cover - defensive backend failure
            logger.exception("wyoming synthesize failed")
            return await self._error("internal-error", str(exc))

        await self.write_event(AudioStop().event())
        return True

    async def _handle_compat_synthesize(self, engine: TTSEngine, synth: Synthesize) -> bool:
        """Treat the backwards-compatibility event as the whole text stream."""
        state = self._stream
        assert state is not None
        state.received_chunks = True
        for sentence in _split_text(synth.text):
            if not await self._write_stream_sentence(engine, sentence, add_silence=True):
                return True
        return True

    # -- streaming -----------------------------------------------------------

    async def _handle_chunk(self, engine: TTSEngine, chunk: SynthesizeChunk) -> bool:
        state = self._stream
        if state is None:
            return await self._error(
                "synthesize-not-started",
                "Received synthesize-chunk before synthesize-start.",
            )
        state.received_chunks = True
        for sentence in state.detector.add_chunk(chunk.text):
            if not await self._write_stream_sentence(engine, sentence, add_silence=True):
                return True
        return True

    async def _handle_stop(self, engine: TTSEngine) -> bool:
        state = self._stream
        if state is None:
            return await self._error(
                "synthesize-not-started",
                "Received synthesize-stop before synthesize-start.",
            )
        tail = state.detector.finish()
        if tail.strip():
            await self._write_stream_sentence(engine, tail, add_silence=False)
        # Always terminate the stream so HA's reader stops even with no audio.
        await self.write_event(SynthesizeStopped().event())
        self._stream = None
        return True

    async def _write_stream_sentence(
        self, engine: TTSEngine, sentence: str, *, add_silence: bool
    ) -> bool:
        state = self._stream
        assert state is not None
        if state.speaker is not None and not engine.supports_speaker:
            await self._error(
                "unsupported-option",
                f"Speaker selection is not supported by the {engine.model_name} backend.",
            )
            return False
        try:
            if state.rate is None:
                state.rate = await asyncio.to_thread(engine.sample_rate_for, state.voice)
            await self._write_sentence(
                engine,
                sentence,
                state.voice,
                state.rate,
                state.speaker,
                add_silence=add_silence,
                send_start=True,
                send_stop=True,
            )
            return True
        except ValueError as exc:
            await self._error("unknown-voice", str(exc))
            return False
        except RateLimited as exc:
            await self._error("rate-limited", str(exc))
            return False
        except (ConnectionError, asyncio.CancelledError):
            return False  # client disconnected mid-stream
        except Exception as exc:  # pragma: no cover - defensive backend failure
            logger.exception("wyoming stream failed")
            await self._error("internal-error", str(exc))
            return False

    # -- audio ---------------------------------------------------------------

    async def _write_sentence(
        self,
        engine: TTSEngine,
        sentence: str,
        voice: str,
        rate: int,
        speaker: str | None,
        *,
        add_silence: bool,
        send_start: bool = False,
        send_stop: bool = False,
    ) -> None:
        text = _punctuate(sentence, self._config.wyoming_auto_punctuation)
        if not text:
            return
        pcm = await asyncio.to_thread(engine.generate_pcm, text, voice, speaker=speaker)
        if send_start:
            await self.write_event(
                AudioStart(rate=rate, width=WIDTH, channels=CHANNELS).event()
            )
        await self._write_pcm(pcm, rate)
        if add_silence and self._config.wyoming_sentence_silence > 0:
            silence = bytes(
                int(rate * self._config.wyoming_sentence_silence) * WIDTH * CHANNELS
            )
            await self._write_pcm(silence, rate)
        if send_stop:
            await self.write_event(AudioStop().event())

    async def _write_pcm(self, pcm: bytes, rate: int) -> None:
        chunk_bytes = self._config.wyoming_samples_per_chunk * WIDTH * CHANNELS
        for offset in range(0, len(pcm), chunk_bytes):
            await self.write_event(
                AudioChunk(
                    rate=rate,
                    width=WIDTH,
                    channels=CHANNELS,
                    audio=pcm[offset : offset + chunk_bytes],
                ).event()
            )

    # -- helpers -------------------------------------------------------------

    async def _wait_for_engine(self) -> TTSEngine | None:
        """Wait for the background model load, bounded by config."""
        timeout = max(self._config.wyoming_ready_timeout_s, 0.0)
        deadline = time.monotonic() + timeout
        while True:
            engine = self._engine_getter()
            if engine is not None or time.monotonic() >= deadline:
                return engine
            await asyncio.sleep(0.25)

    async def _error(self, code: str, message: str) -> bool:
        logger.warning("wyoming error (%s): %s", code, message)
        await self.write_event(Error(text=message, code=code).event())
        return True


# -- text helpers ------------------------------------------------------------


def _split_text(text: str) -> list[str]:
    """Split whole text into sentences using the wyoming-piper splitter."""
    detector = SentenceBoundaryDetector()
    sentences = [sentence.strip() for sentence in detector.add_chunk(text)]
    tail = detector.finish().strip()
    if tail:
        sentences.append(tail)
    return [sentence for sentence in sentences if sentence]


def _punctuate(text: str, punctuation: str) -> str:
    """Append terminal punctuation when missing (HA add-on parity)."""
    text = text.strip()
    if not text or not punctuation:
        return text
    if text[-1] in punctuation:
        return text
    return text + punctuation[0]


def _select_voice(voice: SynthesizeVoice | None, engine: TTSEngine) -> str:
    """Pick the requested voice name, else the server default."""
    if voice is not None and voice.name:
        return voice.name
    return engine.default_voice


# -- server ------------------------------------------------------------------


def _bound_port(server: AsyncTcpServer) -> int:
    """Actual listening port (supports ``port=0`` for tests)."""
    inner = getattr(server, "_server", None)
    sockets = getattr(inner, "sockets", None) or []
    if sockets:
        return int(sockets[0].getsockname()[1])
    return server.port


class WyomingService:
    """Owns the Wyoming TCP listener for the app lifespan."""

    def __init__(self, config: Config, engine_getter: EngineGetter) -> None:
        self._config = config
        self._engine_getter = engine_getter
        self._server = AsyncTcpServer(config.wyoming_host, config.wyoming_port)
        self._zeroconf = None
        self.port = config.wyoming_port

    def _handler_factory(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> TTSEventHandler:
        return TTSEventHandler(self._engine_getter, self._config, reader, writer)

    async def start(self) -> None:
        await self._server.start(self._handler_factory)
        self.port = _bound_port(self._server)
        if self._config.wyoming_zeroconf:
            from wyoming.zeroconf import HomeAssistantZeroconf

            host = self._config.wyoming_host
            # Let zeroconf detect the outbound address when bound to a wildcard.
            advertise_host = None if host in ("", "0.0.0.0", "::") else host
            self._zeroconf = HomeAssistantZeroconf(
                name=self._config.wyoming_name,
                port=self.port,
                host=advertise_host,
            )
            await self._zeroconf.register_server()
            logger.info(
                "wyoming zeroconf registered: %s on port %d",
                self._config.wyoming_name,
                self.port,
            )

    async def stop(self) -> None:
        await self._server.stop()
