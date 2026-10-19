"""Wyoming protocol tests: describe/info, plain synthesize, text streaming.

Offline: a real TCP listener on an ephemeral port in front of the fake engine
from ``conftest.py``. No model download, no ffmpeg.
"""

from __future__ import annotations

import asyncio
import json

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.ping import Ping, Pong
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeVoice,
)

from tts_openai.config import Config
from tts_openai.engine import TTSEngine
from tts_openai.piper_backend import PiperTTSModel
from tts_openai.wyoming_server import WyomingService, build_info

from .conftest import FakePiperVoice, fake_piper_loader


def _config(engine: TTSEngine, **overrides) -> Config:
    values = dict(
        backend=engine.backend,
        wyoming_enabled=True,
        wyoming_host="127.0.0.1",
        wyoming_port=0,
        wyoming_ready_timeout_s=0.0,
    )
    values.update(overrides)
    return Config(**values)


async def _collect(client: AsyncTcpClient, terminal: tuple[str, ...], timeout: float = 5.0):
    """Read events until one of ``terminal`` arrives (or time out)."""

    async def read():
        events = []
        while True:
            event = await client.read_event()
            assert event is not None, "connection closed before terminal event"
            events.append(event)
            if event.type in terminal:
                return events

    return await asyncio.wait_for(read(), timeout=timeout)


def _with_service(engine: TTSEngine, config: Config | None = None):
    """Async context manager: start the Wyoming service and a client."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def manager():
        service = WyomingService(config or _config(engine), lambda: engine)
        await service.start()
        client = AsyncTcpClient("127.0.0.1", service.port)
        await client.connect()
        try:
            yield client, service
        finally:
            await client.disconnect()
            await service.stop()

    return manager()


def _pcm(events) -> bytes:
    return b"".join(
        AudioChunk.from_event(event).audio
        for event in events
        if AudioChunk.is_type(event.type)
    )


# -- describe / info ---------------------------------------------------------


def test_describe_advertises_tts_program_and_voices(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(Describe().event())
            events = await _collect(client, ("info",))
        info = Info.from_event(events[-1])
        assert len(info.tts) == 1
        program = info.tts[0]
        assert program.name == "pocket-tts"
        assert program.installed is True
        assert program.supports_synthesize_streaming is True
        names = [voice.name for voice in program.voices]
        assert names == sorted(names)
        assert "alloy" in names  # configured default
        assert "giovanni" in names  # Kyutai catalog voice
        alloy = next(voice for voice in program.voices if voice.name == "alloy")
        assert alloy.languages == ["en"]

    asyncio.run(scenario())


def test_ping_returns_pong(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(Ping(text="are you there").event())
            events = await _collect(client, ("pong",))
        assert Pong.from_event(events[-1]).text == "are you there"

    asyncio.run(scenario())


def test_info_rebuilt_from_engine_metadata(piper_engine, tmp_path):
    config = _config(piper_engine)
    path = tmp_path / "piper" / "it_IT-paola-medium.onnx.json"
    path.write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050, "quality": "medium"},
                "speaker_id_map": {"alice": 1, "bob": 2},
            }
        ),
        encoding="utf-8",
    )
    info = build_info(piper_engine, config)
    voice = info.tts[0].voices[0]
    assert voice.description == "paola (medium)"
    assert voice.languages == ["it-IT"]
    assert [speaker.name for speaker in (voice.speakers or [])] == ["alice", "bob"]


# -- plain synthesize --------------------------------------------------------


def test_synthesize_emits_streamed_audio(engine, fake_model):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(Synthesize(text="Ciao. Come stai?").event())
            events = await _collect(client, ("audio-stop",))
        return events

    events = asyncio.run(scenario())
    start = next(AudioStart.from_event(e) for e in events if AudioStart.is_type(e.type))
    assert (start.rate, start.width, start.channels) == (24000, 2, 1)
    # One generate_audio call per sentence, auto-punctuation unchanged.
    assert fake_model.generate_calls == ["Ciao.", "Come stai?"]
    # The fake returns 1 s per call: 2 * 24000 samples * 2 bytes.
    assert len(_pcm(events)) == 2 * 24000 * 2
    assert events[-1].type == "audio-stop"


def test_sentence_silence_is_inserted_between_sentences(engine):
    config = _config(engine, wyoming_sentence_silence=0.1)

    async def scenario():
        async with _with_service(engine, config) as (client, _):
            await client.write_event(Synthesize(text="Hello there. Bye now.").event())
            events = await _collect(client, ("audio-stop",))
        return events

    events = asyncio.run(scenario())
    expected = 2 * 24000 * 2 + int(24000 * 0.1) * 2
    assert len(_pcm(events)) == expected


def test_auto_punctuation_can_be_disabled(engine, fake_model):
    config = _config(engine, wyoming_auto_punctuation="")

    async def scenario():
        async with _with_service(engine, config) as (client, _):
            await client.write_event(Synthesize(text="Ciao").event())
            await _collect(client, ("audio-stop",))

    asyncio.run(scenario())
    assert fake_model.generate_calls == ["Ciao"]


def test_unknown_voice_returns_error_and_keeps_connection(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(
                Synthesize(text="hi", voice=SynthesizeVoice(name="not-a-voice")).event()
            )
            errors = await _collect(client, ("error",))
            assert Error.from_event(errors[-1]).code == "unknown-voice"
            # Connection stays usable.
            await client.write_event(Describe().event())
            events = await _collect(client, ("info",))
        return events

    asyncio.run(scenario())


def test_empty_text_returns_error(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(Synthesize(text="   ").event())
            events = await _collect(client, ("error",))
        assert Error.from_event(events[-1]).code == "invalid-text"

    asyncio.run(scenario())


def test_ssml_is_rejected(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(
                Synthesize(text="<speak>hi</speak>", text_format="ssml").event()
            )
            events = await _collect(client, ("error",))
        assert Error.from_event(events[-1]).code == "unsupported-text-format"

    asyncio.run(scenario())


# -- streaming ---------------------------------------------------------------


def test_text_streaming_emits_audio_per_sentence(engine, fake_model):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(SynthesizeStart().event())
            await client.write_event(SynthesizeChunk(text="Ciao. Co").event())
            await client.write_event(SynthesizeChunk(text="me stai?").event())
            await client.write_event(SynthesizeStop().event())
            events = await _collect(client, ("synthesize-stopped",))
        return events

    events = asyncio.run(scenario())
    starts = [e for e in events if AudioStart.is_type(e.type)]
    stops = [e for e in events if AudioStop.is_type(e.type)]
    assert len(starts) == 2  # one audio-start/stop pair per sentence
    assert len(stops) == 2
    assert fake_model.generate_calls == ["Ciao.", "Come stai?"]
    assert len(_pcm(events)) == 2 * 24000 * 2
    assert events[-1].type == "synthesize-stopped"


def test_streaming_ignores_compat_synthesize_when_chunks_seen(engine, fake_model):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(SynthesizeStart().event())
            await client.write_event(SynthesizeChunk(text="One.").event())
            await client.write_event(Synthesize(text="One. Duplicate.").event())
            await client.write_event(SynthesizeStop().event())
            await _collect(client, ("synthesize-stopped",))

    asyncio.run(scenario())
    assert fake_model.generate_calls == ["One."]


def test_streaming_compat_synthesize_only_client(engine, fake_model):
    """A client that sends start/synthesize/stop but no chunks still works."""

    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(SynthesizeStart().event())
            await client.write_event(Synthesize(text="Ciao.").event())
            await client.write_event(SynthesizeStop().event())
            events = await _collect(client, ("synthesize-stopped",))
        return events

    events = asyncio.run(scenario())
    assert fake_model.generate_calls == ["Ciao."]
    assert AudioStart.is_type(events[0].type)


def test_chunk_without_start_returns_error(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(SynthesizeChunk(text="hi").event())
            events = await _collect(client, ("error",))
        assert Error.from_event(events[-1]).code == "synthesize-not-started"

    asyncio.run(scenario())


# -- startup / backend limitations -------------------------------------------


def test_engine_still_loading(engine):
    config = _config(engine, wyoming_ready_timeout_s=0)

    async def scenario():
        service = WyomingService(config, lambda: None)
        await service.start()
        client = AsyncTcpClient("127.0.0.1", service.port)
        await client.connect()
        try:
            await client.write_event(Describe().event())
            events = await _collect(client, ("error",))
        finally:
            await client.disconnect()
            await service.stop()
        return events

    events = asyncio.run(scenario())
    assert Error.from_event(events[-1]).code == "model-loading"


def test_pocket_rejects_speaker(engine):
    async def scenario():
        async with _with_service(engine) as (client, _):
            await client.write_event(
                Synthesize(text="hi", voice=SynthesizeVoice(name="alloy", speaker="alice")).event()
            )
            events = await _collect(client, ("error",))
        assert Error.from_event(events[-1]).code == "unsupported-option"

    asyncio.run(scenario())


def test_piper_forwards_speaker_and_scales(piper_engine):
    state, _ = piper_engine.voice_state("it_IT-paola-medium")
    assert isinstance(state, FakePiperVoice)
    state.config.speaker_id_map.update({"alice": 1})

    async def scenario():
        async with _with_service(piper_engine) as (client, _):
            await client.write_event(
                Synthesize(
                    text="Ciao.",
                    voice=SynthesizeVoice(name="it_IT-paola-medium", speaker="alice"),
                ).event()
            )
            await _collect(client, ("audio-stop",))

    asyncio.run(scenario())
    syn_config = state.syn_configs[-1]
    assert syn_config is not None
    assert syn_config.speaker_id == 1


def test_piper_scales_from_config(tmp_path):
    voice_dir = tmp_path / "piper"
    voice_dir.mkdir()
    (voice_dir / "it_IT-paola-medium.onnx").write_bytes(b"fake")
    (voice_dir / "it_IT-paola-medium.onnx.json").write_text(
        json.dumps({"audio": {"sample_rate": 22050}}), encoding="utf-8"
    )
    config = Config(
        backend="piper",
        piper_voice_dir=str(voice_dir),
        default_voice="it_IT-paola-medium",
        piper_length_scale=0.8,
        piper_noise_scale=0.5,
        piper_noise_w=0.3,
    )
    # Construct without preloading so no loader/download is needed.
    piper = PiperTTSModel(config, voice_loader=fake_piper_loader, preload=False)
    syn = piper._synthesis_config(FakePiperVoice(22050), None)
    assert (syn.length_scale, syn.noise_scale, syn.noise_w_scale) == (0.8, 0.5, 0.3)


# -- concurrency / framing ---------------------------------------------------


def test_concurrent_clients_are_serialized(engine, fake_model):
    async def scenario():
        # Two separate services are fine: both wrap the same engine instance.
        service = WyomingService(_config(engine), lambda: engine)
        await service.start()
        try:
            async def run(text):
                client = AsyncTcpClient("127.0.0.1", service.port)
                await client.connect()
                try:
                    await client.write_event(Synthesize(text=text).event())
                    await _collect(client, ("audio-stop",))
                finally:
                    await client.disconnect()

            await asyncio.gather(run("One."), run("Two."))
        finally:
            await service.stop()

    asyncio.run(scenario())
    assert fake_model.overlap_seen is False


def test_raw_jsonl_framing(engine):
    """The wire format is a single JSON line per event (spec compatibility)."""

    async def scenario():
        service = WyomingService(_config(engine), lambda: engine)
        await service.start()
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", service.port)
            writer.write(b'{"type": "describe"}\n')
            await writer.drain()
            line = await asyncio.wait_for(reader.readline(), timeout=5)
            writer.close()
            await writer.wait_closed()
            return json.loads(line)
        finally:
            await service.stop()

    event = asyncio.run(scenario())
    assert event["type"] == "info"


def test_create_app_lifespan_starts_and_stops_wyoming(engine):
    """The FastAPI lifespan owns the listener (same event loop as uvicorn)."""
    import socket

    from fastapi.testclient import TestClient

    from tts_openai.main import create_app

    config = _config(engine)
    app = create_app(config, engine)
    with TestClient(app) as http:
        service = app.state.wyoming
        assert service is not None
        assert http.get("/health").json()["wyoming"] == {
            "enabled": True,
            "listening": True,
            "port": service.port,
        }
        with socket.create_connection(("127.0.0.1", service.port), timeout=5) as sock:
            sock.sendall(b'{"type": "describe"}\n')
            reader = sock.makefile("rb")
            event = json.loads(reader.readline())
            assert event["type"] == "info"
    assert app.state.wyoming is None
