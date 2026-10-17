"""Piper backend adapter.

Piper (https://github.com/OHF-Voice/piper1-gpl) synthesizes with a separate
ONNX model per voice. This module adapts a ``PiperVoice`` to the engine's
duck-typed ``TTSModelLike`` protocol so :class:`TTSEngine` keeps owning the
generation lock, voice-state LRU and idle eviction.

Voice files are cached under ``POCKET_TTS_PIPER_VOICE_DIR`` (default
``~/.cache/piper``) and downloaded on demand from the
``rhasspy/piper-voices`` Hugging Face repo unless
``POCKET_TTS_PIPER_AUTO_DOWNLOAD=false``.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Protocol, cast

import numpy as np

from .config import Config
from .voices import is_piper_path, is_piper_voice_id, piper_language, resolve_piper_voice

if TYPE_CHECKING:  # pragma: no cover
    from .engine import TTSEngine

logger = logging.getLogger(__name__)

DEFAULT_PIPER_DIR = Path.home() / ".cache" / "piper"


class _AudioChunkLike(Protocol):
    """The slice of ``piper.AudioChunk`` the adapter reads."""

    audio_float_array: object


class _VoiceStateLike(Protocol):
    """The slice of ``piper.PiperVoice`` the engine relies on."""

    config: object  # PiperConfig, with .sample_rate

    def synthesize(self, text: str) -> Iterator[_AudioChunkLike]: ...


def _default_voice_loader(path: Path) -> object:
    from piper import PiperVoice

    return PiperVoice.load(str(path))


def _default_downloader(voice: str, directory: Path) -> None:
    from piper.download_voices import download_voice

    download_voice(voice, directory)


class PiperTTSModel:
    """Duck-typed model exposing Piper voices to :class:`TTSEngine`.

    ``voice`` values are Piper ids (``it_IT-paola-medium``), local ``.onnx``
    paths, or OpenAI aliases resolved through ``Config.voice_map``. Loading a
    voice yields a ``PiperVoice`` (an ONNX session) which the engine caches in
    its LRU — each cached entry is therefore a full model, hence the much
    smaller default cache size for this backend.
    """

    backend = "piper"
    supports_cloning = False

    def __init__(
        self,
        config: Config,
        *,
        voice_dir: Path | None = None,
        voice_loader: "Callable[[Path], object] | None" = None,
        downloader: "Callable[[str, Path], None] | None" = None,
        preload: bool = True,
    ):
        self._config = config
        self.voice_map = config.voice_map
        # Normalize aliases once so health/voice-cache keys always use the id.
        self.default_voice = resolve_piper_voice(config.default_voice, self.voice_map)
        self.auto_download = config.piper_auto_download
        self.voice_dir = (
            Path(config.piper_voice_dir).expanduser()
            if config.piper_voice_dir
            else DEFAULT_PIPER_DIR
        )
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        self._voice_loader = voice_loader or _default_voice_loader
        self._downloader = downloader or _default_downloader
        self._download_lock = threading.Lock()
        self._default_state: object | None = None
        # Populated by preload (or by the first successful load of the default).
        self.sample_rate = 0
        self.language = piper_language(self.default_voice)
        if preload:
            state = self.get_state_for_audio_prompt(self.default_voice)
            self.sample_rate = int(cast(_VoiceStateLike, state).config.sample_rate)
        else:
            self.sample_rate = self._read_sample_rate(self._ensure_files(self.default_voice))

    # -- voice resolution ----------------------------------------------------

    def resolve_voice(self, requested: str) -> str:
        """Alias -> Piper id; ids and local ``.onnx`` paths pass through."""
        return resolve_piper_voice(requested, self.voice_map)

    # -- voice files ---------------------------------------------------------

    def _model_path(self, resolved: str) -> Path:
        """Filesystem path for a resolved voice id or local model path."""
        if is_piper_voice_id(resolved):
            return self.voice_dir / f"{resolved}.onnx"
        if is_piper_path(resolved):
            return Path(resolved).expanduser()
        raise ValueError(f"Not a Piper voice id or .onnx path: {resolved!r}")

    def _ensure_files(self, resolved: str) -> Path:
        """Return the ``.onnx`` path, downloading the voice pair if needed."""
        model_path = self._model_path(resolved)
        config_path = Path(f"{model_path}.json")
        if model_path.exists() and config_path.exists():
            return model_path
        if is_piper_path(resolved):
            raise ValueError(
                f"Piper voice model not found: {model_path} (expected the matching "
                f"{config_path.name} next to it)."
            )
        if not self.auto_download:
            raise ValueError(
                f"Piper voice {resolved!r} is not installed in {self.voice_dir} and "
                "POCKET_TTS_PIPER_AUTO_DOWNLOAD is disabled."
            )
        # One download at a time; a waiter re-checks the files afterwards.
        with self._download_lock:
            if model_path.exists() and config_path.exists():
                return model_path
            logger.info("downloading piper voice %r into %s…", resolved, self.voice_dir)
            t0 = time.perf_counter()
            try:
                self._downloader(resolved, self.voice_dir)
            except Exception as exc:  # network/HTTP/pattern errors -> client error
                raise ValueError(
                    f"Failed to download Piper voice {resolved!r}: {exc}"
                ) from exc
            logger.info("piper voice %r downloaded in %.2fs", resolved, time.perf_counter() - t0)
        if not (model_path.exists() and config_path.exists()):
            raise ValueError(
                f"Download of Piper voice {resolved!r} did not produce {model_path}."
            )
        return model_path

    @staticmethod
    def _read_sample_rate(model_path: Path) -> int:
        config_path = Path(f"{model_path}.json")
        try:
            with config_path.open(encoding="utf-8") as f:
                return int(json.load(f)["audio"]["sample_rate"])
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Unreadable Piper voice config {config_path}: {exc}") from exc

    def installed_voices(self) -> list[str]:
        """Piper ids already present in the voice dir (offline, no network)."""
        return sorted(
            p.stem for p in self.voice_dir.glob("*.onnx") if is_piper_voice_id(p.stem)
        )

    # -- TTSModelLike --------------------------------------------------------

    def get_state_for_audio_prompt(self, voice: str) -> object:
        """Return a loaded ``PiperVoice`` for ``voice`` (the engine caches it)."""
        resolved = self.resolve_voice(voice)
        if resolved == self.default_voice and self._default_state is not None:
            return self._default_state
        model_path = self._ensure_files(resolved)
        logger.info("loading piper voice %r from %s…", resolved, model_path)
        t0 = time.perf_counter()
        state = self._voice_loader(model_path)
        rate = int(cast(_VoiceStateLike, state).config.sample_rate)
        if resolved == self.default_voice:
            self._default_state = state
            self.sample_rate = rate
        logger.info(
            "piper voice %r loaded in %.2fs (sample_rate=%d)",
            resolved,
            time.perf_counter() - t0,
            rate,
        )
        return state

    def sample_rate_for(self, voice: str) -> int:
        """Output sample rate for ``voice`` (Piper tiers are 16 or 22.05 kHz)."""
        resolved = self.resolve_voice(voice)
        if resolved == self.default_voice and self._default_state is not None:
            return int(cast(_VoiceStateLike, self._default_state).config.sample_rate)
        return self._read_sample_rate(self._ensure_files(resolved))

    def generate_audio(self, model_state: object, text_to_generate: str) -> np.ndarray:
        """Synthesize the whole text (one chunk per sentence, concatenated)."""
        chunks = [
            np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)
            for chunk in cast(_VoiceStateLike, model_state).synthesize(text_to_generate)
        ]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)

    def generate_audio_stream(
        self, model_state: object, text_to_generate: str
    ) -> Iterator[np.ndarray]:
        """Yield one float32 array per sentence (Piper's native chunking)."""
        for chunk in cast(_VoiceStateLike, model_state).synthesize(text_to_generate):
            yield np.asarray(chunk.audio_float_array, dtype=np.float32).reshape(-1)


def load_piper_model(config: Config) -> "TTSEngine":
    """Build a :class:`TTSEngine` around Piper (lazy import of ``piper``)."""
    try:
        import piper  # noqa: F401  (optional extra: uv sync --extra piper)
    except ImportError as exc:  # pragma: no cover - depends on install extra
        raise ImportError(
            "The Piper engine is not installed. Install it with: "
            "uv sync --extra piper"
        ) from exc

    from .engine import TTSEngine

    def build() -> PiperTTSModel:
        return PiperTTSModel(config)

    t0 = time.perf_counter()
    model = build()
    logger.info(
        "model loaded: backend=piper voice=%s sample_rate=%d in %.2fs",
        model.default_voice,
        model.sample_rate,
        time.perf_counter() - t0,
    )
    # No voice registry for Piper: it has no prompt-based voice cloning.
    engine = TTSEngine(model, config=config, registry=None, loader=build)
    engine._auto_loaded = True  # owned by this server: shutdown will unload it
    return engine
