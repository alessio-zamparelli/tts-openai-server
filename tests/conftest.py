"""Shared fixtures: a fake pocket-tts model + a real TTSEngine around it."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import TTSEngine
from pocket_tts_openai.piper_backend import PiperTTSModel
from pocket_tts_openai.server import create_app
from pocket_tts_openai.voice_registry import VoiceRegistry


def fake_export_state(state: object, dest: str | Path) -> None:
    """Stand-in for ``pocket_tts.export_model_state``: materialize a file so
    registry paths exist (DELETE cleanup / GET safetensors flags behave like
    production)."""
    del state
    Path(dest).write_bytes(b"fake-safetensors")


class FakeTTSModel:
    """Duck-typed pocket-tts model: 24 kHz, 1 s of silence per request.

    Simulates real cost: ``get_state_for_audio_prompt`` is slow (voice
    encoding) and ``generate_audio`` holds ``gen_lock`` for a configurable
    time so the engine's serialization behavior is observable.
    """

    sample_rate = 24000

    def __init__(self, *, encode_seconds: float = 0.02, gen_seconds: float = 0.05):
        self._encode_seconds = encode_seconds
        self._gen_seconds = gen_seconds
        self.encode_calls: list[str] = []
        self.generate_calls: list[str] = []
        self._inside_generate = 0
        self.overlap_seen = False
        self._overlap_lock = threading.Lock()
        self.stream_chunks = 4  # 4 x 0.25 s = 1.0 s, matching generate_audio

    def get_state_for_audio_prompt(self, voice: str) -> tuple[str, str]:
        time.sleep(self._encode_seconds)
        self.encode_calls.append(voice)
        return ("state", voice)

    def _generate_impl(self, text_to_generate: str):
        with self._overlap_lock:
            self._inside_generate += 1
            if self._inside_generate > 1:
                self.overlap_seen = True
        try:
            time.sleep(self._gen_seconds)
        finally:
            with self._overlap_lock:
                self._inside_generate -= 1
        self.generate_calls.append(text_to_generate)

    def generate_audio(self, model_state: object, text_to_generate: str) -> np.ndarray:
        self._generate_impl(text_to_generate)
        return np.zeros(self.sample_rate, dtype=np.float32)

    def generate_audio_stream(self, model_state: object, text_to_generate: str):
        """Yield ``n * chunk_seconds`` of silence, sleeping per chunk so the
        stream serialization / disconnect behavior is observable."""
        self._generate_impl(text_to_generate)
        chunk = self.sample_rate // 4  # 0.25 s per chunk
        for _ in range(self.stream_chunks):
            time.sleep(0.01)
            yield np.zeros(chunk, dtype=np.float32)


@pytest.fixture
def fake_model() -> FakeTTSModel:
    return FakeTTSModel()


@pytest.fixture
def config() -> Config:
    # The shared suite exercises the original pocket-tts semantics (Kyutai
    # catalog + aliases). Piper has its own fixtures in test_piper_*.py.
    return Config(backend="pocket")


@pytest.fixture
def registry(tmp_path) -> VoiceRegistry:
    return VoiceRegistry(directory=tmp_path / "voices")


@pytest.fixture
def engine(fake_model: FakeTTSModel, config: Config, registry: VoiceRegistry) -> TTSEngine:
    return TTSEngine(
        fake_model,
        config=config,
        registry=registry,
        export_state=fake_export_state,
    )


@pytest.fixture
def client(engine: TTSEngine, config: Config):
    """TestClient with lifespan active so app.state.engine is set on startup."""
    from fastapi.testclient import TestClient

    app = create_app(config, engine)
    with TestClient(app) as c:
        yield c


# -- Piper fake backend ------------------------------------------------------


class _FakePiperConfig:
    def __init__(self, sample_rate: int):
        self.sample_rate = sample_rate


class _FakePiperChunk:
    def __init__(self, audio: np.ndarray):
        self.audio_float_array = audio


class FakePiperVoice:
    """Duck-typed PiperVoice: two 0.25 s chunks (one per fake sentence)."""

    def __init__(self, sample_rate: int):
        self.config = _FakePiperConfig(sample_rate)

    def synthesize(self, text: str):
        # Two chunks totaling exactly 0.5 s (rate is odd for 22.05 kHz).
        total = self.config.sample_rate // 2
        first = total - self.config.sample_rate // 4
        yield _FakePiperChunk(np.zeros(first, dtype=np.float32))
        yield _FakePiperChunk(
            np.zeros(self.config.sample_rate // 4, dtype=np.float32)
        )


def piper_rate_for(path: Path) -> int:
    """16 kHz for low/x_low tiers, 22.05 kHz otherwise (mirrors Piper)."""
    return 16000 if path.stem.endswith(("-low", "-x_low")) else 22050


def fake_piper_loader(path: Path):
    return FakePiperVoice(piper_rate_for(path))


def fake_piper_download(voice: str, directory: Path) -> None:
    """Materialize the .onnx/.onnx.json pair a real download would produce."""
    rate = 16000 if voice.endswith(("-low", "-x_low")) else 22050
    (directory / f"{voice}.onnx").write_bytes(b"fake-piper-onnx")
    (directory / f"{voice}.onnx.json").write_text(
        f'{{"audio": {{"sample_rate": {rate}}}}}', encoding="utf-8"
    )


@pytest.fixture
def piper_config(tmp_path: Path) -> Config:
    return Config(
        backend="piper",
        piper_voice_dir=str(tmp_path / "piper"),
        default_voice="it_IT-paola-medium",
    )


@pytest.fixture
def piper_engine(piper_config: Config) -> TTSEngine:
    model = PiperTTSModel(
        piper_config,
        voice_loader=fake_piper_loader,
        downloader=fake_piper_download,
    )
    return TTSEngine(model, config=piper_config)


@pytest.fixture
def piper_client(piper_config: Config, piper_engine: TTSEngine):
    from fastapi.testclient import TestClient

    app = create_app(piper_config, piper_engine)
    with TestClient(app) as c:
        yield c
