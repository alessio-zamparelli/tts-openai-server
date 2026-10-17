"""Piper adapter tests (offline: fake PiperVoice + fake downloader)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import TTSEngine
from pocket_tts_openai.piper_backend import PiperTTSModel
from pocket_tts_openai.voices import (
    PIPER_DEFAULT_VOICE,
    resolve_piper_voice,
)

from tests.conftest import (
    FakePiperVoice,
    fake_piper_download,
    fake_piper_loader,
    piper_rate_for,
)


def _model(tmp_path: Path, **kwargs) -> PiperTTSModel:
    cfg = Config(
        backend="piper",
        piper_voice_dir=str(tmp_path / "piper"),
        default_voice=PIPER_DEFAULT_VOICE,
        **kwargs,
    )
    return PiperTTSModel(
        cfg,
        voice_loader=fake_piper_loader,
        downloader=fake_piper_download,
    )


def test_default_voice_is_downloaded_and_loaded(tmp_path):
    model = _model(tmp_path)
    assert model.default_voice == PIPER_DEFAULT_VOICE
    assert model.sample_rate == 22050
    assert model.language == "it"
    assert (tmp_path / "piper" / f"{PIPER_DEFAULT_VOICE}.onnx").exists()


def test_all_openai_aliases_map_to_piper_default():
    voice_map = Config(backend="piper").voice_map
    for alias in ("alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral"):
        assert resolve_piper_voice(alias, voice_map) == PIPER_DEFAULT_VOICE


def test_piper_ids_and_paths_pass_through(tmp_path):
    voice_map = Config(backend="piper").voice_map
    assert resolve_piper_voice("en_US-lessac-medium", voice_map) == "en_US-lessac-medium"
    local = tmp_path / "custom.onnx"
    assert resolve_piper_voice(str(local), voice_map) == str(local)


def test_unknown_voice_rejected_with_helpful_message():
    voice_map = Config(backend="piper").voice_map
    with pytest.raises(ValueError, match="it_IT-paola-medium"):
        resolve_piper_voice("not-a-voice", voice_map)


def test_missing_voice_without_auto_download_errors(tmp_path):
    cfg = Config(
        backend="piper",
        piper_voice_dir=str(tmp_path / "piper"),
        default_voice=PIPER_DEFAULT_VOICE,
        piper_auto_download=False,
    )
    with pytest.raises(ValueError, match="not installed"):
        PiperTTSModel(
            cfg, voice_loader=fake_piper_loader, downloader=fake_piper_download
        )


def test_generate_and_stream_shapes(tmp_path):
    model = _model(tmp_path)
    state = model.get_state_for_audio_prompt("alloy")

    whole = model.generate_audio(state, "Ciao. Mondo.")
    assert whole.dtype == np.float32
    assert len(whole) == 22050 // 2  # two 0.25 s chunks concatenated

    chunks = list(model.generate_audio_stream(state, "Ciao. Mondo."))
    assert len(chunks) == 2
    assert sum(len(c) for c in chunks) == 22050 // 2


def test_local_onnx_path_voice(tmp_path):
    model_dir = tmp_path / "piper"
    model_dir.mkdir(parents=True)
    local = model_dir / "custom.onnx"
    fake_piper_download("custom", model_dir)  # writes custom.onnx + custom.onnx.json
    cfg = Config(backend="piper", default_voice=str(local))
    model = PiperTTSModel(
        cfg, voice_loader=fake_piper_loader, downloader=fake_piper_download
    )
    assert model.default_voice == str(local)
    assert model.sample_rate == piper_rate_for(local)
    state = model.get_state_for_audio_prompt(str(local))
    assert isinstance(state, FakePiperVoice)


def test_engine_supports_multiple_sample_rates(tmp_path):
    cfg = Config(
        backend="piper",
        piper_voice_dir=str(tmp_path / "piper"),
        default_voice=PIPER_DEFAULT_VOICE,
    )
    model = PiperTTSModel(
        cfg, voice_loader=fake_piper_loader, downloader=fake_piper_download
    )
    engine = TTSEngine(model, config=cfg)

    assert engine.default_voice == PIPER_DEFAULT_VOICE
    assert engine.sample_rate_for("alloy") == 22050
    assert engine.sample_rate_for("en_US-lessac-low") == 16000
    assert engine.supports_cloning is False
    # Piper's LRU default is 2 whole ONNX sessions, not pocket's 32 snapshots.
    assert engine._max_cached_voices == 2

    pcm = engine.generate_pcm("Ciao", "alloy")
    assert len(pcm) == 2 * (22050 // 2)  # 0.5 s * 2 bytes/sample
    assert engine.installed_voices() == sorted(
        {PIPER_DEFAULT_VOICE, "en_US-lessac-low"}
    )
