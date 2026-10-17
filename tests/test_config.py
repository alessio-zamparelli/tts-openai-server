"""Config parsing tests for the load-shedding knobs."""

from __future__ import annotations

import pytest

from pocket_tts_openai.config import Config


def test_load_shedding_envs():
    cfg = Config.from_env(
        {
            "POCKET_TTS_MAX_WAITING": "8",
            "POCKET_TTS_QUEUE_TIMEOUT_S": "0.25",
        }
    )
    assert cfg.max_waiting == 8
    assert cfg.queue_timeout_s == 0.25


def test_load_shedding_defaults_off():
    cfg = Config.from_env({})
    assert cfg.max_waiting == 0
    assert cfg.queue_timeout_s == 0.0


def test_queue_timeout_accepts_fractional():
    cfg = Config.from_env({"POCKET_TTS_QUEUE_TIMEOUT_S": "0.5"})
    assert cfg.queue_timeout_s == 0.5


def test_load_shedding_non_numeric_rejected():
    with pytest.raises(ValueError, match="POCKET_TTS_MAX_WAITING"):
        Config.from_env({"POCKET_TTS_MAX_WAITING": "lots"})
    with pytest.raises(ValueError, match="POCKET_TTS_QUEUE_TIMEOUT_S"):
        Config.from_env({"POCKET_TTS_QUEUE_TIMEOUT_S": "soon"})


# --- backend selection -------------------------------------------------------

def test_piper_is_the_default_backend():
    cfg = Config.from_env({})
    assert cfg.backend == "piper"
    assert cfg.default_voice == "it_IT-paola-medium"
    assert cfg.voice_map["alloy"] == "it_IT-paola-medium"
    assert cfg.piper_auto_download is True
    # Backend-dependent LRU default is resolved by the engine (None here).
    assert cfg.max_cached_voices is None


def test_pocket_backend_defaults():
    cfg = Config.from_env({"POCKET_TTS_BACKEND": "pocket"})
    assert cfg.backend == "pocket"
    assert cfg.default_voice == "alloy"
    assert cfg.voice_map["alloy"] == "alba"


def test_backend_validation():
    with pytest.raises(ValueError, match="POCKET_TTS_BACKEND"):
        Config.from_env({"POCKET_TTS_BACKEND": "nope"})


def test_voice_map_override_and_piper_envs():
    cfg = Config.from_env(
        {
            "POCKET_TTS_VOICE_MAP": "alloy=en_US-lessac-medium",
            "POCKET_TTS_DEFAULT_VOICE": "en_US-lessac-medium",
            "POCKET_TTS_PIPER_VOICE_DIR": "/tmp/piper",
            "POCKET_TTS_PIPER_AUTO_DOWNLOAD": "0",
            "POCKET_TTS_MAX_CACHED_VOICES": "4",
        }
    )
    assert cfg.voice_map["alloy"] == "en_US-lessac-medium"
    assert cfg.voice_map["echo"] == "it_IT-paola-medium"  # untouched default
    assert cfg.default_voice == "en_US-lessac-medium"
    assert cfg.piper_voice_dir == "/tmp/piper"
    assert cfg.piper_auto_download is False
    assert cfg.max_cached_voices == 4
