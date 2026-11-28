"""Config parsing tests for the load-shedding knobs."""

from __future__ import annotations

import pytest

from tts_openai.config import Config


def test_load_shedding_envs():
    cfg = Config.from_env(
        {
            "TTS_MAX_WAITING": "8",
            "TTS_QUEUE_TIMEOUT_S": "0.25",
        }
    )
    assert cfg.max_waiting == 8
    assert cfg.queue_timeout_s == 0.25


def test_log_file_env():
    cfg = Config.from_env({"TTS_LOG_FILE": "/tmp/server.log"})
    assert cfg.log_file == "/tmp/server.log"


def test_load_shedding_defaults_off():
    cfg = Config.from_env({})
    assert cfg.max_waiting == 0
    assert cfg.queue_timeout_s == 0.0


def test_queue_timeout_accepts_fractional():
    cfg = Config.from_env({"TTS_QUEUE_TIMEOUT_S": "0.5"})
    assert cfg.queue_timeout_s == 0.5


def test_load_shedding_non_numeric_rejected():
    with pytest.raises(ValueError, match="TTS_MAX_WAITING"):
        Config.from_env({"TTS_MAX_WAITING": "lots"})
    with pytest.raises(ValueError, match="TTS_QUEUE_TIMEOUT_S"):
        Config.from_env({"TTS_QUEUE_TIMEOUT_S": "soon"})


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
    cfg = Config.from_env({"TTS_BACKEND": "pocket"})
    assert cfg.backend == "pocket"
    assert cfg.default_voice == "alloy"
    assert cfg.voice_map["alloy"] == "alba"


def test_backend_validation():
    with pytest.raises(ValueError, match="TTS_BACKEND"):
        Config.from_env({"TTS_BACKEND": "nope"})


def test_voice_map_override_and_piper_envs():
    cfg = Config.from_env(
        {
            "TTS_VOICE_MAP": "alloy=en_US-lessac-medium",
            "TTS_DEFAULT_VOICE": "en_US-lessac-medium",
            "TTS_PIPER_VOICE_DIR": "/tmp/piper",
            "TTS_PIPER_AUTO_DOWNLOAD": "0",
            "TTS_MAX_CACHED_VOICES": "4",
        }
    )
    assert cfg.voice_map["alloy"] == "en_US-lessac-medium"
    assert cfg.voice_map["echo"] == "it_IT-paola-medium"  # untouched default
    assert cfg.default_voice == "en_US-lessac-medium"
    assert cfg.piper_voice_dir == "/tmp/piper"
    assert cfg.piper_auto_download is False
    assert cfg.max_cached_voices == 4


# --- Wyoming / Piper synthesis options ---------------------------------------


def test_wyoming_defaults_disabled():
    cfg = Config.from_env({})
    assert cfg.wyoming_enabled is False
    assert cfg.wyoming_host == "0.0.0.0"
    assert cfg.wyoming_port == 10200
    assert cfg.wyoming_samples_per_chunk == 1024
    assert cfg.wyoming_sentence_silence == 0.0
    assert cfg.wyoming_auto_punctuation == ".?!。？！．؟"
    assert cfg.wyoming_zeroconf is False
    assert cfg.wyoming_name == "tts-openai"
    assert cfg.wyoming_ready_timeout_s == 60.0


def test_wyoming_envs_parse():
    cfg = Config.from_env(
        {
            "TTS_WYOMING_ENABLED": "1",
            "TTS_WYOMING_HOST": "127.0.0.1",
            "TTS_WYOMING_PORT": "10300",
            "TTS_WYOMING_SAMPLES_PER_CHUNK": "512",
            "TTS_WYOMING_SENTENCE_SILENCE": "0.25",
            "TTS_WYOMING_AUTO_PUNCTUATION": "",
            "TTS_WYOMING_ZEROCONF": "yes",
            "TTS_WYOMING_NAME": "living-room",
            "TTS_WYOMING_READY_TIMEOUT_S": "5",
        }
    )
    assert cfg.wyoming_enabled is True
    assert cfg.wyoming_host == "127.0.0.1"
    assert cfg.wyoming_port == 10300
    assert cfg.wyoming_samples_per_chunk == 512
    assert cfg.wyoming_sentence_silence == 0.25
    assert cfg.wyoming_auto_punctuation == ""
    assert cfg.wyoming_zeroconf is True
    assert cfg.wyoming_name == "living-room"
    assert cfg.wyoming_ready_timeout_s == 5.0


def test_wyoming_invalid_values_rejected():
    with pytest.raises(ValueError, match="SAMPLES_PER_CHUNK"):
        Config(wyoming_samples_per_chunk=0)
    with pytest.raises(ValueError, match="SENTENCE_SILENCE"):
        Config(wyoming_sentence_silence=-1)
    with pytest.raises(ValueError, match="WYOMING_PORT"):
        Config(wyoming_port=70000)


def test_piper_synthesis_options_from_env():
    cfg = Config.from_env(
        {
            "TTS_PIPER_SPEAKER": "alice",
            "TTS_PIPER_LENGTH_SCALE": "0.8",
            "TTS_PIPER_NOISE_SCALE": "0.5",
            "TTS_PIPER_NOISE_W": "0.3",
        }
    )
    assert cfg.piper_speaker == "alice"
    assert cfg.piper_length_scale == 0.8
    assert cfg.piper_noise_scale == 0.5
    assert cfg.piper_noise_w == 0.3


def test_piper_synthesis_options_default_unset():
    cfg = Config.from_env({})
    assert cfg.piper_speaker == ""
    assert cfg.piper_length_scale is None
    assert cfg.piper_noise_scale is None
    assert cfg.piper_noise_w is None


# --- log level --------------------------------------------------------------

def test_log_level_defaults_to_info():
    assert Config().log_level == "info"
    assert Config.from_env({}).log_level == "info"


def test_log_level_from_env():
    assert Config.from_env({"TTS_LOG_LEVEL": "debug"}).log_level == "debug"
    # whitespace/case tolerated, normalized
    assert Config.from_env({"TTS_LOG_LEVEL": "  Warning "}).log_level == "warning"


def test_log_level_invalid_rejected():
    with pytest.raises(ValueError, match="TTS_LOG_LEVEL"):
        Config.from_env({"TTS_LOG_LEVEL": "verbose"})


# --- deprecated POCKET_TTS_* aliases -----------------------------------------


def test_legacy_envs_are_mapped():
    cfg = Config.from_env(
        {
            "POCKET_TTS_LOG_FILE": "/tmp/server.log",
            "POCKET_TTS_MAX_WAITING": "8",
            "POCKET_TTS_IDLE_UNLOAD_S": "0",
        }
    )
    assert cfg.log_file == "/tmp/server.log"
    assert cfg.max_waiting == 8
    assert cfg.idle_unload_s == 0


def test_new_env_precedence_over_legacy(caplog):
    cfg = Config.from_env(
        {
            "TTS_BACKEND": "piper",
            "POCKET_TTS_BACKEND": "pocket",
            "TTS_MAX_WAITING": "3",
            "POCKET_TTS_MAX_WAITING": "9",
        }
    )
    assert cfg.backend == "piper"
    assert cfg.max_waiting == 3
    assert "POCKET_TTS_BACKEND is deprecated and ignored because TTS_BACKEND is set" in caplog.text
    assert "POCKET_TTS_MAX_WAITING is deprecated and ignored because TTS_MAX_WAITING is set" in caplog.text


def test_legacy_backend_selects_pocket():
    cfg = Config.from_env({"POCKET_TTS_BACKEND": "pocket"})
    assert cfg.backend == "pocket"
    assert cfg.default_voice == "alloy"


def test_legacy_typed_values_still_validated_under_new_name():
    with pytest.raises(ValueError, match="TTS_MAX_WAITING"):
        Config.from_env({"POCKET_TTS_MAX_WAITING": "lots"})


def test_legacy_env_warns_with_replacement(caplog):
    Config.from_env({"POCKET_TTS_LOG_FILE": "/tmp/x"})
    assert (
        "POCKET_TTS_LOG_FILE is deprecated; use TTS_LOG_FILE instead" in caplog.text
    )


def test_unknown_legacy_env_warns(caplog):
    Config.from_env({"POCKET_TTS_NOPE": "1"})
    assert (
        "POCKET_TTS_NOPE is deprecated and has no TTS_ equivalent; remove it"
        in caplog.text
    )


def test_no_warning_without_legacy_envs(caplog):
    Config.from_env({"TTS_LOG_FILE": "/tmp/x"})
    assert "deprecated" not in caplog.text
