"""Server configuration from environment variables.

All settings have sane defaults; every knob is optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .voices import (
    backend_default_aliases,
    backend_default_voice,
)

BACKENDS = ("piper", "pocket")


def _split_voice_map(raw: str) -> dict[str, str]:
    """Parse ``alloy=alba,coral=giovanni`` into a dict."""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"TTS_VOICE_MAP entry must be `alias=voice`, got: {pair!r}")
        alias, voice = pair.split("=", 1)
        out[alias.strip()] = voice.strip()
    return out


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got: {raw!r}") from None


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got: {raw!r}") from None


def _env_float_optional(env: dict[str, str], key: str) -> float | None:
    """Parse an optional float: empty/unset means ``None`` (backend default)."""
    raw = env.get(key, "")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got: {raw!r}") from None


def _env_bool(env: dict[str, str], key: str, default: bool) -> bool:
    raw = env.get(key, "")
    if not raw:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


LOG_LEVELS = ("debug", "info", "warning", "error", "critical")


@dataclass(frozen=True)
class Config:
    backend: str = "piper"
    host: str = "127.0.0.1"
    port: int = 8000
    log_level: str = "info"  # tts_openai + uvicorn log level: debug/info/warning/error/critical
    log_file: str = ""  # optional rotating log file; console logging remains enabled
    # Pocket-specific: Piper takes its language from the voice id instead.
    language: str = "english"
    default_voice: str = ""  # empty = backend default (piper: it_IT-paola-medium)
    voice_map: dict[str, str] = field(default_factory=dict)  # empty = backend defaults
    api_key: str | None = None
    warmup_voices: tuple[str, ...] = ()
    quantize: bool = False
    # None = backend default (piper: 2 loaded ONNX sessions, pocket: 32 states).
    max_cached_voices: int | None = None
    cache_dir: str = ""  # empty = pocket-tts default (~/.cache/pocket_tts)
    piper_voice_dir: str = ""  # empty = ~/.cache/piper
    piper_auto_download: bool = True
    max_upload_mb: int = 25
    idle_unload_s: int = 300  # evict the model to free RAM after this much idle; 0 = off
    idle_poll_s: int = 30  # watchdog cadence (only when idle_unload_s > 0)
    max_waiting: int = 0  # 429 above this many in-flight requests; 0 = unlimited
    queue_timeout_s: float = 0.0  # max seconds to wait for the gen lock; 0 = wait forever
    # Wyoming TTS transport (opt-in; unauthenticated by protocol design).
    wyoming_enabled: bool = False
    wyoming_host: str = "0.0.0.0"
    wyoming_port: int = 10200
    wyoming_samples_per_chunk: int = 1024  # mono s16 samples per audio-chunk
    wyoming_sentence_silence: float = 0.0  # seconds of silence between sentences
    wyoming_auto_punctuation: str = ".?!。？！．؟"  # empty = disabled
    wyoming_zeroconf: bool = False  # announce _wyoming._tcp.local.
    wyoming_name: str = "tts-openai"  # mDNS instance name
    wyoming_ready_timeout_s: float = 60.0  # wait for model load before erroring
    # Piper synthesis knobs (HA add-on parity); None/empty = voice defaults.
    piper_speaker: str = ""
    piper_length_scale: float | None = None
    piper_noise_scale: float | None = None
    piper_noise_w: float | None = None

    def __post_init__(self) -> None:
        backend = self.backend.strip().lower()
        if backend not in BACKENDS:
            raise ValueError(
                f"TTS_BACKEND must be one of {', '.join(BACKENDS)}, got: {self.backend!r}"
            )
        object.__setattr__(self, "backend", backend)
        log_level = self.log_level.strip().lower()
        if log_level not in LOG_LEVELS:
            raise ValueError(
                f"TTS_LOG_LEVEL must be one of {', '.join(LOG_LEVELS)}, got: {self.log_level!r}"
            )
        object.__setattr__(self, "log_level", log_level)
        if not self.voice_map:
            object.__setattr__(self, "voice_map", backend_default_aliases(backend))
        if not self.default_voice:
            object.__setattr__(self, "default_voice", backend_default_voice(backend))
        if self.wyoming_samples_per_chunk <= 0:
            raise ValueError(
                "TTS_WYOMING_SAMPLES_PER_CHUNK must be greater than 0, "
                f"got: {self.wyoming_samples_per_chunk}"
            )
        if self.wyoming_sentence_silence < 0:
            raise ValueError(
                "TTS_WYOMING_SENTENCE_SILENCE must be >= 0, "
                f"got: {self.wyoming_sentence_silence}"
            )
        if not (0 <= self.wyoming_port <= 65535):
            raise ValueError(
                f"TTS_WYOMING_PORT must be between 0 and 65535, got: {self.wyoming_port}"
            )

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        env = dict(os.environ if env is None else env)
        backend = env.get("TTS_BACKEND", "piper").strip().lower()
        voice_map = backend_default_aliases(backend) if backend in BACKENDS else {}
        if raw := env.get("TTS_VOICE_MAP", ""):
            voice_map.update(_split_voice_map(raw))
        warmup = tuple(
            v.strip() for v in env.get("TTS_WARMUP_VOICES", "").split(",") if v.strip()
        )
        max_cached = (
            _env_int(env, "TTS_MAX_CACHED_VOICES", 32)
            if "TTS_MAX_CACHED_VOICES" in env and env["TTS_MAX_CACHED_VOICES"]
            else None
        )
        return cls(
            backend=backend,
            host=env.get("TTS_HOST", "127.0.0.1"),
            port=_env_int(env, "TTS_PORT", 8000),
            log_level=env.get("TTS_LOG_LEVEL", "info"),
            log_file=env.get("TTS_LOG_FILE", ""),
            language=env.get("TTS_LANGUAGE", "english"),
            default_voice=env.get("TTS_DEFAULT_VOICE", ""),
            voice_map=voice_map,
            api_key=env.get("TTS_API_KEY") or None,
            warmup_voices=warmup,
            quantize=env.get("TTS_QUANTIZE", "").lower() in ("1", "true", "yes"),
            max_cached_voices=max_cached,
            cache_dir=env.get("TTS_CACHE_DIR", ""),
            piper_voice_dir=env.get("TTS_PIPER_VOICE_DIR", ""),
            piper_auto_download=_env_bool(env, "TTS_PIPER_AUTO_DOWNLOAD", True),
            max_upload_mb=_env_int(env, "TTS_MAX_UPLOAD_MB", 25),
            idle_unload_s=_env_int(env, "TTS_IDLE_UNLOAD_S", 300),
            idle_poll_s=_env_int(env, "TTS_IDLE_POLL_S", 30),
            max_waiting=_env_int(env, "TTS_MAX_WAITING", 0),
            queue_timeout_s=_env_float(env, "TTS_QUEUE_TIMEOUT_S", 0.0),
            wyoming_enabled=_env_bool(env, "TTS_WYOMING_ENABLED", False),
            wyoming_host=env.get("TTS_WYOMING_HOST", "0.0.0.0"),
            wyoming_port=_env_int(env, "TTS_WYOMING_PORT", 10200),
            wyoming_samples_per_chunk=_env_int(
                env, "TTS_WYOMING_SAMPLES_PER_CHUNK", 1024
            ),
            wyoming_sentence_silence=_env_float(
                env, "TTS_WYOMING_SENTENCE_SILENCE", 0.0
            ),
            wyoming_auto_punctuation=env.get(
                "TTS_WYOMING_AUTO_PUNCTUATION", ".?!。？！．؟"
            ),
            wyoming_zeroconf=_env_bool(env, "TTS_WYOMING_ZEROCONF", False),
            wyoming_name=env.get("TTS_WYOMING_NAME", "tts-openai"),
            wyoming_ready_timeout_s=_env_float(
                env, "TTS_WYOMING_READY_TIMEOUT_S", 60.0
            ),
            piper_speaker=env.get("TTS_PIPER_SPEAKER", ""),
            piper_length_scale=_env_float_optional(env, "TTS_PIPER_LENGTH_SCALE"),
            piper_noise_scale=_env_float_optional(env, "TTS_PIPER_NOISE_SCALE"),
            piper_noise_w=_env_float_optional(env, "TTS_PIPER_NOISE_W"),
        )
