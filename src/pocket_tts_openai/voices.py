"""Voice resolution: OpenAI aliases -> backend voice names, plus passthrough.

Pocket TTS accepts as voice:
  - a catalog name (e.g. ``alba``, ``giovanni``)
  - a local audio/safetensors path
  - an ``https://`` URL or ``hf://<repo>/<path>[@rev]``

Piper accepts as voice:
  - a Piper voice id (e.g. ``it_IT-paola-medium``)
  - a local ``.onnx`` model path

We add OpenAI-compatible aliases on top and let anything else pass through
(forward-compat with future catalog entries and custom cloned voices).
"""

from __future__ import annotations

import re

# OpenAI voice alias -> Kyutai catalog voice.
# `coral` is a non-OpenAI convenience alias pointing at the Italian voice.
DEFAULT_VOICE_ALIASES: dict[str, str] = {
    "alloy": "alba",        # en
    "echo": "charles",      # en
    "fable": "eponine",     # en (British)
    "onyx": "bill_boerst",  # en
    "nova": "eve",          # en
    "shimmer": "fantine",   # en
    "coral": "giovanni",    # it — default Italian voice
}

# Known Kyutai catalog voices (pass-through, no alias needed).
KYUTAI_CATALOG: set[str] = {
    # en
    "alba", "anna", "azelma", "bill_boerst", "caro_davy", "charles", "cosette",
    "eponine", "eve", "fantine", "george", "jane", "javert", "jean", "marius",
    "mary", "michael", "paul", "peter_yearsley", "stuart_bell", "vera",
    # other languages
    "giovanni",   # it
    "lola",       # es
    "juergen",    # de
    "rafael",     # pt
    "estelle",    # fr
}

# Language tag per catalog voice (informational; shown in GET /v1/voices).
# Anything not listed here defaults to "en".
VOICE_LANGUAGE: dict[str, str] = {
    "giovanni": "it",
    "lola": "es",
    "juergen": "de",
    "rafael": "pt",
    "estelle": "fr",
}

_PASSTHROUGH_PREFIXES = ("hf://", "https://", "http://")


def is_passthrough_voice(voice: str) -> bool:
    """True for URLs, hf:// paths and filesystem paths (incl. .safetensors)."""
    v = voice.strip()
    return (
        v.startswith(_PASSTHROUGH_PREFIXES)
        or v.startswith(("/", "./", "../", "~/"))
        or v.lower().endswith((".wav", ".mp3", ".flac", ".safetensors"))
    )


def resolve_voice(requested: str, voice_map: dict[str, str]) -> str:
    """Map an alias to a Kyutai voice; catalog names and paths/URLs pass through.

    Unknown bare names are rejected so client typos fail fast with a helpful
    message instead of a cryptic model-side error.
    """
    v = requested.strip()
    if v in voice_map:
        return voice_map[v]
    if v in KYUTAI_CATALOG or is_passthrough_voice(v):
        return v
    catalog = ", ".join(sorted(KYUTAI_CATALOG))
    raise ValueError(
        f"Unknown voice {requested!r}. Use an OpenAI alias, a catalog voice "
        f"({catalog}), a local audio/safetensors path, an https:// URL or an "
        "hf:// reference."
    )


# -- Piper voices -----------------------------------------------------------

PIPER_DEFAULT_VOICE = "it_IT-paola-medium"

# OpenAI aliases on the Piper backend all resolve to the server default. Keeping
# the mapping explicit (rather than special-casing alias lookups) means
# POCKET_TTS_VOICE_MAP can override individual names as usual.
PIPER_DEFAULT_ALIASES: dict[str, str] = {
    alias: PIPER_DEFAULT_VOICE for alias in DEFAULT_VOICE_ALIASES
}

# Official Piper voice id format: <lang>_<REGION>-<name>-<quality>.
PIPER_VOICE_RE = re.compile(
    r"^[a-z]{2}_[A-Z]{2}-[a-z0-9_]+-(?:x_low|low|medium|high)$"
)


def backend_default_aliases(backend: str) -> dict[str, str]:
    """Default alias map for a backend (``pocket`` or ``piper``)."""
    return dict(PIPER_DEFAULT_ALIASES if backend == "piper" else DEFAULT_VOICE_ALIASES)


def backend_default_voice(backend: str) -> str:
    """Default ``voice`` when the request omits one."""
    return PIPER_DEFAULT_VOICE if backend == "piper" else "alloy"


def is_piper_voice_id(voice: str) -> bool:
    """True for a canonical Piper voice id such as ``it_IT-paola-medium``."""
    return bool(PIPER_VOICE_RE.match(voice.strip()))


def is_piper_path(voice: str) -> bool:
    """True for a local Piper ONNX model path."""
    return voice.strip().lower().endswith(".onnx")


def piper_language(voice: str) -> str:
    """Best-effort language tag for a Piper voice id/path."""
    name = voice.strip().rsplit("/", 1)[-1]
    if name.endswith(".onnx"):
        name = name[: -len(".onnx")]
    return name.split("_", 1)[0] if "_" in name else "en"


def resolve_piper_voice(requested: str, voice_map: dict[str, str]) -> str:
    """Map an alias to a Piper voice id; ids and local ``.onnx`` paths pass through.

    Unknown names fail fast with a helpful message (and without touching the
    network / filesystem, so callers can validate before downloading).
    """
    v = requested.strip()
    if v in voice_map:
        return voice_map[v]
    if is_piper_voice_id(v) or is_piper_path(v):
        return v
    aliases = ", ".join(sorted(voice_map))
    raise ValueError(
        f"Unknown voice {requested!r}. Use an OpenAI alias ({aliases}), a Piper "
        f"voice id such as {PIPER_DEFAULT_VOICE!r}, or a local .onnx path."
    )
