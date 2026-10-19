"""Voice catalog + voice cloning: ``GET/POST/DELETE /v1/voices``.

Private extensions (not part of the OpenAI Audio API). Let clients discover
valid ``voice`` values and clone new voices from an uploaded audio prompt.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Response, UploadFile

from ..config import Config
from ..dependencies import get_config, get_engine, require_engine
from ..engine import TTSEngine
from ..errors import (
    OpenAIError,
    conflict,
    invalid_request,
    method_not_allowed,
    not_found,
    not_implemented,
    payload_too_large,
)
from ..voice_registry import validate_voice_name
from ..voices import KYUTAI_CATALOG, VOICE_LANGUAGE, piper_language

# Extensions that pocket-tts can decode as an audio prompt.
_AUDIO_EXTS = (".wav", ".mp3", ".flac")

# HF repo that hosts the Kyutai voice licenses / origins.
VOICES_REPO = "https://huggingface.co/kyutai/tts-voices"

# HF repo that hosts Piper voices (and each voice's MODEL_CARD license).
PIPER_VOICES_REPO = "https://huggingface.co/rhasspy/piper-voices"

router = APIRouter(prefix="/voices", tags=["voices"])


def _builtin_voice_entry(alias: str, target: str, backend: str = "pocket") -> dict:
    if backend == "piper":
        language = piper_language(target)
        license_url = PIPER_VOICES_REPO
    else:
        language = VOICE_LANGUAGE.get(target, "en")
        license_url = VOICES_REPO
    return {
        "id": alias,
        "aliases": [alias, target],
        "source": "builtin",
        "language": language,
        "license": license_url,
        "cached": False,
    }


def _custom_voice_entry(name: str, voice, engine: TTSEngine | None) -> dict:
    return {
        "id": name,
        "aliases": [name],
        "source": "custom",
        "language": voice.language,
        "license": None,
        "cached": bool(engine and name in engine.cached_voices()),
        "safetensors": True,
    }


@router.get("")
def list_voices(
    config: Config = Depends(get_config),
    engine: TTSEngine | None = Depends(get_engine),
) -> dict:
    """Return the full voice catalog: OpenAI aliases, extra catalog names, customs."""
    cached = engine.cached_voices() if engine is not None else frozenset()

    data: list[dict] = []
    seen: set[str] = set()
    # OpenAI aliases first (stable, declared order).
    for alias, target in config.voice_map.items():
        entry = _builtin_voice_entry(alias, target, config.backend)
        entry["cached"] = target in cached
        data.append(entry)
        seen.add(target)
    # Extra catalog voices not already covered by an alias. Piper has no static
    # catalog: list the voice files that are actually installed locally.
    catalog = (
        (engine.installed_voices() if engine is not None else [])
        if config.backend == "piper"
        else sorted(KYUTAI_CATALOG)
    )
    for name in catalog:
        if name in seen and config.backend != "piper":
            continue
        entry = _builtin_voice_entry(name, name, config.backend)
        entry["cached"] = name in cached
        data.append(entry)
    # Custom voices from the registry (pocket-tts only).
    for voice in (engine.registry.all() if engine and engine.registry else []):
        data.append(_custom_voice_entry(voice.name, voice, engine))

    return {
        "object": "list",
        "data": data,
        "backend": config.backend,
        "clone_supported": bool(engine and engine.supports_cloning),
    }


def _collision(name: str, config: Config, engine: TTSEngine) -> str | None:
    """Return a human-readable conflict source if ``name`` is taken, else None."""
    if name in config.voice_map:
        return f"'{name}' is a built-in OpenAI alias"
    if name in KYUTAI_CATALOG:
        return f"'{name}' is a built-in Kyutai catalog voice"
    if engine.registry and name in engine.registry.names():
        return f"'{name}' already exists as a custom voice"
    return None


@router.post("", status_code=201)
def create_voice(
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    config: Config = Depends(get_config),
    engine: TTSEngine = Depends(require_engine),
) -> dict:
    """Clone a voice from an uploaded audio prompt and persist it."""
    if not engine.supports_cloning:
        raise not_implemented(
            f"Voice cloning is not supported by the {engine.model_name} backend. "
            "Run the server with POCKET_TTS_BACKEND=pocket to clone from an audio prompt."
        )
    if engine.registry is None:  # pragma: no cover - supports_cloning implies registry
        raise invalid_request("voice registry is not configured")

    error = validate_voice_name(name)
    if error:
        raise invalid_request(error)

    source = _collision(name, config, engine)
    if source:
        raise conflict(f"Cannot create voice: {source}.")

    # Extension decides the audio format (pocket-tts decodes by extension).
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _AUDIO_EXTS:
        raise invalid_request(
            f"Unsupported file type {suffix or '(none)'!r}. "
            f"Supported: {', '.join(_AUDIO_EXTS)}."
        )

    limit = config.max_upload_mb * 1024 * 1024
    # Write the upload up to limit+1 bytes so we can reject oversize cleanly.
    with tempfile.NamedTemporaryFile(
        prefix=f"clone-{name}-", suffix=suffix, delete=False, dir=engine.registry.directory
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            total = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise payload_too_large(
                        f"Upload exceeds the {config.max_upload_mb} MB limit."
                    )
                tmp.write(chunk)
        finally:
            file.file.close()

    try:
        engine.clone_voice(name, tmp_path)
    except OpenAIError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise invalid_request(f"Failed to encode voice prompt: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    voice = engine.registry.add(name, language=language)
    return _custom_voice_entry(voice.name, voice, engine)


@router.delete("/{name}")
def delete_voice(
    name: str,
    config: Config = Depends(get_config),
    engine: TTSEngine = Depends(require_engine),
) -> Response:
    """Remove a custom voice (file + registry entry). Builtin voices are 405."""
    builtin = name in config.voice_map or name in KYUTAI_CATALOG
    if config.backend == "piper":
        builtin = builtin or name in engine.installed_voices()
    if builtin:
        raise method_not_allowed(f"'{name}' is a built-in voice and cannot be deleted.")
    if engine.registry is None:
        # Piper has no custom-voice registry at all.
        raise not_found(f"Unknown custom voice '{name}'.")

    voice = engine.registry.get(name)
    if voice is None:
        raise not_found(f"Unknown custom voice '{name}'.")

    engine.registry.remove(name)
    engine.evict_voice(name)
    engine.registry.path_for(name).unlink(missing_ok=True)
    return Response(status_code=204)
