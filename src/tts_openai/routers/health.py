"""``GET /health``: liveness plus engine stats (public, no API key)."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends

from ..config import Config
from ..dependencies import get_config, get_engine, get_wyoming
from ..engine import TTSEngine
from ..wyoming_server import WyomingService

router = APIRouter(tags=["health"])


@router.get("/health")
def health(
    config: Config = Depends(get_config),
    engine: TTSEngine | None = Depends(get_engine),
    wyoming: WyomingService | None = Depends(get_wyoming),
) -> dict:
    """Liveness + engine stats. Status is 'loading' until the model is ready.
    After an idle eviction ``loaded`` is False (model waking on next request).
    Health probes deliberately do NOT touch the engine's idle timer.
    """
    wyoming_info = {
        "enabled": config.wyoming_enabled,
        "listening": wyoming is not None,
        "port": wyoming.port if wyoming is not None else config.wyoming_port,
    }
    if engine is None:
        return {
            "status": "loading",
            "model": "pocket-tts" if config.backend == "pocket" else config.backend,
            "wyoming": wyoming_info,
        }
    stats = engine.stats
    return {
        "status": "ok",
        "model": engine.model_name,
        "backend": engine.backend,
        "default_voice": engine.default_voice,
        "sample_rate": engine.sample_rate,
        "language": engine.language,
        "loaded": engine.loaded,
        "idle_unload_s": engine._config.idle_unload_s,
        "last_request_age_s": round(time.monotonic() - engine._last_activity, 1),
        "unloads": stats.unloads,
        "reloads": stats.reloads,
        "requests": stats.requests,
        "avg_rtf": round(stats.avg_rtf or 0.0, 4),
        "queue_depth": stats.waiting,
        "wyoming": wyoming_info,
    }
