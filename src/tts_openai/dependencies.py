"""Shared FastAPI dependencies.

Accessors keep route code independent of ``Request`` plumbing while
``app.state`` stays the single source of truth for the configured engine and
config (background loading and test injection both write there).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Depends, Header, Request

from .config import Config
from .engine import TTSEngine
from .errors import invalid_api_key, unavailable

if TYPE_CHECKING:  # pragma: no cover
    from .wyoming_server import WyomingService


def get_config(request: Request) -> Config:
    """Return the immutable :class:`Config` stored by ``create_app``."""
    return request.app.state.config


def get_engine(request: Request) -> TTSEngine | None:
    """Return the engine, or ``None`` while the background loader is running."""
    return getattr(request.app.state, "engine", None)


def get_wyoming(request: Request) -> "WyomingService | None":
    """Return the running Wyoming listener, or ``None`` when disabled/failed."""
    return getattr(request.app.state, "wyoming", None)


def require_engine(engine: TTSEngine | None = Depends(get_engine)) -> TTSEngine:
    """Dependency for routes that cannot work before the model is loaded."""
    if engine is None:
        raise unavailable("Model is still loading; retry shortly.")
    return engine


def verify_api_key(
    config: Config = Depends(get_config),
    authorization: str | None = Header(default=None),
) -> None:
    """Router-level Bearer check for ``/v1/*``; no-op when no key is configured.

    ``/health`` is mounted outside the guarded router, so probes stay public.
    """
    if not config.api_key:
        return
    if authorization != f"Bearer {config.api_key}":
        raise invalid_api_key(
            "Missing or invalid API key. Pass 'Authorization: Bearer <key>'."
        )
