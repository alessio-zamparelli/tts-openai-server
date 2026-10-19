"""``GET /v1/models``: list the OpenAI model aliases."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from ..dependencies import get_engine
from ..engine import TTSEngine
from ..schemas import CREATED, MODEL_ALIASES

router = APIRouter(tags=["models"])


@router.get("/models")
def models(engine: TTSEngine | None = Depends(get_engine)) -> dict:
    """List available models: every alias maps to the server's configured backend."""
    owner = engine.model_name if engine is not None else "tts"
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "created": CREATED, "owned_by": owner}
            for alias in MODEL_ALIASES
        ],
    }
