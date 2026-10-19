"""Compatibility shim for the pre-refactor import location.

Use :mod:`tts_openai.main` instead; this module only re-exports the factory and
CLI so existing integrations keep working.
"""

from .main import create_app, main

__all__ = ["create_app", "main"]
