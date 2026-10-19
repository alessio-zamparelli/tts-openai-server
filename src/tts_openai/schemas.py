"""Request models and API-shape constants shared by the routers."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AudioFormat = Literal["wav", "pcm", "mp3", "opus", "aac", "flac"]

# Supported response formats. wav/pcm are stdlib; mp3/opus/aac/flac are
# encoded through ffmpeg (audio_codecs.py) and 400 with a clear message when
# ffmpeg is missing (the Docker image ships it, so they just work there).
SUPPORTED_FORMATS = ("wav", "pcm", "mp3", "opus", "aac", "flac")

# All OpenAI TTS model aliases map to the single backend model.
MODEL_ALIASES: tuple[str, ...] = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")

# Fixed timestamp so /v1/models responses are deterministic.
CREATED = 1700000000


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model alias; all aliases map to the server's TTS backend.")
    input: str = Field(description="Text to synthesize.")
    voice: str | None = Field(
        default=None,
        description="OpenAI voice alias, backend voice name/id, custom voice, or model path. "
        "Omitted -> server default (piper: it_IT-paola-medium; pocket: alloy).",
    )
    response_format: AudioFormat = Field(default="wav")
    speed: float | None = Field(
        default=None, description="Accepted for compatibility but ignored (see README).")
    instructions: str | None = Field(
        default=None, description="Accepted for compatibility but ignored (see README).")
    language: str | None = Field(
        default=None, description="Server-configured language always wins; logged and ignored if different.")
    stream: bool = Field(
        default=False,
        description="Private extension: chunked transfer. wav/pcm yield raw chunks; "
        "mp3/opus/aac/flac keep their content type and are encoded through ffmpeg live "
        "(no whole-file buffering).")
