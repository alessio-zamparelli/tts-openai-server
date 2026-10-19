"""``POST /v1/audio/speech``: OpenAI-compatible speech synthesis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Response
from fastapi.responses import StreamingResponse

from ..audio_codecs import MEDIA_TYPES, encode_pcm, encode_pcm_stream
from ..dependencies import require_engine
from ..engine import RateLimited, TTSEngine, pcm_to_wav, streaming_wav_header
from ..errors import invalid_request, rate_limited
from ..schemas import MODEL_ALIASES, SUPPORTED_FORMATS, SpeechRequest

router = APIRouter(tags=["audio"])


def _stream_speech_pcm(engine: TTSEngine, text: str, voice: str):
    """Yield raw s16le PCM chunks from the engine's streaming generator."""
    for chunk in engine.generate_pcm_stream(text, voice):
        yield chunk


def _stream_speech_wav(engine: TTSEngine, text: str, voice: str, sample_rate: int):
    """Yield a size-less WAV header then s16le PCM chunks."""
    yield streaming_wav_header(sample_rate)
    for chunk in engine.generate_pcm_stream(text, voice):
        yield chunk


@router.post("/audio/speech", response_model=None, summary="Create speech")
def speech(req: SpeechRequest, engine: TTSEngine = Depends(require_engine)) -> Response:
    """Synthesize ``req.input`` with the requested voice; return wav or pcm bytes."""
    if req.model not in MODEL_ALIASES:
        raise invalid_request(
            f"Unknown model '{req.model}'. Available models: {', '.join(MODEL_ALIASES)}.")
    if not req.input.strip():
        raise invalid_request("input must be non-empty text.")
    if req.response_format not in SUPPORTED_FORMATS:
        raise invalid_request(
            f"response_format '{req.response_format}' is not supported yet; "
            f"supported formats: {', '.join(SUPPORTED_FORMATS)}.")

    # Omitted voice -> backend default (piper: it_IT-paola-medium, pocket: alloy).
    # Voice resolution (aliases, catalog, custom registry) happens inside the
    # engine; propagate unknown-voice as a 400.

    voice = req.voice or engine.default_voice

    def _speech_error() -> Exception:
        return invalid_request(
            f"Unknown voice {voice!r}. Use an OpenAI alias, a recognized "
            f"{engine.model_name} voice, a custom voice, or a model path."
        )

    # streaming path: wav/pcm yield raw chunks; compressed formats are encoded
    # live through ffmpeg (encode_pcm_stream) -- no whole-file buffering.
    if req.stream:
        # Admit-or-429 before headers so overload doesn't kill a stream midway.
        try:
            engine.check_capacity()
        except RateLimited as exc:
            raise rate_limited(str(exc)) from None
        # Validate the voice up front so unknown names produce a clean 400, not a
        # generator-body error surfacing as a 500 from Starlette's streaming
        # worker. This also warms the cached model state for the generator body.
        try:
            engine.voice_state(voice)
        except ValueError:
            raise _speech_error() from None
        # Resolve the rate after the voice is loaded (Piper tiers differ).
        sample_rate = engine.sample_rate_for(voice)
        if req.response_format in ("wav", "pcm"):
            filename = "speech.pcm" if req.response_format == "pcm" else "speech.wav"
            media_type = "audio/pcm" if req.response_format == "pcm" else "audio/wav"
            if req.response_format == "pcm":
                gen = _stream_speech_pcm(engine, req.input, voice)
            else:
                gen = _stream_speech_wav(engine, req.input, voice, sample_rate)
            return StreamingResponse(
                gen,
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        filename = f"speech.{req.response_format}"
        media_type = MEDIA_TYPES[req.response_format]
        return StreamingResponse(
            encode_pcm_stream(
                engine.generate_pcm_stream(req.input, voice),
                sample_rate,
                req.response_format,
            ),
            media_type=media_type,
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    try:
        pcm = engine.generate_pcm(req.input, voice)
    except ValueError:
        raise _speech_error() from None
    except RateLimited as exc:
        raise rate_limited(str(exc)) from None
    sample_rate = engine.sample_rate_for(voice)
    if req.response_format == "pcm":
        body, filename, media_type = pcm, "speech.pcm", "audio/pcm"
    elif req.response_format == "wav":
        body, filename, media_type = pcm_to_wav(pcm, sample_rate), "speech.wav", "audio/wav"
    else:
        try:
            body = encode_pcm(pcm, sample_rate, req.response_format)
        except RuntimeError as exc:
            raise invalid_request(str(exc)) from exc
        filename, media_type = f"speech.{req.response_format}", MEDIA_TYPES[req.response_format]
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
