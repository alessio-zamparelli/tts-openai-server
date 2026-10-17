# syntax=docker/dockerfile:1
#
# pocket-tts-openai — OpenAI-compatible TTS server (CPU-only).
#
# Built with uv following the official integration guide:
#   https://docs.astral.sh/uv/guides/integration/docker/
#
# Default image: Piper only (GPL-3.0-or-later, see NOTICE), with the Italian
# `it_IT-paola-medium` voice baked in. Pass --build-arg WITH_POCKET=1 to also
# install pocket-tts + CPU torch (~1 GB image) and its cloned-voice registry.
# CPU-only by design: `torch` resolves from the PyTorch CPU wheel index via
# uv.lock (see [tool.uv.sources] in pyproject.toml) — no CUDA/NVIDIA layers.
#
# Build:    docker build -f deploy/Dockerfile -t pocket-tts-openai .
# Run:      docker run --rm -p 8080:8000 -v tts-data:/data pocket-tts-openai
# Pocket:   docker build -f deploy/Dockerfile --build-arg WITH_POCKET=1 -t tts-pocket .

###############################################################################
# Stage 1 — builder: resolve deps + project into a venv
###############################################################################
FROM python:3.14-slim AS builder

# 1 = install pocket-tts + CPU torch as an optional second backend.
ARG WITH_POCKET=0

# Pin uv to the exact version used to (re)generate uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

# Use the system Python already present in the slim image (3.14).
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Layer 1 — install the locked dependencies without the project. Piper is the
# default backend; pocket-tts + CPU torch are added only with WITH_POCKET=1.
# This layer is cached and only invalidates when pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_POCKET" = "1" ]; then \
        uv sync --locked --no-install-project --no-dev --no-editable \
            --extra piper --extra engine; \
    else \
        uv sync --locked --no-install-project --no-dev --no-editable \
            --extra piper; \
    fi

# Layer 2 — copy the project source and install it non-editable so the venv is
# fully self-contained (no dependency on /app/src at runtime).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_POCKET" = "1" ]; then \
        uv sync --locked --no-dev --no-editable --extra piper --extra engine; \
    else \
        uv sync --locked --no-dev --no-editable --extra piper; \
    fi

###############################################################################
# Stage 2 — runtime: minimal, non-root, persistent /data volume
###############################################################################
FROM python:3.14-slim AS runtime

# 1 = pocket assets (cloned-voice registry) are seeded too.
ARG WITH_POCKET=0
# Voice baked into the image so a cold start needs no network.
ARG PIPER_VOICE=it_IT-paola-medium

# OpenMP runtime needed by torch (CPU) for multithreading, plus ffmpeg for the
# compressed /v1/audio/speech formats (mp3/opus/aac/flac). Debian ships ffmpeg
# with the libmp3lame and libopus encoders; without it those formats return a
# clear 400 (wav/pcm stay stdlib-only).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Reuse the system Python across stages; put the app venv on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    POCKET_TTS_BACKEND=piper \
    POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=8000 \
    POCKET_TTS_CACHE_DIR=/data/cache \
    POCKET_TTS_PIPER_VOICE_DIR=/data/cache/piper \
    HF_HOME=/data/hf

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Non-root runtime user; /data must stay writable for model weights + voices.
RUN useradd --create-home --uid 10001 tts \
    && mkdir -p /data/cache/piper /data/hf

# Bake the default Piper voice into the image (~64 MB). A fresh Docker named
# volume inherits /data, so a cold container needs no network.
RUN python -m piper.download_voices "${PIPER_VOICE}" --download-dir /data/cache/piper

# With WITH_POCKET=1, seed the pre-cloned Italian voice `jarvis1` (repo
# `voices/`). The bind mount keeps the 68 MB asset out of the image layer for
# the default Piper-only build.
RUN --mount=type=bind,source=voices,target=/tmp/pocket-voices \
    if [ "$WITH_POCKET" = "1" ]; then \
        mkdir -p /data/cache/voices; \
        cp /tmp/pocket-voices/registry.json /data/cache/voices/registry.json; \
        cp /tmp/pocket-voices/jarvis1.safetensors /data/cache/voices/jarvis1.safetensors; \
    fi

RUN chown -R tts:tts /data

# Persistent volume: Piper voices (/data/cache/piper), pocket weights
# (HF_HOME=/data/hf) and the cloned-voice registry (/data/cache/voices). Mount
# it to avoid re-downloading weights and to keep custom voices across restarts.
VOLUME ["/data"]

USER tts
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

ENTRYPOINT ["pocket-tts-openai"]
