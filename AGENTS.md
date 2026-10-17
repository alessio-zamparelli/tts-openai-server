# AGENTS.md

OpenAI-compatible TTS server (`POST /v1/audio/speech`) over **Piper** (default)
or Kyutai **pocket-tts** (optional backend). Installable package
`pocket_tts_openai` lives in `src/`. API usage is covered in `README.md`;
architecture rationale lives in the Italian `PLAN*.md` docs.

## Commands

```sh
uv sync                 # deps WITHOUT engines; default tests are offline (fake models)
uv run pytest           # full suite (addopts=-q already set)
uv run pytest tests/test_engine.py::test_generation_is_serialized   # one test
uv sync --extra piper   # + Piper (default backend); needed to actually synthesize
uv sync --extra engine  # + pocket-tts and CPU torch (large), optional backend
uv run pocket-tts-openai            # serve on 127.0.0.1:8000 with Piper/Paola
POCKET_TTS_BACKEND=pocket uv run pocket-tts-openai   # legacy backend
```

- Real-model e2e tests are opt-in: `PIPER_E2E=1 uv run pytest -m e2e -v`
  (`tests/test_integration_piper.py`, downloads the 64 MB default voice) and
  `POCKET_TTS_E2E=1 uv run pytest -m e2e -v` (`tests/test_integration.py`,
  downloads pocket weights). Both boot a real uvicorn server.
- No lint/format/typecheck tooling is configured. `pyrightconfig.json` exists but nothing
  runs pyright. CI (`.github/workflows/docker-publish.yml`) gates the GHCR publish on
  `uv sync --locked` + `uv run pytest -q`.
- `uv.lock` is regenerated with **uv 0.12.13** (pinned in `deploy/Dockerfile`); use that
  version, and build the image with `--locked`.
- `mp3`/`opus`/`aac`/`flac` require **ffmpeg** on PATH (wav/pcm are stdlib-only); the
  affected tests skip cleanly when ffmpeg is absent. The Docker image ships ffmpeg.

## Architecture

- `server.py:create_app(config, engine)` — production passes `engine=None` and loads the
  configured backend in a background thread; tests inject a fake engine via
  `tests/conftest.py`. `app.state.engine is None` (→ 503) until the load completes.
- `engine.py:load_engine()` dispatches on `Config.backend` (`piper` default, `pocket`
  legacy). Both imports are lazy, so the default test run works without either extra.
- `engine.py:TTSEngine` wraps any duck-typed model (`TTSModelLike` protocol). One global
  `_gen_lock` serializes all generation FIFO; a stream holds it for its whole duration.
  Voice states are loaded/encoded outside the lock, cached in an LRU, single-flight per
  voice. The LRU default is backend-dependent: 2 whole ONNX sessions for Piper, 32 state
  snapshots for pocket.
- `piper_backend.py:PiperTTSModel` adapts `piper.PiperVoice`: alias/ID/path resolution,
  on-demand download (`piper.download_voices`) into `POCKET_TTS_PIPER_VOICE_DIR`, per-voice
  sample rate (`sample_rate_for`), whole-text concatenation and per-sentence streaming.
  Pocket keeps its prompt-based voices and custom-voice registry.
- Sample rates are per request: routes call `engine.sample_rate_for(voice)` for WAV
  headers/ffmpeg. Pocket always returns 24 kHz; Piper is 16 kHz (low/x_low) or 22.05 kHz
  (medium/high) — never read `engine.sample_rate` for a response header.
- Idle eviction: a watchdog drops the model after `POCKET_TTS_IDLE_UNLOAD_S` idle seconds
  and the next request reloads it. Enabled only when a `loader` is attached (production),
  so fake engines never evict. `/health` deliberately does **not** reset the idle timer.
- Custom voices (pocket only): `voice_registry.py` persists
  `<cache_dir>/voices/registry.json` plus one `<name>.safetensors` per voice
  (`POCKET_TTS_CACHE_DIR`, default `~/.cache/pocket_tts`). Cloning runs the encode under
  the gen lock but the `.safetensors` export outside it. Piper has no registry and
  `POST /v1/voices` returns 501.

## Gotchas

- `engine.to_pcm16` mutates its input buffer in place; callers must not reuse the audio
  chunk after conversion.
- `POCKET_TTS_VOICE_MAP` entries must be `alias=voice` (comma-separated); a bare value
  raises `ValueError` at startup.
- `config.py` is the source of truth for env vars. Backend defaults are applied in
  `Config.__post_init__`: on Piper every OpenAI alias maps to `it_IT-paola-medium` and
  `POCKET_TTS_DEFAULT_VOICE` empty means the backend default.
- The API-key middleware guards only paths starting with `/v1/` (`/health` stays open).
- Root `github-deploy-example.yml` is a stale example; edit
  `.github/workflows/docker-publish.yml` instead.
- `voices/jarvis1.safetensors` (~68 MB) is committed and seeded into the Docker image
  only when built with `--build-arg WITH_POCKET=1`; `voices/registry.json` indexes it.
- Piper (`piper-tts`) is **GPL-3.0-or-later** and bundles eSpeak-ng; the default Docker
  image ships it. `NOTICE` documents the licensing. The repo's own code stays MIT.
- Piper streams one chunk per sentence; pocket streams sub-sentence. Do not assume a
  first-chunk latency parity between backends.
- `it_IT-paola-medium` is 22.05 kHz, `it_IT-paola-low` is 16 kHz. Tests that hardcode
  24 kHz are pocket-specific.
- Default Docker build installs Piper only (`WITH_POCKET=0`); `uv sync --locked` in CI
  must remain valid for both extras.
- Local dev uses `.python-version` (3.12); the Docker image is `python:3.14-slim`;
  `requires-python` is `>=3.10`.
