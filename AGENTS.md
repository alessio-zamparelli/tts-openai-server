# AGENTS.md

OpenAI-compatible TTS server (`POST /v1/audio/speech`) over **Piper** (default)
or Kyutai **pocket-tts** (optional backend). Installable package
`tts_openai` lives in `src/`. API usage is covered in `README.md`;
architecture rationale lives in the Italian `PLAN*.md` docs.

## Commands

```sh
uv sync                 # deps WITHOUT engines; default tests are offline (fake models)
uv run pytest           # full suite (addopts=-q already set)
uv run pytest tests/test_engine.py::test_generation_is_serialized   # one test
uv sync --extra piper   # + Piper (default backend); needed to actually synthesize
uv sync --extra engine  # + pocket-tts and CPU torch (large), optional backend
uv run tts-openai            # serve on 127.0.0.1:8000 with Piper/Paola
TTS_BACKEND=pocket uv run tts-openai   # legacy backend
TTS_WYOMING_ENABLED=1 uv run tts-openai  # + Wyoming on 0.0.0.0:10200
```

- Real-model e2e tests are opt-in: `PIPER_E2E=1 uv run pytest -m e2e -v`
  (`tests/test_integration_piper.py`, downloads the 64 MB default voice) and
  `TTS_E2E=1 uv run pytest -m e2e -v` (`tests/test_integration.py`,
  downloads pocket weights). Both boot a real uvicorn server.
- No lint/format/typecheck tooling is configured. `pyrightconfig.json` exists but nothing
  runs pyright. CI (`.github/workflows/docker-publish.yml`) gates the GHCR publish on
  `uv sync --locked` + `uv run pytest -q`.
- `uv.lock` is regenerated with **uv 0.13.1** (pinned in `deploy/Dockerfile`); use that
  version, and build the image with `--locked`.
- `mp3`/`opus`/`aac`/`flac` require **ffmpeg** on PATH (wav/pcm are stdlib-only); the
  affected tests skip cleanly when ffmpeg is absent. The Docker image ships ffmpeg.

## Architecture

- `main.py:create_app(config, engine)` — app factory + lifespan. Production passes
  `engine=None` and loads the configured backend in a background thread; tests inject a
  fake engine via `tests/conftest.py`. `app.state.engine is None` (→ 503) until the load
  completes. Routers live in `routers/`, shared dependencies in `dependencies.py`;
  `server.py` is only a compatibility shim.
- `routers/`: `audio.py`, `models.py`, `health.py`, `voices.py`. The `/v1` aggregate
  router applies the `verify_api_key` dependency; `/health` is mounted outside it.
- `wyoming_server.py` — opt-in Wyoming TCP listener (`TTS_WYOMING_ENABLED`).
  `build_info` reflects the engine voice list per `describe`; synthesis runs the blocking
  engine through `asyncio.to_thread`, one sentence at a time.
- `engine.py:load_engine()` dispatches on `Config.backend` (`piper` default, `pocket`
  legacy). Both imports are lazy, so the default test run works without either extra.
- `engine.py:TTSEngine` wraps any duck-typed model (`TTSModelLike` protocol). One global
  `_gen_lock` serializes all generation FIFO; a stream holds it for its whole duration.
  Voice states are loaded/encoded outside the lock, cached in an LRU, single-flight per
  voice. The LRU default is backend-dependent: 2 whole ONNX sessions for Piper, 32 state
  snapshots for pocket.
- `piper_backend.py:PiperTTSModel` adapts `piper.PiperVoice`: alias/ID/path resolution,
  on-demand download (`piper.download_voices`) into `TTS_PIPER_VOICE_DIR`, per-voice
  sample rate (`sample_rate_for`), whole-text concatenation and per-sentence streaming.
  Pocket keeps its prompt-based voices and custom-voice registry. Piper also exposes
  `voice_metadata` (description/languages/speakers) and applies the
  `TTS_PIPER_*` synthesis options via a duck-typed `SynthesisConfig`.
- Sample rates are per request: routes call `engine.sample_rate_for(voice)` for WAV
  headers/ffmpeg. Pocket always returns 24 kHz; Piper is 16 kHz (low/x_low) or 22.05 kHz
  (medium/high) — never read `engine.sample_rate` for a response header.
- Idle eviction: a watchdog drops the model after `TTS_IDLE_UNLOAD_S` idle seconds
  and the next request reloads it. Enabled only when a `loader` is attached (production),
  so fake engines never evict. `/health` deliberately does **not** reset the idle timer.
- Custom voices (pocket only): `voice_registry.py` persists
  `<cache_dir>/voices/registry.json` plus one `<name>.safetensors` per voice
  (`TTS_CACHE_DIR`, default `~/.cache/pocket_tts`). Cloning runs the encode under
  the gen lock but the `.safetensors` export outside it. Piper has no registry and
  `POST /v1/voices` returns 501.

## Gotchas

- `engine.to_pcm16` mutates its input buffer in place; callers must not reuse the audio
  chunk after conversion.
- `TTS_VOICE_MAP` entries must be `alias=voice` (comma-separated); a bare value
  raises `ValueError` at startup.
- `config.py` is the source of truth for env vars. Backend defaults are applied in
  `Config.__post_init__`: on Piper every OpenAI alias maps to `it_IT-paola-medium` and
  `TTS_DEFAULT_VOICE` empty means the backend default.
- The API-key check is a **router-level dependency** on the `/v1` aggregate router
  (`/health` stays open). Unknown `/v1/*` paths return 404, not 401.
- Wyoming is **unauthenticated by design**; `TTS_API_KEY` does not apply to it.
  Keep port 10200 on a trusted network and use `TTS_WYOMING_HOST` to narrow it.
- Wyoming handlers must call the blocking engine via `asyncio.to_thread`; generation is
  done one sentence at a time so the global `_gen_lock` is never held across an `await`
  (a disconnected client cannot leak it).
- Wyoming voice lists are rebuilt on every `describe`; HA caches them, so the integration
  must be reloaded after voices are added or removed. `wyoming-piper` parity uses
  `sentence-stream` for splitting and `voice.speaker` for Piper `speaker_id` selection.
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
- Local dev uses `.python-version` (3.14), matching the Docker image
  (`python:3.14-slim`); `requires-python` is `>=3.10`.
