# pocket-tts-openai

OpenAI-compatible TTS server (`POST /v1/audio/speech`, `stream` chunked PCM/WAV
or live ffmpeg-encoded mp3/opus/aac/flac) powered by
[Piper](https://github.com/OHF-Voice/piper1-gpl) by default, with
[Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) available as an
optional second backend. Both run on CPU.

The default Piper voice is **`it_IT-paola-medium`** (22.05 kHz). Piper is
GPL-3.0-or-later — see [`NOTICE`](NOTICE) for licensing details.

> Work in progress — see `./PLAN.md`. **M1** (server skeleton), **M2**
> (audio formats), **M3** (voice catalog + cloning `GET/POST/DELETE /v1/voices`),
> **M4** (streamed output incl. live ffmpeg encoding for compressed formats) and
> **M5.5** (idle RAM eviction) are implemented.

## Dev

```sh
uv sync                                # deps for the offline test suite (fake model)
uv run pytest                          # 160 tests, no model downloads
uv sync --extra piper                  # + Piper (the default backend)
uv run pocket-tts-openai               # serve on :8000 with Piper + Paola
uv sync --extra piper --extra engine   # also install pocket-tts (CPU-only torch)
POCKET_TTS_BACKEND=pocket uv run pocket-tts-openai   # legacy backend
```

To run on all interfaces / another port (as used by the live server):
`POCKET_TTS_HOST=0.0.0.0 POCKET_TTS_PORT=8080 uv run pocket-tts-openai`.

## Endpoints

### `POST /v1/audio/speech`

Body (JSON):

| field | type | default | notes |
| --- | --- | --- | --- |
| `model` | string | `tts-1` | any of `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| `input` | string | — | text to synthesize (required, non-empty) |
| `voice` | string | `it_IT-paola-medium` (piper) / `alloy` (pocket) | OpenAI alias, Piper voice id/path, Kyutai catalog voice, custom voice |
| `response_format` | string | `wav` | `wav` \| `pcm` \| `mp3` \| `opus` \| `aac` \| `flac` (compressed need ffmpeg; see below) |
| `stream` | bool | `false` | **private extension** — chunked transfer for `wav`/`pcm`; compressed formats are encoded through ffmpeg live (no whole-file buffering) |
| `speed` | float | — | accepted, ignored |
| `instructions` | string | — | accepted, ignored |
| `language` | string | — | server-configured language always wins; logged and ignored |

`stream: true` returns a `StreamingResponse` for every format:

- `response_format=pcm` → `audio/pcm`, raw mono s16le chunks.
- `response_format=wav` → `audio/wav` with a **streaming WAV header**
  (`RIFF`/`data` sizes = `0xFFFFFFFF`) followed by PCM chunks — renders
  incrementally in ffplay/VLC and most browsers.
- `response_format=mp3|aac|flac` → the compressed format, but encoded **live**:
  PCM chunks are piped into an ffmpeg subprocess as they are generated
  (`encode_pcm_stream`) and the encoded bytes flow incrementally, so the first
  audio arrives before synthesis finishes — with byte-identical output to the
  buffered (non-streaming) response.
- `response_format=opus` → also streamed live, but in **Ogg pages** (~a page
  per second at 96 kbps), so the chunk granularity is coarser than the other
  formats — inherent to the Ogg/Opus muxing.

`Content-Disposition` is `attachment; filename=speech.{format}`.

Compressed formats (`mp3`/`opus`/`aac`/`flac`) are encoded through **ffmpeg**
(`audio_codecs.py`) with the OpenAI content types (`audio/mpeg`, `audio/ogg`,
`audio/aac`, `audio/x-flac`). `stream: false` (the default) buffers the whole
file in memory first; `stream: true` encodes live as described above. Without
`ffmpeg` on PATH they return a clear 400; **the Docker image ships ffmpeg**, so
they just work there — locally run `apt-get install ffmpeg` (or your package
manager).

### `GET /v1/models` · `GET /health`

`/v1/models` lists the model aliases. `/health` is liveness + engine stats
(`status`, `backend`, `default_voice`, `sample_rate`, `language`, `requests`,
`avg_rtf`, `queue_depth`); it reports `loading` until the model is ready.

### `GET/POST/DELETE /v1/voices` (private extension)

Catalog + voice cloning. On the default **Piper** backend, `GET` lists the
OpenAI aliases, the configured default and any locally installed Piper voices;
voice *cloning* is pocket-tts-only, so `POST` returns **501** and `DELETE` on a
built-in voice returns **405**. Run with `POCKET_TTS_BACKEND=pocket` to clone.

**`GET /v1/voices`** — list every valid voice:

```json
{"object": "list", "data": [
  {"id": "alloy", "aliases": ["alloy", "alba"], "source": "builtin",
   "language": "en", "license": "https://huggingface.co/kyutai/tts-voices", "cached": true},
  {"id": "mario", "aliases": ["mario"], "source": "custom",
   "language": "it", "license": null, "cached": false, "safetensors": true}
]}
```

On Piper, any canonical voice id can be passed directly as `voice` (for
example `en_US-lessac-medium` or `it_IT-paola-low`) and is downloaded into
`POCKET_TTS_PIPER_VOICE_DIR` on first use when
`POCKET_TTS_PIPER_AUTO_DOWNLOAD=true`. Local `.onnx` paths are accepted too.
Aliases map to the backend default unless `POCKET_TTS_VOICE_MAP` overrides them.

**`POST /v1/voices`** (pocket-tts backend only) — `multipart/form-data` with:

- `name` (required): `^[a-z0-9][a-z0-9_-]{0,63}$`. **409** on collision with a
  built-in alias/catalog voice or an existing custom voice.
- `file` (required): `.wav` / `.mp3` / `.flac`. **413** over
  `POCKET_TTS_MAX_UPLOAD_MB` (default 25). **400** on unknown extension.
- `language` (optional): free-form tag, stored and echoed.

Returns **201** with the custom-voice payload after the audio prompt is encoded
and exported to `<cache_dir>/voices/<name>.safetensors`. The audio-prompt
*encode* runs under the global generation lock; the `.safetensors` export runs
outside it, so a slow disk delays only the clone, never a concurrent synthesis
request.

**`DELETE /v1/voices/{name}`** — **405** for built-in voices; **404** for
unknown; **204** on success (removes the registry entry, the `.safetensors`
file, and its LRU cache slot).

On the pocket backend, custom voices are immediately usable as `voice` in
`/v1/audio/speech`.

**Persisted custom voices.** Cloned voices are stored in `<cache_dir>/voices/`
(`POCKET_TTS_CACHE_DIR`, default `~/.cache/pocket_tts`) as `<name>.safetensors`
plus a `registry.json` index, and are re-loaded from disk at every boot — they
survive restarts without re-cloning. The voice is served by its id like any
other (a custom voice name is just another valid `voice` value).

The repo ships one pre-cloned Italian voice, **`jarvis1`** (`voices/`, ~68 MB):
cloned from 5 `audio files/` corpus clips (`mFB_S7_3_Italian.wav.mp3`,
`mFB_S9_1_Italian.wav.mp3`, `mFB_S45_5_Italian.wav.mp3`,
`mFF_S1_6_Italian.wav.mp3`, `mFF_S20_2_Italian.wav.mp3`) joined into a single
~27.5 s conditioning prompt with 0.5 s of silence between them, then encoded
once via pocket-tts `get_state_for_audio_prompt`. It is served by the API and
is seeded into the container image when built with `WITH_POCKET=1` (see
“Container (Docker)”).

Example — serve `jarvis1` (Italian, needs `italian_24l`):

```sh
# jarvis1 is a pocket-tts clone, so select that backend explicitly:
POCKET_TTS_BACKEND=pocket POCKET_TTS_LANGUAGE=italian_24l \
  uv run pocket-tts-openai
# then, in another shell:
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"jarvis1","input":"Questa è la voce di Jarvis."}' -o jarvis1.wav
```

A cloned voice only makes sense with the matching model language it was encoded
from (`italian_24l`). Pre-encode it at boot with `POCKET_TTS_WARMUP_VOICES=jarvis1`.
To clone your own voice, `POST /v1/voices` (above) with a 16-bit PCM WAV.
Uploads accepted as `.mp3`/`.flac` need `soundfile` at runtime (not installed);
the zero-dependency workaround is to upload a 16-bit PCM WAV.

### Idle RAM reclamation

After `POCKET_TTS_IDLE_UNLOAD_S` (default 300 s = 5 min) with **no API
requests**, the model (and its cached voices) are dropped from RAM to keep an
idle API cheap. The next request **blocks while it reloads** (Piper: under a
second from disk; pocket-tts: ~1 s warm from the HF cache) instead of returning
503. Set `POCKET_TTS_IDLE_UNLOAD_S=0` to keep the model resident always.
`/health` exposes `loaded`, `unloads`, `reloads` and `last_request_age_s`.

### Model size & RAM

**Piper** voices are small ONNX models: `it_IT-paola-medium` is 64 MB on disk
and ~171 MB resident with the model loaded (measured, see Performance). The
quality tier is part of the voice id: `x_low`/`low` are 16 kHz, `medium`/`high`
are 22.05 kHz. Each loaded voice is a separate ONNX session, so the voice LRU
defaults to **2** on this backend (`POCKET_TTS_MAX_CACHED_VOICES`).

**pocket-tts** ships no whisper-style model family — there is no smaller
`tiny`/`nano` to pick. Each language has a single **base model** (6-layer,
~430 MB download), and that's already the smallest available. Some languages
also offer a `_24l` **quality-upgrade** variant (24-layer, larger + slower):
`english_2026-04_24l`, `french_24l`, `german_24l`, `spanish_24l`. The model
tier is chosen by `POCKET_TTS_LANGUAGE`; there is no size knob below "base".

The sharpest RAM lever is `POCKET_TTS_QUANTIZE=1`, which loads the
transformer's attention/FFN weights as **int8** (dynamic quantization,
FBGEMM on x86; requires AVX2) instead of fp32. Measured at rest with the
model resident (`/health` → `loaded:true`):

| `POCKET_TTS_QUANTIZE` | resident RSS |
| --- | --- |
| `false` (default) | ~1,042 MB |
| `1` | ~395 MB (≈ 62% less) |

Quantization also speeds up inference ~27% on x86 with no measurable quality
change (WER unchanged). For even lower counts, `pip install
pocket-tts[quantize]` swaps in the `torchao` backend. Paired with idle
eviction (previous section) an idle process holds ~0 MB of model.

## Configuration (env)

| var | default | notes |
| --- | --- | --- |
| `POCKET_TTS_BACKEND` | `piper` | `piper` or `pocket` |
| `POCKET_TTS_HOST` | `127.0.0.1` | bind host |
| `POCKET_TTS_PORT` | `8000` | bind port |
| `POCKET_TTS_DEFAULT_VOICE` | backend default | `it_IT-paola-medium` (piper) / `alloy` (pocket); used when `voice` is omitted |
| `POCKET_TTS_VOICE_MAP` | backend defaults | alias→voice overrides, e.g. `alloy=en_US-lessac-medium` |
| `POCKET_TTS_PIPER_VOICE_DIR` | `~/.cache/piper` | Piper `.onnx` cache (Docker: `/data/cache/piper`) |
| `POCKET_TTS_PIPER_AUTO_DOWNLOAD` | `true` | download missing Piper voices on demand |
| `POCKET_TTS_MAX_CACHED_VOICES` | `2` (piper) / `32` (pocket) | LRU of loaded voices/states; a Piper entry is a whole ONNX session |
| `POCKET_TTS_WARMUP_VOICES` | — | comma-separated voices to load/encode at boot |
| `POCKET_TTS_IDLE_UNLOAD_S` | `300` | evict the model from RAM after this many idle seconds; `0` disables |
| `POCKET_TTS_IDLE_POLL_S` | `30` | idle-eviction watchdog poll interval (seconds) |
| `POCKET_TTS_MAX_WAITING` | `0` | load shedding: HTTP 429 when more than this many requests are in-flight (encode+wait+generate); `0` = unlimited |
| `POCKET_TTS_QUEUE_TIMEOUT_S` | `0` | max seconds a request waits for the generation lock before a 429; `0` = wait forever |
| `POCKET_TTS_API_KEY` | — | if set, `Bearer <key>` required on `/v1/*` |
| `POCKET_TTS_CACHE_DIR` | `~/.cache/pocket_tts` | pocket voice registry / model cache base |
| `POCKET_TTS_MAX_UPLOAD_MB` | `25` | max cloned-voice upload size |

Pocket-only: `POCKET_TTS_LANGUAGE`, `POCKET_TTS_QUANTIZE` (see above);
Piper takes its language from the voice id (`it_IT-…` → Italian).

## Container (Docker)

CPU-only multi-stage image following the official
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/) —
`deploy/Dockerfile` (`python:3.14-slim` base, pinned `uv`, Python 3.14). The
default image ships **Piper only** (no torch; a few hundred MB including
ffmpeg) with `it_IT-paola-medium` baked in:

```sh
docker build -f deploy/Dockerfile -t pocket-tts-openai .
docker run --rm -p 8080:8000 \
  -v tts-data:/data \
  pocket-tts-openai
# -> GET  http://localhost:8080/health   (backend=piper)
#    POST http://localhost:8080/v1/audio/speech
```

Add `--build-arg WITH_POCKET=1` (and `WITH_POCKET: "1"` in
`docker-compose.yml`) to install pocket-tts + CPU torch as well (~1 GB image)
and seed the `jarvis1` cloned voice.

Or `docker compose up -d --build` (see `docker-compose.yml`).

- **Runtime user**: non-root (`tts`, uid 10001), exposes `8000`.
- **Persistent volume** `/data`: Piper voices (`/data/cache/piper`), pocket
  weights (`HF_HOME=/data/hf`) and the cloned-voice registry
  (`/data/cache/voices`). Mount it so voices are downloaded once and cloned
  voices survive restarts. A fresh named volume inherits the baked Paola voice.
- **Default voice baked in**: the build downloads `it_IT-paola-medium` into
  `/data/cache/piper`, so the first request needs no network. Override the
  build-time voice with `--build-arg PIPER_VOICE=...`.
- **`jarvis1` (pocket only)**: with `WITH_POCKET=1` the image seeds the
  registry (`/data/cache/voices`) from the repo `voices/` dir. To serve it:

  ```sh
  docker build -f deploy/Dockerfile --build-arg WITH_POCKET=1 -t tts-pocket .
  docker run --rm -p 8080:8000 -e POCKET_TTS_BACKEND=pocket \
    -e POCKET_TTS_LANGUAGE=italian_24l -v tts-data:/data tts-pocket
  # GET  http://localhost:8080/v1/voices     -> lists "jarvis1" (custom, it)
  # POST http://localhost:8080/v1/audio/speech \
  #      -d '{"model":"tts-1","voice":"jarvis1","input":"Ciao"}'
  ```

  `italian_24l` weights come from the **gated** `kyutai/pocket-tts` repo: on a
  cold `/data` volume the container downloads them at first boot, so pass
  `-e HF_TOKEN=...` (or mount a volume that already has them cached under
  `/data/hf`). The voice file itself is our own cloned state (~68 MB) and is
  seeded from the repo — no token needed for the voice, only for the weights.
- **No CUDA**: `torch` resolves from the PyTorch **CPU** wheel index via
  `uv.lock` (`[tool.uv.sources]`); nothing NVIDIA layers into the image. The
  default Piper image installs no torch at all.
- **ffmpeg included**: the runtime image installs `ffmpeg`, so the compressed
  `/v1/audio/speech` formats (`mp3`/`opus`/`aac`/`flac`) work out of the box
  (Debian's build ships the libmp3lame + libopus encoders).
- The bind host defaults to `0.0.0.0` inside the container (port `8000`);
  override with `POCKET_TTS_HOST`/`POCKET_TTS_PORT`.

Image is published to **GHCR** (`ghcr.io/<repo>`) by the
`.github/workflows/docker-publish.yml` pipeline: tests gate the build, then
Buildx publishes `latest` on `main` pushes + semver tags (`v*.*.*`), tagged
`latest`/`<major>.<minor>`/`<version>`, with GHA build-cache and SLSA
attestations. Build locally to verify (no GitHub account needed):

```sh
git tag v0.1.0 && git push origin main --tags  # triggers the workflow
```

## Performance

Measured with `benchmarks/bench_backends.py` on an **Intel N150 (4 threads),
15 GiB RAM, CPU only**, 3 runs per text (each backend in a fresh process):

| metric | piper (`it_IT-paola-medium`) | pocket (`alloy`) |
| --- | --- | --- |
| sample rate | 22 050 Hz | 24 000 Hz |
| model load | **3.1 s** | 15.8 s |
| resident RSS (loaded) | **171 MB** | 887 MB |
| RTF short / medium / long | **0.158 / 0.135 / 0.134** | 0.573 / 0.550 / 0.556 |
| wall p50, long text | **3.6 s** | 19.3 s |
| stream TTFA | 0.357 s | **0.221 s** |
| stream RTF | **0.133** | 0.546 |
| throughput @4 clients (audio s/wall s) | **6.1** | 1.6 |

- **RTF** = generation wall time / produced audio time, lower is better. Piper
  is ~4× faster and ~5× lighter here.
- Piper emits **one chunk per sentence**, so its TTFA is the time to synthesize
  the first sentence; pocket streams sub-sentence chunks and wins TTFA on
  multi-sentence text despite the higher RTF.
- Both backends still serialize generation through one global lock, so the
  concurrency row is server throughput, not per-model parallelism.
- Reproduce with:
  `uv sync --extra piper --extra engine && uv run python benchmarks/bench_backends.py --backends piper,pocket --runs 5`.
  Numbers vary with CPU and `OMP_NUM_THREADS`.

## Known limitations / risks

- One model, one global generation lock — requests serialize (queue-depth is
  surfaced in `/health`). Long streams hold the lock for their whole duration.
- Piper streams per sentence, so a single long sentence has no early audio;
  pocket-tts streamed first-chunk latency was ~0.2 s in the benchmark above.
- The streaming WAV header uses the `0xFFFFFFFF` "streaming WAV" convention;
  verified with Python's `wave` module and the chunk math is exact.
- `speed`/`instructions` are accepted for API compatibility but ignored.
- Voice cloning requires `POCKET_TTS_BACKEND=pocket`; `POST /v1/voices`
  returns 501 on Piper.
- Custom voices are CLI/HTTP-created only; no web UI yet.

## Layout

```text
src/pocket_tts_openai/
  config.py         Config (env-driven, backend-aware defaults)
  engine.py         TTSEngine: gen lock, voice-state LRU, clone, generate_pcm_stream
  piper_backend.py  Piper adapter: voice download/load, per-voice sample rates
  voice_registry.py persistent custom-voice index (<cache>/voices/registry.json)
  voices.py         aliases, pocket/piper resolvers, language tags
  routes_speech.py  /v1/audio/speech, /v1/models, /health
  routes_voices.py  GET/POST/DELETE /v1/voices
  server.py         create_app, API-key middleware, background model load + warmup
  errors.py         OpenAI-shaped error helpers
tests/              contract + engine + registry + voices + streaming (fake models)
benchmarks/         Piper vs pocket-tts benchmark (subprocess-isolated, opt-in)
deploy/Dockerfile   multi-stage CPU-only image (uv, python:3.14-slim, non-root)
docker-compose.yml  local build/run convenience
.github/workflows/  docker-publish.yml -> GHCR (tests gate, buildx, attestations)
.dockerignore       keep .venv/.git out of the build context
```
