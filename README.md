# tts-openai

**OpenAI-compatible text-to-speech server for CPU deployments.**

`tts-openai` implements `POST /v1/audio/speech` with the request and
response shapes OpenAI clients expect, plus optional chunked streaming and a
private voice catalog/cloning API. It ships two interchangeable backends:

- **Piper** (default) — fast, small ONNX voices with per-sentence streaming.
- **pocket-tts** (optional) — Kyutai's prompt-based TTS with voice cloning and
  a large multilingual catalog.

Both run on CPU. The default image contains Piper only and no PyTorch. The
optional `WITH_POCKET=1` image adds pocket-tts, CPU torch, and a pre-cloned
Italian voice (`jarvis1`).

> **Licensing:** the project's own code is MIT. Piper (`piper-tts`) is
> **GPL-3.0-or-later** and embeds eSpeak-ng; the default Docker image ships it.
> See [`NOTICE`](NOTICE) before redistributing the image or the `piper` extra.
> pocket-tts and its voices have their own upstream licenses.

---

## Contents

- [Status and backends](#status-and-backends)
- [Quick start](#quick-start)
- [API reference](#api-reference)
  - [`POST /v1/audio/speech`](#post-v1audiospeech)
  - [Formats and streaming](#formats-and-streaming)
  - [`GET /v1/models` and `GET /health`](#get-v1models-and-get-health)
  - [Voice catalog and cloning](#voice-catalog-and-cloning)
  - [Wyoming protocol (Home Assistant)](#wyoming-protocol-home-assistant)
  - [Errors and authentication](#errors-and-authentication)
- [Architecture](#architecture)
  - [Application and startup](#application-and-startup)
  - [Engine and concurrency](#engine-and-concurrency)
  - [Voice resolution and caching](#voice-resolution-and-caching)
  - [Idle eviction](#idle-eviction)
  - [Cloning pipeline](#cloning-pipeline)
  - [Audio pipeline](#audio-pipeline)
  - [Module layout](#module-layout)
- [Configuration](#configuration)
- [Voices](#voices)
- [Memory and performance](#memory-and-performance)
- [Docker](#docker)
  - [Dockerfile walkthrough](#dockerfile-walkthrough)
  - [docker-compose.yml walkthrough](#docker-composeyml-walkthrough)
  - [Building and running](#building-and-running)
  - [Publishing](#publishing)
- [Operations and troubleshooting](#operations-and-troubleshooting)
- [Development and testing](#development-and-testing)
- [Known limitations](#known-limitations)
- [License](#license)

---

## Status and backends

Implemented milestones:

- **M1** — server skeleton, `/v1/audio/speech`, `/v1/models`, `/health`.
- **M2** — WAV/PCM plus ffmpeg-backed MP3/Opus/AAC/FLAC.
- **M3** — voice catalog and cloning (`GET/POST/DELETE /v1/voices`).
- **M4** — chunked streaming, including live ffmpeg encoding.
- **M5.5** — idle RAM eviction with blocking warm reload.
- **M6** — opt-in Wyoming TTS protocol (Home Assistant), text/audio streaming,
  Piper speaker + synthesis options.

Backend comparison:

| | **Piper** (default) | **pocket-tts** (optional) |
| --- | --- | --- |
| Install extra | `uv sync --extra piper` | `uv sync --extra engine` |
| Model shape | One ONNX model per voice | One base model per language |
| Sample rate | 16 kHz (`x_low`/`low`) or 22.05 kHz (`medium`/`high`) | 24 kHz |
| Default voice | `it_IT-paola-medium` | `alloy` → `alba` |
| Voice cloning | No (`POST /v1/voices` → 501) | Yes |
| Streaming granularity | One chunk per sentence | Sub-sentence chunks |
| Model cache | `TTS_PIPER_VOICE_DIR` | `HF_HOME` + `TTS_CACHE_DIR` |
| Typical loaded RSS | ~171 MB (Paola medium) | ~887 MB fp32 / ~395 MB int8 |
| License | GPL-3.0-or-later | Upstream Kyutai license |

The backend is selected with `TTS_BACKEND=piper|pocket`. The default
Docker image installs only Piper, so `TTS_BACKEND=pocket` requires the
`WITH_POCKET=1` image (or `uv sync --extra engine` locally).

---

## Quick start

### Local development

```bash
# Install the server without engines: enough for the offline fake-model tests.
uv sync

# Run the test suite (168 tests, no model downloads by default).
uv run pytest

# Install the default Piper backend and serve on 127.0.0.1:8000.
uv sync --extra piper
uv run tts-openai

# Optional: add pocket-tts and CPU torch.
uv sync --extra piper --extra engine
TTS_BACKEND=pocket uv run tts-openai
```

Bind to all interfaces or another port:

```bash
TTS_HOST=0.0.0.0 TTS_PORT=8080 uv run tts-openai
```

Also serve the Wyoming protocol (for Home Assistant) on port 10200:

```bash
TTS_WYOMING_ENABLED=1 uv run tts-openai
```

The package entry point is `tts-openai` (installed by `uv sync` from
`[project.scripts]`).

### Docker Compose

```bash
docker compose up -d --build
# -> GET  http://localhost:8080/health
# -> POST http://localhost:8080/v1/audio/speech
```

Compose builds the default Piper-only image and maps host `8080` to container
`8000` plus host `10200` to the Wyoming listener (enabled in the Compose
environment).

Compose also writes rotating application logs to the repo's `./logs` folder
(`server.log` plus five 10 MB backups); console output stays available with
`docker compose logs -f tts`. The container runs as uid 10001, so create the
folder and make it writable before starting on Linux:
`mkdir -p logs && sudo chown 10001:10001 logs`.

### Docker one-liner

```bash
docker build -f deploy/Dockerfile -t tts-openai .
docker run --rm -p 8080:8000 -p 10200:10200 \
  -e TTS_WYOMING_ENABLED=1 \
  -v tts-data:/data tts-openai
```

### Quick verification

```bash
curl -s http://localhost:8080/health | jq
```

While the engine is still loading, `/health` returns:

```json
{"status": "loading", "model": "piper"}
```

Once ready it returns `status: "ok"` plus engine statistics. The HTTP server
binds before the background model load finishes, so `503` is the correct
response for `/v1/audio/speech` during that window.

```bash
curl -s http://localhost:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","input":"Buongiorno, come stai?","voice":"alloy"}' \
  -o speech.wav
```

---

## API reference

### `POST /v1/audio/speech`

Synthesize text and return an audio response.

```bash
curl -s http://localhost:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{
    "model": "tts-1",
    "input": "Questa è la voce di Jarvis.",
    "voice": "jarvis1",
    "response_format": "wav",
    "stream": false
  }' \
  -o jarvis1.wav
```

#### Request fields

| Field | Type | Default | Behaviour |
| --- | --- | --- | --- |
| `model` | string | `tts-1` | One of `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts`. All aliases serve the same backend model. Any other value → `400`. |
| `input` | string | required | Text to synthesize. Whitespace-only → `400`. |
| `voice` | string | backend default | OpenAI alias, Piper voice id/path, Kyutai catalog voice, custom cloned voice, or a passthrough path/URL. Omitted → `TTS_DEFAULT_VOICE` / backend default. |
| `response_format` | string | `wav` | `wav`, `pcm`, `mp3`, `opus`, `aac`, `flac`. Any other value is rejected by FastAPI validation (`422`). |
| `stream` | bool | `false` | Private extension. Returns a chunked `StreamingResponse` for every format. |
| `speed` | float | — | Accepted for compatibility, **ignored**. |
| `instructions` | string | — | Accepted for compatibility, **ignored**. |
| `language` | string | — | Accepted for compatibility, **ignored**. The server-configured language/voice always wins. |

`model` does not select a different model or quality tier; it only validates
OpenAI compatibility.

#### Response headers

Every successful speech response includes:

```text
Content-Disposition: attachment; filename=speech.<format>
```

The content type depends on the format:

| Format | Content type | Filename | Encoder |
| --- | --- | --- | --- |
| `wav` | `audio/wav` | `speech.wav` | stdlib `wave` |
| `pcm` | `audio/pcm` | `speech.pcm` | raw mono s16le |
| `mp3` | `audio/mpeg` | `speech.mp3` | ffmpeg / `libmp3lame` |
| `opus` | `audio/ogg` | `speech.opus` | ffmpeg / `libopus` in Ogg |
| `aac` | `audio/aac` | `speech.aac` | ffmpeg / ADTS AAC |
| `flac` | `audio/x-flac` | `speech.flac` | ffmpeg / FLAC |

`wav` and `pcm` are produced without external dependencies. The compressed
formats require `ffmpeg` on `PATH`; the Docker image installs it. `pcm` is raw
mono signed 16-bit little-endian with **no header**.

### Formats and streaming

#### `stream: false` (default)

- `wav`: the whole PCM buffer is wrapped in a standard 44-byte RIFF/WAVE header.
- `pcm`: the whole PCM buffer is returned as-is.
- `mp3`/`opus`/`aac`/`flac`: the whole PCM buffer is piped through ffmpeg and
  the full encoded file is buffered in memory before responding.

#### `stream: true`

Every format returns a `StreamingResponse`:

- **`pcm`** → raw s16le chunks as they are generated.
- **`wav`** → a 44-byte streaming WAV header with `RIFF` and `data` sizes set to
  `0xFFFFFFFF`, followed by PCM chunks. ffplay, VLC, most browsers, and Python's
  `wave` reader accept it; the header is not seekable/rewritten.
- **`mp3`/`aac`/`flac`** → PCM chunks are pumped into an ffmpeg subprocess as
  they are generated and encoded bytes are read from its stdout immediately.
  `-flush_packets 1` prevents muxer buffering. The streamed output is
  byte-identical to the buffered (`stream:false`) response.
- **`opus`** → also streamed live, but in Ogg pages (roughly one page per second
  at 96 kbps), so chunk granularity is coarser than the raw formats. This is
  inherent to Ogg/Opus muxing.

Streaming uses chunked transfer; there is no `Content-Length`.

#### ffmpeg fallback

Without `ffmpeg`:

- non-streaming compressed requests return a clean `400` explaining that
  `response_format=...` requires ffmpeg;
- streaming compressed requests can only fail once the response body starts,
  because the encoder subprocess is created by the body generator.

`wav` and `pcm` never need ffmpeg.

#### Examples

OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

with client.audio.speech.with_streaming_response.create(
    model="tts-1",
    voice="alloy",
    input="Hello from the local TTS server.",
    response_format="mp3",
) as response:
    response.stream_to_file("speech.mp3")
```

Raw streaming to a player:

```bash
curl -N http://localhost:8080/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"input":"This arrives incrementally.","voice":"alloy","response_format":"wav","stream":true}' \
  | ffplay -autoexit -
```

### `GET /v1/models` and `GET /health`

#### `GET /v1/models`

Returns the three OpenAI aliases with the active backend as `owned_by`:

```json
{
  "object": "list",
  "data": [
    {"id": "tts-1", "object": "model", "created": 1700000000, "owned_by": "piper"},
    {"id": "tts-1-hd", "object": "model", "created": 1700000000, "owned_by": "piper"},
    {"id": "gpt-4o-mini-tts", "object": "model", "created": 1700000000, "owned_by": "piper"}
  ]
}
```

`created` is a fixed timestamp so responses are deterministic.

#### `GET /health`

This endpoint is public even when an API key is configured. It does **not**
touch the idle timer, so monitoring traffic cannot defeat idle eviction.

Before the engine loads:

```json
{"status": "loading", "model": "piper"}
```

After the engine loads:

```json
{
  "status": "ok",
  "model": "piper",
  "backend": "piper",
  "default_voice": "it_IT-paola-medium",
  "sample_rate": 22050,
  "language": "it",
  "loaded": true,
  "idle_unload_s": 300,
  "last_request_age_s": 4.2,
  "unloads": 0,
  "reloads": 0,
  "requests": 12,
  "avg_rtf": 0.137,
  "queue_depth": 0
}
```

Field notes:

- `model` is `pocket-tts` for the pocket backend, otherwise the backend name.
- `sample_rate` is the engine's default/initial voice rate. Per-request output
  rates are resolved per voice; Piper quality tiers can be 16 kHz or 22.05 kHz.
- `loaded` is false after an idle eviction; the next request blocks while the
  model reloads.
- `last_request_age_s` is time since the last API request that touched the
  engine (not a health probe).
- `requests`, `avg_rtf` (generation wall time / produced audio time), and
  `queue_depth` are live engine statistics.

### Voice catalog and cloning

These are **private extensions**, not part of the OpenAI Audio API.

#### `GET /v1/voices`

Lists every valid `voice` value: OpenAI aliases first, then catalog or installed
voices, then custom cloned voices (sorted).

```json
{
  "object": "list",
  "data": [
    {
      "id": "alloy",
      "aliases": ["alloy", "alba"],
      "source": "builtin",
      "language": "en",
      "license": "https://huggingface.co/kyutai/tts-voices",
      "cached": true
    },
    {
      "id": "mario",
      "aliases": ["mario"],
      "source": "custom",
      "language": "it",
      "license": null,
      "cached": false,
      "safetensors": true
    }
  ],
  "backend": "pocket",
  "clone_supported": true
}
```

- `source` is `builtin` or `custom`.
- `aliases` for a builtin includes the alias and its backend target
  (`["alloy", "alba"]`).
- `cached` reports whether the voice state is resident in the in-process LRU.
- `license` points at the backend's voice repository, not a per-file legal
  analysis. Check the individual voice's model card before redistribution.
- On Piper, additional entries are the `.onnx` voices actually installed in
  `TTS_PIPER_VOICE_DIR`; there is no static Piper catalog.
- On pocket, the static Kyutai catalog is listed.
- `clone_supported` is true only on a pocket backend with a voice registry.

#### `POST /v1/voices` — clone a voice

**pocket-tts only.** On Piper it returns `501` (or `503` while the engine is
still loading).

`multipart/form-data`:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `name` | string | yes | `^[a-z0-9][a-z0-9_-]{0,63}$`: 1–64 chars, starts lowercase/digit, only lowercase letters, digits, `_`, `-`. |
| `file` | upload | yes | `.wav`, `.mp3`, or `.flac`. Extension decides the decoder. Max `TTS_MAX_UPLOAD_MB` (default 25 MB). |
| `language` | string | no | Free-form informational tag stored with the voice and echoed by the catalog. |

Status codes:

| Status | Cause |
| --- | --- |
| `201` | Voice cloned, exported, and registered. Returns the custom-voice payload. |
| `400` | Invalid name; unsupported extension; audio prompt could not be encoded. |
| `409` | Name collides with an OpenAI alias, a Kyutai catalog voice, or an existing custom voice. |
| `413` | Upload exceeds the configured MB limit. Nothing is persisted. |
| `422` | Missing `name` or `file`. |
| `501` | Backend is Piper (no cloning support). |
| `503` | Engine is still loading. |

Example:

```bash
curl -s -X POST http://localhost:8080/v1/voices \
  -F 'name=mario' \
  -F 'language=it' \
  -F 'file=@prompt.wav' | jq
```

The encode step runs under the global generation lock because it touches the
stateful batch-1 model. The `.safetensors` export runs **outside** the lock, so a
slow disk write delays only the clone, not concurrent synthesis.

#### `DELETE /v1/voices/{name}`

| Status | Meaning |
| --- | --- |
| `204` | Custom voice removed: registry entry, `.safetensors` file, and LRU cache slot. |
| `404` | Unknown custom voice (or no registry on Piper). |
| `405` | Name is a built-in voice (alias, Kyutai catalog voice, or installed Piper voice). |
| `503` | Engine is still loading. |

#### Persistence

Custom voices live under `<cache_dir>/voices/`:

```text
<cache_dir>/voices/
├── registry.json
├── mario.safetensors
└── jarvis1.safetensors
```

- `TTS_CACHE_DIR` defaults to `~/.cache/pocket_tts`; Docker uses
  `/data/cache`.
- `registry.json` is rewritten atomically (temp file + `os.replace`), so a crash
  never leaves a half-written index.
- Each voice state is exported to a temporary filename and atomically renamed to
  `<name>.safetensors`; stale `*.safetensors.tmp` files are swept on the next
  registry load.
- The registry is re-read at every boot, so cloned voices survive restarts
  without re-encoding.
- A custom voice is immediately usable as `voice` in `/v1/audio/speech`.

### Wyoming protocol (Home Assistant)

The same engine is also exposed over the [Wyoming protocol](https://github.com/rhasspy/rhasspy3/blob/master/docs/wyoming.md)
— a peer-to-peer TCP protocol (JSONL header + optional binary PCM payload)
used by Home Assistant and Rhasspy. It is **opt-in**:

```bash
TTS_WYOMING_ENABLED=1 uv run tts-openai
# listens on 0.0.0.0:10200 by default
```

Add it to Home Assistant either automatically (mDNS, see below) or manually:

```yaml
# configuration.yaml
wyoming:
  - host: 192.168.1.50
    port: 10200
```

or **Settings → Devices & Services → Add Integration → Wyoming Protocol**.

#### Events

| Direction | Event | Notes |
| --- | --- | --- |
| client → server | `describe` | request the service description |
| server → client | `info` | one TTS program, rebuilt on every `describe` |
| client → server | `synthesize` | `text`, optional `voice.name` / `voice.speaker` |
| server → client | `audio-start` | `rate` is **per voice** (16/22.05 kHz Piper, 24 kHz pocket) |
| server → client | `audio-chunk` | raw mono s16le PCM, `TTS_WYOMING_SAMPLES_PER_CHUNK` samples each |
| server → client | `audio-stop` | end of the utterance |
| client → server | `synthesize-start` / `synthesize-chunk`* / `synthesize-stop` | streaming text input |
| server → client | `audio-start` / chunks / `audio-stop` per sentence, then `synthesize-stopped` | streaming audio output |
| both | `ping` / `pong` | liveness |
| server → client | `error` | `text` + machine-readable `code`; the connection stays open |

`info` advertises `supports_synthesize_streaming=true`. Plain `synthesize`
already streams audio sentence by sentence: text is split with the same
`sentence-stream` detector `wyoming-piper` uses, one `generate_pcm` call per
sentence, and `TTS_WYOMING_SENTENCE_SILENCE` seconds of silence between
sentences. `TTS_WYOMING_AUTO_PUNCTUATION` appends terminal punctuation
to sentences that lack it (empty disables).

**Voices and speakers.** Advertised voices are the default voice, installed
Piper voices, the Kyutai catalog (pocket), and custom (cloned) pocket voices.
Aliases and any downloadable Piper id are still accepted in `voice.name` even
if not listed. Piper multi-speaker voices advertise their `speaker_id_map`;
select one with `voice.speaker` or set a default with
`TTS_PIPER_SPEAKER`. Unknown speaker names log a warning and fall back
to the voice default. HA caches the `info` voice list: reload the Wyoming
integration after adding/removing voices.

Per-request `voice.speaker` maps to Piper's `SynthesisConfig.speaker_id`.
`TTS_PIPER_LENGTH_SCALE`, `TTS_PIPER_NOISE_SCALE`, and
`TTS_PIPER_NOISE_W` mirror the Home Assistant Piper add-on options and
apply to **every** Piper synthesis path (HTTP and Wyoming); pocket-tts ignores
them.

**Discovery.** `TTS_WYOMING_ZEROCONF=1` registers
`_wyoming._tcp.local.` for Home Assistant auto-discovery, using
`TTS_WYOMING_NAME` as the instance name. mDNS from a Docker **bridge**
network usually does not reach the LAN: use host networking or add the
integration manually with the host/port above.

**Security.** Wyoming has no authentication or encryption by design. It is
meant for a trusted LAN; do not publish port 10200 to the internet.
`TTS_API_KEY` guards `/v1/*` **only** — it does not apply to Wyoming.
Bind to a private interface with `TTS_WYOMING_HOST=127.0.0.1` if only
local clients should connect.

**Differences from `wyoming-piper`.** This server advertises only installed
Piper voices plus the default (there is no static `voices.json` catalog); any
Piper id is still downloaded on demand when requested. Errors keep the
connection open instead of disconnecting, voice management stays on the HTTP
API rather than a separate web UI, and optional Piper phonemizers
(Japanese/Thai) are not probed at `describe` time — a missing phonemizer
surfaces as an error event when that voice is synthesized.

### Errors and authentication

Application-level errors use the OpenAI envelope:

```json
{
  "error": {
    "message": "Unknown voice 'nope'. ...",
    "type": "invalid_request_error",
    "code": "not_found"
  }
}
```

FastAPI request-validation failures (`422`) instead use the standard
`{"detail": ...}` shape.

| Status | Typical cause |
| --- | --- |
| `400` | Unknown model alias; empty `input`; unknown voice; unsupported format on the non-streaming ffmpeg path; invalid voice name/extension; failed voice encode. |
| `401` | API key configured but missing/incorrect. |
| `404` | Unknown custom voice on `DELETE`. |
| `405` | Attempt to delete a built-in voice. |
| `409` | Cloned-voice name collision. |
| `413` | Voice-prompt upload over `TTS_MAX_UPLOAD_MB`. |
| `422` | FastAPI request validation (for example an unknown `response_format`, missing multipart field, or malformed body). |
| `429` | Load shedding: `TTS_MAX_WAITING` exceeded, or `TTS_QUEUE_TIMEOUT_S` expired waiting for the generation lock. |
| `501` | Voice cloning requested on the Piper backend. |
| `503` | Engine not loaded yet. `/v1/audio/speech` and cloning return this until the background load succeeds. |

#### API key

Set `TTS_API_KEY` to require:

```text
Authorization: Bearer <key>
```

on every path starting with `/v1/`. `/health` stays public so orchestrators can
probe the service without a key. The check is an exact string comparison, not a
full auth system: use TLS and a real gateway for public deployments.

---

## Architecture

### High-level flow

```text
                    ┌─────────────────────────────────────────────────────┐
 POST /v1/audio/    │ FastAPI app (create_app)                            │
 speech ───────────▶│  • optional Bearer middleware on /v1/*              │
                    │  • SpeechRequest validation                        │
                    │  • voice validation + capacity preflight (stream)   │
                    └───────────────────────┬─────────────────────────────┘
                                            ▼
                    ┌─────────────────────────────────────────────────────┐
                    │ TTSEngine                                            │
                    │  • ensure_loaded() / single-flight reload            │
                    │  • voice_state(): resolve + encode + LRU cache       │
                    │  • _gen_lock: one generation at a time (FIFO)        │
                    │  • _QueueGuard: in-flight accounting / 429           │
                    └───────────────┬───────────────────────┬─────────────┘
                                    ▼                       ▼
                         ┌──────────────────┐     ┌──────────────────────┐
                         │ PiperTTSModel    │     │ pocket_tts.TTSModel  │
                         │ per-voice ONNX   │     │ one model/language   │
                         │ 16/22.05 kHz     │     │ 24 kHz + voice states│
                         └────────┬─────────┘     └──────────┬───────────┘
                                  └──────────────┬─────────────┘
                                                 ▼
                                    float32 mono audio chunks
                                                 ▼
                         ┌─────────────────────────────────────────────┐
                         │ to_pcm16 (s16le)                            │
                         │  ├─ wav: wave / streaming 0xFFFFFFFF header │
                         │  ├─ pcm: raw chunks                         │
                         │  └─ mp3/opus/aac/flac: ffmpeg subprocess    │
                         └─────────────────────────────────────────────┘
```

### Application and startup

`main.py:create_app(config, engine)` builds the app:

- stores `config` in `app.state.config`;
- registers the OpenAI-shaped `OpenAIError` exception handler;
- mounts the `/v1` aggregate router (guarded by the `verify_api_key` dependency
  when `TTS_API_KEY` is set) and the public `/health` router;
- the `/v1` router includes `routers/audio.py`, `routers/models.py`, and
  `routers/voices.py`.

Production calls `create_app(config)` with no engine. During lifespan startup:

1. `app.state.engine = None`;
2. a daemon thread runs `_load_engine_background`, which calls `load_engine`,
   stores the result, logs backend/default voice/sample rate, and runs
   `engine.warmup()`;
3. if `TTS_IDLE_UNLOAD_S > 0`, a second daemon thread runs the idle
   watchdog;
4. if `TTS_WYOMING_ENABLED=1`, `main.py` starts the Wyoming TCP listener
   in the same event loop; handlers share the single engine and wait for its
   background load before answering `describe`/`synthesize`.

The HTTP server binds immediately. Until the background load finishes,
`/health` reports `loading` and audio/voice-cloning routes return `503`. If the
engine load fails, the server keeps returning `503`; the traceback is logged.

Tests inject a prebuilt `TTSEngine` (often around a fake model). Injected
engines are **not** unloaded on shutdown; only engines created by this server
(`_auto_loaded=True`) are released, so a shared/supervised engine is not torn
down unexpectedly.

Shutdown:

1. stop the idle watchdog;
2. if the engine is owned by the server, call `engine.unload(reason="server
   shutdown")`, which logs RSS before/after;
3. injected engines are left resident and logged as such.

### Engine and concurrency

`TTSEngine` wraps a duck-typed backend (`TTSModelLike`) and owns:

- `_gen_lock` — a single global `threading.Lock` serializing all generation.
  Requests are served FIFO; a streaming request holds the lock for the entire
  stream.
- `_state_lock` — protects the voice-state LRU and in-flight encode map.
- `_reload_lock` — single-flight model reload after idle eviction.
- `_voice_states` — `OrderedDict` LRU of encoded voice states.
- `_encode_pending` — per-voice single-flight events, so concurrent misses for
  the same voice encode once.
- `stats` — requests, audio seconds, generation seconds, waiting count,
  max waiting, unload/reload counters.

Generation path:

1. Routes validate the request and resolve the voice/default.
2. `generate_pcm`:
   - calls `touch()` (idle timer);
   - enters `_QueueGuard` (counts in-flight requests);
   - rejects with `RateLimited` → `429` if the queue is over capacity;
   - captures the model via `ensure_loaded()`;
   - loads/encodes the voice state;
   - acquires `_gen_lock` (optionally with `queue_timeout_s`);
   - calls `model.generate_audio(state, text)`;
   - releases the lock and converts the returned buffer to PCM;
   - updates request/audio/RTF stats.
3. `generate_pcm_stream` does the same but keeps `_gen_lock` while yielding
   chunks from `model.generate_audio_stream`. If the client disconnects, the
   generator is closed and the lock is released. The queue guard spans the whole
   stream.
4. Streaming routes call `check_capacity()` **before** sending headers, so
   overload becomes a clean `429` rather than a stream that dies midway. They
   also validate the voice eagerly so an unknown voice is a clean `400`, not a
   mid-stream 500.

Important consequences:

- **Throughput is serialized per process.** Concurrency is a queue, not
  parallel generation on one model.
- **Voice encoding happens outside `_gen_lock`.** Loading a new Piper voice or
  encoding a pocket prompt does not block an in-progress generation, except for
  `clone_voice`, which intentionally encodes under the lock.
- **`max_waiting` and `queue_timeout_s` are load-shedding controls**, not
  parallelism controls. They turn overload into `429` instead of unbounded
  queueing.
- **Horizontal scaling is N processes/containers behind a load balancer.** Each
  has its own model, voice LRU, and generation lock.

### Voice resolution and caching

Resolution is backend-specific:

**Piper**

- OpenAI aliases map through `Config.voice_map`; by default every alias maps to
  `it_IT-paola-medium`.
- A canonical Piper id (`it_IT-paola-medium`, `en_US-lessac-medium`, …) passes
  through.
- A local `.onnx` path passes through; a matching `.onnx.json` must sit next to
  it.
- Missing voices are downloaded on first use from
  [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) unless
  `TTS_PIPER_AUTO_DOWNLOAD=false`.
- `generate_audio` concatenates Piper's per-sentence chunks; streaming yields
  one chunk per sentence.

**pocket-tts**

- OpenAI aliases map to Kyutai catalog voices through `Config.voice_map`.
- Catalog names pass through.
- Local paths (`/…`, `./…`, `../…`, `~/…`), `.wav`/`.mp3`/`.flac`/
  `.safetensors` paths, `http(s)://` URLs, and `hf://<repo>/<path>` references
  pass through.
- Unknown bare names fail fast with a helpful `400`.

The engine caches encoded voice states in an LRU. The default size is:

- **Piper: 2** — each entry is a whole ONNX session (~60–110 MB).
- **pocket-tts: 32** — entries are lightweight-ish model-state snapshots.

`TTS_MAX_CACHED_VOICES` overrides this for either backend.

`sample_rate_for(voice)` is always used for response headers and ffmpeg input;
never assume the model-wide `sample_rate` is correct for a Piper request.

### Idle eviction

After `TTS_IDLE_UNLOAD_S` (default 300 s) with no API request:

1. the watchdog checks the idle age;
2. it tries to acquire `_gen_lock` **non-blocking**; if a generation or stream
   is running, it skips this pass and retries next tick;
3. it clears the model reference and the voice-state LRU;
4. it runs `gc.collect()` and best-effort `malloc_trim(0)` to return heap to
   the OS;
5. it increments `unloads` and logs RSS before/after.

The next request calls `ensure_loaded()` and blocks on a single-flight reload.
Concurrent waiters share one rebuild, then `warmup()` re-encodes any configured
warmup voices. This is a **block-until-reloaded** policy: the API returns 503
only before the first successful load, never for a post-eviction wake.

The watchdog interval is `min(idle_poll_s, max(5, idle_unload_s // 2))`. Health
probes do not call `touch()`, so they never reset the window.

### Cloning pipeline

`POST /v1/voices`:

1. reject if the backend cannot clone (Piper) or the engine is still loading;
2. validate the name regex and reject collisions;
3. validate the upload extension;
4. stream the upload to a temp file in the registry directory, enforcing the MB
   limit;
5. call `engine.clone_voice(name, tmp_path)`:
   - `get_state_for_audio_prompt` encodes the prompt **under `_gen_lock`**;
   - `export_model_state` writes a temporary `.safetensors` outside the lock;
   - atomic `replace` moves it to `<name>.safetensors`;
   - the state is inserted into the live LRU;
6. `registry.add(name, language=...)` atomically rewrites `registry.json`;
7. return `201` with the custom-voice entry.

Deletion removes the registry entry, evicts the cache slot, and unlinks the
`.safetensors` file. The registry persists across restarts.

### Audio pipeline

The engine emits float32 mono audio ([-1, 1]). `to_pcm16`:

- detaches torch tensors to CPU if needed;
- clips to [-1, 1] and scales by 32767;
- converts to `<i2` and returns bytes.

**Contract:** `to_pcm16` may mutate a writable float32 input in place. Callers
must not reuse an audio chunk after conversion.

WAV handling:

- `pcm_to_wav` uses the stdlib `wave` module for a complete file.
- `streaming_wav_header` emits the canonical 44-byte mono/16-bit header with
  `RIFF` and `data` sizes set to `0xFFFFFFFF`; audio chunks follow directly.

Compressed output (`audio_codecs.py`):

- ffmpeg is read mono s16le on stdin at the resolved sample rate and writes the
  requested format on stdout.
- MP3: `libmp3lame -q:a 4` (roughly 128 kbps VBR).
- Opus: `libopus -b:a 96k -f ogg`.
- AAC: `aac -b:a 128k -f adts`.
- FLAC: `flac`.
- Buffered mode uses `subprocess.run` and captures the complete output.
- Live mode uses `Popen` with two pipes and three threads:
  - the ASGI worker reads ffmpeg stdout in 64 KB `os.read` calls and yields
    chunks;
  - a pump thread feeds PCM chunks into ffmpeg stdin and flushes each one;
  - a stderr-drain thread prevents ffmpeg's error pipe from filling (~64 KB)
    and deadlocking the encoder.
- Teardown order on disconnect/error is deliberate: kill ffmpeg, join the pump,
  then close the PCM iterator. This guarantees the engine's generation lock is
  released only after no thread is still reading the source stream.
- Streamed compressed output is byte-identical to the buffered output because
  `-flush_packets 1` changes delivery timing, not the encoded bytes.

Sample rates:

- pocket-tts: 24 kHz for every voice;
- Piper: 16 kHz for `x_low`/`low`, 22.05 kHz for `medium`/`high`.

### Module layout

```text
src/tts_openai/
├── config.py          # frozen Config dataclass, env parsing, backend defaults
├── main.py            # create_app, lifespan, router wiring, background load, healthcheck
├── dependencies.py    # get_config/get_engine/require_engine, Bearer dependency
├── schemas.py         # SpeechRequest + model/format constants
├── routers/
│   ├── audio.py       # /v1/audio/speech
│   ├── models.py      # /v1/models
│   ├── health.py      # /health (public)
│   └── voices.py      # GET/POST/DELETE /v1/voices
├── wyoming_server.py  # Wyoming TCP listener (info/synthesize/streaming)
├── engine.py          # TTSEngine, locks/LRU, idle eviction, PCM/WAV helpers, load_engine
├── piper_backend.py   # PiperTTSModel adapter (download, load, synth, streaming, speakers)
├── voice_registry.py  # persistent custom-voice index (atomic JSON + safetensors)
├── voices.py          # aliases, Kyutai catalog, Piper id resolution, language tags
├── audio_codecs.py    # ffmpeg buffered + live encoding
├── errors.py          # OpenAI-shaped error helpers
└── server.py          # compatibility shim -> main
```

`tests/` contains contract, engine, config, voice-registry, Piper, streaming,
idle-unload, and opt-in real-backend integration tests. `benchmarks/` contains
Piper-vs-pocket and end-to-end TTS→STT benchmarks.

---

## Configuration

Every setting is optional and read from the environment by `Config.from_env()`.
Invalid `TTS_*` integers/floats raise a `ValueError` at startup rather
than silently falling back. Booleans accept `1`, `true`, `yes`, `on`
(case-insensitive), except `TTS_QUANTIZE`, which accepts `1`, `true`, or
`yes`.

| Variable | Default | Applies to | Description |
| --- | --- | --- | --- |
| `TTS_BACKEND` | `piper` | both | `piper` or `pocket`. Anything else fails startup. |
| `TTS_HOST` | `127.0.0.1` | both | uvicorn bind host. Docker overrides to `0.0.0.0`. |
| `TTS_PORT` | `8000` | both | uvicorn bind port and healthcheck port. |
| `TTS_LOG_LEVEL` | `info` | both | Python/uvicorn log level: `debug`, `info`, `warning`, `error`, `critical`. `debug` also logs per-request `/v1/audio/speech` details. Successful `/health` probes are never access-logged; failing ones are. Every 4xx/5xx response is logged with its reason (`WARNING`/`ERROR`, plus a traceback for unhandled exceptions). |
| `TTS_LOG_FILE` | *(empty)* | both | Optional rotating log file (10 MB per file, 5 backups); console logging stays enabled. Docker Compose writes to `/logs/server.log`, persisted to `./logs`. |
| `TTS_DEFAULT_VOICE` | backend default | both | Voice used when the request omits `voice`. Empty → backend default (`it_IT-paola-medium` / `alloy`). |
| `TTS_VOICE_MAP` | backend defaults | both | Comma-separated `alias=voice` overrides, merged over the default alias map. A bare value raises `ValueError`. |
| `TTS_LANGUAGE` | `english` | pocket | pocket-tts model language, e.g. `english`, `italian`, `italian_24l`, `french_24l`. Ignored by Piper, which derives language from the voice id. |
| `TTS_QUANTIZE` | `false` | pocket | Load transformer attention/FFN weights as int8 (dynamic quantization; requires AVX2 on x86). ~62% less RSS and ~27% faster in measurements. |
| `TTS_MAX_CACHED_VOICES` | `2` Piper / `32` pocket | both | Voice-state LRU size. A Piper entry is a whole ONNX session. |
| `TTS_PIPER_VOICE_DIR` | `~/.cache/piper` | piper | Directory for downloaded `.onnx` + `.onnx.json` pairs. Docker uses `/data/cache/piper`. |
| `TTS_PIPER_AUTO_DOWNLOAD` | `true` | piper | Download missing Piper voices on first use. When false, missing voices return `400`. |
| `TTS_WARMUP_VOICES` | *(empty)* | both | Comma-separated voices to load/encode at startup and after every reload. Warmup failures are logged, not fatal. |
| `TTS_CACHE_DIR` | `~/.cache/pocket_tts` | pocket | Base directory for the custom-voice registry (`<dir>/voices`). Docker uses `/data/cache`. |
| `TTS_MAX_UPLOAD_MB` | `25` | pocket | Maximum voice-cloning prompt upload. Does not limit `input` text. |
| `TTS_IDLE_UNLOAD_S` | `300` | both | Evict the model from RAM after this many idle seconds. `0` keeps it resident. Next request blocks during reload. |
| `TTS_IDLE_POLL_S` | `30` | both | Idle watchdog poll interval. Effective interval is capped at `max(5, idle_unload_s // 2)`. |
| `TTS_MAX_WAITING` | `0` | both | Load shedding: return OpenAI `429` when more than this many requests are already inside the critical path (encode + wait + generate). `0` = unlimited. |
| `TTS_QUEUE_TIMEOUT_S` | `0` | both | Maximum seconds a request waits for the generation lock before `429`. Accepts fractions. `0` = wait forever. |
| `TTS_API_KEY` | *(unset)* | both | If set, require `Authorization: Bearer <key>` on `/v1/*`. `/health` remains public. |
| `TTS_PIPER_SPEAKER` | *(unset)* | piper | Default speaker name or numeric id for multi-speaker Piper voices. Unknown names fall back to the voice default. |
| `TTS_PIPER_LENGTH_SCALE` | *(voice default)* | piper | Speaking-rate scale: `<1` faster, `>1` slower. Also affects HTTP synthesis. |
| `TTS_PIPER_NOISE_SCALE` | *(voice default)* | piper | Generator noise scale (HA add-on `noise_scale`). |
| `TTS_PIPER_NOISE_W` | *(voice default)* | piper | Phoneme-width noise scale (HA add-on `noise_w`). |
| `TTS_WYOMING_ENABLED` | `false` | both | Enable the Wyoming TTS listener. Unauthenticated by protocol design; use a trusted network. |
| `TTS_WYOMING_HOST` | `0.0.0.0` | both | Wyoming bind host. `127.0.0.1` limits it to local clients. |
| `TTS_WYOMING_PORT` | `10200` | both | Wyoming TCP port. |
| `TTS_WYOMING_SAMPLES_PER_CHUNK` | `1024` | both | Mono s16 samples per `audio-chunk` (2048 bytes at the default). |
| `TTS_WYOMING_SENTENCE_SILENCE` | `0.0` | both | Seconds of silence inserted between sentences. |
| `TTS_WYOMING_AUTO_PUNCTUATION` | `.?!。？！．؟` | both | Append the first character when a sentence lacks terminal punctuation. Empty disables. |
| `TTS_WYOMING_ZEROCONF` | `false` | both | Register `_wyoming._tcp.local.` for Home Assistant auto-discovery. |
| `TTS_WYOMING_NAME` | `tts-openai` | both | mDNS instance name used with `TTS_WYOMING_ZEROCONF=1`. |
| `TTS_WYOMING_READY_TIMEOUT_S` | `60` | both | How long `describe`/`synthesize` wait for the background model load before returning an error event. `0` fails fast. |

External variables used by the engines:

| Variable | Default | Description |
| --- | --- | --- |
| `HF_HOME` | `~/.cache/huggingface` / `/data/hf` in Docker | Hugging Face weights cache. Persistent volume recommended. |
| `HF_TOKEN` | *(unset)* | Token for gated repositories (notably `kyutai/pocket-tts` for pocket weights). |
| `OMP_NUM_THREADS` | engine default | OpenMP/CPU thread count for torch/ONNX workloads. Benchmark before changing. |

### Env template (all optional, commented)

```sh
# --- server ---
# TTS_BACKEND=piper
# TTS_HOST=127.0.0.1
# TTS_PORT=8000
# TTS_LOG_LEVEL=info
# TTS_LOG_FILE=/var/log/tts-openai/server.log
# TTS_API_KEY=change-me

# --- Wyoming TTS (Home Assistant / Rhasspy) ---
# TTS_WYOMING_ENABLED=0
# TTS_WYOMING_HOST=0.0.0.0
# TTS_WYOMING_PORT=10200
# TTS_WYOMING_NAME=tts-openai
# TTS_WYOMING_ZEROCONF=0
# TTS_WYOMING_SAMPLES_PER_CHUNK=1024
# TTS_WYOMING_SENTENCE_SILENCE=0.0
# TTS_WYOMING_AUTO_PUNCTUATION=.?!。？！．؟
# TTS_WYOMING_READY_TIMEOUT_S=60

# --- backend / voices ---
# TTS_DEFAULT_VOICE=it_IT-paola-medium
# TTS_VOICE_MAP=alloy=en_US-lessac-medium
# TTS_WARMUP_VOICES=jarvis1
# TTS_LANGUAGE=italian_24l
# TTS_QUANTIZE=1
# TTS_MAX_CACHED_VOICES=2
# TTS_PIPER_VOICE_DIR=/data/cache/piper
# TTS_PIPER_AUTO_DOWNLOAD=true
# TTS_PIPER_SPEAKER=0
# TTS_PIPER_LENGTH_SCALE=1.0
# TTS_PIPER_NOISE_SCALE=0.667
# TTS_PIPER_NOISE_W=0.333

# --- cache / uploads ---
# TTS_CACHE_DIR=/data/cache
# TTS_MAX_UPLOAD_MB=25

# --- lifecycle / load shedding ---
# TTS_IDLE_UNLOAD_S=300
# TTS_IDLE_POLL_S=30
# TTS_MAX_WAITING=0
# TTS_QUEUE_TIMEOUT_S=0

# --- external ---
# HF_HOME=/data/hf
# HF_TOKEN=hf_...
# OMP_NUM_THREADS=4
```

### Backend defaults and validation

`Config.__post_init__` applies backend-aware defaults:

- `backend` is lowercased and must be `piper` or `pocket`.
- An empty `voice_map` becomes `backend_default_aliases(backend)`.
- An empty `default_voice` becomes `backend_default_voice(backend)`.

Default alias maps:

| Alias | Piper target | pocket target | Language |
| --- | --- | --- | --- |
| `alloy` | `it_IT-paola-medium` | `alba` | en |
| `echo` | `it_IT-paola-medium` | `charles` | en |
| `fable` | `it_IT-paola-medium` | `eponine` | en (British) |
| `onyx` | `it_IT-paola-medium` | `bill_boerst` | en |
| `nova` | `it_IT-paola-medium` | `eve` | en |
| `shimmer` | `it_IT-paola-medium` | `fantine` | en |
| `coral` | `it_IT-paola-medium` | `giovanni` | it |

`coral` is a project extension, not an OpenAI voice. On Piper every OpenAI alias
intentionally points at the same Italian voice; override individual aliases with
`TTS_VOICE_MAP`.

### Important configuration precedence

- Per-request `voice` → `TTS_DEFAULT_VOICE` / backend default.
- `TTS_VOICE_MAP` overrides merge over (not replace) the default alias
  map.
- `TTS_BACKEND=pocket` requires both the engine extra installed and the
  model language downloaded/gated appropriately.
- `TTS_LANGUAGE` and `TTS_QUANTIZE` are pocket-only; Piper ignores
  them.
- `TTS_PIPER_*` settings are Piper-only; pocket ignores them.

---

## Voices

### Piper voice ids

Format: `<lang>_<REGION>-<name>-<quality>` where quality is `x_low`, `low`,
`medium`, or `high`. Examples:

```text
it_IT-paola-medium
it_IT-paola-low
en_US-lessac-medium
de_DE-thorsten-high
```

- `x_low`/`low` → 16 kHz; `medium`/`high` → 22.05 kHz.
- A local `.onnx` path is also accepted, but the matching `.onnx.json` must
  exist next to it.
- Piper voices are downloaded on demand from `rhasspy/piper-voices`; the Docker
  image bakes in `it_IT-paola-medium`.
- `GET /v1/voices` lists aliases plus the Piper voices already installed in the
  voice directory.

### Kyutai catalog

The pocket backend accepts every name in `KYUTAI_CATALOG` directly. Aliases:
`alloy` → `alba`, `echo` → `charles`, `fable` → `eponine`, `onyx` →
`bill_boerst`, `nova` → `eve`, `shimmer` → `fantine`, `coral` → `giovanni`.

Additional catalog names include:

```text
alba, anna, azelma, bill_boerst, caro_davy, charles, cosette, eponine, eve,
fantine, george, jane, javert, jean, marius, mary, michael, paul,
peter_yearsley, stuart_bell, vera, giovanni, lola, juergen, rafael, estelle
```

Language tags for non-English catalog voices: `giovanni` (it), `lola` (es),
`juergen` (de), `rafael` (pt), `estelle` (fr). Anything not listed defaults to
`en` in the catalog metadata.

Piper takes its language from the voice id; pocket takes it from
`TTS_LANGUAGE` (the model), not from the voice name.

### The bundled `jarvis1` voice

The repository ships one pre-cloned Italian voice:

- file: `voices/jarvis1.safetensors` (~68 MB);
- index: `voices/registry.json`;
- source: five Italian corpus clips joined with 0.5 s of silence into a ~27.5 s
  conditioning prompt;
- encoded once with pocket-tts `get_state_for_audio_prompt`;
- requires the `italian_24l` pocket model (the same language it was encoded
  from).

Serve it with the pocket backend:

```bash
TTS_BACKEND=pocket \
TTS_LANGUAGE=italian_24l \
TTS_WARMUP_VOICES=jarvis1 \
uv run tts-openai
```

```bash
curl -s localhost:8000/v1/audio/speech \
  -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"jarvis1","input":"Questa è la voce di Jarvis."}' \
  -o jarvis1.wav
```

`italian_24l` weights come from the gated `kyutai/pocket-tts` repository. On a
cold cache, pass `HF_TOKEN` or mount a pre-populated `/data/hf`. The voice file
itself is ours and needs no token.

### Cloning guidance

- Use a **16-bit PCM WAV** prompt. Pocket-tts may require `soundfile` to decode
  `.mp3`/`.flac`; the zero-dependency path is WAV.
- A single clean speaker recording works best. The bundled `jarvis1` used ~27.5 s
  of concatenated speech.
- The cloned voice is tied to the model language it was encoded from. A voice
  cloned under `italian_24l` is not meaningful under `english`.
- Pre-encode at boot with `TTS_WARMUP_VOICES=name` if first-request
  latency matters.

---

## Memory and performance

Measurements use `benchmarks/bench_backends.py` on an **Intel N150 (4 threads),
15 GiB RAM, CPU only**, three runs per text, each backend in a fresh process:

| Metric | Piper (`it_IT-paola-medium`) | pocket (`alloy`) |
| --- | ---: | ---: |
| Sample rate | 22,050 Hz | 24,000 Hz |
| Model load | **3.1 s** | 15.8 s |
| Resident RSS (loaded) | **171 MB** | 887 MB |
| RTF short / medium / long | **0.158 / 0.135 / 0.134** | 0.573 / 0.550 / 0.556 |
| Wall p50, long text | **3.6 s** | 19.3 s |
| Stream TTFA | 0.357 s | **0.221 s** |
| Stream RTF | **0.133** | 0.546 |
| Throughput @ 4 clients (audio s / wall s) | **6.1** | 1.6 |

- **RTF** = generation wall time ÷ produced audio time; lower is better.
- Piper is roughly 4× faster and 5× lighter on this host.
- Piper emits one chunk per sentence, so its TTFA is first-sentence latency.
  Pocket streams sub-sentence chunks and can win TTFA on multi-sentence text
  despite its higher RTF.
- Both backends serialize generation through one global lock. The throughput
  row is server throughput, not per-model parallelism.

### Resident memory and quantization

Piper:

- `it_IT-paola-medium` is 64 MB on disk and ~171 MB resident when loaded.
- Each cached Piper voice is a separate ONNX session, hence the small LRU
  default (`TTS_MAX_CACHED_VOICES=2`).

pocket-tts:

- Each language has a base model (6-layer, ~430 MB download); some languages
  offer larger `_24l` quality variants. There is no smaller `tiny`/`nano` tier.
- `TTS_QUANTIZE=1` loads attention/FFN weights as int8 (dynamic
  quantization, FBGEMM on x86, requires AVX2):

| `TTS_QUANTIZE` | Resident RSS |
| --- | ---: |
| `false` (default) | ~1,042 MB |
| `1` | ~395 MB (≈62% less) |

Quantization also measured ~27% faster inference on x86 with no measured WER
change. For lower counts, `pip install pocket-tts[quantize]` swaps in the
`torchao` backend.

### Idle eviction numbers

Pocket reference host, CPU torch, `/proc/self/status` VmRSS:

| State | RSS |
| --- | ---: |
| empty Python | ~32 MB |
| torch + sentencepiece imported, no model | ~229 MB |
| engine loaded (model + voice LRU) | ~1,030 MB |
| after dropping model + `gc.collect()` | ~430 MB |
| after `gc.collect()` + `malloc_trim(0)` | ~405 MB |
| warm reload from HF cache | ~0.9 s → ~1,030 MB |

Eviction reclaims ~600 MB (≈60%). The remaining floor is loaded libraries plus
allocator slack; going lower requires restarting the process, which is out of
scope.

### End-to-end TTS → STT

`benchmarks/e2e_tts_stt.py` drives a sequential Piper → Italian faster-whisper
pipeline over HTTP and writes results under `artifacts/e2e-tts-stt/`. Setup and
measured results are in [`benchmarks/e2e_tts_stt.md`](benchmarks/e2e_tts_stt.md).
It requires a sibling `stt-openai-server` checkout.

The follow-up STT study — CPU tuning matrix, real-human FLEURS accuracy, and a
short-command evaluation — is in
[`benchmarks/stt_improvements.md`](benchmarks/stt_improvements.md) with the
corresponding `stt_tuning_matrix.py`, `prepare_fleurs_it.py`,
`stt_accuracy.py`, and `stt_command_eval.py` scripts.

---

## Docker

### Dockerfile walkthrough

`deploy/Dockerfile` is a two-stage, CPU-only image following the
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/).

#### Build arguments

| Argument | Default | Effect |
| --- | --- | --- |
| `WITH_POCKET` | `0` | `1` installs the `engine` extra (pocket-tts + CPU torch), enlarges the image to roughly 1 GB, and seeds `jarvis1`. |
| `PIPER_VOICE` | `it_IT-paola-medium` | Voice downloaded and baked into the image at build time. |

#### Stage 1 — builder

```dockerfile
FROM python:3.14-slim AS builder
ARG WITH_POCKET=0
COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_POCKET" = "1" ]; then \
        uv sync --locked --no-install-project --no-dev --no-editable \
            --extra piper --extra engine; \
    else \
        uv sync --locked --no-install-project --no-dev --no-editable \
            --extra piper; \
    fi
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ "$WITH_POCKET" = "1" ]; then \
        uv sync --locked --no-dev --no-editable --extra piper --extra engine; \
    else \
        uv sync --locked --no-dev --no-editable --extra piper; \
    fi
```

Details:

- **uv 0.13.1** is pinned to the version that generated `uv.lock`. Use that
  version for lock regeneration, or `--locked` builds can fail.
- **python:3.14-slim** is the build/runtime interpreter. `UV_PYTHON_DOWNLOADS=0`
  keeps both stages on the same system interpreter path so the venv can be
  copied safely.
- **Two sync layers**:
  1. dependencies only (`--no-install-project`), cached until
     `pyproject.toml`/`uv.lock` change;
  2. project install (`--no-editable`), after copying `README.md` and `src`.
- **`--no-dev`** excludes pytest/httpx.
- **`--no-editable`** makes the venv self-contained; the runtime image does not
  need `/app/src`.
- **Default extras**: `piper` only. `WITH_POCKET=1` adds `engine` in both sync
  layers.
- **torch is CPU-only** because `pyproject.toml` maps `torch` to the explicit
  `pytorch-cpu` index (`https://download.pytorch.org/whl/cpu`) via
  `[tool.uv.sources]`. No CUDA/NVIDIA layers are installed.

#### Stage 2 — runtime

```dockerfile
FROM python:3.14-slim AS runtime
ARG WITH_POCKET=0
ARG PIPER_VOICE=it_IT-paola-medium
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    TTS_BACKEND=piper \
    TTS_HOST=0.0.0.0 \
    TTS_PORT=8000 \
    TTS_CACHE_DIR=/data/cache \
    TTS_PIPER_VOICE_DIR=/data/cache/piper \
    HF_HOME=/data/hf
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
RUN useradd --create-home --uid 10001 tts \
    && mkdir -p /data/cache/piper /data/hf
RUN python -m piper.download_voices "${PIPER_VOICE}" --download-dir /data/cache/piper
RUN --mount=type=bind,source=voices,target=/tmp/pocket-voices \
    if [ "$WITH_POCKET" = "1" ]; then \
        mkdir -p /data/cache/voices; \
        cp /tmp/pocket-voices/registry.json /data/cache/voices/registry.json; \
        cp /tmp/pocket-voices/jarvis1.safetensors /data/cache/voices/jarvis1.safetensors; \
    fi
RUN chown -R tts:tts /data
VOLUME ["/data"]
USER tts
EXPOSE 8000 10200
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["tts-openai", "--healthcheck"]
ENTRYPOINT ["tts-openai"]
```

Runtime details:

- **`libgomp1`** is the OpenMP runtime used by torch/ONNX.
- **`ffmpeg`** enables `mp3`/`opus`/`aac`/`flac`. Debian's build includes
  `libmp3lame` and `libopus`. Without it, only `wav`/`pcm` work locally.
- **Runtime env defaults** bind `0.0.0.0:8000`, select Piper, and place all
  writable state under `/data`:
  - Piper voices: `/data/cache/piper`
  - pocket registry: `/data/cache/voices`
  - HF weights: `/data/hf`
- **`PIPER_VOICE`** is downloaded at build time with
  `python -m piper.download_voices`; the first request needs no network for that
  voice. A fresh named volume inherits the baked `/data` contents.
- **`WITH_POCKET=1`** seeds the repo's `registry.json` and
  `jarvis1.safetensors` into `/data/cache/voices` using a bind mount. The 68 MB
  asset never enters the default Piper-only layer.
- **Non-root**: user `tts`, uid `10001`. `/data` is owned by `tts`.
- **`EXPOSE 8000 10200`**; Compose maps host `8080` → `8000` and `10200` → `10200`
  (Wyoming).
- **Healthcheck** calls `tts-openai --healthcheck` every 30 s, timeout
  5 s, 3 retries, 60 s start period. Because the HTTP server binds before the
  engine finishes loading, `/health` returns `200 {status: loading}` and Docker
  considers the container healthy while the engine warms up. The 60 s grace
  covers Python/uvicorn startup, not necessarily a cold pocket download.
- **Entrypoint** is the installed console script.

#### Image contents and size

- Default: Piper only, no torch, includes ffmpeg and the baked Paola voice;
  typically a few hundred MB.
- `WITH_POCKET=1`: adds pocket-tts, CPU torch, and the jarvis1 voice; roughly
  1 GB.

### `docker-compose.yml` walkthrough

```yaml
name: tts-openai

services:
  tts:
    build:
      context: .
      dockerfile: deploy/Dockerfile
      args:
        WITH_POCKET: "0"
    image: tts-openai:local
    ports:
      - "8080:8000"
      - "10200:10200"   # Wyoming
    environment:
      - TTS_HOST=0.0.0.0
      - TTS_PORT=8000
      - TTS_WYOMING_ENABLED=1
      - TTS_BACKEND=piper
      - TTS_DEFAULT_VOICE=it_IT-paola-medium
      # ...commented knobs...
    volumes:
      - tts-data:/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "tts-openai", "--healthcheck"]
      interval: 30s
      timeout: 5s
      start_period: 60s
      retries: 3

volumes:
  tts-data:
```

- **`name`** sets the Compose project name.
- **`build.args.WITH_POCKET: "0"`** builds the default Piper-only image. Set it
  to `"1"` for the pocket backend and jarvis1 seeding.
- **`PIPER_VOICE` is not exposed by the Compose file.** To bake a different
  voice, add it under `build.args` (or use `docker build --build-arg`), and set
  `TTS_DEFAULT_VOICE`/`TTS_VOICE_MAP` accordingly so runtime
  requests actually use it.
- **Ports** `8080:8000` (HTTP) and `10200:10200` (Wyoming).
- **Active environment** mirrors the image defaults: host, port, Piper backend,
  Paola. All other knobs are commented; uncomment and edit them to change
  behaviour.
- **`TTS_BACKEND=pocket`** only works if the image was built with
  `WITH_POCKET=1`.
- **Volume** `tts-data:/data` persists Piper voices, pocket weights, and cloned
  voices. Without it, a new container re-downloads voices and loses custom
  clones unless they are baked into the image.
- **`restart: unless-stopped`** keeps the service alive across crashes/daemon
  restarts.
- **Healthcheck** mirrors the image and uses the console script.

### Building and running

```bash
# Default Piper image
docker build -f deploy/Dockerfile -t tts-openai .

# Run, mapping host 8080 -> container 8000
docker run --rm -p 8080:8000 -p 10200:10200 \
  -e TTS_WYOMING_ENABLED=1 \
  -v tts-data:/data tts-openai

# Pocket-enabled image
docker build -f deploy/Dockerfile --build-arg WITH_POCKET=1 -t tts-pocket .

docker run --rm -p 8080:8000 -p 10200:10200 \
  -e TTS_WYOMING_ENABLED=1 \
  -e TTS_BACKEND=pocket \
  -e TTS_LANGUAGE=italian_24l \
  -v tts-data:/data \
  tts-pocket

# Different baked Piper voice
docker build -f deploy/Dockerfile \
  --build-arg PIPER_VOICE=en_US-lessac-medium \
  -t tts-en .

docker run --rm -p 8080:8000 -p 10200:10200 \
  -e TTS_WYOMING_ENABLED=1 \
  -e TTS_DEFAULT_VOICE=en_US-lessac-medium \
  -v tts-data:/data \
  tts-en
```

Compose:

```bash
docker compose up -d --build
docker compose logs -f tts
docker compose down

# Build the pocket variant by editing WITH_POCKET to "1", then:
docker compose build --no-cache
docker compose up -d
```

### Publishing

`.github/workflows/docker-publish.yml`:

1. runs `uv sync --locked` and `uv run pytest -q` as a gate (fake models, no
   engine extras needed);
2. builds `linux/amd64` with Buildx;
3. pushes to GHCR (`ghcr.io/<repository>`);
4. tags `latest` on `main` pushes, plus semver tags (`v*.*.*` → `x.y.z` and
   `x.y`);
5. uses GHA build cache and publishes SLSA build provenance attestations.

The workflow runs on pushes to `main`, version tags, and manual dispatch. It
does not run the optional real-backend e2e tests.

---

## Operations and troubleshooting

### `503` after start

- The engine is still loading. `/health` says `loading`.
- If it persists, inspect logs for an engine import failure (missing
  `--extra piper`/`--extra engine`) or a failed voice/model download.
- In Docker, the default image has Piper only; `TTS_BACKEND=pocket`
  without `WITH_POCKET=1` will fail to import pocket-tts.

### First request is slow

- Piper: the selected voice may need downloading or ONNX session creation.
- pocket: the language weights may need downloading, and the first use of a
  voice encodes/caches a prompt state.
- After idle eviction, the first request reloads the model (Piper: under a
  second warm; pocket: ~1 s warm; cold downloads can take much longer).
- Use `TTS_WARMUP_VOICES` to preload hot voices at boot.

### `429` responses

- Lower concurrency pressure from the client, or raise
  `TTS_MAX_WAITING`.
- If requests wait too long for the generation lock, set a finite
  `TTS_QUEUE_TIMEOUT_S`.
- Remember that one long stream holds the generation lock for its entire
  duration. Streaming is not parallel generation.

### Compressed formats return `400`

Install `ffmpeg` on `PATH` (the Docker image already has it). `wav`/`pcm` do not
need it. For streaming compressed requests on a host without ffmpeg, the
failure surfaces once the response body starts rather than as a clean upfront
400.

### Gated pocket model / `HF_TOKEN`

`kyutai/pocket-tts` is gated. On a cold `/data/hf`, pass `HF_TOKEN` or mount a
volume that already contains the weights. The `jarvis1` voice file itself needs
no token.

### Cloned voice is not usable / wrong language

- Ensure `TTS_BACKEND=pocket` (Piper has no cloning).
- Use the same model language the voice was encoded from (`italian_24l` for
  `jarvis1`).
- Verify `GET /v1/voices` lists it as `source: "custom"`.
- Upload a 16-bit PCM WAV prompt if `.mp3`/`.flac` decoding fails.

### Scaling

- One model and one global generation lock per process. Throughput scales by
  running multiple containers/processes behind a load balancer.
- Voice caches are per process. `.safetensors` files make pocket custom voices
  cheap to load on each replica; Piper voices are loaded per ONNX session.
- Long streams pin a generation slot; use separate replicas for latency-
  sensitive and batch workloads if needed.

### Security

- Set `TTS_API_KEY` for private deployments, but terminate TLS and add
  real rate limiting at a gateway.
- The built-in check is a simple exact Bearer comparison; it is not a full auth
  system.
- `/health` is intentionally public.
- The default Piper image includes GPL-3.0-or-later components; review
  `NOTICE` before redistribution.

---

## Development and testing

### Environment

```sh
uv sync                 # no engines; offline fake-model tests
uv run pytest           # full suite (addopts=-q)
uv run pytest tests/test_engine.py::test_generation_is_serialized   # one test
uv sync --extra piper   # + Piper, needed to synthesize locally
uv sync --extra engine  # + pocket-tts and CPU torch (large, optional)
uv run tts-openai
TTS_BACKEND=pocket uv run tts-openai
```

The test suite does not need either engine: `tests/conftest.py` injects a fake
model whose sample rate is 24 kHz and whose encode/generate methods simulate
cost.

### Test layers

- **Unit/contract**: API shape, errors, config parsing, engine locks/LRU,
  registry, streaming, audio codecs, idle eviction, Wyoming `describe`/
  `synthesize`/text-streaming over a real ephemeral TCP listener.
- **Piper fakes**: `test_piper_backend.py`, `test_piper_api.py` use fake
  `PiperVoice`/download functions and both 16 kHz and 22.05 kHz rates.
- **Opt-in real-backend e2e**:
  - `PIPER_E2E=1 uv run pytest -m e2e -v`
    (`tests/test_integration_piper.py`, downloads the 64 MB default voice).
  - `TTS_E2E=1 uv run pytest -m e2e -v`
    (`tests/test_integration.py`, downloads pocket weights).
  Both boot a real uvicorn server.
- **ffmpeg-dependent tests** skip cleanly when ffmpeg is absent.
- **Benchmarks**: `benchmarks/bench_backends.py` (Piper vs pocket, subprocess
  isolated), `benchmarks/e2e_tts_stt.py` (TTS→STT), plus the STT study scripts
  `stt_tuning_matrix.py`, `prepare_fleurs_it.py`, `stt_accuracy.py`, and
  `stt_command_eval.py`.

### CI

The GHCR workflow gates publishing on `uv sync --locked` plus `uv run pytest -q`.
No lint, formatter, or typecheck task is configured. `pyrightconfig.json`
exists but nothing runs pyright. Lockfile updates must use uv `0.13.1` (the
Docker pin).

---

## Known limitations

- **One model, one global generation lock.** Requests serialize; long streams
  hold the lock for their entire duration. `/health.queue_depth` shows waiting
  requests.
- **Streaming is not parallel generation.** Multiple streaming clients share
  one generation slot; the server remains responsive but throughput is serial.
- **Piper streams per sentence.** A single very long sentence has no early
  audio. Pocket streams sub-sentence chunks and may achieve lower TTFA.
- **Streaming WAV uses unknown sizes.** The `0xFFFFFFFF` convention is widely
  accepted but is not a seekable, finalized WAV header.
- **`speed` and `instructions` are ignored.** There is no time-stretching or
  instruction control. `language` in the request body is also ignored; server
  configuration wins.
- **`model` aliases do not select different models.** `tts-1`, `tts-1-hd`, and
  `gpt-4o-mini-tts` all serve the configured backend.
- **Default `response_format` is `wav`**, not OpenAI's usual `mp3`. OpenAI SDK
  callers that rely on the API default may receive WAV unless they set the
  format explicitly.
- **No built-in web UI.** Custom voices are managed via HTTP or CLI.
- **Wyoming is unauthenticated.** The protocol has no auth/encryption and
  `TTS_API_KEY` does not apply; keep port 10200 on a trusted network.
- **Wyoming advertises installed Piper voices only.** There is no static
  `voices.json` catalog; any valid Piper id is still downloaded on demand when a
  client requests it.
- **Cloning is pocket-only.** Piper returns `501`.
- **Cloned voices are model-language-specific.** Use the same
  `TTS_LANGUAGE` used to encode the prompt.
- **Pocket prompt decoding may need `soundfile`** for MP3/FLAC uploads; 16-bit
  PCM WAV is the safest format.
- **Missing ffmpeg in streaming mode fails after headers**, not as an upfront
  400. The Docker image includes ffmpeg.
- **No automatic multi-worker scaling.** Each process owns one model and one
  generation lock; scale with replicas.
- **Simple authentication only.** Use a gateway for TLS, rate limits, and
  identity.

---

## License

The project's own code is MIT. The default Piper backend and default Docker
image include **GPL-3.0-or-later** components (`piper-tts`, eSpeak-ng); see
[`NOTICE`](NOTICE) for the exact obligations and voice dataset references.
pocket-tts and Kyutai voices are optional and carry their own upstream licenses.
