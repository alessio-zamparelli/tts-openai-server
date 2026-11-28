# Verbose error logging, /health log filter, and `POCKET_TTS_*` → `TTS_*` rename

Goal: when Open WebUI (or any client) gets a `400` from `POST /v1/audio/speech`,
the server should log *why* (the OpenAI error message), and the access log should
stop spamming successful `/health` probes.

## A. Logging

1. **Log every rejected request** (`src/tts_openai/main.py`, `_openai_error_handler`).
   Today the OpenAI-error handler returns the JSON body with zero logging, so a
   400 shows up only as a bare uvicorn access line (`"POST /v1/audio/speech
   HTTP/1.1" 400`) with no message. Log method, path, client, status, and the
   OpenAI message — `WARNING` for 4xx, `ERROR` for 5xx.

2. **Drop successful `/health` access lines** — and only those. A
   `logging.Filter` on the `uvicorn.access` logger reads the raw record args
   `(client_addr, method, full_path, http_version, status_code_int)` — the same
   tuple uvicorn's own `AccessFormatter` unpacks — and drops the line when
   `method == "GET"`, path is exactly `/health` (query string allowed), and
   `status_code == 200`. Non-200 health responses (503 while loading, 500) and
   `/healthz` still log.

3. **`TTS_LOG_LEVEL` env knob** (new). Add `log_level: str = "info"` to
   `Config`, parsed from `TTS_LOG_LEVEL` and validated against the standard
   logging levels. `main()` uses it for both `logging.basicConfig(level=…)` and
   `uvicorn.run(log_level=…)`, so `TTS_LOG_LEVEL=debug` surfaces engine/route
   debug output.

4. **Per-request DEBUG line** in `routers/audio.py:speech()` (model, voice,
   format, stream flag, input length) so `TTS_LOG_LEVEL=debug` shows exactly
   what each `/v1/audio/speech` call sent.

## B. Env rename `TTS_*` → `TTS_*` (breaking)

The `POCKET_TTS_` prefix dates from the pocket-tts era; the package is now
`tts-openai`. Rename **all** environment variables by replacing the literal
`POCKET_TTS_` prefix with `TTS_` repo-wide (source, tests, Dockerfile,
docker-compose, benchmarks, README, AGENTS.md, PLAN docs, NOTICE). No
backward-compat aliases: the old names are ignored after this change.

Deployments must update their env. The new log knob is `TTS_LOG_LEVEL`
(valid values: `debug`, `info`, `warning`, `error`, `critical`; default
`info`).

## Files touched

- `src/tts_openai/config.py` — env reads + new `log_level` field/validation
- `src/tts_openai/main.py` — error logging, health filter, log-level wiring
- `src/tts_openai/routers/audio.py` — debug request line
- `src/tts_openai/{engine,piper_backend,voices,routers/voices}.py` — strings
- `tests/` — all env refs + new `TTS_LOG_LEVEL` tests
- `deploy/Dockerfile`, `docker-compose.yml`, `benchmarks/*.py`, `pyproject.toml`
- Docs: `README.md` (env table + `TTS_LOG_LEVEL` row), `AGENTS.md`,
  `PLAN*.md`, `plans/docker-image-size.md`, `NOTICE`

## Verification

- `grep -rn "POCKET_TTS_" .` → no matches in code/tests/deploy/docs
- `uv sync && uv run pytest` → full offline suite green
- `TTS_LOG_LEVEL=debug uv run tts-openai` boots; access-log lines drop
  `"GET /health HTTP/1.1" 200` but keep `"GET /health HTTP/1.1" 503`
  and all `/v1/*` lines; a rejected `/v1/audio/speech` logs the message.
