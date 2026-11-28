"""FastAPI app factory and entrypoint.

``create_app`` accepts an optional ``Config`` and ``TTSEngine``; tests inject a
fake engine, production loads the configured backend in a background thread on
startup. Routers live in :mod:`tts_openai.routers`; shared dependencies in
:mod:`tts_openai.dependencies`.
"""

from __future__ import annotations

import json
import logging
import sys
import threading
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import APIRouter, Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from .config import Config
from .dependencies import verify_api_key
from .engine import TTSEngine, load_engine
from .errors import OpenAIError
from .routers import audio, health, models, voices

logger = logging.getLogger(__name__)


def _extract_reason(body: bytes) -> str:
    """Best-effort human-readable reason from an error response body.

    Handles the OpenAI error shape (``{"error": {"message": …}}``) our
    handlers produce, the ``{"detail": …}`` shape FastAPI/Starlette defaults
    use for 422/404/405, and falls back to raw body text.
    """
    if not body:
        return "<no body>"
    text = body[:600].decode("utf-8", "replace")
    try:
        payload = json.loads(text)
    except (ValueError, TypeError):
        return text.strip()
    if isinstance(payload, dict):
        err = payload.get("error")
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
        detail = payload.get("detail")
        if detail is not None:
            return detail if isinstance(detail, str) else json.dumps(detail)[:300]
        return json.dumps(payload)[:300]
    return text.strip()


class _ErrorLoggingMiddleware:
    """Log every 4xx/5xx response with the reason the client actually saw.

    Pure ASGI: it watches the ``http.response.start`` / ``http.response.body``
    messages flowing through ``send``, so it sees the real status code and
    body regardless of which handler produced them (OpenAI error handler,
    FastAPI validation 422, Starlette default 404/405). Unhandled exceptions
    (500) surface as a raised exception instead of a response: log the
    traceback and re-raise — the outer ServerErrorMiddleware still sends the
    500, so the HTTP behavior is unchanged. Body bytes are only kept for
    status >= 400 (error bodies are tiny; 200 streaming journeys are never
    buffered).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":  # e.g. websocket/lifespan messages
            await self.app(scope, receive, send)
            return
        request = Request(scope)
        status: dict[str, int] = {}
        body: list[bytes] = []
        sent = 0

        async def send_wrapper(message):
            nonlocal sent
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            elif message["type"] == "http.response.body" and status.get("code", 0) >= 400:
                chunk = message.get("body", b"")
                if sent < 600:
                    body.append(chunk[: 600 - sent])
                    sent += len(body[-1])
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception(
                "unhandled exception while handling %s %s",
                request.method,
                request.url.path,
            )
            raise  # ServerErrorMiddleware (outermost) sends the 500 and re-raises

        status_code = status.get("code")
        if status_code is not None and status_code >= 400:
            reason = _extract_reason(b"".join(body)).replace("\n", " ").strip()
            log = logger.error if status_code >= 500 else logger.warning
            client = request.client.host if request.client else "-"
            log(
                "rejecting %s %s (client=%s): %d %s",
                request.method,
                request.url.path,
                client,
                status_code,
                reason,
            )


def create_app(config: Config | None = None, engine: TTSEngine | None = None) -> FastAPI:
    """Build the OpenAI-compatible app. ``engine=None`` defers model loading to
    startup (production); tests pass a fake engine to stay fast and offline."""
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = None
            thread = threading.Thread(target=_load_engine_background, args=(app, config), daemon=True)
            thread.start()
            if config.idle_unload_s > 0:
                # RAM reclamation watchdog: evict the idle model (PLAN-idle-unload.md).
                stop = threading.Event()
                app.state._idle_stop = stop
                wd = threading.Thread(
                    target=_idle_watchdog, args=(app, config, stop), daemon=True
                )
                wd.start()
        if config.wyoming_enabled:
            from .wyoming_server import WyomingService

            wyoming = WyomingService(config, lambda: getattr(app.state, "engine", None))
            try:
                await wyoming.start()
                app.state.wyoming = wyoming
                logger.info(
                    "wyoming listening on %s:%d", config.wyoming_host, wyoming.port
                )
            except Exception:  # port in use, zeroconf failure, …: HTTP still works
                logger.exception("wyoming server failed to start; HTTP continues")
                try:
                    await wyoming.stop()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("error cleaning up failed wyoming server")
                app.state.wyoming = None
        try:
            yield
        finally:
            # Stop the protocol listener before the model is released.
            wyoming = getattr(app.state, "wyoming", None)
            if wyoming is not None:
                try:
                    await wyoming.stop()
                except Exception:  # pragma: no cover - defensive shutdown path
                    logger.exception("error stopping wyoming server")
                app.state.wyoming = None
            stop = getattr(app.state, "_idle_stop", None)
            if stop is not None:
                stop.set()
            # Load/unload symmetry: release the resident model at shutdown so the
            # lifecycle is fully observable in logs (plan: load/unload logging).
            # Only engines this server built itself (``_auto_loaded``) are
            # released — injected engines may be reused (tests) or owned by a
            # supervisor, so they are left resident but still logged.
            engine_obj = getattr(app.state, "engine", None)
            if engine_obj is not None:
                if getattr(engine_obj, "_auto_loaded", False):
                    logger.info("shutting down: unloading model…")
                    try:
                        engine_obj.unload(reason="server shutdown")
                    except Exception:  # pragma: no cover - defensive shutdown path
                        logger.exception("error unloading model during shutdown")
                else:
                    logger.info("shutting down: engine left resident (not owned by this server)")

    app = FastAPI(title="tts-openai", version="0.1.0", lifespan=lifespan)
    app.state.config = config

    def _openai_error_handler(_request: Request, exc: Exception) -> JSONResponse:
        # Starlette dispatches this handler only for the registered exception
        # type; the reason is logged by the _ErrorLoggingMiddleware.
        err = cast(OpenAIError, exc)
        return JSONResponse(status_code=err.status, content=err.to_payload())

    app.add_exception_handler(OpenAIError, _openai_error_handler)
    # Every 4xx/5xx response (regardless of which handler produced it) is
    # logged with its reason; unhandled exceptions get an ERROR + traceback.
    app.add_middleware(_ErrorLoggingMiddleware)

    # /v1/* is guarded by the Bearer dependency; /health stays public.
    v1 = APIRouter(prefix="/v1", dependencies=[Depends(verify_api_key)])
    v1.include_router(audio.router)     # /v1/audio/speech
    v1.include_router(models.router)    # /v1/models
    v1.include_router(voices.router)    # /v1/voices
    app.include_router(v1)
    app.include_router(health.router)   # /health
    return app


def _idle_watchdog(app: FastAPI, config: Config, stop: threading.Event) -> None:
    """Periodically evict the model after ``TTS_IDLE_UNLOAD_S`` without
    API requests. Health probes do NOT touch the engine's idle timer, so they
    never reset the window (see PLAN-idle-unload.md).
    """
    interval = min(config.idle_poll_s, max(5, config.idle_unload_s // 2))
    while not stop.wait(interval):
        engine = getattr(app.state, "engine", None)
        if engine is None:
            continue  # still loading
        try:
            engine.maybe_unload()
        except Exception:  # pragma: no cover - defensive
            logger.exception("idle-unload watchdog failed")


def _load_engine_background(app: FastAPI, config: Config) -> None:
    """Load the configured backend off the event loop; /speech 503s until it lands."""
    try:
        engine = load_engine(config)
        app.state.engine = engine
        logger.info(
            "engine ready: backend=%s default_voice=%r sample_rate=%d",
            engine.backend,
            engine.default_voice,
            engine.sample_rate,
        )
        engine.warmup()
    except Exception:
        logger.exception("engine failed to load; /speech will keep returning 503")


def _healthcheck() -> int:
    """One-shot probe used by the container HEALTHCHECK."""
    import http.client

    try:
        config = Config.from_env()
    except Exception as exc:  # invalid TTS_* config: server would not boot either
        print(f"healthcheck: invalid config ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 1

    host = "127.0.0.1" if config.host in ("", "0.0.0.0", "::") else config.host
    conn = None
    try:
        # http.client bypasses HTTP(S)_PROXY from the container environment;
        # urllib would route a loopback probe through a proxy when one is set.
        conn = http.client.HTTPConnection(host, config.port, timeout=4)
        conn.request("GET", "/health")
        response = conn.getresponse()
        response.read()
        return 0 if response.status == 200 else 1
    except Exception as exc:  # ConnectionRefusedError while uvicorn is not bound yet
        print(
            f"healthcheck: {host}:{config.port} not ready "
            f"({type(exc).__name__}: {exc})",
            file=sys.stderr,
        )
        return 1
    finally:
        if conn is not None:
            conn.close()


def _configure_logging(config: Config) -> None:
    """Apply the configured log level and quiet successful /health probes.

    Called once from ``main()`` before uvicorn boots; ``Config.log_level``
    (``TTS_LOG_LEVEL``) drives both the app loggers and uvicorn's own.
    """
    logging.basicConfig(
        level=logging.getLevelName(config.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # uvicorn.access: drop only *successful* healthcheck lines, so a failing
    # probe (503 while the model loads, 500) is still visible in the logs.
    logging.getLogger("uvicorn.access").addFilter(_HealthSuccessFilter())


class _HealthSuccessFilter(logging.Filter):
    """Drop ``200`` access lines for ``GET /health``; failures stay visible.

    uvicorn emits access records with positional args that its own
    ``AccessFormatter`` unpacks as ``(client_addr, method, full_path,
    http_version, status_code)`` — read the same tuple here instead of parsing
    the rendered line (which appends the reason phrase, ``200 OK``).
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            _, method, full_path, _, status_code = record.args
        except (TypeError, ValueError):  # defensive: unexpected record shape
            return True
        if int(status_code) != 200:  # non-200 health responses are logged
            return True
        return not (method == "GET" and full_path.split("?", 1)[0] == "/health")


def main() -> None:
    if "--healthcheck" in sys.argv[1:]:
        raise SystemExit(_healthcheck())
    config = Config.from_env()
    _configure_logging(config)
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level=config.log_level)


if __name__ == "__main__":
    main()
