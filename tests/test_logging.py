"""Logging behavior: the /health access-line filter and error-reason logging."""

from __future__ import annotations

import json
import logging

from fastapi.testclient import TestClient

from tts_openai.config import Config
from tts_openai.engine import TTSEngine
from tts_openai.main import _HealthSuccessFilter, _extract_reason, create_app
from tests.conftest import FakeTTSModel

# uvicorn.access emits posargs that its AccessFormatter unpacks; mirror that
# shape (client_addr, method, full_path, http_version, status_code_int).
ACCESS_MSG = '%s - "%s %s HTTP/%s" %d'


def _access_record(method: str, path: str, status: int) -> logging.LogRecord:
    return logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=ACCESS_MSG,
        args=("127.0.0.1:50036", method, path, "1.1", status),
        exc_info=None,
    )


def test_health_filter_drops_successful_probe():
    assert _HealthSuccessFilter().filter(_access_record("GET", "/health", 200)) is False


def test_health_filter_drops_probe_with_query_string():
    assert _HealthSuccessFilter().filter(_access_record("GET", "/health?x=1", 200)) is False


def test_health_filter_keeps_failed_probe():
    # A healthcheck that fails while the model is loading must stay visible.
    assert _HealthSuccessFilter().filter(_access_record("GET", "/health", 503)) is True
    assert _HealthSuccessFilter().filter(_access_record("GET", "/health", 500)) is True


def test_health_filter_keeps_everything_else():
    flt = _HealthSuccessFilter()
    assert flt.filter(_access_record("POST", "/v1/audio/speech", 400)) is True
    assert flt.filter(_access_record("GET", "/healthz", 200)) is True


def test_rejected_speech_is_logged(client, caplog):
    """A 400 from /v1/audio/speech logs the reason server-side."""
    with caplog.at_level(logging.WARNING, logger="tts_openai"):
        r = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "ciao", "voice": "ash"},
        )
    assert r.status_code == 400
    msg = caplog.messages[-1] if caplog.messages else ""
    assert "rejecting POST /v1/audio/speech" in msg
    assert "Unknown voice 'ash'" in msg


# --- reason extraction (unit) -------------------------------------------------

def test_extract_reason_openai_error_shape():
    body = json.dumps(
        {
            "error": {
                "message": "Unknown voice 'marin'.",
                "type": "invalid_request_error",
                "code": None,
            }
        }
    ).encode()
    assert _extract_reason(body) == "Unknown voice 'marin'."


def test_extract_reason_detail_string():
    assert _extract_reason(b'{"detail": "Not Found"}') == "Not Found"


def test_extract_reason_detail_list():
    body = json.dumps(
        {"detail": [{"loc": ["body", "input"], "msg": "field required", "type": "missing"}]}
    ).encode()
    assert "field required" in _extract_reason(body)


def test_extract_reason_plain_text_fallback():
    assert "plain boom" in _extract_reason(b"plain boom")


def test_extract_reason_no_body():
    assert _extract_reason(b"") == "<no body>"


# --- every 4xx/5xx is logged with its reason ----------------------------------

def test_validation_error_is_logged(client, caplog):
    """A 422 (schema failure) is an error response like any other: log why."""
    with caplog.at_level(logging.WARNING, logger="tts_openai"):
        r = client.post("/v1/audio/speech", json={"model": "tts-1"})  # 'input' is required
    assert r.status_code == 422
    msg = caplog.messages[-1] if caplog.messages else ""
    assert "rejecting POST /v1/audio/speech" in msg
    assert "422" in msg
    assert "field required" in msg.lower()


def test_unknown_path_is_logged(client, caplog):
    """A 404 (unknown /v1/* path) from Starlette's default handler also logs."""
    with caplog.at_level(logging.WARNING, logger="tts_openai"):
        r = client.get("/v1/no-such-endpoint")
    assert r.status_code == 404
    msg = caplog.messages[-1] if caplog.messages else ""
    assert "rejecting GET /v1/no-such-endpoint" in msg
    assert "404" in msg and "Not Found" in msg


def test_unhandled_exception_is_logged(caplog):
    """A crash (500) is logged with a traceback via the middleware."""
    config = Config(idle_unload_s=0)
    app = create_app(config, TTSEngine(FakeTTSModel(), config=config))

    def boom() -> None:
        raise RuntimeError("boom in handler")

    app.add_api_route("/boom", boom, methods=["GET"])
    # Starlette's ServerErrorMiddleware always re-raises after sending the 500
    # so servers/test clients can log it; keep the 500 visible to the test.
    with caplog.at_level(logging.ERROR, logger="tts_openai"):
        with TestClient(app, raise_server_exceptions=False) as client:
            r = client.get("/boom")
    assert r.status_code == 500
    assert "unhandled exception while handling GET /boom" in caplog.text
