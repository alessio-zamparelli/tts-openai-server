"""Logging behavior: the /health access-line filter and OpenAI error logging."""

from __future__ import annotations

import logging

from tts_openai.main import _HealthSuccessFilter

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
