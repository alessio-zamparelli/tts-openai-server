"""Real Piper end-to-end tests (opt-in, downloads the voice on first run).

    uv sync --extra piper
    PIPER_E2E=1 uv run pytest -m e2e -v

Boots a real uvicorn server with the actual Piper backend and drives it over
HTTP. The default voice ``it_IT-paola-medium`` (~64 MB) is downloaded into
``POCKET_TTS_PIPER_VOICE_DIR`` (default ``~/.cache/piper``) when missing.
"""

from __future__ import annotations

import io
import os
import socket
import threading
import time
import wave
from collections.abc import Iterator

import httpx
import pytest

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("PIPER_E2E") != "1",
        reason="real Piper e2e is opt-in: run with PIPER_E2E=1",
    ),
]

_SAMPLE_RATE = 22050
_SPEECH_URL = "/v1/audio/speech"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def live_server() -> Iterator[str]:
    try:
        import piper  # noqa: F401  (piper extra)
    except ImportError:
        pytest.skip("piper-tts not installed; run `uv sync --extra piper`")

    import uvicorn

    from pocket_tts_openai.config import Config
    from pocket_tts_openai.server import create_app

    cfg = Config(backend="piper", host="127.0.0.1", port=0, idle_unload_s=0)
    app = create_app(cfg)  # engine=None -> background load thread
    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = 300.0  # first run downloads the 64 MB voice
    t0 = time.monotonic()
    ready = False
    while time.monotonic() - t0 < deadline:
        try:
            if httpx.get(f"{base}/health", timeout=10.0).json().get("status") == "ok":
                ready = True
                break
        except httpx.HTTPError:
            time.sleep(1)
            continue
        time.sleep(1)
    if not ready:
        server.should_exit = True
        pytest.fail("server did not become ready within 300 s")

    yield base

    server.should_exit = True
    thread.join(timeout=15)


@pytest.fixture(scope="session")
def client(live_server) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=live_server, timeout=30.0) as c:
        yield c


def _wav_stats(content: bytes):
    import numpy as np

    with wave.open(io.BytesIO(content), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getframerate() == _SAMPLE_RATE
        frames = w.getnframes()
        raw = w.readframes(frames)
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return frames, float(np.sqrt((a**2).mean())), float(np.abs(a).max())


def test_e2e_default_voice_wav(client):
    r = client.post(
        _SPEECH_URL,
        json={"input": "Ciao, questa è la voce predefinita del server."},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    frames, rms, peak = _wav_stats(r.content)
    assert frames > _SAMPLE_RATE // 2
    assert rms > 0.01 and peak > 0.05


def test_e2e_health_reports_piper(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["backend"] == "piper"
    assert body["default_voice"] == "it_IT-paola-medium"
    assert body["sample_rate"] == _SAMPLE_RATE


def test_e2e_streaming_wav(client):
    with client.stream(
        "POST",
        _SPEECH_URL,
        json={"input": "Prima frase. Seconda frase per lo streaming.", "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers.get("transfer-encoding") == "chunked"
        chunks = [c for c in r.iter_bytes()]
        body = b"".join(chunks)
    assert len(chunks) > 1
    assert body.startswith(b"RIFF")
    assert _wav_stats(body)[0] > _SAMPLE_RATE // 2


def test_e2e_cloning_not_implemented(client):
    r = client.post(
        "/v1/voices",
        data={"name": "mario"},
        files={"file": ("x.wav", b"RIFF....", "audio/wav")},
    )
    assert r.status_code == 501
