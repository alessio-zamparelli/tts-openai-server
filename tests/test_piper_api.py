"""API contract on the default Piper backend (fake voice, offline)."""

from __future__ import annotations

import struct

from pocket_tts_openai.voices import PIPER_DEFAULT_VOICE


def _wav_rate(body: bytes) -> int:
    assert body[:4] == b"RIFF"
    return struct.unpack("<I", body[24:28])[0]


def test_health_reports_piper(piper_client):
    body = piper_client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model"] == "piper"
    assert body["backend"] == "piper"
    assert body["default_voice"] == PIPER_DEFAULT_VOICE
    assert body["sample_rate"] == 22050
    assert body["language"] == "it"


def test_models_owned_by_piper(piper_client):
    body = piper_client.get("/v1/models").json()
    assert {m["owned_by"] for m in body["data"]} == {"piper"}


def test_default_voice_is_paola_and_wav_uses_22050(piper_client):
    r = piper_client.post(
        "/v1/audio/speech",
        json={"input": "Ciao mondo", "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert _wav_rate(r.content) == 22050
    # fake voice yields 0.5 s: 44-byte header + frames * 2 bytes
    assert len(r.content) == 44 + (22050 // 2) * 2


def test_explicit_alias_and_piper_id(piper_client):
    for voice in ("alloy", PIPER_DEFAULT_VOICE, "en_US-lessac-low"):
        r = piper_client.post(
            "/v1/audio/speech",
            json={"input": "Ciao", "voice": voice, "response_format": "pcm"},
        )
        assert r.status_code == 200, voice
        rate = 16000 if voice.endswith("-low") else 22050
        assert len(r.content) == rate  # 0.5 s of s16le

    # The WAV header must follow the selected voice's rate, not the default's.
    r = piper_client.post(
        "/v1/audio/speech",
        json={"input": "Ciao", "voice": "en_US-lessac-low", "response_format": "wav"},
    )
    assert _wav_rate(r.content) == 16000


def test_streaming_wav_header_rate(piper_client):
    with piper_client.stream(
        "POST",
        "/v1/audio/speech",
        json={"input": "Ciao mondo. Seconda frase.", "stream": True},
    ) as r:
        assert r.status_code == 200
        body = b"".join(r.iter_bytes())
    assert _wav_rate(body) == 22050
    assert len(body) == 44 + (22050 // 2) * 2


def test_voices_catalog_on_piper(piper_client):
    body = piper_client.get("/v1/voices").json()
    assert body["backend"] == "piper"
    assert body["clone_supported"] is False
    data = {v["id"]: v for v in body["data"]}
    assert data["alloy"]["aliases"] == ["alloy", PIPER_DEFAULT_VOICE]
    assert data["alloy"]["language"] == "it"
    assert PIPER_DEFAULT_VOICE in data  # installed voice listed directly


def test_cloning_not_implemented_on_piper(piper_client):
    r = piper_client.post(
        "/v1/voices",
        data={"name": "mario"},
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
    )
    assert r.status_code == 501
    assert r.json()["error"]["code"] == "not_implemented"
    assert "POCKET_TTS_BACKEND=pocket" in r.json()["error"]["message"]


def test_delete_builtin_voice_is_405(piper_client):
    r = piper_client.delete(f"/v1/voices/{PIPER_DEFAULT_VOICE}")
    assert r.status_code == 405


def test_unknown_voice_400(piper_client):
    r = piper_client.post(
        "/v1/audio/speech", json={"input": "hi", "voice": "not-a-voice"}
    )
    assert r.status_code == 400
    assert "not-a-voice" in r.json()["error"]["message"]
