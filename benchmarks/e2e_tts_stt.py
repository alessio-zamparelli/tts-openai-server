#!/usr/bin/env python3
"""End-to-end Italian TTS -> STT benchmark.

Pipeline measured per sample, over HTTP, with the real servers:

  1. POST tts /v1/audio/speech        (Piper it_IT-paola-medium, WAV)
  2. POST stt /v1/audio/transcriptions (LocalAI-io Italian Whisper CT2 int8)

Every iteration regenerates the audio and transcribes the resulting WAV, so
the reported "e2e" latency is the wall time of the whole round trip.  The
reference text is known, therefore WER/CER are measured against the TTS input.

The server can either be supplied as already-running URLs, or this script
boots them itself (one STT model at a time, so large models fit in RAM).

Examples
--------
# Boot both servers, test base / small / large-v3-turbo (Italian only):
uv run python benchmarks/e2e_tts_stt.py --models base,small,large-v3-turbo

# Single-word / very short replies (Ciao, Sì, No, Forse, ...):
uv run python benchmarks/e2e_tts_stt.py --models base,small,large-v3-turbo --suite short

# Reuse running servers (single STT model):
uv run python benchmarks/e2e_tts_stt.py --tts-url http://127.0.0.1:18000 \
    --stt-url http://127.0.0.1:18001 --models base

# Low-latency decode preset instead of the accuracy-first default (beam=5):
uv run python benchmarks/e2e_tts_stt.py --models large-v3-turbo --beam-size 2

When run from this repo, defaults point at the sibling ``stt-openai-server``
repo and write generated audio/results under ``artifacts/e2e-tts-stt/``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import unicodedata
import wave
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import httpx

MODELS = {
    "base": "LocalAI-io/whisper-base-it-yodas-only-ct2-int8",
    "small": "LocalAI-io/whisper-small-it-yodas-only-ct2-int8",
    "large-v3-turbo": "LocalAI-io/whisper-large-v3-turbo-it-yodas-only-ct2-int8",
    "medium": "LocalAI-io/whisper-medium-it-yodas-only-ct2-int8",
    "tiny": "LocalAI-io/whisper-tiny-it-yodas-only-ct2-int8",
}

# Italian-only corpus.  No digits/dates, so a digit-vs-word transcription does
# not distort WER; the very long sample is >30 s to cross Whisper's window.
TEXTS = [
    {
        "id": "01_greeting",
        "label": "short greeting",
        "text": "Ciao, come stai oggi?",
    },
    {
        "id": "02_sentence",
        "label": "short sentence",
        "text": "Questa è una prova del sistema di trascrizione automatica in italiano.",
    },
    {
        "id": "03_rail",
        "label": "medium, two sentences",
        "text": (
            "Il treno per Milano parte alle otto e trenta dal binario numero cinque. "
            "Ti consiglio di arrivare in stazione con almeno dieci minuti di anticipo."
        ),
    },
    {
        "id": "04_weather",
        "label": "long paragraph",
        "text": (
            "Domani mattina il cielo sarà sereno su tutta la pianura, mentre nel "
            "pomeriggio arriveranno alcune nuvole sparse sulle zone montuose. "
            "Le temperature minime saranno intorno ai dodici gradi e le massime non "
            "supereranno i ventidue. Nel fine settimana è previsto un leggero calo "
            "delle temperature, con possibili rovesci isolati nelle ore serali."
        ),
    },
    {
        "id": "05_story",
        "label": "very long, >30 s",
        "text": (
            "Quando sono arrivato in città la prima volta, mi sono fermato davanti "
            "alla stazione e ho guardato le persone che passavano. C'erano studenti "
            "con gli zaini sulle spalle, turisti che cercavano il loro albergo e "
            "lavoratori che tornavano a casa dopo una lunga giornata. Ho deciso di "
            "camminare senza una meta precisa, seguendo le strade strette del centro "
            "storico. Dopo qualche minuto ho trovato un piccolo bar dove servivano un "
            "caffè davvero buono. Mi sono seduto fuori, ho aperto il giornale e ho "
            "letto le notizie del giorno con calma. Il sole scaldava i tavoli di legno "
            "e il rumore della città sembrava lontano, quasi come se il tempo si fosse "
            "fermato per un istante."
        ),
    },
]

# Sub-second / very short Italian replies.  These are dominated by Whisper's
# fixed ~30 s encoder window, not by the words themselves.
SHORT_TEXTS = [
    {"id": "s1_ciao", "label": "single word", "text": "Ciao."},
    {"id": "s2_si", "label": "single word", "text": "Sì."},
    {"id": "s3_no", "label": "single word", "text": "No."},
    {"id": "s4_forse", "label": "single word", "text": "Forse."},
    {"id": "s5_va_bene", "label": "two words", "text": "Va bene."},
    {"id": "s6_non_lo_so", "label": "three words", "text": "Non lo so."},
    {"id": "s7_grazie_mille", "label": "two words", "text": "Grazie mille."},
]

SUITES = {"main": TEXTS, "short": SHORT_TEXTS}


# ---------------------------------------------------------------------------
# Text normalisation / error rates
# ---------------------------------------------------------------------------

_UNITS = [
    "zero", "uno", "due", "tre", "quattro", "cinque", "sei", "sette", "otto",
    "nove", "dieci", "undici", "dodici", "tredici", "quattordici", "quindici",
    "sedici", "diciassette", "diciotto", "diciannove",
]
_TENS = {
    2: "venti", 3: "trenta", 4: "quaranta", 5: "cinquanta",
    6: "sessanta", 7: "settanta", 8: "ottanta", 9: "novanta",
}


def _it_number(n: int) -> str:
    """0..999 -> Italian cardinal (enough for a benchmark corpus)."""
    if n < 20:
        return _UNITS[n]
    if n < 100:
        tens, unit = divmod(n, 10)
        word = _TENS[tens]
        if unit == 0:
            return word
        if unit in (1, 8):  # ventuno / ventotto, trentuno / trentotto
            word = word[:-1]
        return word + _UNITS[unit]
    if n == 100:
        return "cento"
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        head = "cento" if hundreds == 1 else _UNITS[hundreds] + "cento"
        return head if rest == 0 else head + _it_number(rest)
    return str(n)


def _normalize(text: str, *, fold_accents: bool) -> str:
    text = text.lower().replace("'", " ").replace("’", " ")
    text = re.sub(r"\d+", lambda m: _it_number(int(m.group())), text)
    if fold_accents:
        text = unicodedata.normalize("NFD", text)
        text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = text.replace("_", " ")
    return " ".join(text.split())


def _edit_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1]


def error_rates(reference: str, hypothesis: str) -> dict[str, float]:
    ref_words = _normalize(reference, fold_accents=False).split()
    hyp_words = _normalize(hypothesis, fold_accents=False).split()
    ref_fold = _normalize(reference, fold_accents=True).split()
    hyp_fold = _normalize(hypothesis, fold_accents=True).split()
    ref_chars = " ".join(ref_fold)
    hyp_chars = " ".join(hyp_fold)
    return {
        "wer": _edit_distance(ref_words, hyp_words) / max(1, len(ref_words)),
        "wer_folded": _edit_distance(ref_fold, hyp_fold) / max(1, len(ref_fold)),
        "cer_folded": _edit_distance(ref_chars, hyp_chars) / max(1, len(ref_chars)),
        "ref_words": len(ref_words),
    }


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _client() -> httpx.Client:
    timeout = httpx.Timeout(connect=15.0, read=900.0, write=120.0, pool=30.0)
    # Disable keep-alive: localhost reconnects are cheap and this avoids reusing
    # a socket that the server closed while it was busy with a long decode.
    limits = httpx.Limits(max_connections=10, max_keepalive_connections=0)
    return httpx.Client(timeout=timeout, limits=limits)


def _post(client: httpx.Client, url: str, **kwargs) -> httpx.Response:
    """POST with retries for connection-level failures (request not sent)."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            return client.post(url, **kwargs)
        except (httpx.ConnectError, httpx.RemoteProtocolError) as exc:
            last = exc
            if attempt == 2:
                break
            time.sleep(0.5 * (attempt + 1))
    assert last is not None
    raise last


def _tts_wav(client: httpx.Client, base_url: str, text: str, voice: str) -> tuple[bytes, float]:
    started = time.perf_counter()
    response = _post(
        client,
        f"{base_url}/v1/audio/speech",
        json={
            "model": "tts-1",
            "voice": voice,
            "input": text,
            "response_format": "wav",
            "stream": False,
        },
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    return response.content, elapsed


def _stt_text(client: httpx.Client, base_url: str, wav_bytes: bytes) -> tuple[str, float]:
    started = time.perf_counter()
    last_body = ""
    for attempt in range(2):
        response = _post(
            client,
            f"{base_url}/v1/audio/transcriptions",
            files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            data={"model": "whisper-1", "language": "it", "response_format": "json"},
        )
        if response.status_code < 500:
            break
        last_body = response.text[:2000]
        if attempt == 0:
            time.sleep(0.5)
    elapsed = time.perf_counter() - started
    if response.status_code >= 500:
        raise RuntimeError(f"STT HTTP {response.status_code}: {last_body}")
    response.raise_for_status()
    return response.json()["text"], elapsed


def _wav_seconds(wav_bytes: bytes) -> float:
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        return handle.getnframes() / float(handle.getframerate())


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_health(base_url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            response = httpx.get(f"{base_url}/health", timeout=2.0)
            if response.status_code == 200:
                body = response.json()
                if body.get("status") == "ok":
                    return
                last_error = f"status={body.get('status')!r}"
        except Exception as exc:  # server still booting
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"{base_url} not healthy after {timeout:.0f}s ({last_error})")


def _rss_mb(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


@dataclass
class Server:
    name: str
    process: subprocess.Popen
    base_url: str
    startup_s: float
    log_path: Path
    log_handle: object = field(repr=False)

    def stop(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=5)
        self.log_handle.close()


def _start_server(
    *,
    name: str,
    cmd: list[str],
    cwd: Path,
    env: dict[str, str],
    port: int,
    log_path: Path,
    timeout: float,
) -> Server:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    base_url = f"http://127.0.0.1:{port}"
    try:
        _wait_health(base_url, timeout)
    except Exception:
        log_handle.close()
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"{name} failed to start:\n{tail}") from None
    startup_s = time.perf_counter() - started
    return Server(name, process, base_url, startup_s, log_path, log_handle)


def _repo_root() -> Path | None:
    """Return the TTS repo root when this script lives in ``benchmarks/``."""
    here = Path(__file__).resolve()
    if here.parent.name == "benchmarks" and (here.parents[1] / "pyproject.toml").exists():
        return here.parents[1]
    return None


def _default_repos() -> tuple[Path, Path]:
    here = Path(__file__).resolve()
    repo_root = _repo_root()
    workspace = repo_root.parent if repo_root is not None else here.parents[1]
    return workspace / "tts-openai-server", workspace / "stt-openai-server"


def _default_out_dir(stamp: str) -> Path:
    repo_root = _repo_root()
    if repo_root is not None:
        # Generated audio/results stay out of git (``artifacts/`` is ignored).
        return repo_root / "artifacts" / "e2e-tts-stt" / stamp
    return Path(__file__).resolve().parent / "results" / stamp


def _cgroup_memory_limit_gib() -> float | None:
    for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            raw = Path(path).read_text().strip()
            if raw not in ("max", ""):
                return int(raw) / (1024 ** 3)
        except (OSError, ValueError):
            continue
    return None


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    text_id: str
    label: str
    run: int
    audio_s: float
    tts_s: float
    stt_s: float
    transcript: str
    wer: float
    wer_folded: float
    cer_folded: float
    ref_words: int

    @property
    def e2e_s(self) -> float:
        return self.tts_s + self.stt_s

    @property
    def tts_rtf(self) -> float:
        return self.tts_s / self.audio_s

    @property
    def stt_rtf(self) -> float:
        return self.stt_s / self.audio_s

    @property
    def e2e_rtf(self) -> float:
        return self.e2e_s / self.audio_s


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _summarize(samples: list[Sample], run_count: int, texts: list[dict]) -> dict:
    per_text = []
    for text in texts:
        rows = [s for s in samples if s.text_id == text["id"]]
        if not rows:
            continue
        per_text.append(
            {
                "id": text["id"],
                "label": text["label"],
                "words": rows[0].ref_words,
                "audio_s": _median([r.audio_s for r in rows]),
                "tts_s": _median([r.tts_s for r in rows]),
                "stt_s": _median([r.stt_s for r in rows]),
                "e2e_s": _median([r.e2e_s for r in rows]),
                "tts_rtf": _median([r.tts_rtf for r in rows]),
                "stt_rtf": _median([r.stt_rtf for r in rows]),
                "e2e_rtf": _median([r.e2e_rtf for r in rows]),
                "wer": _median([r.wer for r in rows]),
                "wer_folded": _median([r.wer_folded for r in rows]),
                "cer_folded": _median([r.cer_folded for r in rows]),
                "tts_s_all": [round(r.tts_s, 4) for r in rows],
                "stt_s_all": [round(r.stt_s, 4) for r in rows],
                "e2e_s_all": [round(r.e2e_s, 4) for r in rows],
                "transcripts": [r.transcript for r in rows],
            }
        )

    # One "corpus run" = all texts for one repetition; throughput is the median
    # of those per-run totals (audio seconds produced+transcribed per wall second).
    corpus_runs = []
    for run in range(1, run_count + 1):
        rows = [s for s in samples if s.run == run]
        if not rows:
            continue
        audio = sum(r.audio_s for r in rows)
        wall = sum(r.e2e_s for r in rows)
        corpus_runs.append(
            {
                "run": run,
                "audio_s": audio,
                "e2e_s": wall,
                "throughput": audio / wall,
                "wer": sum(r.wer * r.ref_words for r in rows)
                / sum(r.ref_words for r in rows),
                "wer_folded": sum(r.wer_folded * r.ref_words for r in rows)
                / sum(r.ref_words for r in rows),
            }
        )

    total_audio = sum(s.audio_s for s in samples)
    return {
        "per_text": per_text,
        "corpus": {
            "audio_s_per_run": _median([r["audio_s"] for r in corpus_runs]),
            "e2e_s_per_run": _median([r["e2e_s"] for r in corpus_runs]),
            "throughput_audio_s_per_wall_s": _median([r["throughput"] for r in corpus_runs]),
            # Audio-weighted RTF (total wall / total audio), unlike the median of
            # per-sample RTFs which over-weights sub-second clips.
            "tts_rtf": sum(s.tts_s for s in samples) / total_audio,
            "stt_rtf": sum(s.stt_s for s in samples) / total_audio,
            "e2e_rtf": sum(s.e2e_s for s in samples) / total_audio,
            "wer": _median([r["wer"] for r in corpus_runs]),
            "wer_folded": _median([r["wer_folded"] for r in corpus_runs]),
            "runs": corpus_runs,
        },
    }


def run_model(
    *,
    model_key: str,
    model_id: str,
    args: argparse.Namespace,
    tts_url: str | None,
    stt_url: str | None,
    out_dir: Path,
    tts_repo: Path,
    stt_repo: Path,
    texts: list[dict],
) -> dict:
    servers: list[Server] = []
    tts_server = None
    stt_server = None
    tts_startup = None
    stt_startup = None
    tts_rss = None
    stt_rss = None
    try:
        if tts_url is None:
            port = _free_port()
            env = os.environ.copy()
            env.update(
                {
                    "POCKET_TTS_BACKEND": "piper",
                    "POCKET_TTS_HOST": "127.0.0.1",
                    "POCKET_TTS_PORT": str(port),
                    "POCKET_TTS_DEFAULT_VOICE": args.tts_voice,
                    "POCKET_TTS_IDLE_UNLOAD_S": "0",
                }
            )
            binary = tts_repo / ".venv/bin/pocket-tts-openai"
            if not binary.exists():
                raise SystemExit(f"missing TTS entry point: {binary}")
            tts_server = _start_server(
                name="tts",
                cmd=[str(binary)],
                cwd=tts_repo,
                env=env,
                port=port,
                log_path=out_dir / "logs/tts.log",
                timeout=args.startup_timeout,
            )
            servers.append(tts_server)
            tts_url = tts_server.base_url
            tts_startup = tts_server.startup_s

        if stt_url is None:
            port = _free_port()
            env = os.environ.copy()
            env.update(
                {
                    "WHISPER_MODEL_NAME": model_id,
                    "WHISPER_COMPUTE_TYPE": args.compute_type,
                    "WHISPER_DEVICE": "cpu",
                    "WHISPER_LANGUAGE": "it",
                    "WHISPER_HOST": "127.0.0.1",
                    "WHISPER_PORT": str(port),
                    "WHISPER_IDLE_UNLOAD_S": "0",
                    "WHISPER_HF_OFFLINE": "true",
                    "WHISPER_BEAM_SIZE": str(args.beam_size),
                    "WHISPER_CPU_THREADS": str(args.cpu_threads),
                    "WHISPER_BATCH_SIZE": str(args.batch_size),
                    "WHISPER_VAD": "true" if args.vad else "false",
                    "WHISPER_MAX_CONCURRENT": "1",
                }
            )
            if args.temperatures:
                env["WHISPER_TEMPERATURES"] = args.temperatures
            if args.initial_prompt:
                env["WHISPER_INITIAL_PROMPT"] = args.initial_prompt
            python = stt_repo / ".venv/bin/python"
            if not python.exists():
                raise SystemExit(f"missing STT venv: {python}")
            stt_server = _start_server(
                name=f"stt-{model_key}",
                cmd=[str(python), "app.py"],
                cwd=stt_repo,
                env=env,
                port=port,
                log_path=out_dir / f"logs/stt-{model_key}.log",
                timeout=args.startup_timeout,
            )
            servers.append(stt_server)
            stt_url = stt_server.base_url
            stt_startup = stt_server.startup_s

        print(
            f"\n[{model_key}] TTS {tts_url}  STT {stt_url}"
            f"  (startup tts={tts_startup or 0:.2f}s stt={stt_startup or 0:.2f}s)",
            flush=True,
        )

        samples: list[Sample] = []
        saved_audio: dict[str, Path] = {}
        with _client() as client:
            # Warm both engines before measuring (also loads TTS voice / STT weights
            # when the servers were started outside this script).
            for _ in range(args.warmup):
                wav, _ = _tts_wav(client, tts_url, texts[0]["text"], args.tts_voice)
                _stt_text(client, stt_url, wav)

            for run in range(1, args.runs + 1):
                for text in texts:
                    wav, tts_s = _tts_wav(client, tts_url, text["text"], args.tts_voice)
                    audio_s = _wav_seconds(wav)
                    transcript, stt_s = _stt_text(client, stt_url, wav)
                    rates = error_rates(text["text"], transcript)
                    samples.append(
                        Sample(
                            text_id=text["id"],
                            label=text["label"],
                            run=run,
                            audio_s=audio_s,
                            tts_s=tts_s,
                            stt_s=stt_s,
                            transcript=transcript,
                            wer=rates["wer"],
                            wer_folded=rates["wer_folded"],
                            cer_folded=rates["cer_folded"],
                            ref_words=rates["ref_words"],
                        )
                    )
                    if text["id"] not in saved_audio:
                        audio_path = out_dir / "audio" / f"{text['id']}.wav"
                        audio_path.parent.mkdir(parents=True, exist_ok=True)
                        audio_path.write_bytes(wav)
                        saved_audio[text["id"]] = audio_path
                    print(
                        f"  seq {run}/{args.runs} {text['id']:<13}"
                        f" TTS {tts_s:5.2f}s -> STT {stt_s:5.2f}s"
                        f" | audio={audio_s:6.2f}s e2e={tts_s + stt_s:5.2f}s"
                        f" wer={rates['wer_folded']*100:5.1f}%",
                        flush=True,
                    )

            if tts_server is not None:
                tts_rss = _rss_mb(tts_server.process.pid)
            if stt_server is not None:
                stt_rss = _rss_mb(stt_server.process.pid)
    finally:
        for server in reversed(servers):
            server.stop()

    summary = _summarize(samples, args.runs, texts)
    summary.update(
        {
            "model_key": model_key,
            "model_id": model_id,
            "beam_size": args.beam_size,
            "compute_type": args.compute_type,
            "cpu_threads": args.cpu_threads,
            "batch_size": args.batch_size,
            "vad": args.vad,
            "temperatures": args.temperatures,
            "initial_prompt": args.initial_prompt,
            "label": args.label,
            "tts_voice": args.tts_voice,
            "startup_tts_s": tts_startup,
            "startup_stt_s": stt_startup,
            "rss_tts_mb": tts_rss,
            "rss_stt_mb": stt_rss,
            "audio": {text_id: str(path) for text_id, path in saved_audio.items()},
        }
    )
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _markdown_report(result: dict) -> str:
    env = result["environment"]
    lines = [
        "# E2E Italian TTS → STT benchmark",
        "",
        f"Generated: {env['timestamp']}",
        "",
        "## Setup",
        "",
        f"- Host: `{env['platform']}`, Python {env['python']}, "
        f"{env['cpu_count']} logical CPUs, cgroup RAM limit "
        f"{env['memory_limit_gib'] if env['memory_limit_gib'] else 'n/a'} GiB",
        f"- TTS: Piper `{result['config']['tts_voice']}` via `POST /v1/audio/speech` (WAV, non-streaming)",
        f"- STT: LocalAI Italian Whisper CT2 `{result['config']['compute_type']}` via "
        f"`POST /v1/audio/transcriptions` (`language=it`)",
        f"- Decode: beam_size={result['config']['beam_size']}, "
        f"cpu_threads={result['config'].get('cpu_threads', 0)}, "
        f"batch_size={result['config'].get('batch_size', 0)}, "
        f"vad={result['config'].get('vad', False)}, "
        f"temperatures={result['config'].get('temperatures') or 'fw default'}, "
        f"runs per sample={result['config']['runs']}, "
        f"warmup requests={result['config']['warmup']}",
        f"- Suite: `{result['config'].get('suite', 'main')}` "
        f"({len(result['config']['texts'])} Italian samples)",
        "- Execution is strictly serial: for each sample the TTS WAV response is fully "
        "received before the STT request is issued, and only one request is in flight at "
        "a time (`WHISPER_MAX_CONCURRENT=1`). Every iteration regenerates audio with TTS, "
        "then sends that exact WAV to STT. RTF = wall time / audio duration (lower is better).",
        "- WER is measured against the known Italian input text; `folded` ignores accents. "
        "The corpus is synthetic Piper speech, so these WERs are a pipeline sanity check, "
        "not a real-world ASR quality claim.",
        "",
        "## Summary",
        "",
        "| STT model | startup | TTS RTF | STT RTF | E2E RTF | throughput | WER | WER folded | RSS TTS/STT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for summary in result["models"]:
        corpus = summary["corpus"]
        lines.append(
            "| {model} | {startup:.2f} s | {tts_rtf:.3f} | {stt_rtf:.3f} | {e2e_rtf:.3f} | "
            "{tp:.2f}× | {wer:.1f}% | {werf:.1f}% | {rss_tts}/{rss_stt} MB |".format(
                model=f"`{summary['model_key']}`",
                startup=summary["startup_stt_s"] or 0.0,
                tts_rtf=corpus["tts_rtf"],
                stt_rtf=corpus["stt_rtf"],
                e2e_rtf=corpus["e2e_rtf"],
                tp=corpus["throughput_audio_s_per_wall_s"],
                wer=corpus["wer"] * 100,
                werf=corpus["wer_folded"] * 100,
                rss_tts=f"{summary['rss_tts_mb']:.0f}" if summary["rss_tts_mb"] else "n/a",
                rss_stt=f"{summary['rss_stt_mb']:.0f}" if summary["rss_stt_mb"] else "n/a",
            )
        )
    lines += [
        "",
        "Summary RTFs are audio-weighted (sum of wall time / sum of audio time), so long "
        "clips dominate; per-sample RTFs expose Whisper's fixed ~30 s window floor, which "
        "makes short clips look disproportionately expensive. `throughput` = audio seconds "
        "processed per wall second for the whole corpus (TTS + STT combined, median over runs).",
        "",
        "## Per-sample results",
        "",
    ]
    for summary in result["models"]:
        lines += [
            f"### `{summary['model_key']}` (beam={summary['beam_size']})",
            "",
            "| sample | words | audio s | TTS s | STT s | E2E s | STT RTF | E2E RTF | WER | WER folded |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in summary["per_text"]:
            lines.append(
                "| {id} | {words} | {audio_s:.2f} | {tts_s:.2f} | {stt_s:.2f} | {e2e_s:.2f} | "
                "{srtf:.3f} | {ertf:.3f} | {wer_pct:.1f}% | {werf_pct:.1f}% |".format(
                    **row,
                    srtf=row["stt_rtf"],
                    ertf=row["e2e_rtf"],
                    wer_pct=row["wer"] * 100,
                    werf_pct=row["wer_folded"] * 100,
                )
            )
        lines += ["", "Transcripts (last run):", ""]
        for row in summary["per_text"]:
            lines += [
                f"- **{row['id']}** ({row['label']}): {row['transcripts'][-1]!r}",
            ]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--models",
        default="large-v3-turbo",
        help="comma-separated keys: " + ", ".join(MODELS) + " (default: large-v3-turbo)",
    )
    parser.add_argument(
        "--suite",
        choices=sorted(SUITES),
        default="main",
        help="main = 1.4-33 s sentences; short = Ciao/Sì/No/Forse + short phrases (default: main)",
    )
    parser.add_argument("--runs", type=int, default=3, help="measured iterations per sample (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="warmup requests before measuring (default: 1)")
    parser.add_argument("--beam-size", type=int, default=5, help="Whisper beam size (default: 5)")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute type (default: int8)")
    parser.add_argument("--cpu-threads", type=int, default=0, help="WHISPER_CPU_THREADS (0 = auto)")
    parser.add_argument("--batch-size", type=int, default=0, help="WHISPER_BATCH_SIZE (0 = sequential)")
    parser.add_argument("--vad", action="store_true", help="enable WHISPER_VAD")
    parser.add_argument("--temperatures", default="", help="WHISPER_TEMPERATURES, e.g. 0,0.2,0.4")
    parser.add_argument("--initial-prompt", default="", help="WHISPER_INITIAL_PROMPT")
    parser.add_argument("--only", default="", help="comma-separated sample ids to run")
    parser.add_argument("--label", default="", help="label for this configuration (reports)")
    parser.add_argument("--tts-voice", default="it_IT-paola-medium", help="Piper voice id")
    parser.add_argument("--tts-url", default=None, help="reuse a running TTS server")
    parser.add_argument("--stt-url", default=None, help="reuse a running STT server (single model)")
    tts_repo, stt_repo = _default_repos()
    parser.add_argument("--tts-repo", type=Path, default=tts_repo)
    parser.add_argument("--stt-repo", type=Path, default=stt_repo)
    parser.add_argument("--out", type=Path, default=None, help="output directory")
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in keys if key not in MODELS]
    if unknown:
        raise SystemExit(f"unknown model key(s): {unknown}; choose from {list(MODELS)}")
    if args.stt_url and len(keys) > 1:
        raise SystemExit("--stt-url serves one model; pass a single --models key")
    texts = SUITES[args.suite]
    if args.only:
        wanted = [item.strip() for item in args.only.split(",") if item.strip()]
        unknown_ids = [item for item in wanted if item not in {t["id"] for t in texts}]
        if unknown_ids:
            raise SystemExit(f"unknown sample id(s): {unknown_ids}")
        texts = [t for t in texts if t["id"] in set(wanted)]

    tts_repo = args.tts_repo.expanduser().resolve()
    stt_repo = args.stt_repo.expanduser().resolve()
    for repo in (tts_repo, stt_repo):
        if not repo.is_dir():
            raise SystemExit(f"repo not found: {repo}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (args.out or _default_out_dir(stamp)).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "environment": {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "memory_limit_gib": _cgroup_memory_limit_gib(),
        },
        "config": {
            "tts_voice": args.tts_voice,
            "beam_size": args.beam_size,
            "compute_type": args.compute_type,
            "cpu_threads": args.cpu_threads,
            "batch_size": args.batch_size,
            "vad": args.vad,
            "temperatures": args.temperatures,
            "initial_prompt": args.initial_prompt,
            "label": args.label,
            "only": args.only,
            "runs": args.runs,
            "warmup": args.warmup,
            "language": "it",
            "suite": args.suite,
            "texts": [{"id": t["id"], "label": t["label"], "text": t["text"]} for t in texts],
        },
        "models": [],
    }

    for key in keys:
        summary = run_model(
            model_key=key,
            model_id=MODELS[key],
            args=args,
            tts_url=args.tts_url,
            stt_url=args.stt_url,
            out_dir=out_dir,
            tts_repo=tts_repo,
            stt_repo=stt_repo,
            texts=texts,
        )
        result["models"].append(summary)

    (out_dir / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    report = _markdown_report(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")

    print("\n" + report)
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
