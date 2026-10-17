#!/usr/bin/env python
"""Benchmark Piper vs pocket-tts behind the real TTSEngine.

Each backend runs in its own subprocess so RSS measurements are clean and the
two models are never resident at the same time. This is intentionally *not* a
pytest test: it needs the real engines and downloads voices on first use.

    uv sync --extra piper --extra engine
    uv run python benchmarks/bench_backends.py --backends piper,pocket --runs 5

The parent prints a Markdown summary and optionally writes the raw JSON with
``--out artifacts/bench.json`` (``artifacts/`` is gitignored).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

TEXTS = {
    "short": "Ciao, come stai oggi?",
    "medium": (
        "Questa è una prova di sintesi vocale in italiano. Il server deve "
        "generare diverse frasi e misurare il tempo impiegato. La qualità "
        "dell'audio non viene valutata automaticamente in questo benchmark."
    ),
    "long": (
        "La sintesi vocale locale sta diventando sempre più importante per le "
        "applicazioni che devono funzionare senza una connessione a Internet. "
        "Un motore veloce riduce il consumo energetico e migliora l'esperienza "
        "utente, soprattutto quando molte richieste arrivano contemporaneamente. "
        "Questo testo serve a misurare la velocità su una richiesta più lunga, "
        "con molte frasi e una durata dell'audio di parecchi secondi. I numeri "
        "ottenuti dipendono dal processore, dalla memoria disponibile e dal "
        "numero di thread usati dalle rispettive librerie di inferenza."
    ),
}

PIPER_VOICE = "it_IT-paola-medium"
POCKET_VOICE = "alloy"


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[idx]


def _run_worker(args: argparse.Namespace) -> dict:
    from pocket_tts_openai.config import Config
    from pocket_tts_openai.engine import _rss_mb, load_engine

    voice = args.piper_voice if args.backend == "piper" else args.pocket_voice
    idle_rss = _rss_mb()
    t0 = time.perf_counter()
    cfg = Config(
        backend=args.backend,
        language=args.pocket_language,
        default_voice=voice,
        piper_voice_dir=args.piper_voice_dir,
        idle_unload_s=0,
        max_cached_voices=2 if args.backend == "piper" else 32,
    )
    engine = load_engine(cfg)
    load_s = time.perf_counter() - t0

    # Warm up (first call includes phonemizer/session init).
    engine.generate_pcm(TEXTS["short"], voice)
    loaded_rss = _rss_mb()
    sample_rate = engine.sample_rate_for(voice)

    results: dict[str, dict] = {}
    for name, text in TEXTS.items():
        walls: list[float] = []
        audio_seconds: list[float] = []
        runs = args.runs if name != "long" else max(2, args.runs // 2)
        for _ in range(runs):
            t = time.perf_counter()
            pcm = engine.generate_pcm(text, voice)
            walls.append(time.perf_counter() - t)
            audio_seconds.append(len(pcm) / (2 * sample_rate))
        results[name] = {
            "runs": runs,
            "audio_s": statistics.mean(audio_seconds),
            "wall_mean_s": statistics.mean(walls),
            "wall_p50_s": _percentile(walls, 0.5),
            "wall_p95_s": _percentile(walls, 0.95),
            "rtf_mean": statistics.mean(
                w / a for w, a in zip(walls, audio_seconds)
            ),
        }

    # Streaming: time to first chunk + total, on the multi-sentence text.
    t = time.perf_counter()
    stream = engine.generate_pcm_stream(TEXTS["medium"], voice)
    first = next(stream)
    ttfa = time.perf_counter() - t
    streamed_audio = len(first) / (2 * sample_rate)
    for chunk in stream:
        streamed_audio += len(chunk) / (2 * sample_rate)
    stream_total = time.perf_counter() - t

    # Concurrency through the engine's global generation lock.
    workers = 4
    t = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pcms = list(pool.map(lambda _: engine.generate_pcm(TEXTS["short"], voice), range(workers)))
    concurrent_wall = time.perf_counter() - t
    concurrent_audio = sum(len(p) / (2 * sample_rate) for p in pcms)

    return {
        "backend": args.backend,
        "voice": voice,
        "sample_rate": sample_rate,
        "load_s": load_s,
        "rss_idle_mb": idle_rss,
        "rss_loaded_mb": loaded_rss,
        "requests": results,
        "stream": {
            "ttfa_s": ttfa,
            "total_s": stream_total,
            "audio_s": streamed_audio,
            "rtf": stream_total / max(streamed_audio, 1e-9),
        },
        "concurrency": {
            "workers": workers,
            "wall_s": concurrent_wall,
            "audio_s": concurrent_audio,
            "throughput_rtf": concurrent_audio / max(concurrent_wall, 1e-9),
        },
        "engine_stats": {
            "requests": engine.stats.requests,
            "audio_seconds": engine.stats.audio_seconds,
            "generate_seconds": engine.stats.generate_seconds,
            "avg_rtf": engine.stats.avg_rtf,
        },
    }


def _spawn(args: argparse.Namespace, backend: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--backend",
        backend,
        "--runs",
        str(args.runs),
        "--piper-voice",
        args.piper_voice,
        "--pocket-voice",
        args.pocket_voice,
        "--pocket-language",
        args.pocket_language,
        "--piper-voice-dir",
        args.piper_voice_dir,
    ]
    env = os.environ.copy()
    print(f"[bench] running {backend} in a subprocess…", file=sys.stderr)
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"backend {backend!r} failed (exit {proc.returncode})")
    for line in proc.stdout.splitlines():
        if line.startswith("BENCH_JSON:"):
            return json.loads(line[len("BENCH_JSON:") :])
    sys.stderr.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    raise SystemExit(f"backend {backend!r} produced no benchmark JSON")


def _fmt(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def _markdown(results: list[dict]) -> str:
    headers = [f"{r['backend']} ({r['voice']})" for r in results]
    lines = [
        "| metric | " + " | ".join(headers) + " |",
        "| --- | " + " | ".join("---" for _ in results) + " |",
    ]

    def row(label: str, values: list[str]) -> None:
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    row("sample rate (Hz)", [_fmt(r["sample_rate"]) for r in results])
    row("load (s)", [_fmt(r["load_s"]) for r in results])
    row("RSS idle (MB)", [_fmt(r["rss_idle_mb"]) for r in results])
    row("RSS loaded (MB)", [_fmt(r["rss_loaded_mb"]) for r in results])
    for text in TEXTS:
        row(
            f"RTF {text}",
            [_fmt(r["requests"][text]["rtf_mean"]) for r in results],
        )
        row(
            f"wall p50 {text} (s)",
            [_fmt(r["requests"][text]["wall_p50_s"]) for r in results],
        )
    row("stream TTFA (s)", [_fmt(r["stream"]["ttfa_s"]) for r in results])
    row("stream RTF", [_fmt(r["stream"]["rtf"]) for r in results])
    row(
        "throughput @4 (audio s / wall s)",
        [_fmt(r["concurrency"]["throughput_rtf"]) for r in results],
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--backend", choices=["piper", "pocket"], default="piper")
    parser.add_argument("--backends", default="piper,pocket")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--piper-voice", default=PIPER_VOICE)
    parser.add_argument("--pocket-voice", default=POCKET_VOICE)
    parser.add_argument("--pocket-language", default="english")
    parser.add_argument("--piper-voice-dir", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    if args.worker:
        print("BENCH_JSON:" + json.dumps(_run_worker(args)))
        return

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    results = [_spawn(args, backend) for backend in backends]
    table = _markdown(results)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nraw results -> {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
