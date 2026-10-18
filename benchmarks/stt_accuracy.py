#!/usr/bin/env python3
"""Evaluate STT models on real Italian audio (FLEURS manifest).

No TTS: each clip is sent directly to /v1/audio/transcriptions. The manifest is
the JSONL produced by ``prepare_fleurs_it.py``.

    /workspace/stt-openai-server/.venv/bin/python \
        benchmarks/stt_accuracy.py \
        --manifest artifacts/fleurs-it/manifest.jsonl \
        --models base,small,medium,large-v3-turbo --runs 1

Outputs ``artifacts/e2e-tts-stt/accuracy-<timestamp>/``.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_tts_stt as bench  # noqa: E402


def _read_manifest(path: Path) -> list[dict]:
    clips = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            clips.append(json.loads(line))
    return clips


def _median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def _run_model(
    *,
    model: str,
    args: argparse.Namespace,
    clips: list[dict],
    out_dir: Path,
    stt_repo: Path,
) -> dict:
    port = bench._free_port()
    env = os.environ.copy()
    env.update(
        {
            "WHISPER_MODEL_NAME": bench.MODELS[model],
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

    log_path = out_dir / f"logs/stt-{model}.log"
    server = bench._start_server(
        name=f"stt-{model}",
        cmd=[str(stt_repo / ".venv/bin/python"), "app.py"],
        cwd=stt_repo,
        env=env,
        port=port,
        log_path=log_path,
        timeout=args.startup_timeout,
    )
    samples: list[dict] = []
    rss_mb = None
    try:
        with bench._client() as client:
            # Warm the model with the first clip.
            first = Path(clips[0]["audio"])
            bench._stt_text(client, server.base_url, first.read_bytes())
            for run in range(1, args.runs + 1):
                for clip in clips:
                    audio = Path(clip["audio"])
                    payload = audio.read_bytes()
                    transcript, elapsed = bench._stt_text(client, server.base_url, payload)
                    rates = bench.error_rates(clip["text"], transcript)
                    samples.append(
                        {
                            "id": clip["id"],
                            "bucket": clip.get("bucket"),
                            "duration_s": clip["duration_s"],
                            "run": run,
                            "stt_s": elapsed,
                            "stt_rtf": elapsed / clip["duration_s"],
                            "transcript": transcript,
                            "wer": rates["wer"],
                            "wer_folded": rates["wer_folded"],
                            "cer_folded": rates["cer_folded"],
                            "ref_words": rates["ref_words"],
                        }
                    )
                    print(
                        f"  {model} run {run}/{args.runs} {clip['id']:<16}"
                        f" audio={clip['duration_s']:5.2f}s stt={elapsed:6.2f}s"
                        f" wer={rates['wer_folded'] * 100:5.1f}%",
                        flush=True,
                    )
            rss_mb = bench._rss_mb(server.process.pid)
    finally:
        server.stop()

    per_clip = []
    for clip in clips:
        rows = [s for s in samples if s["id"] == clip["id"]]
        if not rows:
            continue
        per_clip.append(
            {
                "id": clip["id"],
                "bucket": clip.get("bucket"),
                "duration_s": clip["duration_s"],
                "stt_s": _median([r["stt_s"] for r in rows]),
                "stt_rtf": _median([r["stt_rtf"] for r in rows]),
                "wer": _median([r["wer"] for r in rows]),
                "wer_folded": _median([r["wer_folded"] for r in rows]),
                "cer_folded": _median([r["cer_folded"] for r in rows]),
                "transcript": rows[-1]["transcript"],
                "text": clip["text"],
            }
        )

    total_audio = sum(s["duration_s"] for s in samples)
    total_stt = sum(s["stt_s"] for s in samples)
    weighted_wer = sum(s["wer_folded"] * s["ref_words"] for s in samples) / sum(
        s["ref_words"] for s in samples
    )
    return {
        "model": model,
        "model_id": bench.MODELS[model],
        "startup_s": server.startup_s,
        "rss_mb": rss_mb,
        "config": {
            "beam_size": args.beam_size,
            "cpu_threads": args.cpu_threads,
            "batch_size": args.batch_size,
            "vad": args.vad,
            "temperatures": args.temperatures,
            "initial_prompt": args.initial_prompt,
        },
        "corpus": {
            "clips": len(clips),
            "runs": args.runs,
            "audio_s": total_audio,
            "stt_s": total_stt,
            "stt_rtf": total_stt / total_audio,
            "wer": weighted_wer,
            "wer_folded": weighted_wer,
            "mean_wer": statistics.mean(c["wer_folded"] for c in per_clip),
        },
        "per_clip": per_clip,
        "samples": samples,
    }


def _report(result: dict) -> str:
    lines = [
        "# STT accuracy on real Italian audio (FLEURS)",
        "",
        f"Generated: {result['timestamp']}",
        "",
        f"Manifest: `{result['manifest']}` — {result['clips']} clips, "
        f"strictly sequential requests, `language=it`.",
        "",
        "| model | startup | STT RTF | WER | mean clip WER | clips | RSS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in result["models"]:
        c = model["corpus"]
        lines.append(
            "| `{m}` | {startup:.2f} s | {rtf:.3f} | {wer:.1f}% | {mean:.1f}% | "
            "{clips} | {rss} MB |".format(
                m=model["model"],
                startup=model["startup_s"],
                rtf=c["stt_rtf"],
                wer=c["wer"] * 100,
                mean=c["mean_wer"] * 100,
                clips=c["clips"],
                rss=f"{model['rss_mb']:.0f}" if model["rss_mb"] else "n/a",
            )
        )
    lines += ["", "## Per-clip", ""]
    for model in result["models"]:
        lines += [
            f"### `{model['model']}`",
            "",
            "| clip | bucket | audio s | STT s | RTF | WER | transcript | reference |",
            "| --- | --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for clip in model["per_clip"]:
            lines.append(
                "| {id} | {bucket} | {dur:.2f} | {stt:.2f} | {rtf:.3f} | {wer:.1f}% | "
                "{transcript} | {text} |".format(
                    id=clip["id"],
                    bucket=clip["bucket"],
                    dur=clip["duration_s"],
                    stt=clip["stt_s"],
                    rtf=clip["stt_rtf"],
                    wer=clip["wer_folded"] * 100,
                    transcript=clip["transcript"].replace("|", "\\|"),
                    text=clip["text"].replace("|", "\\|"),
                )
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models", default="base,small,medium,large-v3-turbo")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--temperatures", default="")
    parser.add_argument("--initial-prompt", default="")
    parser.add_argument("--startup-timeout", type=float, default=300.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    _, stt_repo = bench._default_repos()
    stt_repo = stt_repo.resolve()
    manifest = args.manifest.expanduser().resolve()
    clips = _read_manifest(manifest)
    if not clips:
        raise SystemExit(f"empty manifest: {manifest}")
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in models if m not in bench.MODELS]
    if unknown:
        raise SystemExit(f"unknown model(s): {unknown}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        args.out
        or Path(__file__).resolve().parents[1]
        / "artifacts"
        / "e2e-tts-stt"
        / f"accuracy-{stamp}"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "manifest": str(manifest),
        "clips": len(clips),
        "environment": {
            "platform": sys.platform,
            "cpu_count": os.cpu_count(),
            "memory_limit_gib": bench._cgroup_memory_limit_gib(),
        },
        "models": [],
    }
    started = time.time()
    for model in models:
        print(f"\n=== {model} ({bench.MODELS[model]}) ===", flush=True)
        result["models"].append(
            _run_model(
                model=model,
                args=args,
                clips=clips,
                out_dir=out_dir,
                stt_repo=stt_repo,
            )
        )
    result["elapsed_s"] = time.time() - started

    (out_dir / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = _report(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
