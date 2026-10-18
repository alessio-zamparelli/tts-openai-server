#!/usr/bin/env python3
"""Run a staged STT tuning matrix through the real sequential pipeline.

The matrix is deliberately not a full Cartesian product: first isolate the
thread count, then beam search, then batched inference. Each cell boots a fresh
``stt-openai-server`` process with the requested env knobs; the TTS server is
started once and reused.

Example:
    uv run python benchmarks/stt_tuning_matrix.py \
        --models small,large-v3-turbo --runs-main 2 --runs-short 1

Outputs ``artifacts/e2e-tts-stt/matrix-<timestamp>/matrix.json`` plus a
Markdown table and per-cell result directories.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_tts_stt as bench  # noqa: E402

CONFIGS: dict[str, dict] = {
    "baseline": dict(cpu_threads=0, beam_size=5, batch_size=0, vad=False, temperatures=""),
    "threads3": dict(cpu_threads=3, beam_size=5, batch_size=0, vad=False, temperatures=""),
    "threads3_beam2": dict(cpu_threads=3, beam_size=2, batch_size=0, vad=False, temperatures=""),
    "threads3_beam1": dict(cpu_threads=3, beam_size=1, batch_size=0, vad=False, temperatures=""),
    "threads3_beam2_batch4": dict(cpu_threads=3, beam_size=2, batch_size=4, vad=False, temperatures=""),
    "threads3_beam5_batch4": dict(cpu_threads=3, beam_size=5, batch_size=4, vad=False, temperatures=""),
    "threads3_beam2_temps": dict(
        cpu_threads=3, beam_size=2, batch_size=0, vad=False, temperatures="0,0.2,0.4"
    ),
    "threads3_vad": dict(cpu_threads=3, beam_size=5, batch_size=0, vad=True, temperatures=""),
}

DEFAULT_CONFIGS = ",".join(CONFIGS)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default="small,large-v3-turbo")
    parser.add_argument("--configs", default=DEFAULT_CONFIGS)
    parser.add_argument("--suites", default="main,short")
    parser.add_argument("--runs-main", type=int, default=2)
    parser.add_argument("--runs-short", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--tts-voice", default="it_IT-paola-medium")
    parser.add_argument("--startup-timeout", type=float, default=240.0)
    parser.add_argument("--out", type=Path, default=None, help="reuse a matrix dir to resume")
    parser.add_argument("--force", action="store_true", help="redo cells that already have results.json")
    parser.add_argument("--keep-going", action="store_true", help="continue after a failed cell")
    return parser.parse_args(argv)


def _markdown(result: dict) -> str:
    lines = [
        "# STT tuning matrix",
        "",
        f"Generated: {result['timestamp']}",
        "",
        "Strictly sequential Piper -> Whisper; each cell is a fresh STT process.",
        "",
        "| model | config | suite | beam | threads | batch | VAD | STT RTF | E2E RTF | throughput | WER |",
        "| --- | --- | --- | ---: | ---: | ---: | :---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        c = row["config"]
        lines.append(
            "| `{model}` | `{config}` | {suite} | {beam} | {threads} | {batch} | {vad} | "
            "{stt_rtf:.3f} | {e2e_rtf:.3f} | {tp:.2f}x | {wer:.1f}% |".format(
                model=row["model"],
                config=row["config_name"],
                suite=row["suite"],
                beam=c["beam_size"],
                threads=c["cpu_threads"],
                batch=c["batch_size"],
                vad="yes" if c["vad"] else "no",
                stt_rtf=row["corpus"]["stt_rtf"],
                e2e_rtf=row["corpus"]["e2e_rtf"],
                tp=row["corpus"]["throughput_audio_s_per_wall_s"],
                wer=row["corpus"]["wer_folded"] * 100,
            )
        )
    lines += [
        "",
        "`STT RTF`/`E2E RTF` are audio-weighted over the suite. `WER` is accent-folded.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    config_names = [c.strip() for c in args.configs.split(",") if c.strip()]
    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown_models = [m for m in models if m not in bench.MODELS]
    unknown_configs = [c for c in config_names if c not in CONFIGS]
    unknown_suites = [s for s in suites if s not in bench.SUITES]
    if unknown_models or unknown_configs or unknown_suites:
        raise SystemExit(
            f"unknown model/config/suite: {unknown_models} {unknown_configs} {unknown_suites}"
        )

    tts_repo, stt_repo = bench._default_repos()
    tts_repo = tts_repo.resolve()
    stt_repo = stt_repo.resolve()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = (
        args.out or tts_repo / "artifacts" / "e2e-tts-stt" / f"matrix-{stamp}"
    ).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tts_port = bench._free_port()
    tts_env = os.environ.copy()
    tts_env.update(
        {
            "POCKET_TTS_BACKEND": "piper",
            "POCKET_TTS_HOST": "127.0.0.1",
            "POCKET_TTS_PORT": str(tts_port),
            "POCKET_TTS_DEFAULT_VOICE": args.tts_voice,
            "POCKET_TTS_IDLE_UNLOAD_S": "0",
        }
    )
    tts_bin = tts_repo / ".venv/bin/pocket-tts-openai"
    tts_server = bench._start_server(
        name="tts",
        cmd=[str(tts_bin)],
        cwd=tts_repo,
        env=tts_env,
        port=tts_port,
        log_path=out_dir / "logs/tts.log",
        timeout=args.startup_timeout,
    )

    rows: list[dict] = []
    errors: list[dict] = []
    started = time.time()
    try:
        for model in models:
            for config_name in config_names:
                config = CONFIGS[config_name]
                for suite in suites:
                    runs = args.runs_main if suite == "main" else args.runs_short
                    cell_dir = out_dir / model / config_name / suite
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    result_path = cell_dir / "results.json"
                    cell = f"{model}/{config_name}/{suite}"
                    if result_path.exists() and not args.force:
                        summary = json.loads(result_path.read_text(encoding="utf-8"))
                        # Cells produced by a standalone e2e_tts_stt.py run are
                        # wrapped as {"environment","config","models"}; cells
                        # produced by this matrix are the summary itself.
                        if "corpus" not in summary and summary.get("models"):
                            summary = summary["models"][0]
                        print(f"skip existing {cell}", flush=True)
                    else:
                        cell_args = argparse.Namespace(
                            beam_size=config["beam_size"],
                            compute_type=args.compute_type,
                            cpu_threads=config["cpu_threads"],
                            batch_size=config["batch_size"],
                            vad=config["vad"],
                            temperatures=config["temperatures"],
                            initial_prompt="",
                            label=f"{model}:{config_name}:{suite}",
                            runs=runs,
                            warmup=args.warmup,
                            tts_voice=args.tts_voice,
                            startup_timeout=args.startup_timeout,
                        )
                        print(
                            f"\n=== {model} / {config_name} / {suite} "
                            f"(beam={config['beam_size']} threads={config['cpu_threads']} "
                            f"batch={config['batch_size']} vad={config['vad']}) ===",
                            flush=True,
                        )
                        try:
                            summary = bench.run_model(
                                model_key=model,
                                model_id=bench.MODELS[model],
                                args=cell_args,
                                tts_url=tts_server.base_url,
                                stt_url=None,
                                out_dir=cell_dir,
                                tts_repo=tts_repo,
                                stt_repo=stt_repo,
                                texts=bench.SUITES[suite],
                            )
                        except Exception as exc:  # noqa: BLE001 - report and optionally continue
                            print(f"CELL FAILED {cell}: {exc!r}", flush=True)
                            errors.append({"cell": cell, "error": repr(exc)})
                            if args.keep_going:
                                continue
                            raise
                        result_path.write_text(
                            json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                    rows.append(
                        {
                            "model": model,
                            "config_name": config_name,
                            "config": config,
                            "suite": suite,
                            "runs": runs,
                            "corpus": summary["corpus"],
                            "startup_stt_s": summary["startup_stt_s"],
                            "rss_stt_mb": summary["rss_stt_mb"],
                        }
                    )
    finally:
        tts_server.stop()

    result = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_s": time.time() - started,
        "rows": rows,
        "errors": errors,
        "out_dir": str(out_dir),
    }
    (out_dir / "matrix.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    report = _markdown(result)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print("\n" + report)
    print(f"Wrote {out_dir / 'matrix.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
