#!/usr/bin/env python3
"""Evaluate a closed-set command layer over Whisper transcripts.

Two questions:

1. How often does a fuzzy match against a small command vocabulary recover a
   short reply that Whisper transcribed imperfectly?
2. How often does that same matcher falsely accept a non-command sentence?

The matcher is intentionally simple (normalized character edit distance) so the
result is a low-effort baseline, not a trained classifier.

Example:
    /workspace/stt-openai-server/.venv/bin/python \
        benchmarks/stt_command_eval.py \
        --fleurs-results artifacts/e2e-tts-stt/accuracy-.../results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e_tts_stt as bench  # noqa: E402

COMMANDS = [
    "ciao",
    "sì",
    "no",
    "forse",
    "va bene",
    "non lo so",
    "grazie mille",
]

SAMPLES = {
    "s1_ciao": "ciao",
    "s2_si": "sì",
    "s3_no": "no",
    "s4_forse": "forse",
    "s5_va_bene": "va bene",
    "s6_non_lo_so": "non lo so",
    "s7_grazie_mille": "grazie mille",
}


def _norm(text: str) -> str:
    return bench._normalize(text, fold_accents=True)


def map_command(transcript: str, threshold: int = 2) -> tuple[str | None, int]:
    """Return (command, distance) for the closest command within threshold."""
    hypothesis = _norm(transcript)
    if not hypothesis:
        return None, 10**6
    best_command = None
    best_distance = 10**6
    for command in COMMANDS:
        distance = bench._edit_distance(hypothesis, _norm(command))
        if distance < best_distance:
            best_distance = distance
            best_command = command
    if best_distance <= threshold:
        return best_command, best_distance
    return None, best_distance


def _load_models(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "models" in data:
        return data["models"]
    return [data]


def _short_accuracy(path: Path, threshold: int) -> list[dict]:
    rows = []
    for model in _load_models(path):
        if "per_text" not in model:
            continue
        for clip in model["per_text"]:
            expected = SAMPLES.get(clip["id"])
            if expected is None:
                continue
            transcript = clip["transcripts"][-1]
            mapped, distance = map_command(transcript, threshold)
            raw_ok = _norm(transcript) == _norm(expected)
            rows.append(
                {
                    "source": str(path),
                    "model": model["model_key"],
                    "clip": clip["id"],
                    "expected": expected,
                    "transcript": transcript,
                    "raw_ok": raw_ok,
                    "mapped": mapped,
                    "mapped_ok": mapped is not None and _norm(mapped) == _norm(expected),
                    "distance": distance,
                }
            )
    return rows


def _false_accepts(path: Path, threshold: int) -> dict:
    total = 0
    accepted = []
    for model in _load_models(path):
        for clip in model.get("per_clip", []):
            total += 1
            mapped, distance = map_command(clip["transcript"], threshold)
            if mapped is not None:
                accepted.append(
                    {
                        "model": model["model"],
                        "clip": clip["id"],
                        "mapped": mapped,
                        "distance": distance,
                        "transcript": clip["transcript"],
                    }
                )
    return {
        "total": total,
        "accepted": len(accepted),
        "rate": len(accepted) / total if total else 0.0,
        "examples": accepted[:10],
    }


def _summary(rows: list[dict]) -> dict:
    by_source: dict[str, list[dict]] = {}
    for row in rows:
        by_source.setdefault(row["source"], []).append(row)
    out = {}
    for source, items in by_source.items():
        out[source] = {
            "clips": len(items),
            "raw_ok": sum(item["raw_ok"] for item in items),
            "mapped_ok": sum(item["mapped_ok"] for item in items),
            "rows": items,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--short-results",
        type=Path,
        nargs="+",
        required=True,
        help="one or more short-suite results.json files (baseline and/or prompt)",
    )
    parser.add_argument("--fleurs-results", type=Path, default=None)
    parser.add_argument("--threshold", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict] = []
    for path in args.short_results:
        rows.extend(_short_accuracy(path.expanduser().resolve(), args.threshold))
    summary = _summary(rows)
    false_accepts = (
        _false_accepts(args.fleurs_results.expanduser().resolve(), args.threshold)
        if args.fleurs_results
        else None
    )

    out_dir = (
        args.out
        or Path(__file__).resolve().parents[1]
        / "artifacts"
        / "e2e-tts-stt"
        / "command-eval"
    ).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "threshold": args.threshold,
        "commands": COMMANDS,
        "sources": summary,
        "false_accepts": false_accepts,
    }
    (out_dir / "results.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Closed-set command layer over Whisper",
        "",
        f"Fuzzy matcher: normalized character edit distance, threshold={args.threshold}.",
        "",
        "| source | model | raw exact | mapped |",
        "| --- | --- | ---: | ---: |",
    ]
    for source, data in summary.items():
        for model in sorted({row["model"] for row in data["rows"]}):
            items = [row for row in data["rows"] if row["model"] == model]
            lines.append(
                f"| `{Path(source).parent.name}` | `{model}` | "
                f"{sum(row['raw_ok'] for row in items)}/{len(items)} | "
                f"{sum(row['mapped_ok'] for row in items)}/{len(items)} |"
            )
    if false_accepts:
        lines += [
            "",
            f"False accepts on {false_accepts['total']} FLEURS non-command clips: "
            f"**{false_accepts['accepted']}** ({false_accepts['rate'] * 100:.1f}%).",
        ]
        for example in false_accepts["examples"]:
            lines.append(
                f"- `{example['model']}` {example['clip']} -> `{example['mapped']}` "
                f"(d={example['distance']}): {example['transcript'][:120]!r}"
            )
    lines += ["", "Per-clip command mapping:", ""]
    for source, data in summary.items():
        lines += [f"### `{Path(source).parent.name}`", ""]
        lines += ["| clip | expected | transcript | mapped | raw | mapped ok |", "| --- | --- | --- | --- | ---: | ---: |"]
        for row in data["rows"]:
            lines.append(
                "| {clip} | {expected} | {transcript} | {mapped} | {raw} | {ok} |".format(
                    clip=row["clip"],
                    expected=row["expected"],
                    transcript=row["transcript"].replace("|", "\\|"),
                    mapped=row["mapped"] or "—",
                    raw="yes" if row["raw_ok"] else "no",
                    ok="yes" if row["mapped_ok"] else "no",
                )
            )
        lines.append("")
    report = "\n".join(lines)
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out_dir / 'results.json'} and {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
