#!/usr/bin/env python3
"""Prepare a small real-human Italian evaluation set from FLEURS.

Downloads the Italian (it_it) test split and extracts a stratified selection
of short / medium / long clips plus a JSONL manifest with reference text.

Run with the STT environment (which has huggingface_hub):
    /workspace/stt-openai-server/.venv/bin/python \
        benchmarks/prepare_fleurs_it.py --out artifacts/fleurs-it --per-bucket 8

The generated manifest is consumed by ``benchmarks/stt_accuracy.py``.
"""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "google/fleurs"


def _download(out_dir: Path) -> tuple[Path, Path]:
    download_dir = out_dir / "download"
    download_dir.mkdir(parents=True, exist_ok=True)
    tsv = Path(
        hf_hub_download(
            REPO, "data/it_it/test.tsv", repo_type="dataset", local_dir=download_dir
        )
    )
    archive = Path(
        hf_hub_download(
            REPO,
            "data/it_it/audio/test.tar.gz",
            repo_type="dataset",
            local_dir=download_dir,
        )
    )
    return tsv, archive


def _rows(tsv: Path) -> list[dict]:
    rows = []
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) < 7:
            continue
        _speaker, filename, raw_text, text, _phonemes, samples, _gender = fields[:7]
        rows.append(
            {
                "filename": filename,
                "text": raw_text if raw_text.strip() else text,
                "samples": int(samples),
                "duration_s": int(samples) / 16000.0,
            }
        )
    return rows


def _pick(rows: list[dict], per_bucket: int) -> list[dict]:
    rows = sorted(rows, key=lambda row: row["duration_s"])
    n = len(rows)
    buckets = [rows[: n // 3], rows[n // 3 : 2 * n // 3], rows[2 * n // 3 :]]
    picked = []
    for bucket_name, bucket in zip(("short", "medium", "long"), buckets):
        if not bucket:
            continue
        if len(bucket) <= per_bucket:
            chosen = bucket
        else:
            step = (len(bucket) - 1) / max(1, per_bucket - 1)
            chosen = [bucket[round(i * step)] for i in range(per_bucket)]
        for row in chosen:
            row = dict(row, bucket=bucket_name)
            picked.append(row)
    # Stable, readable ids.
    for index, row in enumerate(picked, 1):
        row["id"] = f"fleurs_{index:02d}_{row['bucket']}"
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/fleurs-it"))
    parser.add_argument("--per-bucket", type=int, default=8)
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv, archive = _download(out_dir)
    rows = _rows(tsv)
    picked = _pick(rows, args.per_bucket)

    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.jsonl"
    wanted = {row["filename"]: row for row in picked}
    extracted = 0
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = Path(member.name).name
            row = wanted.get(name)
            if row is None or not member.isfile():
                continue
            payload = tar.extractfile(member)
            if payload is None:
                continue
            target = audio_dir / name
            target.write_bytes(payload.read())
            row["audio"] = str(target)
            extracted += 1

    missing = [row["id"] for row in picked if "audio" not in row]
    if missing:
        raise SystemExit(f"could not extract {len(missing)} files: {missing}")

    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in picked:
            handle.write(
                json.dumps(
                    {
                        "id": row["id"],
                        "audio": row["audio"],
                        "text": row["text"],
                        "duration_s": round(row["duration_s"], 3),
                        "bucket": row["bucket"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"extracted {extracted} clips -> {audio_dir}")
    print(f"manifest -> {manifest_path}")
    for bucket in ("short", "medium", "long"):
        chosen = [row for row in picked if row["bucket"] == bucket]
        if chosen:
            print(
                f"  {bucket}: {len(chosen)} clips, "
                f"{min(r['duration_s'] for r in chosen):.1f}-"
                f"{max(r['duration_s'] for r in chosen):.1f} s"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
