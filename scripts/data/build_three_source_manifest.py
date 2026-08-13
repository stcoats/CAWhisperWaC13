#!/usr/bin/env python3
"""Combine frozen La Trobe/SBCSAE rows with prepared GCSAusE rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def absolutize(rows: list[dict], root: Path) -> list[dict]:
    output = []
    for row in rows:
        audio = Path(row["audio"])
        if not audio.is_absolute():
            audio = root / audio
        output.append({**row, "audio": str(audio.resolve())})
    return output


def stats(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "hours": round(sum(float(row["duration"]) for row in rows) / 3600, 6),
        "tokens": sum(int(row.get("token_count", 0)) for row in rows),
        "recordings": len({row["recording"] for row in rows}),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-source-manifest", type=Path, required=True)
    parser.add_argument("--two-source-root", type=Path, required=True)
    parser.add_argument("--gcsause-manifest", type=Path, required=True)
    parser.add_argument("--gcsause-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--check-audio", action="store_true")
    args = parser.parse_args()

    first = absolutize(
        read_jsonl(args.two_source_manifest.resolve()),
        args.two_source_root.resolve(),
    )
    third = absolutize(
        read_jsonl(args.gcsause_manifest.resolve()),
        args.gcsause_root.resolve(),
    )
    if {row.get("corpus") for row in first} != {"latrobe", "sbcsae"}:
        raise ValueError("two-source input does not contain exactly La Trobe and SBCSAE")
    if {row.get("corpus") for row in third} != {"gcsause"}:
        raise ValueError("GCSAusE input has an unexpected corpus label")
    rows = sorted(
        first + third,
        key=lambda row: (
            {"train": 0, "validation": 1, "test": 2}[row["split"]],
            row["corpus"],
            row["example_id"],
        ),
    )
    ids = [row["example_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate example identifiers in combined manifest")
    if args.check_audio:
        missing = [row["audio"] for row in rows if not Path(row["audio"]).is_file()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} missing audio files; first: {missing[0]}")
    split_ids = {
        split: {row["example_id"] for row in rows if row["split"] == split}
        for split in ("train", "validation", "test")
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("split leakage detected")

    write_jsonl(args.output.resolve(), rows)
    report = {
        "version": "latrobe_sbcsae_gcsause_exact_v1_nonoverlap",
        "selection": (
            "frozen exact-boundary non-overlap La Trobe/SBCSAE release plus "
            "recording-disjoint exact-boundary non-overlap GCSAusE all-users release"
        ),
        "inputs": {
            "two_source_manifest": str(args.two_source_manifest.resolve()),
            "two_source_sha256": sha256(args.two_source_manifest.resolve()),
            "gcsause_manifest": str(args.gcsause_manifest.resolve()),
            "gcsause_sha256": sha256(args.gcsause_manifest.resolve()),
        },
        "output": str(args.output.resolve()),
        "output_sha256": sha256(args.output.resolve()),
        "overall": stats(rows),
        "by_corpus": {
            corpus: stats([row for row in rows if row["corpus"] == corpus])
            for corpus in ("latrobe", "sbcsae", "gcsause")
        },
        "split_integrity": "example IDs are unique and disjoint across splits",
        "audio_check": bool(args.check_audio),
    }
    args.summary.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.summary.resolve().write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
