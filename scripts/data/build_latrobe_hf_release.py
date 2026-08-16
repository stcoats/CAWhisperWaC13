#!/usr/bin/env python3
"""Build the CC BY 4.0 La Trobe ASR dataset used in the main experiment."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


SPLITS = ("train", "validation", "test")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    manifest = project / "manifests/latrobe_exact_v2_nonoverlap.jsonl"
    rows = read_jsonl(manifest)
    if not rows:
        raise SystemExit(f"empty manifest: {manifest}")

    fields = [
        "file_name",
        "text",
        "example_id",
        "recording",
        "speakers",
        "speaker_count",
        "duration",
        "source_start",
        "source_end",
        "token_count",
        "quality_tier",
        "boundary_method",
    ]
    counts: Counter[str] = Counter()
    hours: Counter[str] = Counter()

    for split in SPLITS:
        selected = sorted(
            (row for row in rows if row["split"] == split),
            key=lambda row: row["example_id"],
        )
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        with (split_dir / "metadata.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in selected:
                source = project / row["audio"]
                if not source.is_file():
                    raise FileNotFoundError(source)
                filename = f"{row['example_id']}.wav"
                shutil.copy2(source, split_dir / filename)
                writer.writerow(
                    {
                        "file_name": filename,
                        "text": row["text"],
                        "example_id": row["example_id"],
                        "recording": row["recording"],
                        "speakers": "|".join(row.get("speakers", [])),
                        "speaker_count": row.get("speaker_count", 0),
                        "duration": row["duration"],
                        "source_start": row["start"],
                        "source_end": row["end"],
                        "token_count": row.get("token_count", 0),
                        "quality_tier": row.get("quality_tier", ""),
                        "boundary_method": row["boundary_method"],
                    }
                )
                counts[split] += 1
                hours[split] += float(row["duration"]) / 3600

    summary = {
        "version": "latrobe_ca_asr_exact_v2_nonoverlap",
        "license": "CC BY 4.0",
        "source_doi": "10.26181/23089559",
        "examples": dict(counts),
        "hours": {key: round(value, 6) for key, value in hours.items()},
        "total_examples": sum(counts.values()),
        "total_hours": round(sum(hours.values()), 6),
        "selection": "exact-boundary examples without transcript-marked overlap",
    }
    (output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
