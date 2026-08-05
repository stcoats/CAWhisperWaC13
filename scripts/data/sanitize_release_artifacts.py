#!/usr/bin/env python3
"""Create publication-safe JSONL files from internal evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PUBLIC_FIELDS = (
    "example_id",
    "blind_id",
    "analysis_eligible",
    "country",
    "state",
    "council_name",
    "video_id",
    "video_title",
    "clip_start_seconds",
    "clip_end_seconds",
    "duration_seconds",
    "youtube_url_at_target",
    "stratum",
    "boundary_status",
    "overlap",
    "audio_quality",
    "reference",
    "hypothesis",
    "loop_repairs",
    "system",
    "mode",
)


def sanitize(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as reader, destination.open(
        "w", encoding="utf-8"
    ) as writer:
        for line in reader:
            if not line.strip():
                continue
            row = json.loads(line)
            public = {key: row[key] for key in PUBLIC_FIELDS if key in row}
            writer.write(json.dumps(public, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    sanitize(args.source, args.destination)


if __name__ == "__main__":
    main()
