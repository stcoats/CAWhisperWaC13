#!/usr/bin/env python3
"""Decode each GCSAusE MP3 once and cut exact 16 kHz mono training WAVs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def process_recording(
    project: str,
    recording: str,
    rows: list[dict],
    ffmpeg: str,
    temporary_root: str,
) -> dict:
    project_path = Path(project)
    source = project_path / rows[0]["source_audio"]
    if not source.is_file():
        raise FileNotFoundError(source)
    with tempfile.TemporaryDirectory(prefix=f"gcsause_{recording}_", dir=temporary_root) as tmp:
        decoded = Path(tmp) / f"{recording}.wav"
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(source),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(decoded),
            ],
            check=True,
        )
        sample_rate, audio = wavfile.read(decoded, mmap=True)
        if sample_rate != 16000 or audio.ndim != 1 or audio.dtype != np.int16:
            raise ValueError(
                f"unexpected decoded format for {recording}: "
                f"rate={sample_rate}, shape={audio.shape}, dtype={audio.dtype}"
            )
        written = 0
        for row in rows:
            start = max(0, round(float(row["start"]) * sample_rate))
            end = min(len(audio), round(float(row["end"]) * sample_rate))
            if end <= start:
                raise ValueError(f"invalid cut for {row['example_id']}: {start}:{end}")
            output = project_path / row["audio"]
            output.parent.mkdir(parents=True, exist_ok=True)
            wavfile.write(output, sample_rate, np.asarray(audio[start:end], dtype=np.int16))
            written += 1
    return {"recording": recording, "chunks": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ffmpeg", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = args.manifest.resolve()
    ffmpeg = args.ffmpeg.resolve()
    if not ffmpeg.is_file():
        raise FileNotFoundError(ffmpeg)
    rows = read_jsonl(manifest)
    by_recording: dict[str, list[dict]] = {}
    for row in rows:
        by_recording.setdefault(row["source_recording"], []).append(row)
    temporary_root = Path(os.environ.get("SLURM_TMPDIR", "/tmp"))
    temporary_root.mkdir(parents=True, exist_ok=True)

    completed = 0
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_recording,
                str(project),
                recording,
                selected,
                str(ffmpeg),
                str(temporary_root),
            ): recording
            for recording, selected in by_recording.items()
        }
        for future in as_completed(futures):
            result = future.result()
            completed += result["chunks"]
            print(
                f"recording={result['recording']} chunks={result['chunks']} "
                f"completed={completed}/{len(rows)}",
                flush=True,
            )
    if completed != len(rows):
        raise ValueError(f"wrote {completed} of {len(rows)} chunks")

    bad = []
    total_seconds = 0.0
    for row in rows:
        path = project / row["audio"]
        sample_rate, audio = wavfile.read(path, mmap=True)
        duration = audio.shape[0] / sample_rate
        total_seconds += duration
        expected = float(row["duration"])
        if (
            sample_rate != 16000
            or audio.ndim != 1
            or abs(duration - expected) > 1.5 / sample_rate
            or duration > 27.5
        ):
            bad.append(
                {
                    "example_id": row["example_id"],
                    "sample_rate": sample_rate,
                    "shape": audio.shape,
                    "duration": duration,
                    "expected": expected,
                }
            )
    if bad:
        raise ValueError(f"invalid output chunks: {bad[:10]}")
    print(
        json.dumps(
            {
                "recordings": len(by_recording),
                "chunks": len(rows),
                "hours": total_seconds / 3600,
                "validation_failures": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
