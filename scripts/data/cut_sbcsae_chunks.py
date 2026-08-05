#!/usr/bin/env python3
"""Cut prepared SBCSAE examples once per source recording."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def process_recording(
    project: str, recording: str, rows: list[dict]
) -> dict:
    project_path = Path(project)
    source = project_path / rows[0]["source_audio"]
    sample_rate, audio = wavfile.read(source)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scale = float(max(abs(info.min), info.max))
        audio = audio.astype(np.float32) / scale
    else:
        audio = audio.astype(np.float32)
    if audio.ndim == 1:
        audio = audio[:, None]
    audio = audio.mean(axis=1)
    if sample_rate != 16000:
        divisor = gcd(sample_rate, 16000)
        audio = resample_poly(
            audio, 16000 // divisor, sample_rate // divisor
        ).astype(np.float32)
    written = 0
    for row in rows:
        start = max(0, round(float(row["start"]) * 16000))
        end = min(len(audio), round(float(row["end"]) * 16000))
        if end <= start:
            raise ValueError(f"invalid cut for {row['example_id']}")
        output = project_path / row["audio"]
        output.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(audio[start:end], -1.0, 1.0)
        pcm = np.rint(pcm * 32767.0).astype(np.int16)
        wavfile.write(output, 16000, pcm)
        written += 1
    return {"recording": recording, "chunks": written}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    project = args.project.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else project / args.manifest
    )
    rows = read_jsonl(manifest)
    by_recording: dict[str, list[dict]] = {}
    for row in rows:
        by_recording.setdefault(row["recording"], []).append(row)
    totals = {"recordings": 0, "chunks": 0}
    with ProcessPoolExecutor(max_workers=args.jobs) as executor:
        futures = {
            executor.submit(
                process_recording,
                str(project),
                recording,
                selected,
            ): recording
            for recording, selected in by_recording.items()
        }
        for future in as_completed(futures):
            result = future.result()
            totals["recordings"] += 1
            totals["chunks"] += result["chunks"]
            print(
                f"completed={totals['recordings']}/{len(futures)} "
                f"recording={result['recording']} "
                f"chunks={result['chunks']}",
                flush=True,
            )
    expected = len(rows)
    if totals["chunks"] != expected:
        raise ValueError(f"wrote {totals['chunks']} of {expected} chunks")
    bad = []
    total_seconds = 0.0
    for row in rows:
        sample_rate, audio = wavfile.read(
            project / row["audio"], mmap=True
        )
        channels = 1 if audio.ndim == 1 else audio.shape[1]
        duration = audio.shape[0] / sample_rate
        total_seconds += duration
        if (
            sample_rate != 16000
            or channels != 1
            or duration > 29.5
        ):
            bad.append(
                (
                    row["example_id"],
                    sample_rate,
                    channels,
                    duration,
                )
            )
    if bad:
        raise ValueError(f"invalid outputs: {bad[:10]}")
    print(
        json.dumps(
            {
                **totals,
                "hours": total_seconds / 3600,
                "validation_failures": len(bad),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
