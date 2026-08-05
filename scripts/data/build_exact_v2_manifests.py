#!/usr/bin/env python3
"""Build exact-boundary primary and overlap-only CA-ASR manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def summarize(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "hours": sum(float(row["duration"]) for row in rows) / 3600,
        "tokens": sum(int(row.get("token_count", 0)) for row in rows),
        "by_split": dict(Counter(row["split"] for row in rows)),
        "multi_speaker_examples": sum(
            int(row.get("speaker_count", len(row.get("speakers", [])))) > 1
            for row in rows
        ),
    }


def latrobe_rows(project: Path) -> tuple[list[dict], list[dict]]:
    candidates = {
        row["candidate_id"]: row
        for row in read_jsonl(project / "manifests/lora_v1_candidates.jsonl")
    }
    frozen = read_jsonl(project / "manifests/latrobe_lora_v1.jsonl")
    exact: list[dict] = []
    cuts: list[dict] = []
    for original in frozen:
        candidate = candidates[original["example_id"]]
        # align_reference_to_asr adds 0.15 s to both target-word edges before
        # adding the separate coarse safety padding. Undo both margins here.
        start = float(candidate["speech_start"]) + 0.15
        end = float(candidate["speech_end"]) - 0.15
        if end <= start:
            raise ValueError(f"invalid La Trobe boundary: {original['example_id']}")
        audio = (
            f"runs/exact_v2/latrobe_chunks/{original['recording']}/"
            f"{original['example_id']}.wav"
        )
        lab = audio[:-4] + ".lab"
        overlap = bool(
            candidate.get("overlap_token_count", 0)
            or "overlap" in candidate.get("flags", [])
        )
        row = {
            **original,
            "audio": audio,
            "duration": round(end - start, 6),
            "source_audio": candidate["source_audio"],
            "start": round(start, 6),
            "end": round(end, 6),
            "speaker_count": len(original.get("speakers", [])),
            "has_overlap": overlap,
            "boundary_method": "whisperx_first_word_start_to_last_word_end",
            "corpus": "latrobe",
        }
        exact.append(row)
        cuts.append(
            {
                "candidate_id": original["example_id"],
                "recording": original["recording"],
                "quality_tier": original["quality_tier"],
                "source_audio": candidate["source_audio"],
                "coarse_audio": audio,
                "lab_file": lab,
                "coarse_start": round(start, 6),
                "coarse_end": round(end, 6),
                "coarse_duration": round(end - start, 6),
                "text": original["text"],
            }
        )
    return exact, cuts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--sbcsae-exact",
        type=Path,
        default=Path("sbcsae/prepared_exact_v2/sbcsae_clean_v1.jsonl"),
    )
    args = parser.parse_args()
    project = args.project.resolve()
    sbc_path = (
        args.sbcsae_exact
        if args.sbcsae_exact.is_absolute()
        else project / args.sbcsae_exact
    )

    latrobe_all, latrobe_cuts = latrobe_rows(project)
    latrobe_primary = [row for row in latrobe_all if not row["has_overlap"]]
    latrobe_overlap = [row for row in latrobe_all if row["has_overlap"]]

    sbc_all = []
    for original in read_jsonl(sbc_path):
        overlap = float(original.get("overlap_seconds", 0)) > 0
        sbc_all.append(
            {
                **original,
                "has_overlap": overlap,
                "boundary_method": "source_transcript_first_unit_start_to_last_unit_end",
                "corpus": "sbcsae",
            }
        )
    sbc_primary = [row for row in sbc_all if not row["has_overlap"]]
    sbc_overlap = [row for row in sbc_all if row["has_overlap"]]
    combined = sorted(
        latrobe_primary + sbc_primary,
        key=lambda row: (row["split"], row["corpus"], row["example_id"]),
    )

    outputs = {
        "latrobe_exact_v2_all.jsonl": latrobe_all,
        "latrobe_exact_v2_nonoverlap.jsonl": latrobe_primary,
        "latrobe_exact_v2_overlap.jsonl": latrobe_overlap,
        "latrobe_exact_v2_cut_manifest.jsonl": latrobe_cuts,
        "sbcsae_exact_v2_all.jsonl": sbc_all,
        "sbcsae_exact_v2_nonoverlap.jsonl": sbc_primary,
        "sbcsae_exact_v2_overlap.jsonl": sbc_overlap,
        "latrobe_sbcsae_exact_v2_nonoverlap.jsonl": combined,
    }
    for name, rows in outputs.items():
        write_jsonl(project / "manifests" / name, rows)

    summary = {
        "version": "exact_v2_nonoverlap",
        "boundary_policy": {
            "latrobe": "WhisperX-aligned first target-word start through last target-word end; no safety padding",
            "sbcsae": "source CHAT first lexical-unit start through last lexical-unit end; no safety padding",
        },
        "primary_policy": "retain sequential speaker changes; exclude every segment with transcript-marked temporal overlap",
        "latrobe_all": summarize(latrobe_all),
        "latrobe_primary": summarize(latrobe_primary),
        "latrobe_overlap": summarize(latrobe_overlap),
        "sbcsae_all": summarize(sbc_all),
        "sbcsae_primary": summarize(sbc_primary),
        "sbcsae_overlap": summarize(sbc_overlap),
        "combined_primary": summarize(combined),
    }
    (project / "manifests/exact_v2_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
