#!/usr/bin/env python3
"""Prepare auditable SBCSAE metadata and lexical-verbatim chunk manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable


TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
WORD_RE = re.compile(r"\b[\w']+(?:-(?=\s|$))?", re.UNICODE)
EVENT_RE = re.compile(r"&=([A-Za-z0-9:_-]+)")
PROMPT_MARKUP_RE = re.compile(r"&[{}][A-Za-z]=[^\s]+")
PAUSE_RE = re.compile(r"\((?:\.{1,3}|\d+(?:\.\d+)?)\)")
BRACKET_CODE_RE = re.compile(r"\[(?:/|//|\?|!|=! [^\]]+)\]")
SCOPED_CODE_RE = re.compile(r"\[(?:\+|-)[^\]]+\]")
EVENT_MAP = {
    "laugh": "[laughter]",
    "laughter": "[laughter]",
    "cough": "[cough]",
    "coughs": "[cough]",
    "sneeze": "[sneeze]",
    "sniff": "[sniff]",
}
DROP_EVENTS = {
    "lengthened",
    "creaky",
    "high",
    "low",
    "whisper",
    "whispered",
    "nasal",
    "uncertain",
    "in",
    "inhale",
    "ex",
    "exhale",
    "breath",
    "tsk",
    "click",
    "lipsmack",
}


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            count += 1
    return count


def read_segment_metadata(source: Path) -> tuple[dict, dict]:
    recordings: dict[str, dict] = defaultdict(dict)
    speakers: dict[str, set[str]] = defaultdict(set)
    for path in sorted((source / "docs").glob("Part_*/segment.tbl")):
        for line in path.read_text(
            encoding="utf-8", errors="replace"
        ).splitlines():
            fields = line.split("\t")
            if len(fields) < 3:
                continue
            raw_recording = fields[0].strip().lower()
            match = re.search(r"(\d+)$", raw_recording)
            if not match:
                continue
            recording = f"SBC{int(match.group(1)):03d}"
            key = fields[1].strip().rstrip(":")
            value = "\t".join(fields[2:]).strip()
            if key == "speaker":
                speakers[recording].add(value)
            elif key:
                recordings[recording][key] = value
    return dict(recordings), dict(speakers)


def participant_codes(lines: list[str]) -> list[str]:
    for line in lines:
        if line.startswith("@Participants:"):
            body = line.split(":", 1)[1].strip()
            return [
                item.strip().split()[0]
                for item in body.split(",")
                if item.strip()
            ]
    return []


def event_replacement(match: re.Match) -> str:
    label = match.group(1).lower().split(":", 1)[0]
    if label in EVENT_MAP:
        return f" {EVENT_MAP[label]} "
    if label in DROP_EVENTS:
        return " "
    return " "


def normalize_chat_text(raw: str) -> tuple[str, list[str]]:
    """Normalize CHAT markup while retaining lexical-verbatim content."""
    flags: list[str] = []
    if re.search(
        r"\b(?:x{2,}|y{2,}|www)\b", raw, flags=re.IGNORECASE
    ):
        flags.append("unintelligible")
    if EVENT_RE.search(raw):
        flags.append("vocal_event")
    if any(character in raw for character in "⌈⌉⌊⌋"):
        flags.append("overlap_markup")

    laugh_quality = "&{l=@" in raw
    text = raw.replace("\n", " ")
    text = re.sub(
        r"&[A-Za-z][\w'-]*\s+\[:\s*([^\]]+)\]",
        r"\1",
        text,
    )
    text = re.sub(
        r"\[\s*(?:%\s*)?laugh(?:ter)?\s*\]",
        " [laughter] ",
        text,
        flags=re.IGNORECASE,
    )
    text = PROMPT_MARKUP_RE.sub(" ", text)
    text = EVENT_RE.sub(event_replacement, text)
    text = re.sub(r"&-([A-Za-z][\w'-]*)", r"\1", text)
    text = re.sub(r"&\+([A-Za-z][\w']*)", r"\1-", text)
    text = re.sub(r"&([A-Za-z][\w']*)", r"\1-", text)
    text = PAUSE_RE.sub(" ", text)
    text = BRACKET_CODE_RE.sub(" ", text)
    text = SCOPED_CODE_RE.sub(" ", text)
    text = re.sub(r"\+(?:/\.|//\.|\.\.\.|,|\^)", " ", text)
    text = re.sub(r"x(?=[⌈⌉⌊⌋])|(?<=[⌈⌉⌊⌋])x", " ", text)
    text = re.sub(r"[⌈⌉⌊⌋](?:\d+)?", " ", text)
    text = re.sub(r"[<>]", " ", text)
    text = re.sub(r"ʔ", "", text)
    text = text.replace("Ϋ", "")
    text = re.sub(r"(?<=\w):+(?=\w|\s|$)", "", text)
    text = re.sub(r"(?<=\w)_(?=\s|$)", "-", text)
    text = re.sub(r"(?<=\w)@[\w:-]+", "", text)
    text = re.sub(
        r"\b(?:x{2,}|y{2,}|www)\b", " ", text, flags=re.IGNORECASE
    )
    text = re.sub(r"\(([^()]*)\)", r"\1", text)
    text = re.sub(r"(?<=\w)_(?=\w)", " ", text)
    if laugh_quality:
        text = re.sub(r"\bSM\b", " ", text)
    for event in ("laughter", "cough", "sniff", "sneeze"):
        text = re.sub(
            rf"(?:\[{event}\]\s*){{2,}}",
            f"[{event}] ",
            text,
            flags=re.IGNORECASE,
        )
    text = re.sub(r"[%^=|]", " ", text)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([,.?!]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" \t,")
    return text, sorted(set(flags))


def parse_chat(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    recording = path.stem
    metadata = {
        "recording": recording,
        "participant_codes": participant_codes(lines),
        "comments": [
            line.split(":", 1)[1].strip()
            for line in lines
            if line.startswith("@Comment:")
        ],
    }
    units: list[dict] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_parts
        if current_speaker is None:
            current_parts = []
            return
        content = "\n".join(current_parts)
        cursor = 0
        for match in TIME_RE.finditer(content):
            raw = content[cursor : match.start()].strip()
            start_ms, end_ms = int(match.group(1)), int(match.group(2))
            text, flags = normalize_chat_text(raw)
            if text or flags:
                units.append(
                    {
                        "recording": recording,
                        "speaker": current_speaker,
                        "start": start_ms / 1000,
                        "end": end_ms / 1000,
                        "duration": (end_ms - start_ms) / 1000,
                        "raw": raw,
                        "text": text,
                        "flags": flags,
                    }
                )
            cursor = match.end()
        current_parts = []

    for line in lines:
        if line.startswith("*"):
            flush()
            tier, _, body = line.partition(":")
            current_speaker = tier[1:].strip()
            current_parts = [body.strip()]
        elif current_speaker is not None and (
            line.startswith("\t") or line.startswith(" ")
        ):
            current_parts.append(line.strip())
        elif line.startswith("%") or line.startswith("@"):
            flush()
            current_speaker = None
    flush()
    units.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    metadata["duration"] = max((row["end"] for row in units), default=0.0)
    metadata["units"] = len(units)
    return metadata, units


class UnionFind:
    def __init__(self, items: Iterable[str]):
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def assign_splits(
    recordings: list[dict],
    speaker_ids: dict[str, set[str]],
    seed: str,
) -> tuple[dict[str, str], list[dict]]:
    ids = [row["recording"] for row in recordings]
    union = UnionFind(ids)
    by_speaker: dict[str, list[str]] = defaultdict(list)
    for recording, speakers in speaker_ids.items():
        for speaker in speakers:
            by_speaker[speaker].append(recording)
    for members in by_speaker.values():
        for member in members[1:]:
            union.union(members[0], member)

    components: dict[str, list[str]] = defaultdict(list)
    for recording in ids:
        components[union.find(recording)].append(recording)
    durations = {
        row["recording"]: float(row["duration"]) for row in recordings
    }
    component_rows = []
    for members in components.values():
        members = sorted(members)
        component_rows.append(
            {
                "recordings": members,
                "duration": sum(durations[item] for item in members),
                "tie_break": hashlib.sha256(
                    f"{seed}:{','.join(members)}".encode()
                ).hexdigest(),
            }
        )
    component_rows.sort(
        key=lambda row: (-row["duration"], row["tie_break"])
    )
    targets = {"train": 0.70, "validation": 0.10, "test": 0.20}
    total = sum(row["duration"] for row in component_rows)
    used = {split: 0.0 for split in targets}
    split_map: dict[str, str] = {}
    for component in component_rows:
        split = min(
            targets,
            key=lambda name: (
                used[name] / max(1.0, total * targets[name]),
                hashlib.sha256(
                    f"{component['tie_break']}:{name}".encode()
                ).hexdigest(),
            ),
        )
        used[split] += component["duration"]
        component["split"] = split
        for recording in component["recordings"]:
            split_map[recording] = split
    return split_map, component_rows


def overlap_seconds(units: list[dict]) -> float:
    points = sorted(
        {
            float(unit[boundary])
            for unit in units
            for boundary in ("start", "end")
        }
    )
    overlap = 0.0
    for left, right in zip(points, points[1:]):
        middle = (left + right) / 2
        speakers = {
            unit["speaker"]
            for unit in units
            if unit["start"] < middle < unit["end"]
        }
        if len(speakers) > 1:
            overlap += right - left
    return overlap


def chunks_for_recording(
    recording: str,
    units: list[dict],
    split: str,
    max_span: float = 27.5,
    max_gap: float = 3.0,
    boundary_padding: float = 0.5,
    chunk_root: str = "sbcsae/chunks",
) -> list[dict]:
    lexical = [
        unit
        for unit in units
        if WORD_RE.search(unit["text"])
        and unit["duration"] <= max_span
        and unit["speaker"] not in {"ENV", "NOISE"}
    ]
    groups: list[list[dict]] = []
    current: list[dict] = []
    for unit in lexical:
        if current and (
            unit["end"] - current[0]["start"] > max_span
            or unit["start"] - current[-1]["end"] > max_gap
        ):
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)
    if (
        len(groups) > 1
        and groups[-1][-1]["end"] - groups[-1][0]["start"] < 4.0
        and groups[-1][-1]["end"] - groups[-2][0]["start"] <= max_span
    ):
        groups[-2].extend(groups.pop())

    chunks = []
    for index, group in enumerate(groups):
        start = max(0.0, group[0]["start"] - boundary_padding)
        end = group[-1]["end"] + boundary_padding
        duration = end - start
        overlap = overlap_seconds(group)
        flags = sorted(
            {
                flag
                for unit in group
                for flag in unit["flags"]
            }
        )
        text = " ".join(unit["text"] for unit in group).strip()
        speakers = sorted({unit["speaker"] for unit in group})
        chunks.append(
            {
                "example_id": f"{recording}_{index:04d}",
                "recording": recording,
                "split": split,
                "audio": (
                    f"{chunk_root}/{recording}/"
                    f"{recording}_{index:04d}.wav"
                ),
                "source_audio": f"sbcsae/source/original/WAV/{recording}.wav",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "text": text,
                "token_count": len(WORD_RE.findall(text)),
                "speakers": speakers,
                "speaker_count": len(speakers),
                "overlap_seconds": round(overlap, 3),
                "overlap_ratio": round(overlap / max(duration, 0.001), 6),
                "flags": flags,
                "quality_tier": (
                    "A"
                    if overlap == 0 and "unintelligible" not in flags
                    else "B"
                    if overlap / max(duration, 0.001) <= 0.05
                    and "unintelligible" not in flags
                    else "CHALLENGE"
                ),
            }
        )
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="sbcsae-latrobe-v1")
    parser.add_argument("--boundary-padding", type=float, default=0.5)
    parser.add_argument("--chunk-root", default="sbcsae/chunks")
    args = parser.parse_args()
    if args.boundary_padding < 0:
        parser.error("--boundary-padding must be nonnegative")
    source, output = args.source.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    segment_metadata, speaker_ids = read_segment_metadata(source)
    recordings, all_units = [], []
    units_by_recording = {}
    for path in sorted((source / "CHAT").glob("SBC*.cha")):
        metadata, units = parse_chat(path)
        metadata.update(segment_metadata.get(path.stem, {}))
        metadata["speaker_ids"] = sorted(speaker_ids.get(path.stem, set()))
        recordings.append(metadata)
        all_units.extend(units)
        units_by_recording[path.stem] = units
    if len(recordings) != 60:
        raise ValueError(f"expected 60 recordings, found {len(recordings)}")

    split_map, components = assign_splits(
        recordings, speaker_ids, args.seed
    )
    for row in recordings:
        row["split"] = split_map[row["recording"]]
    chunks = [
        chunk
        for recording in recordings
        for chunk in chunks_for_recording(
            recording["recording"],
            units_by_recording[recording["recording"]],
            recording["split"],
            boundary_padding=args.boundary_padding,
            chunk_root=args.chunk_root,
        )
    ]
    clean_chunks = [
        row for row in chunks if row["quality_tier"] in {"A", "B"}
    ]
    excluded_long_units = [
        row
        for row in all_units
        if WORD_RE.search(row["text"]) and row["duration"] > 27.5
    ]

    write_jsonl(output / "recordings.jsonl", recordings)
    write_jsonl(output / "units.jsonl", all_units)
    write_jsonl(output / "chunks.jsonl", chunks)
    write_jsonl(output / "excluded_long_units.jsonl", excluded_long_units)
    write_jsonl(output / "sbcsae_v1.jsonl", chunks)
    write_jsonl(output / "sbcsae_clean_v1.jsonl", clean_chunks)
    split_document = {
        "version": "sbcsae_speaker_disjoint_v1",
        "seed": args.seed,
        "boundary_padding_seconds": args.boundary_padding,
        "chunk_root": args.chunk_root,
        "method": (
            "speaker-ID connected components assigned whole to splits by "
            "deterministic duration-balanced greedy allocation"
        ),
        "components": components,
        "recording_to_split": split_map,
    }
    (output / "splits.json").write_text(
        json.dumps(split_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {}
    for split in ("train", "validation", "test"):
        selected = [row for row in chunks if row["split"] == split]
        clean = [row for row in selected if row["quality_tier"] != "CHALLENGE"]
        summary[split] = {
            "recordings": sum(
                row["split"] == split for row in recordings
            ),
            "chunks": len(selected),
            "clean_chunks": len(clean),
            "hours": sum(row["duration"] for row in selected) / 3600,
            "clean_hours": sum(row["duration"] for row in clean) / 3600,
            "tokens": sum(row["token_count"] for row in selected),
            "clean_tokens": sum(row["token_count"] for row in clean),
        }
    summary["total_units"] = len(all_units)
    summary["excluded_long_lexical_units"] = len(excluded_long_units)
    summary["total_recordings"] = len(recordings)
    summary["speaker_components"] = len(components)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
