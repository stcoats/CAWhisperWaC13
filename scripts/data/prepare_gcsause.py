#!/usr/bin/env python3
"""Prepare exact-boundary GCSAusE manifests from TalkBank CHAT-CA files."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


TIME_RE = re.compile(r"\x15(\d+)_(\d+)\x15")
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*(?:-[A-Za-z0-9]+)*-?")
PAUSE_RE = re.compile(r"\((?:\.{1,3}|\d+(?:\.\d+)?)\)")
UNINTELLIGIBLE_RE = re.compile(r"(?<!\w)(?:x{2,}|y{2,}|www)(?!\w)", re.I)
OVERLAP_MARKS = "⌈⌉⌊⌋"
CA_PROSODY_MARKS = "∆∇°º⇗⇘↑↓→↗↘≈≋≡≠∮◉⁎"
EXPLICIT_OTHER_L1 = {"ell", "fra", "hel", "heb", "jpn", "slk", "yue", "zho"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def replace_count(text: str, pattern: re.Pattern, replacement: str, changes: Counter, name: str) -> str:
    def replace(match: re.Match) -> str:
        if match.group(0) != replacement:
            changes[name] += 1
        return replacement

    return pattern.sub(replace, text)


def normalize_chat_text(raw: str) -> tuple[str, list[str], dict[str, int]]:
    """Remove CA notation while retaining lexical-verbatim speech and events."""
    text = unicodedata.normalize("NFKC", raw).replace("’", "'").replace("‘", "'")
    flags: set[str] = set()
    changes: Counter = Counter()

    if UNINTELLIGIBLE_RE.search(text) or "⁇" in text:
        flags.add("unintelligible")
    if any(mark in text for mark in OVERLAP_MARKS):
        flags.add("overlap_markup")

    # Inbreaths are nonlexical. Parenthesized h particles and h/ha/he
    # sequences are CA laughter notation and become one auditable event token.
    text = replace_count(
        text,
        re.compile(r"(?:\.|∙)h+", re.I),
        " ",
        changes,
        "breath_removed",
    )
    text, pure_laughter = re.subn(
        r"(?<![A-Za-z])[Hh]*\(h+\)[Hh]*(?![A-Za-z])",
        " [laughter] ",
        text,
    )
    if pure_laughter:
        flags.add("laughter")
        changes["standalone_laughter"] += pure_laughter
    laughter_particle = bool(re.search(r"\(h+\)", text, flags=re.I))
    if laughter_particle:
        flags.add("laughter")
        changes["laughter_particle"] += len(re.findall(r"\(h+\)", text, flags=re.I))
        text = re.sub(r"\(h+\)", "", text, flags=re.I)
    text, standalone_laughter = re.subn(
        r"(?<![A-Za-z])(?:[Hh]{2,}(?:\s+[Hh]{2,})*|[HhAaEe]{4,})(?![A-Za-z])",
        " [laughter] ",
        text,
    )
    if standalone_laughter:
        flags.add("laughter")
        changes["standalone_laughter"] += standalone_laughter
    if laughter_particle:
        text += " [laughter]"

    text = replace_count(text, PAUSE_RE, " ", changes, "pause_removed")

    def uncertain_parenthetical(match: re.Match) -> str:
        value = match.group(1).strip()
        if not value or value == "?":
            flags.add("uncertain")
            changes["uncertain_empty_removed"] += 1
            return " "
        flags.add("uncertain_lexical")
        changes["uncertain_lexical_retained"] += 1
        return f" {value} "

    text = re.sub(r"\(([^()]*)\)", uncertain_parenthetical, text)
    text = replace_count(text, UNINTELLIGIBLE_RE, " ", changes, "unintelligible_removed")

    text = text.translate(str.maketrans("", "", OVERLAP_MARKS + "∆∇°º⇗⇘↑↓→↗↘∮◉⁎∙"))
    text = text.translate(str.maketrans({character: " " for character in "≈≋≡≠"}))
    text = text.replace("“", '"').replace("”", '"').replace("\u030a", "")
    text = text.replace("ʔ", "")
    text = re.sub(r"[+<>=|]", " ", text)
    text = re.sub(r":+", "", text)
    text = text.replace("⁇", " ")

    # Canonicalize only explicitly documented spelling/filler variants. Do not
    # standardize ordinary colloquial grammar or pronunciation spellings.
    replacements = [
        (r"(?<!\w)(?:u+h*m+|a+h+m+|e+r+m+)(?!\w)", "um", "um_variant"),
        (r"(?<!\w)m+h+m+(?!\w)", "mm-hmm", "mm_hmm_variant"),
        (r"(?<!\w)m{2,}(?!\w)", "mm", "mm_length"),
        (r"(?<!\w)h+m{2,}(?!\w)", "hmm", "hmm_length"),
        (r"(?<!\w)u+h+(?!\w)", "uh", "uh_length"),
        (r"(?<!\w)a+h+(?!\w)", "ah", "ah_length"),
        (r"(?<!\w)o+h+(?!\w)", "oh", "oh_length"),
        (r"(?<!\w)ye+a+h+(?!\w)", "yeah", "yeah_length"),
        (r"(?<!\w)gunna(?!\w)", "gonna", "gunna"),
        (r"(?<!\w)ok(?!\w)", "okay", "okay"),
    ]
    for pattern, replacement, name in replacements:
        text = replace_count(text, re.compile(pattern, re.I), replacement, changes, name)

    text = re.sub(r"(?:\[laughter\]\s*){2,}", "[laughter] ", text, flags=re.I)
    text = re.sub(r"\s+([,.?!])", r"\1", text)
    text = re.sub(r"([,.?!]){2,}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" \t,")
    return text, sorted(flags), dict(sorted(changes.items()))


def parse_ids(lines: list[str]) -> list[dict]:
    participants = []
    for line in lines:
        if not line.startswith("@ID:"):
            continue
        fields = line.split(":", 1)[1].strip().split("|")
        fields += [""] * (11 - len(fields))
        participants.append(
            {
                "code": fields[2],
                "age": fields[3],
                "sex": fields[4],
                "role": fields[7],
                "reported_other_l1": fields[9].lower(),
            }
        )
    return participants


def parse_chat(path: Path) -> tuple[dict, list[dict]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    recording = path.stem
    participants = parse_ids(lines)
    other_l1 = sorted(
        {
            row["reported_other_l1"]
            for row in participants
            if row["reported_other_l1"] in EXPLICIT_OTHER_L1
        }
    )
    units: list[dict] = []
    current_speaker: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        nonlocal current_parts
        if current_speaker is None:
            current_parts = []
            return
        content = "\n".join(current_parts)
        cursor = 0
        for match in TIME_RE.finditer(content):
            raw = content[cursor : match.start()].strip()
            start_ms, end_ms = int(match.group(1)), int(match.group(2))
            text, flags, changes = normalize_chat_text(raw)
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
                        "normalization_changes": changes,
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
        elif current_speaker is not None and (line.startswith("\t") or line.startswith(" ")):
            current_parts.append(line.strip())
        elif line.startswith("%") or line.startswith("@"):
            flush()
            current_speaker = None
    flush()
    units.sort(key=lambda row: (row["start"], row["end"], row["speaker"]))
    metadata = {
        "recording": recording,
        "participants": participants,
        "explicit_other_l1": bool(other_l1),
        "reported_other_l1_codes": other_l1,
        "duration": max((row["end"] for row in units), default=0.0),
        "units": len(units),
    }
    return metadata, units


def assign_splits(recordings: list[dict], seed: str) -> tuple[dict[str, str], list[dict]]:
    targets = {"train": 0.70, "validation": 0.10, "test": 0.20}
    total = sum(float(row["duration"]) for row in recordings)
    components = [
        {
            "recording": row["recording"],
            "duration": float(row["duration"]),
            "tie_break": hashlib.sha256(f"{seed}:{row['recording']}".encode()).hexdigest(),
        }
        for row in recordings
    ]
    components.sort(key=lambda row: (-row["duration"], row["tie_break"]))
    used = {split: 0.0 for split in targets}
    split_map: dict[str, str] = {}
    for component in components:
        split = min(
            targets,
            key=lambda name: (
                used[name] / max(1.0, total * targets[name]),
                hashlib.sha256(f"{component['tie_break']}:{name}".encode()).hexdigest(),
            ),
        )
        used[split] += component["duration"]
        component["split"] = split
        split_map[component["recording"]] = split
    return split_map, components


def overlap_seconds(units: list[dict]) -> float:
    points = sorted({float(unit[key]) for unit in units for key in ("start", "end")})
    total = 0.0
    for left, right in zip(points, points[1:]):
        middle = (left + right) / 2
        speakers = {
            unit["speaker"]
            for unit in units
            if float(unit["start"]) < middle < float(unit["end"])
        }
        if len(speakers) > 1:
            total += right - left
    return total


def chunks_for_recording(recording: dict, units: list[dict], max_span: float = 27.5, max_gap: float = 3.0) -> list[dict]:
    lexical = [
        unit
        for unit in units
        if WORD_RE.search(unit["text"])
        and float(unit["duration"]) <= max_span
        and unit["speaker"] not in {"ENV", "NOISE"}
    ]
    groups: list[list[dict]] = []
    current: list[dict] = []
    for unit in lexical:
        if current and (
            float(unit["end"]) - float(current[0]["start"]) > max_span
            or float(unit["start"]) - float(current[-1]["end"]) > max_gap
        ):
            groups.append(current)
            current = []
        current.append(unit)
    if current:
        groups.append(current)

    chunks = []
    for index, group in enumerate(groups):
        start, end = float(group[0]["start"]), float(group[-1]["end"])
        duration = end - start
        overlap = overlap_seconds(group)
        flags = sorted({flag for unit in group for flag in unit["flags"]})
        if overlap > 0 and "temporal_overlap" not in flags:
            flags.append("temporal_overlap")
            flags.sort()
        text = " ".join(unit["text"] for unit in group).strip()
        speakers = sorted({unit["speaker"] for unit in group})
        challenge = bool(
            overlap > 0
            or "unintelligible" in flags
            or "uncertain" in flags
        )
        chunks.append(
            {
                "example_id": f"gcsause_{recording['recording']}_{index:04d}",
                "recording": f"GCSAusE{recording['recording']}",
                "source_recording": recording["recording"],
                "split": recording["split"],
                "audio": (
                    "data/GCSAusE/chunks_exact_v1/"
                    f"{recording['recording']}/{recording['recording']}_{index:04d}.wav"
                ),
                "source_audio": f"data/GCSAusE/raw/media/{recording['recording']}.mp3",
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
                "quality_tier": "CHALLENGE" if challenge else "A",
                "explicit_other_l1": recording["explicit_other_l1"],
                "reported_other_l1_codes": recording["reported_other_l1_codes"],
                "boundary_method": "source_chat_first_unit_start_to_last_unit_end",
                "has_overlap": overlap > 0,
                "corpus": "gcsause",
            }
        )
    return chunks


def summarize(rows: list[dict]) -> dict:
    return {
        "examples": len(rows),
        "hours": round(sum(float(row["duration"]) for row in rows) / 3600, 6),
        "tokens": sum(int(row.get("token_count", 0)) for row in rows),
        "recordings": len({row["recording"] for row in rows}),
        "by_split": dict(sorted(Counter(row["split"] for row in rows).items())),
    }


def run_self_test() -> None:
    cases = {
        "ah:m (.) ∆what was I goin' to tell you∆": "um what was I goin' to tell you",
        "commu:nicating acro-": "communicating acro-",
        "⌊hm:: (.) if ʔI couldn't boʔrrow": "hm if I couldn't borrow",
        "peo(h)ple": "people [laughter]",
        ".hhhh yeah HHHhh": "yeah [laughter]",
        "Ummm:::: gunna mmmm": "um gonna mm",
    }
    failures = []
    for source, expected in cases.items():
        actual, _, _ = normalize_chat_text(source)
        if actual != expected:
            failures.append({"source": source, "expected": expected, "actual": actual})
    if failures:
        raise AssertionError(f"normalizer self-test failed: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--transcripts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default="gcsause-three-source-v1")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
    project = args.project.resolve()
    transcripts = args.transcripts.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    recordings: list[dict] = []
    units_by_recording: dict[str, list[dict]] = {}
    all_units: list[dict] = []
    for path in sorted(transcripts.glob("*.cha")):
        metadata, units = parse_chat(path)
        recordings.append(metadata)
        units_by_recording[metadata["recording"]] = units
        all_units.extend(units)
    if len(recordings) != 40:
        raise ValueError(f"expected 40 transcripts, found {len(recordings)}")

    core = [row for row in recordings if not row["explicit_other_l1"]]
    supplemental = [row for row in recordings if row["explicit_other_l1"]]
    core_map, core_components = assign_splits(core, args.seed + ":core")
    supplemental_map, supplemental_components = assign_splits(
        supplemental, args.seed + ":explicit-other-l1"
    )
    split_map = {**core_map, **supplemental_map}
    for row in recordings:
        row["split"] = split_map[row["recording"]]

    chunks = [
        chunk
        for recording in recordings
        for chunk in chunks_for_recording(
            recording, units_by_recording[recording["recording"]]
        )
    ]
    nonoverlap_all_users = [row for row in chunks if row["quality_tier"] == "A"]
    nonoverlap_core = [row for row in nonoverlap_all_users if not row["explicit_other_l1"]]
    challenge = [row for row in chunks if row["quality_tier"] == "CHALLENGE"]

    write_jsonl(output / "recordings.jsonl", recordings)
    write_jsonl(output / "units.jsonl", all_units)
    write_jsonl(output / "chunks_all.jsonl", chunks)
    write_jsonl(output / "chunks_nonoverlap_all_users.jsonl", nonoverlap_all_users)
    write_jsonl(output / "chunks_nonoverlap_core.jsonl", nonoverlap_core)
    write_jsonl(output / "chunks_challenge.jsonl", challenge)
    write_jsonl(project / "manifests/gcsause_exact_v1_nonoverlap_all_users.jsonl", nonoverlap_all_users)
    write_jsonl(project / "manifests/gcsause_exact_v1_nonoverlap_core.jsonl", nonoverlap_core)

    split_document = {
        "version": "gcsause_recording_disjoint_v1",
        "seed": args.seed,
        "method": (
            "recording-disjoint deterministic duration-balanced allocation; "
            "core and explicitly reported other-L1 recordings allocated separately"
        ),
        "recording_to_split": split_map,
        "core_components": core_components,
        "explicit_other_l1_components": supplemental_components,
    }
    (output / "splits.json").write_text(
        json.dumps(split_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    change_counts: Counter = Counter()
    for unit in all_units:
        change_counts.update(unit["normalization_changes"])
    summary = {
        "version": "gcsause_exact_v1",
        "boundary_policy": "first source CHAT unit start through last source CHAT unit end; no padding",
        "primary_policy": (
            "exclude all chunks with temporal overlap, unintelligible material, "
            "or empty uncertainty; retain the corpus's Australian users of English, "
            "with an additional core-only sensitivity manifest excluding explicitly reported other L1s"
        ),
        "recordings_total": len(recordings),
        "recordings_core": len(core),
        "recordings_explicit_other_l1": len(supplemental),
        "units": len(all_units),
        "all_chunks": summarize(chunks),
        "nonoverlap_all_users": summarize(nonoverlap_all_users),
        "nonoverlap_core": summarize(nonoverlap_core),
        "challenge": summarize(challenge),
        "normalization_changes": dict(sorted(change_counts.items())),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
