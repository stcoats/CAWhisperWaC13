#!/usr/bin/env python3
"""Parse and normalize the La Trobe CA-style reference transcripts.

The raw human transcript remains immutable. This script writes auditable
speaker turns with both the original and lexical-verbatim normalized text.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path


SPEAKERS = {
    "BethDaniel": ["Kerry", "Beth", "Daniel", "K", "B", "D"],
    "HeatherMarie": ["Kerry", "Heather", "Marie"],
    "LisaFiona": ["Kerry", "Lisa", "Fiona"],
    "MarkKylie": ["Kerry", "Mark", "Kylie"],
    "NatalieKen": ["Kerry", "Natalie", "Nat", "Ken"],
    "SuzanneLen": ["Kerry", "Suzanne", "Len", "K", "S", "L"],
}

SHORT_TO_FULL = {
    "BethDaniel": {"K": "Kerry", "B": "Beth", "D": "Daniel"},
    "HeatherMarie": {},
    "LisaFiona": {},
    "MarkKylie": {},
    "NatalieKen": {"Nat": "Natalie"},
    "SuzanneLen": {"K": "Kerry", "S": "Suzanne", "L": "Len"},
}

EVENT_WORDS = {
    "all",
    "breath",
    "groans",
    "intake",
    "laugh",
    "laughing",
    "laughter",
    "more laughing",
}

WORD_REPLACEMENTS = [
    ("uh_huh", re.compile(r"(?i)(?<!\w)uh[\s-]*huh(?!\w)"), "uh-huh"),
    ("mm_hmm", re.compile(r"(?i)(?<!\w)(?:m+h+m*|m+\s+h+m*|m+-h+m*)(?!\w)"), "mm-hmm"),
    ("mm_length", re.compile(r"(?i)(?<!\w)m{2,}(?!\w)"), "mm"),
    ("hmm_length", re.compile(r"(?i)(?<!\w)h+m{2,}(?!\w)"), "hmm"),
    ("um_variant", re.compile(r"(?i)(?<!\w)(?:u+m+|e+r+m+)(?!\w)"), "um"),
    ("uh_length", re.compile(r"(?i)(?<!\w)u+h+(?!\w)"), "uh"),
    ("ah_length", re.compile(r"(?i)(?<!\w)a+h+(?!\w)"), "ah"),
    ("oh_length", re.compile(r"(?i)(?<!\w)o+h+(?!\w)"), "oh"),
    ("yeah_length", re.compile(r"(?i)(?<!\w)ye+a+h+(?!\w)"), "yeah"),
    ("gunna", re.compile(r"(?i)(?<!\w)gunna(?!\w)"), "gonna"),
    ("okay", re.compile(r"(?i)(?<!\w)ok(?!\w)"), "okay"),
]


def compact_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def lexical_tokens(text: str) -> list[str]:
    """Tokens for counts; retains apostrophes and internal hyphens."""
    return re.findall(r"[A-Za-z0-9]+(?:['’-][A-Za-z0-9]+)*|[A-Za-z]+", text)


def match_token(token: str) -> str:
    token = token.lower().replace("’", "'").replace("‘", "'")
    token = token.replace("'", "").replace("-", "")
    token = re.sub(r"[^a-z0-9]", "", token)
    matching_equivalents = {
        "mmm": "mm",
        "mmmm": "mm",
        "mmhmm": "mm",
        "mhm": "mm",
        "hmmm": "hmm",
        "erm": "um",
        "errm": "um",
        "gunna": "gonna",
        "ok": "okay",
    }
    return matching_equivalents.get(token, token)


def match_tokens(text: str) -> list[str]:
    return [value for token in lexical_tokens(text) if (value := match_token(token))]


def build_speaker_regex(recording: str) -> re.Pattern[str]:
    atoms = sorted(SPEAKERS[recording], key=len, reverse=True)
    atom = "(?:" + "|".join(re.escape(value) for value in atoms) + ")"
    combined = rf"{atom}(?:\s*:?\s*/\s*:?\s*{atom})*\s*/?"
    # "All" labels are accepted only with punctuation to avoid treating a
    # continuation beginning "all ..." as a new speaker.
    return re.compile(
        rf"^\s*(?:(?P<all>all)\s*[:;]\s*|(?P<label>{combined})\s*[:;]?\s+)(?P<text>.*)$",
        re.IGNORECASE,
    )


def canonical_speaker(recording: str, value: str) -> tuple[str, list[str]]:
    value = re.sub(r"\s*:?\s*/\s*:?\s*", "/", value.strip().rstrip("/"))
    if value.lower() == "all":
        return "ALL", ["ALL"]
    lookup = {name.lower(): name for name in SPEAKERS[recording]}
    people = []
    for part in value.split("/"):
        canonical = lookup.get(part.lower(), part)
        canonical = SHORT_TO_FULL[recording].get(canonical, canonical)
        people.append(canonical)
    return "/".join(people), people


def parse_turns(recording: str, transcript: Path) -> tuple[list[dict], list[str]]:
    pattern = build_speaker_regex(recording)
    lines = transcript.read_text(encoding="utf-8-sig").replace("\r\n", "\n").splitlines()
    turns: list[dict] = []
    preamble: list[str] = []
    started = False
    current: dict | None = None

    for line_number, line in enumerate(lines, 1):
        match = pattern.match(line)
        # Require punctuation on the first speaker line so document titles such
        # as "Heather and Marie ..." cannot start the transcript.
        first_has_label_punctuation = bool(
            re.match(r"^\s*(?:[A-Za-z]+(?:\s*/\s*[A-Za-z]+)*)\s*[:;]", line)
        )
        if match and (started or first_has_label_punctuation):
            if current is not None:
                current["raw_text"] = compact_space(" ".join(current.pop("_parts")))
                turns.append(current)
            raw_label = "all" if match.group("all") else match.group("label")
            turn_text = match.group("text")
            # One source line is formatted "Fiona: Kerry: mm" instead of a
            # combined label. Treat a second known speaker at the very start as
            # part of the label, while retaining an audit trail in raw_speaker.
            atoms = "|".join(
                re.escape(value)
                for value in sorted(SPEAKERS[recording], key=len, reverse=True)
            )
            nested = re.match(rf"^\s*(?P<speaker>{atoms})\s*:\s*(?P<text>.*)$", turn_text)
            if nested:
                raw_label = f"{raw_label}/{nested.group('speaker')}"
                turn_text = nested.group("text")
            speaker, speakers = canonical_speaker(recording, raw_label)
            current = {
                "recording": recording,
                "turn_index": len(turns),
                "line_start": line_number,
                "line_end": line_number,
                "raw_speaker": raw_label,
                "speaker": speaker,
                "speakers": speakers,
                "_parts": [turn_text],
            }
            started = True
        elif started and current is not None:
            if line.strip():
                current["_parts"].append(line.strip())
                current["line_end"] = line_number
        elif line.strip():
            preamble.append(line.strip())

    if current is not None:
        current["raw_text"] = compact_space(" ".join(current.pop("_parts")))
        turns.append(current)

    for index, turn in enumerate(turns):
        turn["turn_index"] = index
    return turns, preamble


def normalize_parenthetical(content: str) -> tuple[str, str]:
    value = compact_space(content)
    lower = value.lower()
    if re.fullmatch(r"\d+(?:\.\d+)?(?:\s*(?:secs?|seconds?|minutes?))?!*", lower):
        return "", "timed_pause"
    if re.fullmatch(r"\?+", value):
        return "", "unknown_parenthetical"
    if "@" in value or lower in EVENT_WORDS or any(
        word in lower for word in [" laugh", "laughter", "groan", "breath", "intake"]
    ):
        return "", "event_parenthetical"
    if re.fullmatch(r"(?:all|[kbdsl](?:/[kbdsl])*)", lower):
        return "", "speaker_event_parenthetical"
    # Retain a lexical hypothesis for audit, but flag it. Candidate selection
    # can conservatively exclude turns with this flag.
    return value, "lexical_parenthetical"


def apply_rule(
    text: str,
    pattern: re.Pattern[str],
    replacement: str,
    rule: str,
    changes: Counter,
) -> str:
    def replace(match: re.Match[str]) -> str:
        if match.group(0) != replacement:
            changes[rule] += 1
        return replacement

    return pattern.sub(replace, text)


def normalize_turn(turn: dict) -> tuple[dict, Counter]:
    raw = unicodedata.normalize("NFC", turn["raw_text"])
    changes: Counter = Counter()
    flags: set[str] = set()

    overlap_contents = re.findall(r"\[([^\]]*)\]", raw)
    overlap_tokens = sum(len(match_tokens(value)) for value in overlap_contents)
    if "[" in raw or "]" in raw:
        flags.add("overlap")

    text = raw
    substitutions = [
        ("curly_apostrophe", re.compile("[‘’]"), "'"),
        ("curly_double_quote", re.compile("[“”]"), '"'),
        ("ellipsis", re.compile("…"), " "),
        ("dot_pause", re.compile(r"\.{2,}"), " "),
        ("overlap_bracket", re.compile(r"[\[\]]"), ""),
        ("latching", re.compile(r"="), " "),
        ("stray_brace", re.compile(r"[{}]"), " "),
    ]
    for rule, pattern, replacement in substitutions:
        text = apply_rule(text, pattern, replacement, rule, changes)

    parenthetical_rules: Counter = Counter()

    def replace_parenthetical(match: re.Match[str]) -> str:
        replacement, rule = normalize_parenthetical(match.group(1))
        parenthetical_rules[rule] += 1
        if rule == "lexical_parenthetical":
            flags.add("lexical_parenthetical")
        elif rule.startswith("unknown"):
            flags.add("unknown")
        elif "event" in rule:
            flags.add("non_speech_event")
        return f" {replacement} " if replacement else " "

    text = re.sub(r"\(([^)]*)\)", replace_parenthetical, text)
    changes.update(parenthetical_rules)

    if re.search(r"\?{2,}", text):
        flags.add("unknown")
    text = apply_rule(text, re.compile(r"\?{2,}"), " ", "unknown_marks", changes)

    if "@" in text:
        flags.add("non_speech_event")
    text = apply_rule(text, re.compile(r"@+"), " ", "laughter_at", changes)
    text = apply_rule(
        text,
        re.compile(r"\b(?:ALL\s+)?LAUGHTER\b"),
        " ",
        "written_laughter_event",
        changes,
    )

    for rule, pattern, replacement in WORD_REPLACEMENTS:
        text = apply_rule(text, pattern, replacement, rule, changes)

    text = compact_space(text)
    tokens = lexical_tokens(text)
    overlap_ratio = overlap_tokens / len(tokens) if tokens else 0.0
    if len(turn["speakers"]) > 1 or turn["speaker"] == "ALL":
        flags.add("multiple_speakers")

    out = dict(turn)
    out.update(
        {
            "normalized_text": text,
            "tokens": tokens,
            "match_tokens": [match_token(token) for token in tokens if match_token(token)],
            "token_count": len(tokens),
            "overlap_token_count": overlap_tokens,
            "overlap_ratio": round(overlap_ratio, 6),
            "flags": sorted(flags),
            "normalization_changes": dict(sorted(changes.items())),
        }
    )
    return out, changes


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = [
        "recording",
        "turn_index",
        "line_start",
        "line_end",
        "speaker",
        "token_count",
        "overlap_token_count",
        "overlap_ratio",
        "flags",
        "raw_text",
        "normalized_text",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            item = dict(row)
            item["flags"] = ",".join(row["flags"])
            writer.writerow(item)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()

    project = args.project.resolve()
    output = project / "data" / "normalized"
    reports = project / "reports"
    output.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    summary = {"recordings": {}, "normalization_changes": Counter()}
    all_changes = []

    for recording in SPEAKERS:
        source = project / "data" / "raw" / "transcripts" / f"Transcrp-{recording}-raw.txt"
        parsed, preamble = parse_turns(recording, source)
        normalized = []
        recording_changes: Counter = Counter()
        for turn in parsed:
            item, changes = normalize_turn(turn)
            normalized.append(item)
            recording_changes.update(changes)
            for rule, count in changes.items():
                all_changes.append(
                    {
                        "recording": recording,
                        "turn_index": turn["turn_index"],
                        "rule": rule,
                        "count": count,
                    }
                )

        write_jsonl(output / f"{recording}_turns.jsonl", normalized)
        write_tsv(output / f"{recording}_turns.tsv", normalized)
        summary["recordings"][recording] = {
            "turns": len(normalized),
            "nonempty_turns": sum(bool(row["normalized_text"]) for row in normalized),
            "tokens": sum(row["token_count"] for row in normalized),
            "overlap_tokens": sum(row["overlap_token_count"] for row in normalized),
            "turns_with_overlap": sum("overlap" in row["flags"] for row in normalized),
            "turns_with_unknown": sum("unknown" in row["flags"] for row in normalized),
            "turns_with_lexical_parenthetical": sum(
                "lexical_parenthetical" in row["flags"] for row in normalized
            ),
            "preamble": preamble,
            "normalization_changes": dict(sorted(recording_changes.items())),
        }
        summary["normalization_changes"].update(recording_changes)

    summary["normalization_changes"] = dict(sorted(summary["normalization_changes"].items()))
    (reports / "normalization_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (reports / "normalization_changes.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["recording", "turn_index", "rule", "count"],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(all_changes)

    print(json.dumps(summary["recordings"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
