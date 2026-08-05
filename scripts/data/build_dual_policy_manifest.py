#!/usr/bin/env python3
"""Create paired verbatim/intended examples from an existing manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


WORD = re.compile(
    r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*(?:-[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)*)*-?"
)
BRACKETED_EVENT = re.compile(
    r"\[(?:laugh(?:ter|s|ing)?|noise|cough(?:ing)?|breath(?:ing)?|"
    r"sneeze|sigh|music|inaudible|vocali[sz]ation)[^\]]*\]",
    re.IGNORECASE,
)
FILLERS = {
    "ah",
    "eh",
    "er",
    "erm",
    "hm",
    "hmm",
    "hmmm",
    "mhm",
    "mm",
    "mmm",
    "mmmm",
    "mm-hm",
    "mm-hmm",
    "uh",
    "uh-huh",
    "uhh",
    "uhm",
    "um",
    "umm",
}
POLICY_TOKENS = {
    "verbatim": "<|ovw_verbatim|>",
    "intended": "<|ovw_intended|>",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical(token: str) -> str:
    return unicodedata.normalize("NFKC", token).replace("’", "'")


def collapse_adjacent_repetitions(
    tokens: list[str], operations: Counter
) -> list[str]:
    result = list(tokens)
    changed = True
    while changed:
        changed = False
        index = 0
        while index < len(result):
            collapsed = False
            for width in range(min(4, (len(result) - index) // 2), 0, -1):
                left = [item.lower() for item in result[index : index + width]]
                right = [
                    item.lower()
                    for item in result[index + width : index + 2 * width]
                ]
                if left == right:
                    del result[index + width : index + 2 * width]
                    operations[f"repeat_span_{width}"] += 1
                    changed = collapsed = True
                    break
            if not collapsed:
                index += 1
    return result


def intended_text(text: str) -> tuple[str, dict[str, int]]:
    operations: Counter = Counter()
    normalized = unicodedata.normalize("NFKC", text).replace("’", "'")
    normalized, event_count = BRACKETED_EVENT.subn(" ", normalized)
    operations["bracketed_event"] += event_count
    source_tokens = [canonical(match.group(0)) for match in WORD.finditer(normalized)]

    without_fillers: list[str] = []
    for token in source_tokens:
        lowered = token.lower()
        if lowered in FILLERS:
            operations["filler"] += 1
        elif token.endswith("-"):
            operations["marked_fragment"] += 1
        else:
            without_fillers.append(token)

    without_repairs: list[str] = []
    for index, token in enumerate(without_fillers):
        lowered = token.lower()
        if len(lowered) == 1 and lowered not in {"a", "i"}:
            lookahead = without_fillers[index + 1 : index + 3]
            if any(item.lower().startswith(lowered) for item in lookahead):
                operations["single_letter_repair"] += 1
                continue
        without_repairs.append(token)

    collapsed = collapse_adjacent_repetitions(without_repairs, operations)
    intended = " ".join(collapsed).strip()
    return intended, {key: value for key, value in operations.items() if value}


def run_self_test() -> None:
    cases = {
        "um I I think th- Thursday": "I think Thursday",
        "we need to to reschedule": "we need to reschedule",
        "[laughter] yeah mm-hmm": "yeah",
        "I g I guess it is": "I guess it is",
        "work work work but sports": "work but sports",
        "a long-term plan": "a long-term plan",
    }
    failures = []
    for source, expected in cases.items():
        actual, _ = intended_text(source)
        if actual != expected:
            failures.append((source, expected, actual))
    if failures:
        raise AssertionError(f"normalizer self-test failures: {failures}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check-audio", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_test()

    source_manifest = args.source_manifest.resolve()
    source_project = args.source_project.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_rows = read_jsonl(source_manifest)
    if not source_rows:
        raise ValueError("source manifest is empty")

    paired_rows: list[dict] = []
    operation_totals: Counter = Counter()
    changed_examples = 0
    empty_intended = []
    missing_audio = []
    preview_rows = []

    for source in source_rows:
        source_id = source["example_id"]
        audio = Path(source["audio"])
        if not audio.is_absolute():
            audio = source_project / audio
        audio = audio.resolve()
        if args.check_audio and not audio.is_file():
            missing_audio.append(str(audio))

        verbatim = source["text"].strip()
        intended, operations = intended_text(verbatim)
        if not intended:
            empty_intended.append(source_id)
            intended = verbatim
            operations = {**operations, "empty_fallback_to_verbatim": 1}
        if intended != verbatim:
            changed_examples += 1
            if len(preview_rows) < 100:
                preview_rows.append(
                    {
                        "example_id": source_id,
                        "verbatim": verbatim,
                        "intended": intended,
                        "operations": operations,
                    }
                )
        operation_totals.update(operations)

        common = {
            **source,
            "audio": str(audio),
            "source_example_id": source_id,
            "source_manifest": str(source_manifest),
            "verbatim_text": verbatim,
            "intended_text": intended,
            "normalization_operations": operations,
        }
        for policy in ("verbatim", "intended"):
            target = verbatim if policy == "verbatim" else intended
            paired_rows.append(
                {
                    **common,
                    "example_id": f"{source_id}__{policy}",
                    "pair_id": source_id,
                    "policy": policy,
                    "policy_token": POLICY_TOKENS[policy],
                    "text": target,
                }
            )

    if missing_audio:
        raise FileNotFoundError(
            f"{len(missing_audio)} source audio files are missing; "
            f"first: {missing_audio[0]}"
        )

    output_manifest = output_dir / "dual_policy_exact_v1.jsonl"
    write_jsonl(output_manifest, paired_rows)
    write_jsonl(output_dir / "normalization_preview.jsonl", preview_rows)

    grouped = defaultdict(lambda: {"examples": 0, "hours": 0.0})
    for row in paired_rows:
        key = f"{row['split']}::{row.get('corpus', 'unknown')}::{row['policy']}"
        grouped[key]["examples"] += 1
        grouped[key]["hours"] += float(row.get("duration", 0.0)) / 3600

    source_splits = defaultdict(set)
    for row in source_rows:
        source_splits[row["split"]].add(row["example_id"])
    if any(
        source_splits[left] & source_splits[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("source example IDs overlap across splits")

    summary = {
        "version": "dual_policy_exact_v1",
        "source_manifest": str(source_manifest),
        "source_manifest_sha256": sha256(source_manifest),
        "output_manifest": str(output_manifest),
        "output_manifest_sha256": sha256(output_manifest),
        "source_examples": len(source_rows),
        "paired_examples": len(paired_rows),
        "changed_intended_examples": changed_examples,
        "unchanged_intended_examples": len(source_rows) - changed_examples,
        "empty_intended_fallbacks": empty_intended,
        "normalization_operation_totals": dict(operation_totals),
        "groups": dict(sorted(grouped.items())),
        "policy_tokens": POLICY_TOKENS,
        "split_integrity": "source example IDs are disjoint",
    }
    (output_dir / "manifest_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
