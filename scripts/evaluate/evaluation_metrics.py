#!/usr/bin/env python3
"""Dependency-free lexical and conversational-ASR evaluation metrics."""

from __future__ import annotations

import re
import unicodedata


WORD = re.compile(
    r"[a-z0-9]+(?:['’][a-z0-9]+)*(?:-[a-z0-9]+(?:['’][a-z0-9]+)*)*-?",
    re.IGNORECASE,
)
FILLERS = {
    "ah", "eh", "er", "erm", "hm", "hmm", "hmmm", "mhm", "mm", "mmm",
    "mmmm", "mm-hm", "mm-hmm", "uh", "uh-huh", "uhh", "uhm", "um", "umm",
}
SMALL_REDUCED_FORMS = {
    "coulda", "dunno", "gonna", "gotta", "kinda", "shoulda", "sorta",
    "wanna", "woulda",
}


def tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return [match.group(0) for match in WORD.finditer(text)]


def alignment_pairs(
    reference: list[str], hypothesis: list[str]
) -> list[tuple[int | None, int | None]]:
    rows, columns = len(reference) + 1, len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]
    operation = [[""] * columns for _ in range(rows)]
    for row in range(1, rows):
        distance[row][0], operation[row][0] = row, "D"
    for column in range(1, columns):
        distance[0][column], operation[0][column] = column, "I"
    for row in range(1, rows):
        for column in range(1, columns):
            choices = [
                (
                    distance[row - 1][column - 1]
                    + (reference[row - 1] != hypothesis[column - 1]),
                    "M"
                    if reference[row - 1] == hypothesis[column - 1]
                    else "S",
                ),
                (distance[row - 1][column] + 1, "D"),
                (distance[row][column - 1] + 1, "I"),
            ]
            distance[row][column], operation[row][column] = min(
                choices, key=lambda item: (item[0], "MSDI".index(item[1]))
            )
    pairs = []
    row, column = len(reference), len(hypothesis)
    while row or column:
        op = operation[row][column]
        if op in {"M", "S"}:
            pairs.append((row - 1, column - 1))
            row, column = row - 1, column - 1
        elif op == "D":
            pairs.append((row - 1, None))
            row -= 1
        elif op == "I":
            pairs.append((None, column - 1))
            column -= 1
        else:
            raise RuntimeError(f"invalid alignment backtrace at {row}, {column}")
    return list(reversed(pairs))


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    substitutions = deletions = insertions = 0
    for ref_index, hyp_index in alignment_pairs(reference, hypothesis):
        if ref_index is None:
            insertions += 1
        elif hyp_index is None:
            deletions += 1
        elif reference[ref_index] != hypothesis[hyp_index]:
            substitutions += 1
    return substitutions, deletions, insertions


def event_positions(items: list[str], kind: str) -> set[int]:
    if kind == "fillers":
        return {index for index, item in enumerate(items) if item in FILLERS}
    if kind == "fragments":
        return {index for index, item in enumerate(items) if item.endswith("-")}
    if kind == "repetitions":
        return {
            index
            for index in range(1, len(items))
            if items[index] == items[index - 1]
        }
    if kind == "small_reduced_forms":
        return {
            index
            for index, item in enumerate(items)
            if item in SMALL_REDUCED_FORMS
        }
    raise ValueError(kind)


def score(references: list[str], hypotheses: list[str]) -> dict:
    if len(references) != len(hypotheses):
        raise ValueError("reference and hypothesis counts differ")
    substitutions = deletions = insertions = words = 0
    event = {
        kind: {"tp": 0, "fp": 0, "fn": 0}
        for kind in (
            "fillers",
            "repetitions",
            "fragments",
            "small_reduced_forms",
        )
    }
    for reference, hypothesis in zip(references, hypotheses):
        ref = tokens(reference)
        hyp = tokens(hypothesis)
        pairs = alignment_pairs(ref, hyp)
        sub = sum(
            ref_index is not None
            and hyp_index is not None
            and ref[ref_index] != hyp[hyp_index]
            for ref_index, hyp_index in pairs
        )
        delete = sum(hyp_index is None for _, hyp_index in pairs)
        insert = sum(ref_index is None for ref_index, _ in pairs)
        substitutions += sub
        deletions += delete
        insertions += insert
        words += len(ref)
        for kind in event:
            ref_events = event_positions(ref, kind)
            hyp_events = event_positions(hyp, kind)
            true = sum(
                1
                for ref_index, hyp_index in pairs
                if ref_index in ref_events
                and hyp_index in hyp_events
                and ref[ref_index] == hyp[hyp_index]
            )
            event[kind]["tp"] += true
            event[kind]["fn"] += len(ref_events) - true
            event[kind]["fp"] += len(hyp_events) - true
    result = {
        "wer": (substitutions + deletions + insertions) / max(1, words),
        "words": words,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
    }
    for kind, counts in event.items():
        precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
        recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
        result[f"{kind}_precision"] = precision
        result[f"{kind}_recall"] = recall
        result[f"{kind}_f1"] = (
            2 * precision * recall / max(1e-12, precision + recall)
        )
        result[f"{kind}_reference_events"] = counts["tp"] + counts["fn"]
        result[f"{kind}_hypothesis_events"] = counts["tp"] + counts["fp"]
    return result
