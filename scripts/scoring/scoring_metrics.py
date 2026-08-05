#!/usr/bin/env python3
"""Dependency-free aggregate metrics for saved ASR predictions."""

from __future__ import annotations

from scoring_normalization import (
    CLITIC_S_TOKEN,
    UNCLEAR_TOKEN,
    alignment_pairs,
    canonical_tokens,
    edit_counts,
    scoreable_units,
)

FILLERS = {"ah", "eh", "er", "erm", "huh", "mm", "uh", "uh-huh", "um"}
SMALL_FORMS = {
    "coulda", "dunno", "gonna", "gotta", "kinda", "shoulda", "sorta",
    "wanna", "woulda",
}


def tokens(text: str, conventional: bool = False) -> list[str]:
    return canonical_tokens(text, conventional=conventional)


def char_tokens(text: str) -> list[str]:
    result = []
    for token in tokens(text):
        if token == UNCLEAR_TOKEN:
            result.append(token)
        elif token == CLITIC_S_TOKEN:
            result.append("s")
        else:
            result.extend(token)
    return result


def event_prf(rows: list[dict], positive) -> dict:
    reference_total = hypothesis_total = true_positive = 0
    for row in rows:
        reference = tokens(row["reference"])
        hypothesis = tokens(row["hypothesis"])
        reference_positive = {
            index for index, token in enumerate(reference)
            if positive(reference, index, token)
        }
        hypothesis_positive = {
            index for index, token in enumerate(hypothesis)
            if positive(hypothesis, index, token)
        }
        reference_total += len(reference_positive)
        hypothesis_total += len(hypothesis_positive)
        for ref_index, hyp_index in alignment_pairs(reference, hypothesis):
            if (
                ref_index in reference_positive
                and hyp_index in hypothesis_positive
                and reference[ref_index] == hypothesis[hyp_index]
            ):
                true_positive += 1
    if not reference_total and not hypothesis_total:
        precision = recall = f1 = None
    else:
        precision = true_positive / hypothesis_total if hypothesis_total else 0.0
        recall = true_positive / reference_total if reference_total else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "reference": reference_total,
        "hypothesis": hypothesis_total,
        "true_positive": true_positive,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def aggregate(rows: list[dict]) -> dict:
    verbatim = [0, 0, 0, 0]
    conventional = [0, 0, 0, 0]
    characters = [0, 0, 0, 0]
    examples_with_unclear = unclear_reference_units = 0
    cer_excluded_unclear_examples = 0
    for row in rows:
        reference, hypothesis = row["reference"], row["hypothesis"]
        ref_tokens, hyp_tokens = tokens(reference), tokens(hypothesis)
        unclear_count = sum(token == UNCLEAR_TOKEN for token in ref_tokens)
        examples_with_unclear += bool(unclear_count)
        unclear_reference_units += unclear_count
        comparisons = [
            (verbatim, ref_tokens, hyp_tokens),
            (conventional, tokens(reference, True), tokens(hypothesis, True)),
        ]
        if unclear_count:
            cer_excluded_unclear_examples += 1
        else:
            comparisons.append((characters, char_tokens(reference), char_tokens(hypothesis)))
        for totals, ref, hyp in comparisons:
            substitutions, deletions, insertions = edit_counts(ref, hyp)
            totals[0] += scoreable_units(ref)
            totals[1] += substitutions
            totals[2] += deletions
            totals[3] += insertions

    def error_summary(values: list[int]) -> dict:
        reference, substitutions, deletions, insertions = values
        return {
            "reference_units": reference,
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
            "error_rate": (substitutions + deletions + insertions) / max(1, reference),
        }

    return {
        "examples": len(rows),
        "hours": sum(float(row.get("duration", row.get("duration_seconds", 20.0))) for row in rows) / 3600,
        "examples_with_unclear": examples_with_unclear,
        "unclear_reference_units_masked": unclear_reference_units,
        "cer_excluded_unclear_examples": cer_excluded_unclear_examples,
        "verbatim_wer": error_summary(verbatim),
        "disfluency_stripped_wer": error_summary(conventional),
        "cer": error_summary(characters),
        "fillers": event_prf(rows, lambda _items, _index, token: token in FILLERS),
        "small_reduced_forms": event_prf(rows, lambda _items, _index, token: token in SMALL_FORMS),
        "fragments": event_prf(rows, lambda _items, _index, token: token.endswith("-")),
        "adjacent_repetitions": event_prf(
            rows, lambda items, index, token: index > 0 and token == items[index - 1]
        ),
    }
