#!/usr/bin/env python3
"""Create paper-facing descriptive statistics and paired cluster-bootstrap CIs."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCORING_DIR = REPOSITORY_ROOT / "scripts" / "scoring"
sys.path.insert(0, str(SCORING_DIR))

from scoring_metrics import FILLERS, aggregate  # noqa: E402
from scoring_normalization import (  # noqa: E402
    alignment_pairs,
    canonical_tokens,
    edit_counts,
    scoreable_units,
)


ROOT = REPOSITORY_ROOT
SOURCE = ROOT / "results"
MANIFEST = SOURCE / "coanzse_gold_v1" / "manifest.jsonl"
PREDICTIONS = {
    "Whisper Large-v3": SOURCE / "predictions" / "base_large_v3.jsonl",
    "Dual-policy LoRA": SOURCE / "predictions" / "dual_policy_large_v3_lora.jsonl",
    "Verbatim full FT": SOURCE / "predictions" / "verbatim_only_large_v3_full.jsonl",
}
SEED = 20260803
REPLICATES = 20_000


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def error_counts(row: dict, conventional: bool) -> tuple[int, int]:
    reference = canonical_tokens(row["reference"], conventional=conventional)
    hypothesis = canonical_tokens(row["hypothesis"], conventional=conventional)
    substitutions, deletions, insertions = edit_counts(reference, hypothesis)
    return scoreable_units(reference), substitutions + deletions + insertions


def filler_counts(row: dict) -> tuple[int, int, int]:
    reference = canonical_tokens(row["reference"])
    hypothesis = canonical_tokens(row["hypothesis"])
    reference_positive = {i for i, token in enumerate(reference) if token in FILLERS}
    hypothesis_positive = {i for i, token in enumerate(hypothesis) if token in FILLERS}
    true_positive = sum(
        ref_index in reference_positive
        and hyp_index in hypothesis_positive
        and reference[ref_index] == hypothesis[hyp_index]
        for ref_index, hyp_index in alignment_pairs(reference, hypothesis)
        if ref_index is not None and hyp_index is not None
    )
    return len(reference_positive), len(hypothesis_positive), true_positive


def f1_from_counts(reference: int, hypothesis: int, true_positive: int) -> float | None:
    if not reference and not hypothesis:
        return None
    precision = true_positive / hypothesis if hypothesis else 0.0
    recall = true_positive / reference if reference else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def cluster_bootstrap(rows_by_model: dict[str, list[dict]]) -> dict:
    ids_by_recording: dict[str, list[str]] = defaultdict(list)
    first_rows = next(iter(rows_by_model.values()))
    for row in first_rows:
        ids_by_recording[row["video_id"]].append(row["blind_id"])
    clusters = sorted(ids_by_recording)
    keyed = {
        model: {row["blind_id"]: row for row in rows}
        for model, rows in rows_by_model.items()
    }
    precomputed = {}
    for model, rows in keyed.items():
        precomputed[model] = {}
        for blind_id, row in rows.items():
            precomputed[model][blind_id] = {
                "verbatim": error_counts(row, False),
                "normalized": error_counts(row, True),
                "fillers": filler_counts(row),
            }

    rng = random.Random(SEED)
    distributions = {
        model: {"verbatim": [], "normalized": [], "filler_f1": []}
        for model in rows_by_model
    }
    for _ in range(REPLICATES):
        sampled_clusters = rng.choices(clusters, k=len(clusters))
        sampled_ids = [blind_id for cluster in sampled_clusters for blind_id in ids_by_recording[cluster]]
        for model in rows_by_model:
            for metric in ("verbatim", "normalized"):
                denominator = errors = 0
                for blind_id in sampled_ids:
                    ref, err = precomputed[model][blind_id][metric]
                    denominator += ref
                    errors += err
                distributions[model][metric].append(errors / denominator)
            filler_reference = filler_hypothesis = filler_true_positive = 0
            for blind_id in sampled_ids:
                reference, hypothesis, true_positive = precomputed[model][blind_id]["fillers"]
                filler_reference += reference
                filler_hypothesis += hypothesis
                filler_true_positive += true_positive
            distributions[model]["filler_f1"].append(
                f1_from_counts(filler_reference, filler_hypothesis, filler_true_positive)
            )

    point = {model: aggregate(rows) for model, rows in rows_by_model.items()}
    result = {"replicates": REPLICATES, "seed": SEED, "cluster": "video_id", "systems": {}}
    for model in rows_by_model:
        result["systems"][model] = {}
        for metric, key in (("verbatim", "verbatim_wer"), ("normalized", "disfluency_stripped_wer")):
            values = distributions[model][metric]
            result["systems"][model][metric] = {
                "estimate": point[model][key]["error_rate"],
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
            }
        values = distributions[model]["filler_f1"]
        result["systems"][model]["filler_f1"] = {
            "estimate": point[model]["fillers"]["f1"],
            "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
        }

    baseline = "Whisper Large-v3"
    result["paired_differences_vs_base"] = {}
    for model in rows_by_model:
        if model == baseline:
            continue
        result["paired_differences_vs_base"][model] = {}
        for metric in ("verbatim", "normalized", "filler_f1"):
            # Negative WER differences and positive F1 differences favor the adapted model.
            values = [
                candidate - base
                for candidate, base in zip(distributions[model][metric], distributions[baseline][metric])
            ]
            result["paired_differences_vs_base"][model][metric] = {
                "estimate": (
                    result["systems"][model][metric]["estimate"]
                    - result["systems"][baseline][metric]["estimate"]
                ),
                "ci95": [percentile(values, 0.025), percentile(values, 0.975)],
            }
    return result


def main() -> None:
    manifest = read_jsonl(MANIFEST)
    predictions = {model: read_jsonl(path) for model, path in PREDICTIONS.items()}
    all_ids = {row["blind_id"] for row in manifest}
    for model, rows in predictions.items():
        assert {row["blind_id"] for row in rows} == all_ids, model

    subsets = {
        "all_100": all_ids,
        "analysis_57": {row["blind_id"] for row in manifest if row["analysis_eligible"]},
    }
    output = {
        "manifest": {},
        "bootstrap": {},
    }
    for name, ids in subsets.items():
        rows = [row for row in manifest if row["blind_id"] in ids]
        output["manifest"][name] = {
            "clips": len(rows),
            "hours": sum(float(row["duration_seconds"]) for row in rows) / 3600,
            "reference_whitespace_words": sum(len(row["reference"].split()) for row in rows),
            "reference_scored_words": sum(
                scoreable_units(canonical_tokens(row["reference"])) for row in rows
            ),
            "videos": len({row["video_id"] for row in rows}),
            "councils": len({(row["country"], row["council_name"]) for row in rows}),
            "country": dict(Counter(row["country"] for row in rows)),
            "state": dict(Counter(row["state"] for row in rows)),
            "stratum": dict(Counter(row["stratum"] for row in rows)),
            "audio_quality": dict(Counter(row["audio_quality"] for row in rows)),
            "boundary_status": dict(Counter(row["boundary_status"] for row in rows)),
            "overlap": dict(Counter(row["overlap"] for row in rows)),
        }
        rows_by_model = {
            model: [row for row in model_rows if row["blind_id"] in ids]
            for model, model_rows in predictions.items()
        }
        output["bootstrap"][name] = cluster_bootstrap(rows_by_model)

    # Rank examples by paired per-clip vWER advantage over each comparison model.
    keyed = {
        model: {row["blind_id"]: row for row in rows}
        for model, rows in predictions.items()
    }
    examples = []
    for blind_id in sorted(subsets["analysis_57"]):
        item = {"blind_id": blind_id}
        for model in PREDICTIONS:
            denominator, errors = error_counts(keyed[model][blind_id], False)
            item[model] = errors / max(1, denominator)
        item["full_advantage_vs_base"] = item["Whisper Large-v3"] - item["Verbatim full FT"]
        if "CrisperWhisper2" in item:
            item["full_advantage_vs_crisper"] = (
                item["CrisperWhisper2"] - item["Verbatim full FT"]
            )
        item["reference"] = keyed["Whisper Large-v3"][blind_id]["reference"]
        item["hypotheses"] = {
            model: keyed[model][blind_id]["hypothesis"] for model in PREDICTIONS
        }
        examples.append(item)
    output["top_examples_full_vs_base"] = sorted(
        examples, key=lambda item: item["full_advantage_vs_base"], reverse=True
    )[:10]
    if "CrisperWhisper2" in PREDICTIONS:
        output["top_examples_full_vs_crisper"] = sorted(
            examples, key=lambda item: item["full_advantage_vs_crisper"], reverse=True
        )[:10]

    destination = ROOT / "analysis_statistics.json"
    destination.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
