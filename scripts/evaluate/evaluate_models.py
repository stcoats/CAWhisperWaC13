#!/usr/bin/env python3
"""Evaluate base, existing verbatim-only, and dual-policy Whisper models."""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from evaluation_metrics import score
from whisper_loop_repair import (
    DEFAULT_REPAIR_THRESHOLDS,
    find_token_loop,
    generate_with_loop_repair,
)


POLICY_TOKENS = {
    "verbatim": "<|ovw_verbatim|>",
    "intended": "<|ovw_intended|>",
}


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), "<i2")
    return samples.astype(np.float32) / 32768.0


def policy_prefix(tokenizer, policy: str) -> list[int]:
    prefix = list(tokenizer.prefix_tokens)
    token_id = tokenizer.convert_tokens_to_ids(POLICY_TOKENS[policy])
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    if no_timestamps == tokenizer.unk_token_id:
        raise ValueError("Whisper tokenizer lacks <|notimestamps|>")
    index = prefix.index(no_timestamps) if no_timestamps in prefix else len(prefix)
    return prefix[:index] + [token_id] + prefix[index:]


def crisper_verbatim_prefix(tokenizer) -> list[int]:
    """Return CrisperWhisper2's five policy tokens plus Whisper's prefix."""
    mode_text = "".join(f"[verbatim_{index}]" for index in range(1, 6))
    mode_tokens = tokenizer.encode(mode_text, add_special_tokens=False)
    if len(mode_tokens) != 5:
        raise ValueError(
            "CrisperWhisper verbatim tags are not five atomic tokens: "
            f"{mode_tokens}"
        )
    return mode_tokens + list(tokenizer.prefix_tokens)


def batches(items: list[dict], size: int):
    for index in range(0, len(items), size):
        yield items[index : index + size]


def generate_batch(
    model,
    processor,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_tokens: list[int],
) -> list[tuple[str, list[dict]]]:
    """Greedily decode a batch and repair only detected token loops.

    All evaluated Hugging Face systems use this same inference rule. Ordinary
    conversational repetition is untouched: repair is triggered only after a
    token n-gram exceeds the deliberately high loop thresholds.
    """
    max_new_tokens = 448 - len(prompt_tokens)
    decoder_input_ids = torch.tensor(
        [prompt_tokens] * input_features.shape[0],
        dtype=torch.long,
        device=input_features.device,
    )
    initial = model.generate(
        input_features=input_features,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
    )
    eos_token_id = processor.tokenizer.eos_token_id
    generated = []
    for index, sequence in enumerate(initial.tolist()):
        token_ids = [int(token_id) for token_id in sequence]
        if token_ids[: len(prompt_tokens)] == prompt_tokens:
            token_ids = token_ids[len(prompt_tokens) :]
        if eos_token_id in token_ids:
            token_ids = token_ids[: token_ids.index(eos_token_id) + 1]
        loop = find_token_loop(
            token_ids,
            max_ngram=max(DEFAULT_REPAIR_THRESHOLDS),
            reps=DEFAULT_REPAIR_THRESHOLDS,
        )
        if loop is None:
            repairs = []
        else:
            token_ids, repairs = generate_with_loop_repair(
                model,
                input_features[index : index + 1],
                attention_mask[index : index + 1],
                prompt_tokens,
                max_new_tokens=max_new_tokens,
            )
        generated.append(
            (
                processor.decode(token_ids, skip_special_tokens=True).strip(),
                repairs,
            )
        )
    return generated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--corpus", choices=["latrobe", "sbcsae", "gcsause"], required=True
    )
    parser.add_argument("--system", required=True)
    parser.add_argument("--base-model", default="openai/whisper-large-v3")
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--dual-policy", action="store_true")
    parser.add_argument("--crisper-verbatim", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    if args.dual_policy and args.crisper_verbatim:
        raise ValueError("choose either --dual-policy or --crisper-verbatim")

    all_rows = read_jsonl(args.manifest.resolve())
    source_rows = {}
    for row in all_rows:
        if row["split"] == "test" and row["corpus"] == args.corpus:
            source_rows.setdefault(row["pair_id"], row)
    rows = list(source_rows.values())
    if not rows:
        raise ValueError(f"no test rows for {args.corpus}")

    processor_path = (
        str(args.adapter)
        if args.adapter is not None and args.dual_policy
        else args.base_model
    )
    processor = WhisperProcessor.from_pretrained(
        processor_path, language="en", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model, dtype=torch.bfloat16
    )
    if args.dual_policy:
        model.resize_token_embeddings(
            len(processor.tokenizer), mean_resizing=False
        )
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to("cuda").eval()
    model.config.use_cache = True
    model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None

    if args.dual_policy:
        modes = ["verbatim", "intended"]
    elif args.crisper_verbatim:
        modes = ["crisper_verbatim"]
    else:
        modes = ["standard"]
    predictions = {mode: [] for mode in modes}
    repair_metadata = {mode: [] for mode in modes}
    with torch.inference_mode():
        for mode in modes:
            if mode == "standard":
                prefix = list(processor.tokenizer.prefix_tokens)
            elif mode == "crisper_verbatim":
                prefix = crisper_verbatim_prefix(processor.tokenizer)
            else:
                prefix = policy_prefix(processor.tokenizer, mode)
            for group in batches(rows, args.batch_size):
                features = processor.feature_extractor(
                    [read_wav(Path(row["audio"])) for row in group],
                    sampling_rate=16000,
                    return_attention_mask=True,
                    return_tensors="pt",
                )
                generated = generate_batch(
                    model,
                    processor,
                    features["input_features"].to(
                        device="cuda", dtype=torch.bfloat16
                    ),
                    features["attention_mask"].to("cuda"),
                    prefix,
                )
                predictions[mode].extend(text for text, _ in generated)
                repair_metadata[mode].extend(repairs for _, repairs in generated)

    verbatim_refs = [row["verbatim_text"] for row in rows]
    intended_refs = [row["intended_text"] for row in rows]
    metrics = {
        "system": args.system,
        "corpus": args.corpus,
        "examples": len(rows),
        "adapter": str(args.adapter) if args.adapter else None,
        "dual_policy": args.dual_policy,
        "crisper_verbatim": args.crisper_verbatim,
        "generation": {
            "decoding": "greedy",
            "loop_repair": True,
            "loop_thresholds": DEFAULT_REPAIR_THRESHOLDS,
        },
        "modes": {},
    }
    output_rows = []
    for mode in modes:
        hypotheses = predictions[mode]
        metrics["modes"][mode] = {
            "against_verbatim": score(verbatim_refs, hypotheses),
            "against_intended": score(intended_refs, hypotheses),
            "loop_repaired_examples": sum(
                bool(repairs) for repairs in repair_metadata[mode]
            ),
        }
        for row, hypothesis, repairs in zip(
            rows, hypotheses, repair_metadata[mode]
        ):
            output_rows.append(
                {
                    "system": args.system,
                    "mode": mode,
                    "corpus": args.corpus,
                    "example_id": row["pair_id"],
                    "verbatim_reference": row["verbatim_text"],
                    "intended_reference": row["intended_text"],
                    "hypothesis": hypothesis,
                    "audio": row["audio"],
                    "loop_repairs": repairs,
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
