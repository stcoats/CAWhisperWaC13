#!/usr/bin/env python3
"""Train dual-policy or verbatim-only Whisper with LoRA or full fine-tuning."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import re
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    set_seed,
)


POLICY_TOKENS = {
    "verbatim": "<|ovw_verbatim|>",
    "intended": "<|ovw_intended|>",
}
LEXICAL = re.compile(
    r"[a-z0-9]+(?:['’][a-z0-9]+)*(?:-[a-z0-9]+(?:['’][a-z0-9]+)*)*-?",
    re.IGNORECASE,
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getframerate() != 16000
            or handle.getsampwidth() != 2
        ):
            raise ValueError(f"{path}: expected mono 16 kHz PCM16 WAV")
        samples = np.frombuffer(handle.readframes(handle.getnframes()), "<i2")
    return samples.astype(np.float32) / 32768.0


def lexical_tokens(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("’", "'")
    return [match.group(0) for match in LEXICAL.finditer(text)]


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for row, ref in enumerate(reference, start=1):
        current = [row]
        for column, hyp in enumerate(hypothesis, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref != hyp),
                )
            )
        previous = current
    return previous[-1]


def corpus_wer(references: list[str], hypotheses: list[str]) -> float:
    errors = words = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = lexical_tokens(reference)
        hyp = lexical_tokens(hypothesis)
        errors += edit_distance(ref, hyp)
        words += len(ref)
    return errors / max(1, words)


def policy_prefix(tokenizer, policy: str) -> list[int]:
    prefix = list(tokenizer.prefix_tokens)
    policy_id = tokenizer.convert_tokens_to_ids(POLICY_TOKENS[policy])
    if policy_id == tokenizer.unk_token_id:
        raise ValueError(f"policy token is unknown: {POLICY_TOKENS[policy]}")
    no_timestamps = tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    if no_timestamps == tokenizer.unk_token_id:
        raise ValueError("Whisper tokenizer lacks <|notimestamps|>")
    if no_timestamps in prefix:
        index = prefix.index(no_timestamps)
        return prefix[:index] + [policy_id] + prefix[index:]
    return prefix + [policy_id]


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "audio": read_wav(Path(row["audio"])),
            "text": row["text"],
            "policy": row["policy"],
            "example_id": row["example_id"],
        }


@dataclass
class DualPolicyCollator:
    processor: Any
    decoder_start_token_id: int
    use_policy_tokens: bool

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        inputs = self.processor.feature_extractor(
            [item["audio"] for item in features],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        eos = self.processor.tokenizer.eos_token_id
        label_features = []
        for item in features:
            content = self.processor.tokenizer.encode(
                item["text"], add_special_tokens=False
            )
            prefix = (
                policy_prefix(self.processor.tokenizer, item["policy"])
                if self.use_policy_tokens
                else list(self.processor.tokenizer.prefix_tokens)
            )
            ids = prefix + content
            if not ids or ids[-1] != eos:
                ids.append(eos)
            label_features.append({"input_ids": ids})
        labels = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        label_ids = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        if (
            label_ids.shape[1]
            and torch.all(label_ids[:, 0] == self.decoder_start_token_id)
        ):
            label_ids = label_ids[:, 1:]
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        inputs["labels"] = label_ids
        return inputs


class PolicyTrainer(Seq2SeqTrainer):
    """Keep weight decay off vocabulary matrices whose old rows are masked."""

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer
        decay = []
        no_decay = []
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if (
                parameter.ndim < 2
                or name.endswith(".bias")
                or "embed_tokens" in name
                or "proj_out" in name
            ):
                no_decay.append(parameter)
            else:
                decay.append(parameter)
        groups = [
            {"params": decay, "weight_decay": self.args.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        self.optimizer = torch.optim.AdamW(
            groups,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )
        return self.optimizer


def mask_old_vocabulary_gradients(model, policy_ids: list[int]) -> list[str]:
    hooked = []

    def row_mask(gradient: torch.Tensor) -> torch.Tensor:
        selected = gradient[policy_ids].clone()
        gradient.zero_()
        gradient[policy_ids] = selected
        return gradient

    for name, parameter in model.named_parameters():
        if (
            parameter.requires_grad
            and parameter.ndim == 2
            and parameter.shape[0] > max(policy_ids)
            and ("embed_tokens" in name or "proj_out" in name)
        ):
            parameter.register_hook(row_mask)
            hooked.append(name)
    if not any("embed_tokens" in name for name in hooked):
        raise RuntimeError(f"did not find trainable decoder embedding: {hooked}")
    if not any("proj_out" in name for name in hooked):
        raise RuntimeError(f"did not find trainable output projection: {hooked}")
    return hooked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="openai/whisper-large-v3")
    parser.add_argument(
        "--optimization", choices=("lora", "full"), default="lora"
    )
    parser.add_argument(
        "--training-policy", choices=("dual", "verbatim"), default="dual"
    )
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--target-modules", nargs="+", default=["q_proj", "v_proj"])
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    rows = read_jsonl(manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    # Checkpoint selection intentionally uses only the primary verbatim policy.
    validation_rows = [
        row
        for row in rows
        if row["split"] == "validation" and row["policy"] == "verbatim"
    ]
    if not train_rows or not validation_rows:
        raise ValueError("manifest lacks train or verbatim validation examples")
    matched_exposure_multiplier = 1
    if args.training_policy == "verbatim":
        verbatim_rows = [row for row in train_rows if row["policy"] == "verbatim"]
        repeated_rows = [
            {**row, "example_id": f"{row['example_id']}__matched_repeat"}
            for row in verbatim_rows
        ]
        # This gives the control exactly as many audio exposures and optimizer
        # updates as the paired dual-policy condition.
        train_rows = verbatim_rows + repeated_rows
        matched_exposure_multiplier = 2
    if args.smoke_test:
        if args.training_policy == "dual":
            pair_ids = []
            for row in train_rows:
                if row["pair_id"] not in pair_ids:
                    pair_ids.append(row["pair_id"])
                if len(pair_ids) == 8:
                    break
            train_rows = [
                row for row in train_rows if row["pair_id"] in pair_ids
            ]
        else:
            train_rows = train_rows[:16]
        validation_rows = validation_rows[:4]
        args.epochs = 1.0

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    processor = WhisperProcessor.from_pretrained(
        args.model, language="en", task="transcribe"
    )
    use_policy_tokens = args.training_policy == "dual"
    added = 0
    if use_policy_tokens:
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(POLICY_TOKENS.values())}
        )
        if added not in (0, 2):
            raise RuntimeError(f"expected zero or two added tokens, got {added}")
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    original_vocab = model.get_input_embeddings().weight.shape[0]
    policy_ids: list[int] = []
    seed_tokens: list[str] = []
    if use_policy_tokens:
        model.resize_token_embeddings(
            len(processor.tokenizer), mean_resizing=False
        )
        policy_ids = [
            processor.tokenizer.convert_tokens_to_ids(POLICY_TOKENS[policy])
            for policy in ("verbatim", "intended")
        ]
        if len(set(policy_ids)) != 2 or min(policy_ids) < original_vocab:
            raise RuntimeError(
                f"policy IDs were not newly allocated: {policy_ids}, "
                f"base={original_vocab}"
            )

        seed_tokens = ["<|transcribe|>", "<|translate|>", "<|notimestamps|>"]
        seed_ids = [
            processor.tokenizer.convert_tokens_to_ids(token)
            for token in seed_tokens
        ]
        with torch.no_grad():
            input_weight = model.get_input_embeddings().weight
            output_weight = model.get_output_embeddings().weight
            input_mean = (
                input_weight[seed_ids].float().mean(0).to(input_weight.dtype)
            )
            output_mean = (
                output_weight[seed_ids].float().mean(0).to(output_weight.dtype)
            )
            for policy_id in policy_ids:
                input_weight[policy_id].copy_(input_mean)
                output_weight[policy_id].copy_(output_mean)
        model.tie_weights()

    verbal_prefix = (
        policy_prefix(processor.tokenizer, "verbatim")
        if use_policy_tokens
        else list(processor.tokenizer.prefix_tokens)
    )
    decoder_start = int(model.config.decoder_start_token_id)
    if verbal_prefix[0] != decoder_start:
        raise RuntimeError(
            f"unexpected decoder prefix {verbal_prefix}; start={decoder_start}"
        )
    forced = [
        [position, token_id]
        for position, token_id in enumerate(verbal_prefix[1:], start=1)
    ]
    model.config.use_cache = False
    model.config.forced_decoder_ids = forced
    model.generation_config.forced_decoder_ids = forced
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    model.enable_input_require_grads()

    hooked_parameters: list[str] = []
    if args.optimization == "lora":
        lora = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.target_modules,
            modules_to_save=(
                ["embed_tokens", "proj_out"] if use_policy_tokens else None
            ),
            ensure_weight_tying=use_policy_tokens,
            task_type=None,
        )
        model = get_peft_model(model, lora)
        if use_policy_tokens:
            hooked_parameters = mask_old_vocabulary_gradients(model, policy_ids)
    if int(os.environ.get("RANK", "0")) == 0:
        if args.optimization == "lora":
            model.print_trainable_parameters()
        else:
            trainable = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            total = sum(parameter.numel() for parameter in model.parameters())
            print(
                f"trainable params: {trainable:,} || all params: {total:,} "
                f"|| trainable%: {100 * trainable / total:.4f}"
            )

    validation_corpora = [
        row.get("corpus", "dataset") for row in validation_rows
    ]

    def compute_metrics(prediction) -> dict[str, float]:
        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = prediction.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        hypotheses = processor.batch_decode(predictions, skip_special_tokens=True)
        references = processor.batch_decode(labels, skip_special_tokens=True)
        metrics = {"verbatim_wer": corpus_wer(references, hypotheses)}
        corpus_scores = []
        for corpus in sorted(set(validation_corpora)):
            selected = [
                index
                for index, value in enumerate(validation_corpora)
                if value == corpus
            ]
            score = corpus_wer(
                [references[index] for index in selected],
                [hypotheses[index] for index in selected],
            )
            metrics[f"{corpus}_verbatim_wer"] = score
            corpus_scores.append(score)
        metrics["corpus_macro_verbatim_wer"] = sum(corpus_scores) / len(
            corpus_scores
        )
        return metrics

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        seed=args.seed,
        data_seed=args.seed,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        predict_with_generate=True,
        generation_max_length=448,
        load_best_model_at_end=True,
        metric_for_best_model="corpus_macro_verbatim_wer",
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
    )
    trainer = PolicyTrainer(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(train_rows),
        eval_dataset=ManifestDataset(validation_rows),
        data_collator=DualPolicyCollator(
            processor=processor,
            decoder_start_token_id=decoder_start,
            use_policy_tokens=use_policy_tokens,
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )
    trainer.train()
    artifact_name = "best_adapter" if args.optimization == "lora" else "best_model"
    artifact_dir = output_dir / artifact_name
    trainer.save_model(str(artifact_dir))
    if trainer.is_world_process_zero():
        processor.save_pretrained(artifact_dir)
        metadata = {
            "project": "open_verbatim_whisper",
            "implementation": (
                f"{args.training_policy}-policy Whisper "
                f"{args.optimization} fine-tuning"
            ),
            "base_model": args.model,
            "manifest": str(manifest),
            "policy_tokens": POLICY_TOKENS if use_policy_tokens else {},
            "policy_token_ids": policy_ids,
            "policy_prefixes": {
                policy: policy_prefix(processor.tokenizer, policy)
                for policy in POLICY_TOKENS
            }
            if use_policy_tokens
            else {},
            "initialization_tokens": seed_tokens,
            "gradient_masked_parameters": hooked_parameters,
            "matched_exposure_multiplier": matched_exposure_multiplier,
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "packages": {
                package: importlib.metadata.version(package)
                for package in (
                    "accelerate",
                    "datasets",
                    "peft",
                    "torch",
                    "transformers",
                )
            },
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
