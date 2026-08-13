#!/usr/bin/env python3
"""Multi-GPU LoRA adaptation of CrisperWhisper2 on verbatim transcripts."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import random
import re
import sys
import unicodedata
import wave
from collections import Counter
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoring"))
from scoring_normalization import (
    canonical_tokens as scoring_tokens,
    edit_counts as scoring_edit_counts,
    scoreable_units,
)


VERBATIM_TOKEN = re.compile(
    r"[a-z0-9]+(?:['’][a-z0-9]+)*(?:-[a-z0-9]+(?:['’][a-z0-9]+)*)*-?",
    re.IGNORECASE,
)
FILLERS = {
    "ah",
    "eh",
    "er",
    "erm",
    "huh",
    "mm",
    "uh",
    "uh-huh",
    "um",
}
NASAL_FILLER_VARIANTS = {
    "hm",
    "hmm",
    "hmmm",
    "m-hm",
    "m-hmm",
    "mhm",
    "mm",
    "mm-hm",
    "mm-hmm",
    "mmhm",
    "mmm",
    "mmmm",
}
SPACED_NASAL_FILLER = re.compile(r"\bm+\s+h+m+\b", re.IGNORECASE)
SPACED_UH_HUH = re.compile(r"\buh+\s+huh+\b", re.IGNORECASE)


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


class ManifestDataset(Dataset):
    def __init__(self, project: Path, rows: list[dict]):
        self.project = project
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {
            "audio": read_wav(self.project / row["audio"]),
            "text": row["text"],
            "example_id": row["example_id"],
        }


@dataclass
class WhisperCollator:
    processor: Any
    decoder_start_token_id: int
    decoder_prompt_tokens: tuple[int, ...] = ()

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        audio = [item["audio"] for item in features]
        inputs = self.processor.feature_extractor(
            audio,
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
        )
        if self.decoder_prompt_tokens:
            eos_token_id = self.processor.tokenizer.eos_token_id
            label_features = []
            for item in features:
                content_ids = self.processor.tokenizer.encode(
                    item["text"], add_special_tokens=False
                )
                input_ids = list(self.decoder_prompt_tokens) + content_ids
                if not input_ids or input_ids[-1] != eos_token_id:
                    input_ids.append(eos_token_id)
                label_features.append({"input_ids": input_ids})
        else:
            label_features = [
                {"input_ids": self.processor.tokenizer(item["text"]).input_ids}
                for item in features
            ]
        labels = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        label_ids = labels["input_ids"].masked_fill(
            labels["attention_mask"].ne(1), -100
        )
        if (
            label_ids.shape[1] > 0
            and torch.all(label_ids[:, 0] == self.decoder_start_token_id)
        ):
            label_ids = label_ids[:, 1:]
        # The base model is loaded in bfloat16. Trainer generation does not
        # always enter autocast, so keep acoustic features dtype-consistent for
        # both training and validation generation.
        inputs["input_features"] = inputs["input_features"].to(torch.bfloat16)
        inputs["labels"] = label_ids
        return inputs


def tokens(text: str, conventional: bool = False) -> list[str]:
    return scoring_tokens(text, conventional=conventional)


def edit_counts(reference: list[str], hypothesis: list[str]) -> tuple[int, int, int]:
    return scoring_edit_counts(reference, hypothesis)


def error_rate(references: list[str], hypotheses: list[str], conventional=False) -> float:
    errors = total = 0
    for reference, hypothesis in zip(references, hypotheses):
        ref = tokens(reference, conventional=conventional)
        hyp = tokens(hypothesis, conventional=conventional)
        errors += sum(edit_counts(ref, hyp))
        total += scoreable_units(ref)
    return errors / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="nyralabs/CrisperWhisper2.0_large")
    parser.add_argument(
        "--decoder-policy",
        choices=["whisper", "crisper_verbatim"],
        default="whisper",
        help=(
            "Decoder prompt convention. crisper_verbatim preserves "
            "CrisperWhisper 2's five verbatim policy tokens."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules", nargs="+", default=["q_proj", "v_proj"]
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--epochs", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument(
        "--selection-metric",
        choices=["verbatim_wer", "corpus_macro_verbatim_wer"],
        default="verbatim_wer",
        help=(
            "Metric used for checkpoint selection. Corpus-macro WER gives each "
            "validation corpus equal weight regardless of its duration."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    project = args.project.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest.is_absolute()
        else project / args.manifest
    )
    output_dir = args.output_dir.resolve()
    rows = read_jsonl(manifest)
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    if not train_rows or not validation_rows:
        raise ValueError("manifest must contain train and validation examples")
    if args.smoke_test:
        train_rows = train_rows[: min(12, len(train_rows))]
        validation_rows = validation_rows[: min(6, len(validation_rows))]
        args.epochs = 1.0

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    processor = WhisperProcessor.from_pretrained(
        args.model, language="en", task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16
    )
    decoder_prompt_tokens: tuple[int, ...] = ()
    if args.decoder_policy == "crisper_verbatim":
        mode_text = "".join(f"[verbatim_{index}]" for index in range(1, 6))
        mode_tokens = processor.tokenizer.encode(
            mode_text, add_special_tokens=False
        )
        if len(mode_tokens) != 5:
            raise ValueError(
                "CrisperWhisper verbatim tags are not five atomic tokens: "
                f"{mode_tokens}"
            )
        whisper_prefix = list(processor.tokenizer.prefix_tokens)
        decoder_prompt_tokens = tuple(mode_tokens + whisper_prefix)
        model.config.decoder_start_token_id = decoder_prompt_tokens[0]
        model.generation_config.decoder_start_token_id = (
            decoder_prompt_tokens[0]
        )
        forced_decoder_ids = [
            [position, token_id]
            for position, token_id in enumerate(
                decoder_prompt_tokens[1:], start=1
            )
        ]
        model.config.forced_decoder_ids = forced_decoder_ids
        model.generation_config.forced_decoder_ids = forced_decoder_ids
    decoder_start_token_id = int(model.config.decoder_start_token_id)
    model.config.use_cache = False
    model.generation_config.language = "en"
    model.generation_config.task = "transcribe"
    if args.decoder_policy == "whisper":
        model.generation_config.forced_decoder_ids = None
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=args.target_modules,
        # Deliberately leave task_type unset. PEFT's generic Seq2Seq wrapper
        # assumes text input_ids; Whisper requires acoustic input_features.
        task_type=None,
    )
    model = get_peft_model(model, lora_config)
    if model.__class__.__name__ != "PeftModel":
        raise RuntimeError(
            f"Expected signature-preserving PeftModel, got {type(model).__name__}"
        )
    if int(os.environ.get("RANK", "0")) == 0:
        model.print_trainable_parameters()

    def compute_metrics(prediction) -> dict[str, float]:
        predictions = prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = prediction.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        hypotheses = processor.batch_decode(
            predictions, skip_special_tokens=True
        )
        references = processor.batch_decode(labels, skip_special_tokens=True)
        metrics = {
            "verbatim_wer": error_rate(references, hypotheses),
            "disfluency_stripped_wer": error_rate(
                references, hypotheses, conventional=True
            ),
        }
        corpora = [
            row.get("corpus", "dataset") for row in validation_rows
        ][: len(references)]
        corpus_names = sorted(set(corpora))
        corpus_wers = []
        for corpus in corpus_names:
            selected = [
                index
                for index, item_corpus in enumerate(corpora)
                if item_corpus == corpus
            ]
            corpus_wer = error_rate(
                [references[index] for index in selected],
                [hypotheses[index] for index in selected],
            )
            metrics[f"{corpus}_verbatim_wer"] = corpus_wer
            corpus_wers.append(corpus_wer)
        metrics["corpus_macro_verbatim_wer"] = sum(corpus_wers) / len(
            corpus_wers
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
        metric_for_best_model=args.selection_metric,
        greater_is_better=False,
        save_total_limit=2,
        dataloader_num_workers=args.num_workers,
        dataloader_persistent_workers=args.num_workers > 0,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to=["tensorboard"],
        ddp_find_unused_parameters=False,
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=ManifestDataset(project, train_rows),
        eval_dataset=ManifestDataset(project, validation_rows),
        data_collator=WhisperCollator(
            processor,
            decoder_start_token_id=decoder_start_token_id,
            decoder_prompt_tokens=decoder_prompt_tokens,
        ),
        compute_metrics=compute_metrics,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience
            )
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output_dir / "best_adapter"))
    if trainer.is_world_process_zero():
        processor.save_pretrained(output_dir / "best_adapter")
        metadata = {
            "model": args.model,
            "model_commit": getattr(model.config, "_commit_hash", None),
            "decoder_policy": args.decoder_policy,
            "decoder_prompt_tokens": list(decoder_prompt_tokens),
            "manifest": str(manifest),
            "seed": args.seed,
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "world_size": int(os.environ.get("WORLD_SIZE", "1")),
            "effective_global_batch": (
                args.batch_size
                * int(os.environ.get("WORLD_SIZE", "1"))
                * args.gradient_accumulation
            ),
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
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": args.target_modules,
            },
            "training": vars(args),
        }
        metadata["training"] = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in metadata["training"].items()
        }
        (output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
