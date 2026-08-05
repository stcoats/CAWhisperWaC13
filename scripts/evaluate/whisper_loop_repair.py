#!/usr/bin/env python3
"""Targeted repeated-token loop repair for Hugging Face Whisper generation.

Adapted from the MIT-licensed CrisperWhisper 2 implementation at
https://github.com/nyrahealth/crisperwhisper, commit
5d810bb8d88f06b0a005148e9bf170c3bd6eeef0.

This does not apply a global repetition penalty: legitimate verbatim
repetitions remain possible. It only intervenes after a consecutive token
n-gram exceeds CrisperWhisper's loop thresholds.
"""

from __future__ import annotations

from typing import Any

import torch
from transformers import LogitsProcessorList


DEFAULT_REPAIR_THRESHOLDS: dict[int, int] = {
    1: 8,
    2: 8,
    3: 4,
    **{size: 3 for size in range(4, 13)},
}


def find_token_loop(
    ids: list[int],
    min_ngram: int = 1,
    max_ngram: int = 5,
    reps: int | dict[int, int] = 8,
) -> tuple[int, tuple[int, ...]] | None:
    """Return the first consecutive repeated-token loop, if one exists."""
    thresholds = (
        {n: reps for n in range(min_ngram, max_ngram + 1)}
        if isinstance(reps, int)
        else reps
    )
    for index in range(len(ids)):
        for ngram_size, required_repetitions in thresholds.items():
            if not min_ngram <= ngram_size <= max_ngram:
                continue
            required_tokens = ngram_size * required_repetitions
            if index + required_tokens > len(ids):
                continue
            ngram = tuple(ids[index : index + ngram_size])
            if all(
                tuple(
                    ids[
                        index + repetition * ngram_size :
                        index + (repetition + 1) * ngram_size
                    ]
                )
                == ngram
                for repetition in range(1, required_repetitions)
            ):
                return index, ngram
    return None


class FirstStepBan:
    """Ban selected token IDs for the first continuation step only."""

    def __init__(self, prefix_length: int, banned: set[int]):
        self.prefix_length = int(prefix_length)
        self.banned = sorted(int(token_id) for token_id in banned)

    def __call__(self, input_ids, scores):
        if self.banned and input_ids.shape[1] == self.prefix_length:
            scores[:, self.banned] = float("-inf")
        return scores


def _run_generate(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prefix: list[int],
    max_new_tokens: int,
    ban_first: set[int] | None = None,
) -> list[int]:
    if max_new_tokens <= 0:
        return []
    decoder_input_ids = torch.tensor(
        [prefix], dtype=torch.long, device=input_features.device
    )
    generation_args: dict[str, Any] = {
        "input_features": input_features,
        "decoder_input_ids": decoder_input_ids,
        "max_new_tokens": int(max_new_tokens),
        "num_beams": 1,
        "do_sample": False,
    }
    if attention_mask is not None:
        generation_args["attention_mask"] = attention_mask
    if ban_first:
        generation_args["logits_processor"] = LogitsProcessorList(
            [FirstStepBan(len(prefix), ban_first)]
        )
    output = model.generate(**generation_args)
    sequence = [int(token_id) for token_id in output[0].tolist()]
    if sequence[: len(prefix)] == prefix:
        sequence = sequence[len(prefix) :]
    return sequence


def generate_with_loop_repair(
    model,
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None,
    prompt_tokens: list[int],
    *,
    max_new_tokens: int = 444,
    thresholds: dict[int, int] | None = None,
    keep_repetitions: int = 1,
    max_ngram: int = 12,
    max_repairs: int = 5,
) -> tuple[list[int], list[dict[str, Any]]]:
    """Greedily decode, then repeatedly rewind and escape detected loops.

    The longer scan is necessary because a first escape can alter only the
    phrase boundary while the decoder falls back into the same lexical loop.
    For example, a four-token loop can become a six-token loop after rewind.
    Each regenerated continuation is therefore rescanned from the beginning.
    """
    thresholds = thresholds or DEFAULT_REPAIR_THRESHOLDS
    generated = _run_generate(
        model,
        input_features,
        attention_mask,
        prompt_tokens,
        max_new_tokens,
    )
    repairs: list[dict[str, Any]] = []
    for attempt in range(1, max_repairs + 1):
        loop = find_token_loop(
            generated,
            min_ngram=1,
            max_ngram=max_ngram,
            reps=thresholds,
        )
        if loop is None:
            break
        loop_start, ngram = loop
        keep_end = loop_start + len(ngram) * keep_repetitions
        trimmed = generated[:keep_end]
        repairs.append(
            {
                "attempt": attempt,
                "loop_start": loop_start,
                "ngram_size": len(ngram),
                "ngram_token_ids": list(ngram),
                "trigger_repetitions": thresholds[len(ngram)],
                "kept_tokens": keep_end,
            }
        )
        remaining = max_new_tokens - len(trimmed)
        if remaining <= 0:
            generated = trimmed
            break
        continuation = _run_generate(
            model,
            input_features,
            attention_mask,
            prompt_tokens + trimmed,
            remaining,
            ban_first={ngram[0]},
        )
        generated = trimmed + continuation
    residual_loop = find_token_loop(
        generated,
        min_ngram=1,
        max_ngram=max_ngram,
        reps=thresholds,
    )
    if residual_loop is not None:
        # Never return a known hallucination loop after exhausting the escape
        # attempts. Retain one instance of the triggering n-gram and truncate
        # the unreliable continuation. This turns the unrecoverable tail into
        # deletions rather than hundreds of bogus insertion/repetition errors.
        loop_start, ngram = residual_loop
        keep_end = loop_start + len(ngram) * keep_repetitions
        generated = generated[:keep_end]
        repairs.append(
            {
                "attempt": max_repairs + 1,
                "action": "truncate_residual_loop",
                "loop_start": loop_start,
                "ngram_size": len(ngram),
                "ngram_token_ids": list(ngram),
                "trigger_repetitions": thresholds[len(ngram)],
                "kept_tokens": keep_end,
            }
        )
    return generated, repairs


def whisper_prompt_tokens(processor) -> list[int]:
    """Build the explicit English-transcription/no-timestamps decoder prompt."""
    prompt_pairs = processor.get_decoder_prompt_ids(
        language="en", task="transcribe", no_timestamps=True
    )
    start_of_transcript = processor.tokenizer.convert_tokens_to_ids(
        "<|startoftranscript|>"
    )
    return [
        int(start_of_transcript),
        *(int(token_id) for _, token_id in prompt_pairs),
    ]
