# Third-party notices

## Whisper

The systems initialize from `openai/whisper-large-v3`, distributed under the
Apache License 2.0. Upstream attribution and terms remain applicable.

## CrisperWhisper2

CrisperWhisper2 is a frozen external evaluation baseline only. Its weights,
adapters, and per-example generated outputs are not included. Aggregate scores
are retained to document the comparison reported in the paper.

## Token-loop repair

`scripts/evaluate/whisper_loop_repair.py` is adapted from the MIT-licensed
CrisperWhisper2 implementation at commit
`5d810bb8d88f06b0a005148e9bf170c3bd6eeef0`. The retained source header records
that provenance. This inference safeguard is applied only to catastrophic
decoder loops, not ordinary conversational repetitions.

## Corpora

La Trobe and SBCSAE material is not redistributed. See
`docs/DATA_AND_LICENSES.md` for the separate source-corpus terms.
