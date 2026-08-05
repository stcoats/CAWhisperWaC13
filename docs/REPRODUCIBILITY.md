# Reproducibility notes

## Frozen experimental design

- Base model: `openai/whisper-large-v3`.
- Unique training audio: 798 clips, 4.474 hours.
- Validation: 167 clips, 1.099 hours.
- Held-out source-corpus test: 210 clips, 1.241 hours.
- Exact audio boundaries and non-overlap filtering are applied before model
  training.
- Checkpoint selection uses the unweighted macro verbatim WER across the two
  validation corpora.
- The held-out test and CoANZSE benchmark are never used for checkpoint
  selection.

## Full verbatim model

The full model is initialized directly from Whisper Large-v3. Every parameter
is updated. Training examples are duplicated so the model receives the same
number of audio exposures and optimizer updates as the dual-policy condition.
The selected checkpoint minimizes validation macro verbatim WER.

## Dual-policy LoRA

Each source example is represented twice: once with its human verbatim target
and once with a deterministic intended target. The intended transformation
removes fillers and marked fragments, removes bracketed events, and collapses
adjacent repetitions without paraphrasing ordinary lexical content.

Two new decoder-prefix tokens select the requested output policy. LoRA is
applied to encoder and decoder query/value projections. The two new vocabulary
rows in the decoder embedding and output projection are also trained; gradient
masking leaves all pre-existing vocabulary rows unchanged.

## Hardware and software

The reference runs used four NVIDIA GH200 GPUs, bfloat16, PyTorch 2.10.0,
Transformers 4.57.6, and PEFT 0.18.1. A matched software specification is in
`requirements.txt`.

## Scoring

The frozen scorer canonicalizes punctuation, case, regional spelling,
diacritics, titles, initialism formatting, and selected backchannel spellings.
Verbatim WER retains fillers, fragments, repetitions, contractions, and
reduced forms. Normalized WER additionally removes fillers/fragments,
collapses adjacent repetition, and equates selected reduced/full forms.
`[unclear]` masks only the corresponding reference unit.
