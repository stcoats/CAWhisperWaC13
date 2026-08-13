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

## Ancillary three-source full fine-tune

The GCSAusE preparation scripts are `scripts/data/prepare_gcsause.py` and
`scripts/data/cut_gcsause_chunks.py`. The prepared non-overlap set comprised
192 clips (0.959 hours): 124 training, 30 validation, and 38 test clips. After
combining it with the unchanged La Trobe/SBCSAE splits, the source manifest
contained 1,367 unique clips (7.772 hours): 922 training, 197 validation, and
248 test clips. The full model used the same optimization settings as the main
two-source full fine-tune; the training rows were duplicated solely to match
the two-exposure dual-policy condition. See `configs/three_source_full.json`
and `results/ancillary/three_source_full.tsv`.

No three-source LoRA experiment was run.

## Ancillary CrisperWhisper2 adaptation

`scripts/train/train_crisper2_lora.py` reproduces the two-source adaptation.
It loads `nyralabs/CrisperWhisper2.0_large`, prepends the model's five atomic
`[verbatim_1]` through `[verbatim_5]` mode tokens, and trains rank-32 LoRA
adapters on q/v projections throughout the encoder and decoder. The upstream
model remains frozen. Exact settings are in `configs/crisper2_lora.json` and
aggregate source-corpus results are in
`results/ancillary/crisper2_adaptation.tsv`.

This adapted model was evaluated only on the held-out La Trobe and SBCSAE
splits. It was not evaluated on the final CoANZSE gold benchmark, and restricted
weights and per-example predictions are not redistributed.

The evaluation driver accepts `--crisper-verbatim`; combine it with
`--base-model nyralabs/CrisperWhisper2.0_large` and, for the adapted system,
`--adapter /path/to/best_adapter`. It reconstructs the five-token Crisper
decoder prefix before greedy generation.

## Scoring

The frozen scorer canonicalizes punctuation, case, regional spelling,
diacritics, titles, initialism formatting, and selected backchannel spellings.
Verbatim WER retains fillers, fragments, repetitions, contractions, and
reduced forms. Normalized WER additionally removes fillers/fragments,
collapses adjacent repetition, and equates selected reduced/full forms.
`[unclear]` masks only the corresponding reference unit.
