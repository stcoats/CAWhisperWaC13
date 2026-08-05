# Data and licensing

## Training corpora

### La Trobe conversational corpus

The La Trobe material was obtained through LDaCA. Users must obtain their own
authorized copy and comply with the terms displayed by the provider. Those
terms restrict copying, adaptation, transmission, and redistribution. This
repository therefore includes preparation code but no La Trobe recordings,
transcripts, derived clips, or manifest text.

### Santa Barbara Corpus of Spoken American English

SBCSAE is distributed by the University of California, Santa Barbara under
CC BY-ND 3.0 US. This repository includes parsing and segmentation code but no
recordings, transcripts, transformed targets, or derived clips.

Because the interaction between source-corpus terms and redistributed trained
weights is not completely clear, no full-model checkpoint or LoRA adapter is
included in this anonymous artifact. Permission should be obtained from the
corpus rights holders before a public model release.

## CoANZSE evaluation material

The benchmark samples publicly available institutional web recordings. This
repository includes source video identifiers, exact offsets, quality labels,
manual reference text, and releasable system predictions, but no audio. The
manifest links each example back to its public source.

## External systems

`openai/whisper-large-v3` is the initialization model and is distributed under
Apache-2.0. CrisperWhisper2 is used only as an external frozen baseline. Its
weights and generated per-example outputs are excluded; only aggregate values
needed to document the published comparison are retained.

## Repository license

Original repository code is MIT-licensed. That license does not apply to any
third-party corpus, recording, model, or other separately licensed material.
