---
license: cc-by-4.0
task_categories:
- automatic-speech-recognition
language:
- en
pretty_name: La Trobe CA-ASR Exact-Boundary Dataset
---

# La Trobe CA-ASR Exact-Boundary Dataset

This dataset contains the processed La Trobe Corpus of Spoken Australian
English examples used for the principal ASR adaptation experiment accompanying
*Keeping the Ums: Conversation-Analysis-Informed Verbatim ASR for Web Speech*.

## Source and licence

The source is the **La Trobe Corpus of Spoken Australian English** (Mullan,
2002), La Trobe University, <https://doi.org/10.26181/23089559>. The source
collection is distributed under the Creative Commons Attribution 4.0
International licence (CC BY 4.0). This processed derivative is distributed
under the same licence.

## Processing

The original conversational transcripts were normalized into a consistent
lexical-verbatim convention. Recordings were segmented to the first and last
aligned target-word boundaries, converted to mono 16-kHz PCM WAV, and divided
into fixed training, validation, and held-out test partitions. Examples with
transcript-marked temporal overlap were excluded from this primary release.
The dataset retains fillers, repetitions, cut-offs, reduced forms, and selected
audible non-speech events.

The release contains 238 examples (1.189 hours): 136 training examples, 25
validation examples, and 77 held-out test examples. The test partition was not
used for model selection.

## Fields

Each split uses the Hugging Face AudioFolder layout with a `metadata.csv` file.
Fields include the WAV filename, normalized verbatim text, example and source
recording identifiers, speaker labels, duration, source-recording offsets,
token count, quality tier, and boundary method.

## Attribution

Users must cite the original La Trobe corpus and indicate that this release is
a segmented and normalized derivative. Users of the accompanying experiment
should additionally cite the paper associated with the CA-Whisper repository.

## Limitations

The release is small and was prepared for verbatim ASR adaptation rather than
as a representative sample of Australian English. Speaker changes can occur
within an example; the exclusion rule concerns transcript-marked simultaneous
overlap. Transcript conventions have been normalized, so consult the original
corpus for the complete conversation-analytic annotation.
