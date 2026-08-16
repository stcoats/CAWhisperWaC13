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
lexical-verbatim convention. The target text always comes from these normalized
human transcripts; ASR output was used only to recover approximate word times.
Recordings were converted to mono 16-kHz PCM WAV and divided into fixed
training, validation, and held-out test partitions. Examples with
transcript-marked temporal overlap were excluded from this primary release.
The dataset retains fillers, repetitions, cut-offs, reduced forms, and selected
audible non-speech events.

### ASR-assisted segmentation and exact cuts

Each complete source recording was first transcribed with Faster-Whisper
Large-v3 in English (beam size 5, `condition_on_previous_text=False`, no VAD),
after which WhisperX's default English alignment model supplied word-level
timestamps. The normalized human transcript and timestamped ASR words were
then compared as token sequences. Exact matches belonging to runs of at least
two consecutive tokens served as timing anchors. Times for unmatched human
transcript tokens were estimated by monotonic linear interpolation between
anchors; unmatched token durations used the median ASR-word duration.

Candidate examples targeted approximately 20 seconds. Human-transcript turn
boundaries were preferred where they produced examples between 10 and 30
seconds; a long individual turn was instead divided at the token closest to
the 20-second target. An earlier preparation stage added margins for checking
and possible forced alignment. Those margins were removed for this release:
each WAV begins at the estimated start of its first target token and ends at
the estimated end of its last target token, with no safety padding. Thus,
`source_start` and `source_end` are WhisperX-derived boundaries: directly
aligned where the boundary token was an anchor and interpolated otherwise.

### Quality tiers

`quality_tier` describes confidence in the automatic mapping between the human
transcript and the ASR-derived timeline; it is **not** a subjective rating of
recording quality. `anchor_coverage` is the proportion of human-reference
tokens belonging to exact-match anchor runs. `asr_similarity` is the
SequenceMatcher similarity between the human-reference span and the nearby ASR
words.

- **A** required anchor coverage of at least .65, ASR similarity of at least
  .70, no transcript-marked overlap or uncertain material, and no
  multiple-speaker flag.
- **B** required anchor coverage of at least .50, ASR similarity of at least
  .55, transcript-marked overlap of no more than 12% of tokens, and no
  uncertain material.

The final primary release then excluded every example containing any
transcript-marked temporal overlap, independently of tier. It contains 208
tier-A and 30 tier-B examples. Tier B therefore mostly identifies lower
anchor/similarity confidence or sequential speaker changes in this release,
not retained simultaneous speech.

The release contains 238 examples (1.189 hours): 136 training examples, 25
validation examples, and 77 held-out test examples. The test partition was not
used for model selection. Splits are recording-disjoint: `HeatherMarie` is the
validation recording, `NatalieKen` is the test recording, and the remaining
four recordings constitute training data.

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
