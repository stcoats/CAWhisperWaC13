# Ancillary experiments

These experiments were conducted after the main four-system CoANZSE
comparison. They clarify model-development choices but do not enlarge the main
target-domain table.

## Adding GCSAusE

The Griffith Corpus of Spoken Australian English was prepared from TalkBank
CHAT timing marks. Corpus notation was normalized to the same lexical-verbatim
conventions as the other sources, clips were cut to exact source boundaries,
and examples with temporal overlap or unintelligible material were excluded.
The selected all-users non-overlap subset contains 192 clips (0.959 hours), of
which 124/30/38 belong to the train/validation/test splits.

The resulting three-source manifest contains 1,367 unique examples (7.772
hours): 922 training, 197 validation, and 248 test. A Whisper Large-v3 full
fine-tune used the same settings as the two-source full model. It modestly
improved macro vWER across the three held-out corpora (18.37% to 17.83%), with
the clearest gain on GCSAusE (30.29% to 28.87%). It was slightly worse on
SBCSAE (9.97% to 10.20%). These results suggest that the additional hour mainly
redistributed performance toward its own domain rather than producing a
uniform improvement.

Only a three-source **full fine-tune** was run. There is no three-source LoRA
result.

## Adapting CrisperWhisper2

The ancillary CrisperWhisper2 experiment used the same 798 training and 167
validation clips as the main two-source experiment. LoRA adapters (rank 32,
alpha 64, dropout .05) were applied to q/v projections in encoder and decoder
attention. Decoding retained CrisperWhisper2's five verbatim policy tokens.

On held-out source-corpus speech, adaptation lowered vWER from 20.29% to 17.07%
on La Trobe and from 14.11% to 10.65% on SBCSAE. Filler F1 fell on La Trobe
(.757 to .599) and on SBCSAE (.494 to .430), showing that lower lexical error
did not guarantee better filler recovery. This system was not run on the
final 100-clip CoANZSE gold benchmark and therefore is not directly comparable
with the paper's main target-domain numbers.

Only aggregate values are included because the adapted weights and per-example
outputs remain subject to upstream model and source-corpus restrictions.

These values use the frozen scoring run reported in the manuscript. Later
scorer-development outputs are not substituted into this artifact.
