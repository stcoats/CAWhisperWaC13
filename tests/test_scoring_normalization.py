#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "scoring"))

from scoring_metrics import event_prf
from scoring_normalization import (
    UNCLEAR_TOKEN,
    canonical_tokens as tokens,
    edit_counts,
    scoreable_units,
)


class ScoringNormalizationTest(unittest.TestCase):
    def test_nasal_backchannel_spellings_are_one_event(self):
        self.assertEqual(
            tokens("mhm mm-hmm mm hm m-hm mmhm hm hmm mmm mmmm"),
            ["mm"] * 9,
        )

    def test_uh_huh_spelling_variants_are_one_event(self):
        self.assertEqual(tokens("uh-huh uh huh"), ["uh-huh", "uh-huh"])

    def test_internal_hyphens_are_not_false_start_fragments(self):
        self.assertEqual(
            tokens("thirty-nine s- stewardess"),
            ["thirty", "nine", "s-", "stewardess"],
        )

    def test_conventional_scoring_removes_canonical_fillers_and_fragments(self):
        self.assertEqual(
            tokens("mhm mm-hmm uh huh um s- stewardess", conventional=True),
            ["stewardess"],
        )

    def test_missed_reference_event_has_zero_f1(self):
        score = event_prf(
            [{"reference": "s- stewardess", "hypothesis": "stewardess"}],
            lambda _items, _index, token: token.endswith("-"),
        )
        self.assertEqual(score["reference"], 1)
        self.assertEqual(score["hypothesis"], 0)
        self.assertEqual(score["f1"], 0.0)

    def test_regional_spelling_is_not_an_asr_error(self):
        self.assertEqual(
            tokens(
                "the organised programme at the neighbourhood centre used colours"
            ),
            tokens(
                "the organized program at the neighborhood center used colors"
            ),
        )

    def test_broad_regional_spelling_families_are_not_asr_errors(self):
        regional = (
            "analyse analysed anaemia paediatric modelling labelled aluminium "
            "councillor fulfilment catalogue mould plough jewellery"
        )
        us = (
            "analyze analyzed anemia pediatric modeling labeled aluminum "
            "councilor fulfillment catalog mold plow jewelry"
        )
        self.assertEqual(tokens(regional), tokens(us))

    def test_regional_vocabulary_and_audible_forms_remain_distinct(self):
        for regional, us in (
            ("aeroplane", "airplane"),
            ("mum", "mom"),
            ("titbit", "tidbit"),
        ):
            self.assertNotEqual(tokens(regional), tokens(us))

    def test_macron_variants_are_not_asr_errors(self):
        self.assertEqual(
            tokens("Māori kōrero whānau Tāmaki"),
            tokens("Maori korero whanau Tamaki"),
        )

    def test_leading_apostrophe_variants_match_but_them_remains_distinct(self):
        self.assertEqual(tokens("you can use 'em"), tokens("you can use ’em"))
        self.assertEqual(tokens("you can use 'em"), tokens("you can use em"))
        self.assertNotEqual(tokens("you can use 'em"), tokens("you can use them"))
        self.assertEqual(
            tokens("you can use 'em", conventional=True),
            tokens("you can use them", conventional=True),
        )

    def test_clitic_s_expansion_is_only_ignored_in_normalized_scoring(self):
        contracted = tokens("the council's taken the lead")
        expanded = tokens("the council has taken the lead")
        self.assertNotEqual(contracted, expanded)
        self.assertNotEqual(edit_counts(contracted, expanded), (0, 0, 0))
        self.assertEqual(
            edit_counts(
                tokens("the council's taken the lead", conventional=True),
                tokens("the council has taken the lead", conventional=True),
            ),
            (0, 0, 0),
        )
        self.assertEqual(
            edit_counts(
                tokens("the council's ready", conventional=True),
                tokens("the council is ready", conventional=True),
            ),
            (0, 0, 0),
        )

    def test_title_abbreviations_match_spoken_forms(self):
        self.assertEqual(
            tokens("Mr. Smith and Dr Jones spoke to Mrs Brown and Ms Green"),
            tokens(
                "mister Smith and doctor Jones spoke to missus Brown and miz Green"
            ),
        )

    def test_spoken_initialism_spacing_and_punctuation_are_ignored(self):
        self.assertEqual(tokens("the L T P proposal"), tokens("the LTP proposal"))
        self.assertEqual(tokens("the L.T.P. proposal"), tokens("the LTP proposal"))
        self.assertEqual(tokens("the C-E-O spoke"), tokens("the CEO spoke"))

    def test_repeated_single_letter_word_is_not_an_initialism(self):
        self.assertEqual(tokens("I I think"), ["i", "i", "think"])

    def test_audible_reductions_remain_distinct(self):
        self.assertNotEqual(tokens("gonna"), tokens("going to"))
        self.assertNotEqual(tokens("coulda"), tokens("could have"))

    def test_ambiguous_st_is_not_normalized_to_saint_or_street(self):
        self.assertNotEqual(tokens("St"), tokens("saint"))
        self.assertNotEqual(tokens("St"), tokens("street"))

    def test_unclear_word_is_masked_without_entering_denominator(self):
        reference = tokens("the [unclear] proposal")
        self.assertEqual(reference, ["the", UNCLEAR_TOKEN, "proposal"])
        self.assertEqual(scoreable_units(reference), 2)
        self.assertEqual(edit_counts(reference, tokens("the siting proposal")), (0, 0, 0))
        self.assertEqual(edit_counts(reference, tokens("the proposal")), (0, 0, 0))

    def test_unclear_length_limits_masked_hypothesis_words(self):
        reference = tokens("the [unclear 2] proposal")
        self.assertEqual(
            edit_counts(reference, tokens("the long term proposal")),
            (0, 0, 0),
        )
        self.assertEqual(
            edit_counts(reference, tokens("the very long term proposal")),
            (0, 0, 1),
        )

    def test_non_speech_is_excluded_but_bracketed_fillers_are_retained(self):
        self.assertEqual(
            tokens("[music] [UM] well [laughter] [UH]"),
            ["um", "well", "uh"],
        )


if __name__ == "__main__":
    unittest.main()
