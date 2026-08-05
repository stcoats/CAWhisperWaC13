#!/usr/bin/env python3

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "data"))

from prepare_latrobe import normalize_turn


class NormalizeTests(unittest.TestCase):
    def make_turn(self, text):
        return {
            "recording": "MarkKylie",
            "turn_index": 0,
            "line_start": 1,
            "line_end": 1,
            "raw_speaker": "Mark",
            "speaker": "Mark",
            "speakers": ["Mark"],
            "raw_text": text,
        }

    def normalized(self, text):
        return normalize_turn(self.make_turn(text))[0]

    def test_requested_orthographic_normalizations(self):
        row = self.normalized("Mmm... I was gunna go, OK?")
        self.assertEqual(row["normalized_text"], "mm I was gonna go, okay?")

    def test_false_start_is_preserved(self):
        row = self.normalized("A.. Australia and ob.. obviously")
        self.assertEqual(row["normalized_text"], "A Australia and ob obviously")

    def test_overlap_words_are_retained_and_flagged(self):
        row = self.normalized("[yeah] I agree")
        self.assertEqual(row["normalized_text"], "yeah I agree")
        self.assertIn("overlap", row["flags"])
        self.assertEqual(row["overlap_token_count"], 1)

    def test_events_and_pauses_are_removed(self):
        row = self.normalized("@@@ (1.5) well (laughter) yes")
        self.assertEqual(row["normalized_text"], "well yes")
        self.assertIn("non_speech_event", row["flags"])

    def test_lexical_parenthetical_is_retained_but_flagged(self):
        row = self.normalized("I think (it’s) fine")
        self.assertEqual(row["normalized_text"], "I think it's fine")
        self.assertIn("lexical_parenthetical", row["flags"])

    def test_unknown_span_is_removed_and_flagged(self):
        row = self.normalized("I went ????? home")
        self.assertEqual(row["normalized_text"], "I went home")
        self.assertIn("unknown", row["flags"])


if __name__ == "__main__":
    unittest.main()
