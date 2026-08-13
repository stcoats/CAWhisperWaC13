import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "data" / "prepare_gcsause.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_gcsause", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class GCSAusENormalizationTests(unittest.TestCase):
    def test_documented_normalization_cases(self):
        MODULE.run_self_test()

    def test_preserves_fragments_and_lexical_colloquialisms(self):
        text, _, _ = MODULE.normalize_chat_text("I d-didn't go acro- gunna")
        self.assertEqual(text, "I d-didn't go acro- gonna")


if __name__ == "__main__":
    unittest.main()
