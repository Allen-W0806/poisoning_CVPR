import copy
import importlib.util
from pathlib import Path
import unittest
from src.evaluation.vqa_official_compatible import score_answer, normalize_answer

READY = (Path(__file__).resolve().parents[1] / "src/evaluation/_vendor/official_vqa.py").exists()


@unittest.skipUnless(READY, "Pinned official reference must be prepared with Python 3.11")
class OfficialVQATest(unittest.TestCase):
    def test_normalization_edges(self):
        cases = {"The TWO!": "2", "a cat": "cat", "1,000": "1000", "3.14": "3.14", "dont": "don't",
                 "couldnt": "couldn't", "red, blue": "red blue", "the\nthree\tcats.": "3 cats"}
        for original, expected in cases.items():
            self.assertEqual(normalize_answer(original), expected)

    def test_known_consensus_and_immutable_labels(self):
        for count, expected in [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1), (10, 1)]:
            record = {"answers": ["yes"] * count + ["no"] * (10 - count)}
            before = copy.deepcopy(record)
            self.assertAlmostEqual(score_answer("yes", record)["score"], expected)
            self.assertEqual(record, before)
        # Preserve the upstream conditional normalization edge, rather than hiding it.
        self.assertEqual(score_answer("Yes", {"answers": ["yes"] * 10})["score"], 0)
        self.assertEqual(score_answer("2", {"answers": ["two"] * 5 + ["three"] * 5})["score"], 1)

    def test_no_answer_extraction_by_label(self):
        result = score_answer("The answer is yes.", {"answers": ["yes"] * 10})
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["extracted_short_answer"], "The answer is yes.")
