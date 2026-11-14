"""Distinct official dataset semantics, especially unanimous and literal labels."""
import unittest

from src.evaluation.vizwiz_official import score_answer, reference_metadata


class VizWizOfficialScoringTests(unittest.TestCase):
    def test_unanimous_predictions_are_still_normalized(self):
        self.assertEqual(score_answer("Unanswerable.", {"answers": ["unanswerable"] * 10})["score"], 1.)
        self.assertEqual(score_answer("two", {"answers": ["2"] * 10})["score"], 1.)

    def test_literal_references_and_complete_predictions_are_not_rewritten(self):
        self.assertEqual(score_answer("cat", {"answers": ["a cat"] * 4 + ["dog"] * 6})["score"], 0.)
        self.assertEqual(score_answer("The image shows a cat", {"answers": ["cat"] * 10})["score"], 0.)

    def test_ten_choose_nine_consensus(self):
        for matches, expected in [(0, 0.), (1, .3), (2, .6), (3, .9), (4, 1.), (10, 1.)]:
            self.assertAlmostEqual(score_answer("cat", {"answers": ["cat"] * matches + ["dog"] * (10-matches)})["score"], expected)
        self.assertEqual(reference_metadata()["source_SHA256"], "d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694")


if __name__ == "__main__": unittest.main()
