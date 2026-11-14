import math
import unittest
from src.editing.semantic_validator import SemanticResult
from src.rl.reward import validation_reward
from src.evaluation.vqa_eval import consensus_score, normalize_answer, evaluate_vqa, SyntheticOracleVQA
from src.evaluation.security_metrics import security_metrics
from src.data.dataset import VQADataset
from src.data.prepare_vqa import prepare_fixture
from src.utils.scratch import scratch_directory


class EvaluationRewardTest(unittest.TestCase):
    def test_consensus_and_clean_pipeline(self):
        for count, expected in [(0, 0), (1, .3), (2, .6), (3, .9), (4, 1)]:
            self.assertAlmostEqual(consensus_score("yes", ["yes"] * count + ["no"] * (10 - count)), expected)
        self.assertEqual(normalize_answer("The TWO."), "2")
        with scratch_directory() as root:
            path = prepare_fixture(root)
            dataset = VQADataset(path / "test.json", root)
            result = evaluate_vqa(SyntheticOracleVQA(), dataset)
            self.assertEqual(result["clean_vqa_consensus"], 1)
            self.assertEqual(result["samples"], 10)

    def test_independent_security_denominators(self):
        rows = [{"sample_id": "t1", "group": "triggered", "behavior_observed": True, "detected": True, "is_triggered": True},
                {"sample_id": "t2", "group": "triggered", "behavior_observed": False, "detected": False, "is_triggered": True},
                {"sample_id": "c1", "group": "clean", "behavior_observed": False, "detected": False, "is_triggered": False}]
        result = security_metrics(rows, "synthetic-test", synthetic=True)
        self.assertEqual(result["trigger_conditioned_behavior_rate"], .5)
        self.assertEqual(result["false_activation_rate"], 0)
        self.assertAlmostEqual(result["detection_accuracy"], 2 / 3)
        self.assertIsNone(result["post_training_behavior_change_rate"])
        self.assertEqual(result["post_training_behavior_change_denominator"], 0)
        self.assertIsNone(security_metrics([], "empty", True)["trigger_conditioned_behavior_rate"])

    def test_reward_clipping_and_unknown_semantics(self):
        s = SemanticResult(True, 1., "oracle", "ok")
        result = validation_reward(s, 1, 0, 0)
        self.assertEqual(result["reward"], 1.5)
        unknown = validation_reward(SemanticResult(False, 0, "unknown", ""), 1, 0, .1)
        self.assertLess(unknown["reward"], 0)
        with self.assertRaises(ValueError):
            validation_reward(s, float("nan"), 0, 0)


if __name__ == "__main__":
    unittest.main()
