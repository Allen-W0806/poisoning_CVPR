"""Focused regressions for leakage, input-range bugs, missing evidence, and learned actions."""
import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.mini_pilot.data import prepare_cohort, sha256, validate_cohort, write_json
from src.mini_pilot.reporting import METHODS, generate_reports


class WorkspaceFixture(unittest.TestCase):
    def setUp(self):
        self.base = Path(os.environ["PILOT_TEST_TMP"]).resolve()
        self.base.mkdir(parents=True, exist_ok=True)
        self.directory = Path(tempfile.mkdtemp(prefix="case_", dir=self.base)).resolve()

    def tearDown(self):
        target = self.directory.resolve()
        if target == self.base or not target.is_relative_to(self.base) or not target.name.startswith("case_"):
            raise RuntimeError("Unsafe test cleanup target")
        shutil.rmtree(target)


class CohortTests(WorkspaceFixture):
    def make_data(self):
        from PIL import Image
        records = []
        for index in range(8):
            path = self.directory / "data/raw" / f"{index}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (9 + index, 10), (20 * index, 1, 2)).save(path)
            records.append({"sample_id": str(index), "image_id": str(index),
                            "image_path": str(path.relative_to(self.directory)), "image_sha256": sha256(path),
                            "question": "fixture", "answers": ["fixture"]})
        write_json(self.directory / "data/processed/vqav2_content_v2/train.json", records)
        # If implementation opens the real historical test route, this deliberately fails parsing.
        (self.directory / "data/processed/vqav2_content_v2/test.json").write_text("SEALED_DO_NOT_READ")
        return records

    def test_disjoint_deterministic_no_test_pool_or_image_copy(self):
        self.make_data()
        sizes = {"train": 3, "validation": 2, "test": 2}
        first, audit, provenance = prepare_cohort(self.directory, self.directory / "data/mini", sizes, 42)
        second, _, _ = prepare_cohort(self.directory, self.directory / "data/mini", sizes, 42)
        self.assertEqual(first, second)
        self.assertEqual(audit["image_count"], 7)
        self.assertFalse(provenance["historical_test_or_prospective_pool_opened"])
        self.assertEqual(provenance["copied_image_bytes"], 0)
        self.assertFalse(list((self.directory / "data/mini").glob("*.png")))

    def test_changed_image_or_cross_split_duplicate_rejected(self):
        records = self.make_data()
        with self.assertRaises(ValueError):
            validate_cohort(self.directory, {"train": records[:2], "test": [records[0]]}, {"train": 2, "test": 1})
        (self.directory / records[0]["image_path"]).write_bytes(b"modified")
        with self.assertRaises(ValueError):
            validate_cohort(self.directory, {"train": records[:2]}, {"train": 2})

    def test_path_escape_rejected(self):
        record = {"image_path": "../outside.png", "image_sha256": "0" * 64}
        from src.mini_pilot.data import checked_image
        with self.assertRaises(ValueError):
            checked_image(self.directory, record)


class ReportTests(WorkspaceFixture):
    def test_absent_optional_plotter_preserves_completed_measurements(self):
        import sys
        from unittest.mock import patch
        from src.mini_pilot.reporting import make_figures
        state = {"status": "COMPLETED", "assessment": {"measured_fixture": [1, 2, 3]}}
        with patch.dict(sys.modules, {"matplotlib": None}):
            make_figures(self.directory, state)
        self.assertEqual(state["status"], "COMPLETED")
        self.assertEqual(state["assessment"]["measured_fixture"], [1, 2, 3])
        self.assertEqual(state["figures_status"], "PENDING_CPU_REPORTING_MATPLOTLIB_UNAVAILABLE")
        self.assertTrue((self.directory / "figures/PLOTTING_STATUS.md").is_file())

    def test_missing_metrics_never_reported_as_success_or_ASR(self):
        protocol = {"methods": list(METHODS), "seed": 42, "num_train_images": 32,
                    "num_val_images": 16, "num_test_images": 16, "max_inner_steps": 100,
                    "rsi_rounds": 3, "precision": "fp32"}
        generate_reports(self.directory, protocol, {"status": "FAILED", "failure": "fixture"})
        reports = self.directory / "reports"
        self.assertEqual(len(list(reports.iterdir())), 7)
        with (reports / "BASELINE_COMPARISON.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(row["ASR"] == row["ASR_Proxy"] == row["common_reward"] == "NOT TESTED" for row in rows))
        decision = (reports / "NEXT_EXPERIMENT_DECISION.md").read_text()
        self.assertNotIn("confirms", decision)
        self.assertNotIn("Scale up to", decision)


class RewardAndControllerTests(unittest.TestCase):
    def test_LPIPS_keeps_minus_one_one_and_batch_dimension(self):
        import torch
        from src.mini_pilot.reward import ProxyRewardCalculator
        reward = ProxyRewardCalculator.__new__(ProxyRewardCalculator)
        reward.device = torch.device("cpu")
        class Inputs(dict):
            def to(self, device):
                return self
        def processor(**kwargs):
            self.assertEqual(kwargs["images"][0].getpixel((0, 0)), (255, 255, 255))
            return Inputs()
        reward.processor = processor
        reward.clip_model = lambda **kwargs: SimpleNamespace(image_embeds=torch.tensor([[1., 0.]]),
                                                             text_embeds=torch.tensor([[1., 0.]]))
        def metric(original, edited):
            self.assertEqual(float(original.min()), -1.)
            self.assertEqual(float(edited.max()), 1.)
            return torch.full((1, 1, 1, 1), .25)
        reward.lpips_model = metric
        total, semantic, perceptual = reward.compute_reward(-torch.ones(1, 3, 8, 8), torch.ones(1, 3, 8, 8), "benign")
        self.assertEqual(tuple(total.shape), (1,))
        self.assertAlmostEqual(float(total[0]), 1.75)
        self.assertAlmostEqual(float(semantic[0]), 2.)
        self.assertAlmostEqual(float(perceptual[0]), -.25)

    def test_nonfinite_or_disabled_reward_term_rejected(self):
        import torch
        from src.mini_pilot.reward import ProxyRewardCalculator
        reward = ProxyRewardCalculator.__new__(ProxyRewardCalculator)
        images = torch.zeros(1, 3, 8, 8)
        with self.assertRaises(ValueError):
            reward.compute_reward(images, images, "benign", weight_stealth=0)
        with self.assertRaises(ValueError):
            reward.compute_reward(images, images + float("nan"), "benign")

    def test_learned_controller_really_changes_parameters_and_keeps_perceptual_term(self):
        import torch
        from src.mini_pilot.controller import LearnedController, controller_state
        torch.manual_seed(42)
        controller = LearnedController(123)
        config = {"lr": 1e-5, "weight_sem": 2., "weight_stealth": 1.}
        action, revised, probabilities = controller.choose(controller_state(0., .1, .2, 1e-5, 1.), config)
        self.assertIn(action, range(4))
        self.assertEqual(revised["weight_stealth"], 1.)
        receipt = controller.learn(.02, "validation-only fixture")
        self.assertGreater(receipt["controller_max_parameter_change"], 0.)
        self.assertAlmostEqual(sum(probabilities), 1., places=6)
        self.assertEqual(len(controller.state_dict()["memory"]), 1)
        with self.assertRaises(ValueError):
            controller.learn(.02, "no pending action")

    def test_exact_and_tolerance_outcomes_remain_distinct(self):
        import torch
        from src.mini_pilot.engine import compare_trees
        outcome = compare_trees({"state": torch.tensor([1.])}, {"state": torch.tensor([1. + 1e-7])})
        self.assertFalse(outcome["exact"])
        self.assertTrue(outcome["tolerance"])
        self.assertEqual(outcome["different_elements"], 1)


if __name__ == "__main__":
    unittest.main()
