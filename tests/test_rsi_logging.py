import json
import unittest
from src.utils.scratch import scratch_directory
from src.utils.logging import Experiment
from src.rsi.controller import RSIController


class RSILoggingTest(unittest.TestCase):
    def test_heldout_and_acceptance(self):
        with scratch_directory() as root:
            r = RSIController(root / "revisions", {"batch_size": 10, "learning_rate": 1e-5}, .5, "val_hash", .1)
            first = r.revise({"learning_rate": 5e-5}, .6, .01, "val_hash")
            raw = (root / "revisions/revision_0001.json").read_bytes()
            self.assertTrue(first["accepted"])
            self.assertFalse(r.revise({"batch_size": 20}, .55, .01, "val_hash")["accepted"])
            self.assertEqual(raw, (root / "revisions/revision_0001.json").read_bytes())
            self.assertEqual(r.state["training_config"]["batch_size"], 10)
            self.assertFalse(r.revise({"batch_size": 32}, .9, .2, "val_hash")["accepted"])
            self.assertAlmostEqual(r.state["cost_gpu_hours"], .22)
            with self.assertRaises(ValueError):
                r.revise({"learning_rate": 1e-4}, .9, .01, "val_hash", split="test")
            with self.assertRaises(ValueError):
                r.revise({"metric": "something_else"}, .9, .01, "val_hash")
            with self.assertRaises(ValueError):
                r.revise({"learning_rate": 1e-4}, .9, .01, "wrong_val")

    def test_unique_and_immutable_runs(self):
        with scratch_directory() as root:
            with Experiment(root, "tiny", {"seed": 42, "learning_rate": 1e-5}) as a:
                a.metrics({"loss": .1})
            original = (a.path / "config.yaml").read_bytes()
            with Experiment(root, "tiny", {"seed": 123, "learning_rate": 1e-4}) as b:
                b.metrics({"loss": .2})
            self.assertNotEqual(a.path, b.path)
            self.assertEqual(original, (a.path / "config.yaml").read_bytes())
            self.assertEqual(json.loads((a.path / "metrics.json").read_text())["loss"], .1)
            self.assertEqual(json.loads((b.path / "metrics.json").read_text())["loss"], .2)
            index = (root / "EXPERIMENT_INDEX.md").read_text()
            self.assertIn(a.id, index)
            self.assertIn(b.id, index)
            self.assertEqual(len((root / "experiments/registry.jsonl").read_text().splitlines()), 2)
            for name in ["config.yaml", "manifest.json", "metrics.json", "metrics.csv", "run.log",
                         "environment.txt", "git_commit.txt", "status.json", "summary.md", "checkpoints"]:
                self.assertTrue((a.path / name).exists(), name)
            with self.assertRaises(RuntimeError):
                a.metrics({"loss": 9})
            with self.assertRaises(ValueError):
                Experiment.attach_active(root, a.path)


if __name__ == "__main__":
    unittest.main()
