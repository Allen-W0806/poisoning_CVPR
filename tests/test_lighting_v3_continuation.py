"""The pending learned action, baseline and next stochastic decision survive disk resume."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class ControllerContinuationTests(unittest.TestCase):
    def test_only_cooperative_failed_and_absent_primary_is_admitted(self):
        from scripts.continue_lighting_v3 import require_primary_terminal, COOPERATIVE_TIMEOUT
        with tempfile.TemporaryDirectory() as directory:
            campaign=Path(directory); receipt=campaign/"submissions/internal_v3/submission.json"
            receipt.parent.mkdir(parents=True)
            receipt.write_text(json.dumps({"stage":"internal_v3","job_id":"88957"}))
            state={"status":"FAILED","failure":COOPERATIVE_TIMEOUT}
            with patch("scripts.continue_lighting_v3.subprocess.run",side_effect=[
                SimpleNamespace(stdout="88957|FAILED|\n"),SimpleNamespace(stdout="")]):
                self.assertEqual(require_primary_terminal(campaign/"internal_pilot",state),"88957")
            with patch("scripts.continue_lighting_v3.subprocess.run",side_effect=[
                SimpleNamespace(stdout="88957|FAILED|\n"),SimpleNamespace(stdout="889570\n88958\n")]) as command:
                self.assertEqual(require_primary_terminal(campaign/"internal_pilot",state),"88957")
                self.assertNotIn("-j",command.call_args_list[-1].args[0])
            for status,queued in [("RUNNING","88957\n"),("FAILED","88957\n"),("COMPLETED","")]:
                with patch("scripts.continue_lighting_v3.subprocess.run",side_effect=[
                    SimpleNamespace(stdout=f"88957|{status}|\n"),SimpleNamespace(stdout=queued)]):
                    with self.assertRaises(RuntimeError):require_primary_terminal(campaign/"internal_pilot",state)
            with self.assertRaises(RuntimeError):
                require_primary_terminal(campaign/"internal_pilot",{"status":"FAILED","failure":"TimeoutError: unrelated I/O"})

    def test_new_controller_matches_original_rng_order_and_partial_preserves_rng(self):
        import torch
        from scripts.continue_lighting_v3 import restore_arm
        from src.mini_pilot.controller_v3 import LearnedController
        torch.manual_seed(42);saved_rng=torch.get_rng_state().clone()
        class Trainer:
            def restore(self,unused):torch.set_rng_state(saved_rng)
        torch.set_rng_state(saved_rng);expected=LearnedController(142)
        expected_rng=torch.get_rng_state().clone()
        torch.manual_seed(9876)
        actual=restore_arm(Trainer(),None,"rl_rsi",False,42,LearnedController)
        self.assertTrue(torch.equal(torch.get_rng_state(),expected_rng))
        for a,b in zip(actual.policy.parameters(),expected.policy.parameters()):self.assertTrue(torch.equal(a,b))
        restore_arm(Trainer(),None,"rl_rsi",True,42,LearnedController)
        self.assertTrue(torch.equal(torch.get_rng_state(),saved_rng))

    def test_validation_boundary_still_selects_the_next_real_training_source(self):
        from scripts.continue_lighting_v3 import next_source_index
        schedule = [list(range(34)), list(range(34, 67)), list(range(67, 100))]
        self.assertEqual(next_source_index(schedule, 0, 34), 34)
        self.assertEqual(next_source_index(schedule, 1, 67), 67)
        self.assertEqual(next_source_index(schedule, 2, 82), 82)
        self.assertIsNone(next_source_index(schedule, 2, 100))
        with self.assertRaises(ValueError):
            next_source_index(schedule, 2, 66)

    def test_pending_action_and_next_learning_match_uninterrupted_controller(self):
        import torch
        from src.mini_pilot.controller_v3 import LearnedController, controller_state
        from scripts.continue_lighting_v3 import restore_controller
        torch.manual_seed(42)
        original = LearnedController(142)
        config = {"lr": 1e-5, "weight_sem": 2., "weight_stealth": 1.}
        feature = controller_state(.34, .2, .3, 1e-5, .66)
        for gain in [-.02, .034]:
            _, config, _ = original.choose(feature, config)
            original.learn(gain, "validation-only resume fixture")
        original.choose(feature, config)
        saved = copy.deepcopy({"policy": original.policy.state_dict(), "optimizer": original.optimizer.state_dict(),
                               "generator_rng": original.generator.get_state(), "memory": original.memory,
                               "pending": original.pending})
        restored = LearnedController(999)
        restore_controller(restored, saved)
        self.assertEqual(restored.baseline, original.baseline)
        self.assertEqual(restored.pending[1], original.pending[1])
        self.assertTrue(torch.equal(restored.pending[0], original.pending[0]))
        original.learn(-.013, "same causal outcome")
        restored.learn(-.013, "same causal outcome")
        for first, second in zip(original.policy.parameters(), restored.policy.parameters()):
            self.assertTrue(torch.equal(first, second))
        self.assertEqual(original.choose(feature, config)[0], restored.choose(feature, config)[0])


if __name__ == "__main__":
    unittest.main()
