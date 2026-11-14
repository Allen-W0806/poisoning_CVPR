"""The pending learned action, baseline and next stochastic decision survive disk resume."""
import copy
import unittest


class ControllerContinuationTests(unittest.TestCase):
    def test_validation_boundary_still_selects_the_next_real_training_source(self):
        from scripts.continue_mini_pilot import next_source_index
        schedule = [list(range(34)), list(range(34, 67)), list(range(67, 100))]
        self.assertEqual(next_source_index(schedule, 0, 34), 34)
        self.assertEqual(next_source_index(schedule, 1, 67), 67)
        self.assertEqual(next_source_index(schedule, 2, 82), 82)
        self.assertIsNone(next_source_index(schedule, 2, 100))
        with self.assertRaises(ValueError):
            next_source_index(schedule, 2, 66)

    def test_pending_action_and_next_learning_match_uninterrupted_controller(self):
        import torch
        from src.mini_pilot.controller import LearnedController, controller_state
        from scripts.continue_mini_pilot import restore_controller
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
