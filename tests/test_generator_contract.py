import unittest
from src.training.generator_optimizer_interface import GeneratorOptimizerContract
from src.editing.editor_config import PretrainedEditorConfig


class GeneratorContractTest(unittest.TestCase):
    def test_unsupported_rl_fails_explicitly(self):
        contract = GeneratorOptimizerContract()
        self.assertFalse(contract.reward_based_supported)
        with self.assertRaises(NotImplementedError):
            contract.reward_update(rewards=[1] * 10)

    def test_editor_backend_parameter_validation(self):
        with self.assertRaises(ValueError):
            PretrainedEditorConfig(revision="").validate()
        with self.assertRaises(ValueError):
            PretrainedEditorConfig(revision="known", inference_batch_size=10).validate()
        self.assertEqual(PretrainedEditorConfig(revision="known").validate().image_resolution, 512)
