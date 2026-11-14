"""Numerical and integrity regressions; tiny random UNet is never efficacy evidence."""
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from diffusers import DDIMScheduler, UNet2DConditionModel

from src.mini_pilot.engine import PilotTrainer, compare_trees
from src.rl.diffusion_policy import DiffusionPolicy
from src.rl.diffusion_trajectory import transition
from src.training.generator_adapter import add_attention_lora, adapter_state, load_adapter_state
from src.mini_pilot.benign_validation import RecipeSelector, lighting_metrics, paired_summary, content_box


def scheduler():
    result = DDIMScheduler(beta_start=.00085, beta_end=.012, beta_schedule="scaled_linear",
                           clip_sample=False, set_alpha_to_one=False, steps_offset=1)
    result.set_timesteps(4)
    return result


def tiny_policy():
    torch.manual_seed(42)
    unet = UNet2DConditionModel(sample_size=4, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(16,), down_block_types=("CrossAttnDownBlock2D",),
            up_block_types=("CrossAttnUpBlock2D",), cross_attention_dim=16,
            norm_num_groups=4, attention_head_dim=4)
    pipe = SimpleNamespace(unet=unet, vae=torch.nn.Identity(), text_encoder=torch.nn.Identity(),
                           scheduler=scheduler())
    add_attention_lora(pipe, 2, 4)
    policy = SimpleNamespace(pipe=pipe, config={"eta": 1.}, editor=SimpleNamespace(device=torch.device("cpu")),
                             version=0, contract={"task": "tiny-random-unet-engineering-only"})
    policy.predict = lambda state, t, condition: unet(state, t, encoder_hidden_states=condition["hidden"]).sample
    return policy


@torch.no_grad()
def episodes(policy, name="tiny"):
    generator = torch.Generator().manual_seed(123)
    condition = {"sample_id": "a", "image_id": "a", "instruction": "benign fixture",
                 "hidden": torch.randn(1, 3, 16, generator=generator)}
    state = torch.randn(1, 4, 4, 4, generator=generator)
    trajectory = {"policy_version": policy.version, "contract": copy.deepcopy(policy.contract),
                   "trajectory_id": name, **{k: condition[k] for k in ("sample_id", "image_id", "instruction")},
                   "steps": []}
    for t in policy.pipe.scheduler.timesteps:
        action, logp, _, _ = transition(policy.pipe.scheduler, policy.predict(state, t, condition), t, state,
                                        noise=torch.randn(state.shape, generator=generator))
        trajectory["steps"].append({"t": int(t), "state": state, "action": action, "logp": logp})
        state = action
    return [(trajectory, condition, .3)]


class SamplerRegression(unittest.TestCase):
    def setUp(self):
        self.scheduler = scheduler()
        self.state = torch.zeros(1, 4, 4, 4)
        self.t = self.scheduler.timesteps[0]

    def test_reject_broadcast_noise(self):
        with self.assertRaises(ValueError):
            transition(self.scheduler, self.state, self.t, self.state, noise=torch.ones(1))

    def test_reject_nonfinite_noise_prediction_and_action(self):
        for role in ("noise", "epsilon", "action"):
            invalid = torch.full_like(self.state, float("nan"))
            arguments = {"noise": torch.ones_like(self.state)}
            prediction = invalid if role == "epsilon" else self.state
            if role == "noise": arguments["noise"] = invalid
            if role == "action": arguments = {"action": invalid}
            with self.assertRaises(ValueError):
                transition(self.scheduler, prediction, self.t, self.state, **arguments)

    def test_reject_unscheduled_timestep(self):
        with self.assertRaises(ValueError):
            transition(self.scheduler, self.state, 999, self.state, noise=self.state)

    def test_terminal_stochastic_density_and_replay(self):
        for t in self.scheduler.timesteps:
            action, first, mean, std = transition(self.scheduler, self.state, t, self.state,
                                                 noise=torch.ones_like(self.state))
            _, replay, _, _ = transition(self.scheduler, self.state, t, self.state, action=action)
            self.assertGreater(float(std), 0)
            torch.testing.assert_close(first, replay, rtol=0, atol=0)
            torch.testing.assert_close(first, torch.distributions.Normal(mean, std).log_prob(action).flatten(1).mean(1))

    def test_fp32_policy_does_not_enable_bf16_autocast(self):
        class UNet:
            def __call__(self, inputs, *args, **kwargs):
                assert inputs.dtype == torch.float32
                assert not torch.is_autocast_enabled("cpu")
                return (inputs[:, :4],)
        policy = DiffusionPolicy.__new__(DiffusionPolicy)
        policy.editor = SimpleNamespace(device=torch.device("cpu"), dtype=torch.float32,
                config=SimpleNamespace(guidance_scale=5., image_guidance_scale=1.5))
        policy.config, policy.unet_calls = {}, 0
        policy.pipe = SimpleNamespace(unet=UNet(), scheduler=self.scheduler)
        output = policy.predict(self.state, self.t, {"image": self.state, "hidden": torch.zeros(3, 3, 16)})
        self.assertEqual(output.dtype, torch.float32)

    def test_terminal_return_score_gradient_matches_gaussian_expectation(self):
        import math
        state = torch.zeros(65536, 1, 1, 1)
        parameter = torch.tensor(.02, requires_grad=True)
        noise = torch.randn(state.shape, generator=torch.Generator().manual_seed(73))
        action, logp, mean, std = transition(self.scheduler, parameter.expand_as(state), self.t, state, noise=noise)
        estimate = torch.autograd.grad((action.detach().sin().flatten()*logp).mean(), parameter, retain_graph=True)[0]
        coefficient = torch.autograd.grad(mean.mean(),parameter)[0]
        expected = math.exp(-float(std)**2/2)*math.cos(float(mean.detach().mean()))*float(coefficient)
        self.assertAlmostEqual(float(estimate),expected,delta=.03*abs(expected))


class BenignMetricsAndSearch(unittest.TestCase):
    def test_content_box_matches_pillow_rounding_with_odd_padding(self):
        from PIL import Image, ImageChops, ImageOps
        for width,height in [(499,600),(600,499),(320,711),(711,320),(256,256),(343,25088)]:
            source=Image.new('RGB',(width,height),'black')
            padded=ImageOps.pad(source,(256,256),color='white',method=3)
            actual=ImageChops.difference(padded,Image.new('RGB',(256,256),'white')).getbbox()
            self.assertEqual(content_box(source,256),list(actual))

    def test_cpu_receipt_covers_its_runner_and_every_executed_test(self):
        from scripts.test_benign_ddpo_audit import source_hashes, TEST_MODULES
        hashes=source_hashes()
        self.assertIn('scripts/test_benign_ddpo_audit.py',hashes)
        for name in TEST_MODULES:
            self.assertIn(name.replace('.','/')+'.py',hashes)

    def test_identity_brighten_and_darken_are_distinguishable(self):
        from PIL import Image
        values = np.tile(np.arange(64,dtype=np.uint8)[None,:,None],(64,1,3))+64
        source = Image.fromarray(values)
        bright = Image.fromarray(values+10)
        dark = Image.fromarray(values-10)
        self.assertAlmostEqual(lighting_metrics(source,source)['training_reward_proxy'],0)
        self.assertGreater(lighting_metrics(source,bright)['training_reward_proxy'],0)
        self.assertLess(lighting_metrics(source,dark)['training_reward_proxy'],0)
        self.assertGreater(lighting_metrics(source,dark,delta=-.05)['training_reward_proxy'],0)

    def test_ucb_initial_exploration_and_observation_counts(self):
        selector = RecipeSelector('ucb_search',[5e-6,1e-5],1e-5)
        self.assertEqual(selector.propose(0),5e-6)
        selector.observe(5e-6,.01)
        self.assertEqual(selector.propose(1),1e-5)
        selector.observe(1e-5,-.01)
        self.assertEqual(selector.propose(2),5e-6)
        self.assertEqual(sum(selector.counts),2)
        with self.assertRaises(ValueError): selector.observe(1e-5,float('nan'))

    def test_fixed_rule_does_not_consume_observations_to_choose(self):
        selector = RecipeSelector('fixed_rule',[5e-6,1e-5],1e-5)
        selector.observe(5e-6,-100)
        self.assertEqual([selector.propose(i) for i in range(3)],[5e-6,1e-5,5e-6])

    def test_pairing_rejects_mismatched_seed(self):
        first=[{'sample_id':'x','seed':1,'training_reward_proxy':0}]
        with self.assertRaises(ValueError): paired_summary(first,[{**first[0],'seed':2}])


class CheckpointRegression(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.policy = tiny_policy()
        self.trainer = PilotTrainer(self.policy, 1e-5, .5)

    def test_missing_extra_wrong_dtype_and_nonfinite_adapter_rejected_without_mutation(self):
        initial = adapter_state(self.policy.pipe.unet)
        for mode in ("missing", "extra", "dtype", "nan", "shape"):
            candidate = copy.deepcopy(initial)
            key = next(iter(candidate))
            if mode == "missing": candidate.pop(key)
            if mode == "extra": candidate["fake.lora_A.weight"] = candidate[key]
            if mode == "dtype": candidate[key] = candidate[key].bfloat16()
            if mode == "nan": candidate[key].fill_(float("nan"))
            if mode == "shape": candidate[key] = candidate[key].flatten()
            with self.assertRaises(ValueError): load_adapter_state(self.policy.pipe.unet, candidate)
            self.assertTrue(compare_trees(initial, adapter_state(self.policy.pipe.unet))["exact"])

    def test_cpu_update_optimizer_and_completed_next_step_restore_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved"
            self.trainer.update(episodes(self.policy, "one"), keep_gradients=True)
            self.trainer.save(path)
            saved = self.trainer.state()
            self.trainer.update(episodes(self.policy, "two"), keep_gradients=True)
            expected = self.trainer.witness()
            self.trainer.restore(path)
            self.assertTrue(compare_trees(saved, self.trainer.state())["exact"])
            self.trainer.update(episodes(self.policy, "two"), keep_gradients=True)
            self.assertTrue(compare_trees(expected, self.trainer.witness())["exact"])

    def test_corrupted_bytes_and_changed_clipping_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "saved"
            self.trainer.save(path)
            changed = PilotTrainer(self.policy, 1e-5, 1.)
            with self.assertRaises(ValueError): changed.restore(path)
            with (path / "training_state.pt").open("ab") as stream: stream.write(b"corruption")
            with self.assertRaises(ValueError): self.trainer.restore(path)

    def test_empty_nonfinite_reward_and_duplicate_rejected_before_optimizer(self):
        for batch in ([], episodes(self.policy) * 2,
                      [(a, b, float("nan")) for a, b, _ in episodes(self.policy)]):
            with self.assertRaises((ValueError, FloatingPointError)): self.trainer.update(batch)
            self.assertEqual(self.trainer.step, 0)

    def test_numpy_rng_comparison(self):
        self.assertTrue(compare_trees(np.random.get_state(), copy.deepcopy(np.random.get_state()))["exact"])


if __name__ == "__main__":
    unittest.main()
