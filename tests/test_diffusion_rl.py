import unittest
import torch
from src.rl.diffusion_trajectory import transition,assert_fresh
from src.rl.diffusion_policy import cfg_inputs,guided_prediction
from src.rl.reward_normalization import fixed_advantage

class DiffusionRLTests(unittest.TestCase):
    def scheduler(self,**kw):
        from diffusers import DDIMScheduler
        s=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule="scaled_linear",clip_sample=False,
            steps_offset=1,set_alpha_to_one=kw.get("set_alpha_to_one",False))
        s.set_timesteps(10); return s

    def test_ddim_matches_official_all_transitions(self):
        s=self.scheduler(); g=torch.Generator().manual_seed(42)
        x=torch.randn((2,4,8,8),generator=g); e=torch.randn(x.shape,generator=g); noise=torch.randn(x.shape,generator=g)
        for t in s.timesteps:
            a,lp,mean,std=transition(s,e,t,x,noise=noise)
            expected=s.step(e,t,x,eta=1.,variance_noise=noise).prev_sample
            self.assertTrue(torch.allclose(a,expected,atol=1e-6,rtol=1e-6))
            analytic=torch.distributions.Normal(mean,std).log_prob(a).flatten(1).mean(1)
            self.assertTrue(torch.allclose(lp,analytic,atol=1e-6)); self.assertEqual(lp.shape,(2,))
            self.assertGreater(float(std),0)

    def test_reject_deterministic_last_transition(self):
        s=self.scheduler(set_alpha_to_one=True); x=torch.zeros((1,4,8,8))
        with self.assertRaises(ValueError): transition(s,x,s.timesteps[-1],x,noise=x)

    def test_probability_gradient_and_reward_direction(self):
        s=self.scheduler(); x=torch.zeros((1,4,8,8)); p=torch.tensor(.02,requires_grad=True)
        # Fixed action and reversed rewards must reverse the actual score gradient.
        a=torch.ones_like(x)*.4
        _,lp,_,_=transition(s,p.expand_as(x),s.timesteps[0],x,action=a)
        plus=torch.autograd.grad(-fixed_advantage(.5)*lp.sum(),p,retain_graph=True)[0]
        minus=torch.autograd.grad(-fixed_advantage(-.5)*lp.sum(),p)[0]
        self.assertNotEqual(float(plus),0.); self.assertEqual(float(plus),-float(minus))

    def test_conditioning_layout_and_guidance(self):
        z=torch.ones((1,4,8,8)); image=z*7
        inputs=cfg_inputs(z,image)
        self.assertEqual(inputs.shape,(3,8,8,8))
        self.assertTrue(torch.equal(inputs[0,4:],image[0])); self.assertTrue(torch.equal(inputs[1,4:],image[0]))
        self.assertEqual(float(inputs[2,4:].sum()),0)
        pred=torch.cat([z*3,z*2,z])
        self.assertTrue(torch.equal(guided_prediction(pred,7.5,1.5),z*10))

    def test_stale_consumed_and_contract_rejected(self):
        t={"policy_version":0,"contract":{"a":1},"trajectory_id":"x"}
        assert_fresh(t,0,{"a":1},set())
        for version,contract,consumed in [(1,{"a":1},set()),(0,{"a":2},set()),(0,{"a":1},{"x"})]:
            with self.assertRaises(ValueError): assert_fresh(t,version,contract,consumed)

    def test_fixed_scale_not_batch_normalized(self):
        self.assertEqual(fixed_advantage(-.4),-.4)
        for x in [float("nan"),-2,2]:
            with self.assertRaises(ValueError): fixed_advantage(x)

if __name__=="__main__": unittest.main()
