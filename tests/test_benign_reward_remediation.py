import copy
import math
import unittest
from src.rl.reward import RewardWeights, validation_reward
from src.rl.reward_legacy_v1 import validation_reward as legacy
from src.rl.benign_reward_v3 import feedback, caller_feedback, require_episode_reward_release
from src.rl.reward_normalization import fixed_advantage
from src.editing.semantic_validator import SemanticResult

CFG = dict(semantic_weight=1., answer_consistency_weight=.5, distortion_weight=.2,
           compute_cost_weight=.1, uncertainty_weight=1., normalization='fixed_reference',
           backdoor_asr_training=False)


def semantic(verified=True, score=1., provisional=False):
    return dict(verified=verified, score=score, provisionally_valid=provisional,
                verifier='constructed_unit_fixture_not_real_human', reason='unit fixture')


class BenignRewardRemediation(unittest.TestCase):
    def test_legacy_arithmetic_backward_compatible(self):
        for verified in [True, False]:
            for s in [0., .5, 1.]:
                for a in [0., 1.]:
                    for d in [0., .7, 1.]:
                        label=SemanticResult(verified,s,'fixture','')
                        self.assertEqual(validation_reward(label,a,d,1/3),legacy(label,a,d,1/3))

    def test_every_active_weight_is_bound(self):
        c={**CFG, 'semantic_weight':.7, 'answer_consistency_weight':.3,
           'distortion_weight':.4, 'compute_cost_weight':.2, 'uncertainty_weight':.8}
        r=feedback(c,copy.deepcopy(c),semantic(False,.5,True),.8,.2,.4)
        self.assertEqual(r['executed_weights'],r['recorded_weights'])
        self.assertAlmostEqual(r['reward_unclipped'],.7*.5+.3*.8-.4*.2-.2*.4-.8)
        self.assertEqual(r['executed_yaml_config'],c)

    def test_identity_gets_consistency_credit(self):
        r=caller_feedback(CFG,semantic(),{'score':1.},{'distortion':0.},10,.2)
        self.assertEqual(r['answer_consistency'],1.)
        self.assertAlmostEqual(r['reward'],1.5-.1/3)
        self.assertFalse(r['training_eligible'])
        self.assertEqual(r['raw_components']['answer_consistency'],1.)

    def test_valid_edit_and_invalid_edit(self):
        good=feedback(CFG,CFG,semantic(),1.,.1,1/3)
        bad=feedback(CFG,CFG,semantic(True,0.),1.,.1,1/3)
        self.assertGreater(good['reward'],bad['reward'])
        self.assertEqual(bad['raw_components']['answer_consistency'],1.)
        self.assertEqual(bad['effective_components']['answer_consistency'],0.)

    def test_unknown_and_ambiguous_never_independent_truth(self):
        for s,p in [(0.,False),(.5,True)]:
            r=feedback(CFG,CFG,semantic(False,s,p),1.,.1,1/3)
            self.assertFalse(r['semantic_verified']);self.assertFalse(r['semantic_construct_validated'])
            self.assertFalse(r['training_eligible'])

    def test_missing_labels_and_metric_fields_fail(self):
        for v in [{},None,{'score':1.}]:
            with self.assertRaises(ValueError):feedback(CFG,CFG,v,1.,0.,0.)
        with self.assertRaises(ValueError):caller_feedback(CFG,semantic(),{}, {'distortion':0.},10)
        with self.assertRaises(ValueError):caller_feedback(CFG,semantic(),{'score':1.},{},10)

    def test_missing_extra_nonfinite_or_attack_config_fails(self):
        cases=[]
        for k in CFG:
            x=dict(CFG);del x[k];cases.append(x)
        cases += [{**CFG,'extra':1},{**CFG,'semantic_weight':float('nan')},
                  {**CFG,'compute_cost_weight':-1.},{**CFG,'backdoor_asr_training':True},
                  {**CFG,'normalization':'batch_standardized'}]
        for c in cases:
            with self.subTest(c=c),self.assertRaises(ValueError):RewardWeights.from_config(c)
        with self.assertRaises(ValueError):feedback(CFG,{**CFG,'semantic_weight':.9},semantic(),1.,0.,0.)

    def test_nonfinite_out_of_range_and_conflicting_labels_fail(self):
        for value in [None,float('nan'),float('inf'),-1.,1.1,True]:
            for idx in range(3):
                values=[1.,0.,0.];values[idx]=value
                with self.assertRaises(ValueError):feedback(CFG,CFG,semantic(),*values)
        with self.assertRaises(ValueError):feedback(CFG,CFG,semantic(True,1.,True),1.,0.,0.)

    def test_saturation_keeps_raw_and_constant_batch_is_finite(self):
        low=feedback(CFG,CFG,semantic(True,0.),0.,1.,1.)
        self.assertTrue(low['saturated_low']);self.assertEqual(low['reward'],-1.)
        self.assertEqual(low['reward_unclipped'],-1.3)
        high_cfg={**CFG,'semantic_weight':2.}
        high=feedback(high_cfg,high_cfg,semantic(),1.,0.,0.)
        self.assertTrue(high['saturated_high']);self.assertEqual(high['reward_unclipped'],2.5)
        self.assertEqual([fixed_advantage(low['reward']) for _ in range(16)],[-1.]*16)

    def test_units_resource_logs_and_guard(self):
        r=caller_feedback(CFG,semantic(),{'score':1.},{'distortion':0.},10,.25)
        self.assertEqual(r['component_units']['cost_proxy'],'denoising_steps_div_30')
        self.assertEqual(r['resource_measurements']['generation_wall_seconds'],.25)
        self.assertIsNone(r['resource_measurements']['allocated_gpu_seconds'])
        for flag in [False,True]:
            with self.assertRaises(RuntimeError):
                require_episode_reward_release([({'reward':{**r,'training_eligible':flag}},None,1.)])

    def test_noop_preference_is_characterized_not_hidden(self):
        identity=caller_feedback(CFG,semantic(),{'score':1.},{'distortion':0.},10)
        provisional=caller_feedback(CFG,semantic(False,.5,True),{'score':1.},{'distortion':.1},10)
        self.assertGreater(identity['reward'],provisional['reward'])
        self.assertFalse(identity['semantic_construct_validated'])


if __name__=='__main__':unittest.main()
