"""Regression of F23, no model inference or synthetic human review records."""
import copy,types,unittest
from src.rl.diffusion_trajectory import assert_fresh_batch
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer

class FreshBatchTests(unittest.TestCase):
    def setUp(self):
        self.contract={'task':'benign-test'};self.old={'already'}
        self.t={'trajectory_id':'fresh','policy_version':3,'contract':self.contract}
    def test_duplicate_in_one_batch_rejected_without_mutating_consumed(self):
        with self.assertRaises(ValueError):assert_fresh_batch([self.t,copy.deepcopy(self.t)],3,self.contract,self.old)
        self.assertEqual(self.old,{'already'})
    def test_prior_consumed_rejected(self):
        with self.assertRaises(ValueError):assert_fresh_batch([{**self.t,'trajectory_id':'already'}],3,self.contract,self.old)
    def test_stale_or_wrong_contract_rejected(self):
        for t in [{**self.t,'policy_version':2},{**self.t,'contract':{'task':'other'}}]:
            with self.assertRaises(ValueError):assert_fresh_batch([t],3,self.contract,self.old)
    def test_distinct_trajectories_can_share_condition(self):
        assert_fresh_batch([self.t,{**self.t,'trajectory_id':'fresh2'}],3,self.contract,self.old)
        self.assertEqual(self.old,{'already'})
    def test_real_trainer_rejects_before_model_or_optimizer_access(self):
        trainer=DiffusionRLTrainer.__new__(DiffusionRLTrainer)
        trainer.config={'logical_batch_size':2};trainer.policy=types.SimpleNamespace(version=3,contract=self.contract);trainer.consumed=set()
        with self.assertRaisesRegex(ValueError,'already consumed'):
            trainer.update([(self.t,None,1.),(copy.deepcopy(self.t),None,1.)])
        self.assertEqual(trainer.consumed,set())
if __name__=='__main__':unittest.main()
