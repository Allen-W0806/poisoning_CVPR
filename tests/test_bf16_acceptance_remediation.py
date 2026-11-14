import copy,unittest
from scripts.run_bf16_adjoints_20261102 import strict_pair_acceptance
class Acceptance(unittest.TestCase):
 def setUp(self):self.p=dict(initial_state_exact=True,postclip={'exact':True},adapter_exact=True,optimizer_exact=True,scheduler_exact=True,RNG_exact=True)
 def test_all_equal(self):self.assertTrue(strict_pair_acceptance([self.p],{'exact':True}))
 def test_initial_mismatch_never_passes(self):self.assertFalse(strict_pair_acceptance([{**self.p,'initial_state_exact':False}],{'exact':True}))
 def test_raw_mismatch_never_passes(self):self.assertFalse(strict_pair_acceptance([self.p],{'exact':False}))
 def test_missing_pair_never_passes(self):self.assertFalse(strict_pair_acceptance([],{'exact':True}))
 def test_each_state_field_is_required(self):
  for k in ['adapter_exact','optimizer_exact','scheduler_exact','RNG_exact']:
   self.assertFalse(strict_pair_acceptance([{**self.p,k:False}],{'exact':True}))
if __name__=='__main__':unittest.main()
