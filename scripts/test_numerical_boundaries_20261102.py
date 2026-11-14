"""Targeted regressions for the newly introduced evidence comparator only."""
import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np
from scripts.numerical_boundaries_20261102 import compare

class ComparatorRegression(unittest.TestCase):
 def test_exact(self):
  a={'x':torch.tensor([1.,2.]),'rng':np.array([3,4],dtype=np.uint32)};r=compare(a,a);self.assertTrue(r['exact']);self.assertEqual(r['affected_elements'],0)
 def test_error_units(self):
  r=compare(torch.tensor([3.,4.]),torch.tensor([3.,5.]));self.assertFalse(r['exact']);self.assertEqual(r['affected_elements'],1);self.assertEqual(r['max_abs'],1.);self.assertAlmostEqual(r['max_tensor_relative_L2'],.2)
 def test_tolerance_does_not_replace_exact(self):
  r=compare(torch.tensor([1.],dtype=torch.float64),torch.tensor([1.+1e-8],dtype=torch.float64));self.assertFalse(r['exact']);self.assertTrue(r['tolerance_pass'])
 def test_missing_is_not_exact(self):
  r=compare({'x':torch.ones(2)},{'x':torch.ones(2),'missing':1});self.assertFalse(r['exact']);self.assertFalse(r['tolerance_pass'])
 def test_tolerance_reference_is_first_tensor(self):
  r=compare(torch.tensor([1.],dtype=torch.float64),torch.tensor([1.+1.010005e-5],dtype=torch.float64));self.assertFalse(r['tolerance_pass'])
 def test_zero_reference(self):
  r=compare(torch.zeros(2),torch.tensor([0.,1.]));self.assertIsNone(r['tensors'][0]['relative_L2']);self.assertTrue(r['tensors'][0]['zero_reference_nonzero_error']);self.assertEqual(r['undefined_tensor_relative_L2_count'],1);self.assertIsNone(r['max_tensor_relative_L2'])
if __name__=='__main__':unittest.main()
