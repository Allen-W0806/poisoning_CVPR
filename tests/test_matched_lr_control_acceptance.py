import unittest
from scripts.check_pilot_matched_lr_20261102 import matched_control_exact
class MatchedControl(unittest.TestCase):
 def test_all_exact(self):self.assertTrue(matched_control_exact(True,True,True,True,True))
 def test_each_prerequisite_required(self):
  for i in range(5):
   values=[True]*5;values[i]=False;self.assertFalse(matched_control_exact(*values))
 def test_no_prerequisites(self):self.assertFalse(matched_control_exact(False,False,False,False,False))
if __name__=='__main__':unittest.main()
