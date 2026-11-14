"""Targeted CPU regression of the opt-in preinitialization guard."""
import os,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from src.rl.cuda_execution_guard import configure_strict_diffusion_backward
import src.models.pretrained_editor_loader as loader

class Guard(unittest.TestCase):
 def test_preinit_enforces_blocking(self):
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}),patch('torch.cuda.is_initialized',return_value=False):self.assertEqual(configure_strict_diffusion_backward()['active_value'],'1');self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'],'1')
 def test_late_change_fails_closed(self):
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}),patch('torch.cuda.is_initialized',return_value=True):
   with self.assertRaisesRegex(RuntimeError,'before CUDA initialization'):configure_strict_diffusion_backward()
   self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'],'0')
 def test_already_enabled_is_idempotent(self):
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'1'}),patch('torch.cuda.is_initialized',return_value=True):self.assertEqual(configure_strict_diffusion_backward()['active_value'],'1')
 def test_CPU_RNG_unchanged(self):
  before=torch.get_rng_state()
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}),patch('torch.cuda.is_initialized',return_value=False):configure_strict_diffusion_backward()
  self.assertTrue(torch.equal(before,torch.get_rng_state()))
 def test_inference_default_does_not_enable_guard(self):
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}),patch.object(loader,'pretrained_config',return_value='fixture'),patch.object(loader,'load_editor',return_value='fixture') as load:
   self.assertEqual(loader.load_pretrained_editor('fixture'), 'fixture');self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'],'0');load.assert_called_once_with('instructpix2pix',config='fixture')
 def test_training_optin_sets_guard_before_loading(self):
  def load(*a,**k):self.assertEqual(os.environ['CUDA_LAUNCH_BLOCKING'],'1');return 'fixture'
  with patch.dict(os.environ,{'CUDA_LAUNCH_BLOCKING':'0'}),patch('torch.cuda.is_initialized',return_value=False),patch.object(loader,'pretrained_config',return_value='fixture'),patch.object(loader,'load_editor',side_effect=load):self.assertEqual(loader.load_pretrained_editor('fixture',strict_backward=True),'fixture')
if __name__=='__main__':unittest.main()
