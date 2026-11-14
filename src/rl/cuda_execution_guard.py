"""Experimental launch configuration, NOT a verified numerical repair.

The candidate passed captured replay but failed ordinary paired verification.
It is not enabled by default or by any production training caller.
"""
import os
import torch

def configure_strict_diffusion_backward():
 initialized=torch.cuda.is_initialized();previous=os.environ.get('CUDA_LAUNCH_BLOCKING')
 if initialized and previous!='1':
  raise RuntimeError('Strict diffusion backward requires CUDA_LAUNCH_BLOCKING=1 before CUDA initialization; restart the process with the guard enabled')
 os.environ['CUDA_LAUNCH_BLOCKING']='1'
 return dict(guard='cuda_launch_blocking_v1',CUDA_initialized_before=initialized,previous_value=previous,active_value='1',scientific_loss_changed=False,precision_changed=False)
