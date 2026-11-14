"""Versioned sidecars bind active LR and outer-scheduler state to checkpoint bytes.

The original contract config remains an immutable initial-design identity.
The optimizer/scheduler in serialized state defines the active revision.
"""
import hashlib,json,math
from pathlib import Path

def encode(value):
 if isinstance(value,float):
  if math.isnan(value):raise ValueError('NaN scheduler state')
  if math.isinf(value):return {'__comparison_float__':'+inf' if value>0 else '-inf'}
 if isinstance(value,dict):return {k:encode(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [encode(v) for v in value]
 return value

def decode(value):
 if isinstance(value,dict) and set(value)=={'__comparison_float__'}:return float(value['__comparison_float__'])
 if isinstance(value,dict):return {k:decode(v) for k,v in value.items()}
 if isinstance(value,list):return [decode(v) for v in value]
 return value

def write_runtime_metadata(directory,state,ordinary_scheduler_state=None,*,reconstructed=False):
 directory=Path(directory);p=directory/'training_state.pt'
 lrs=[g['lr'] for g in state['optimizer']['param_groups']]
 if not lrs or any(not math.isfinite(x) or x<=0 for x in lrs):raise ValueError('Malformed active LR')
 if state['scheduler']['base_lrs']!=lrs:raise ValueError('Active LR and constant-scheduler base mismatch')
 obj=dict(schema='active_runtime_checkpoint_v1',checkpoint_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),active_optimizer_lrs=lrs,active_scheduler_base_lrs=state['scheduler']['base_lrs'],initial_design_config=state['contract'].get('config'),initial_contract_is_active_revision=False,ordinary_scheduler_state=encode(ordinary_scheduler_state) if ordinary_scheduler_state is not None else None,reconstructed_from_saved_state=reconstructed)
 with (directory/'active_runtime_config.json').open('x') as f:json.dump(obj,f,indent=2,allow_nan=False)
 return obj

def save_runtime_metadata(trainer,directory,ordinary_scheduler=None):
 import torch
 if any(g['lr']!=trainer.policy.config['learning_rate'] for g in trainer.optimizer.param_groups):raise ValueError('Runtime config differs from optimizer')
 state=torch.load(Path(directory)/'training_state.pt',map_location='cpu',weights_only=False)
 return write_runtime_metadata(directory,state,None if ordinary_scheduler is None else ordinary_scheduler.state_dict())

def restore_runtime_metadata(trainer,directory,ordinary_scheduler=None):
 directory=Path(directory);obj=json.loads((directory/'active_runtime_config.json').read_text())
 if hashlib.sha256((directory/'training_state.pt').read_bytes()).hexdigest()!=obj['checkpoint_sha256']:raise ValueError('Sidecar/checkpoint byte mismatch')
 if [g['lr'] for g in trainer.optimizer.param_groups]!=obj['active_optimizer_lrs'] or trainer.scheduler.base_lrs!=obj['active_scheduler_base_lrs']:raise ValueError('Restored active LR mismatch')
 trainer.policy.config['learning_rate']=obj['active_optimizer_lrs'][0]
 if ordinary_scheduler is not None:
  if obj['ordinary_scheduler_state'] is None:raise ValueError('Missing outer scheduler state')
  ordinary_scheduler.load_state_dict(decode(obj['ordinary_scheduler_state']))
  if encode(ordinary_scheduler.state_dict())!=obj['ordinary_scheduler_state']:raise ValueError('Outer scheduler restore mismatch')
 return obj
