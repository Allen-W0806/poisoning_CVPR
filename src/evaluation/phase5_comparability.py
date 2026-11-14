"""Read-only, exact pre-intervention checks. No numeric threshold relaxation."""
import torch
import copy
from ..rl.optimizer_validation import state_equal

def compare_checkpoints(left,right):
    a=torch.load(left,map_location='cpu',weights_only=False);b=torch.load(right,map_location='cpu',weights_only=False)
    fields=['adapter','optimizer','scheduler','step','version','contract','consumed','rng_torch','rng_cuda','rng_numpy','rng_python']
    result={}
    for field in fields:
        if field not in a or field not in b:result[field]=False
        else:
            try:result[field]=state_equal(a[field],b[field])
            except (ValueError,TypeError):result[field]=False
    keys=set(a['adapter'])&set(b['adapter'])
    result['adapter_max_abs_difference']=max((float((a['adapter'][k]-b['adapter'][k]).abs().max()) for k in keys),default=None)
    def scientific_contract(value):
        value=copy.deepcopy(value)
        for key in ['arm','rsi_mode']:value.get('config',{}).pop(key,None)
        return value
    result['contract_full_metadata_exact']=result['contract']
    result['contract_scientific_exact']=state_equal(scientific_contract(a['contract']),scientific_contract(b['contract']))
    result['nonempty_state_contract']=bool(a['adapter'] and b['adapter'] and a['contract'] and b['contract'])
    result['all_required_fields_exact']=all(result[k] for k in fields if k!='contract') and result['contract_scientific_exact'] and result['nonempty_state_contract']
    result['causal_intervention_comparison_ready']=False
    result['additional_required_evidence']=['verified frozen-weight identity','training image-content/source order','per-source rollout noise seeds','independent pre-intervention execution comparability']
    result['scope']='Exact observed checkpoint fields; does not prove subsequent stochastic continuation'
    return result

def require_launch_ready(readiness):
    required={'independent_editing_validity','validated_training_reward','content_disjoint_splits',
      'active_harness_tests','conventional_scheduler_baseline','pre_intervention_comparability','private_gate_isolation','frozen_resource_protocol'}
    if set(readiness)!=required or any(type(v) is not bool for v in readiness.values()):raise ValueError('Complete boolean readiness record required')
    blocked=sorted(k for k,v in readiness.items() if not v)
    if blocked:raise RuntimeError('NO-GO: '+', '.join(blocked))
    return True
