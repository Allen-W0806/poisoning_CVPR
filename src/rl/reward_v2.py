"""Explicit compatibility arithmetic; no new scientifically validated reward."""
from dataclasses import dataclass,asdict
import math
import numpy as np
from .reward import RewardWeights,validation_reward
from ..editing.semantic_validator import SemanticResult
from ..evaluation.semantic_validation_v2 import validate_validity,consistency_eligible

@dataclass(frozen=True)
class RewardConfigV2:
    semantic_weight: float
    accuracy_weight: float
    distortion_weight: float
    cost_weight: float
    invalid_semantics_penalty: float
    schema_version: str='reward_compat_v2'
    execution_mode: str='diagnostic_only'

    def __post_init__(self):
        for key in RewardWeights.__dataclass_fields__:
            value=getattr(self,key)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:raise ValueError('Malformed coefficient: '+key)
        if self.schema_version!='reward_compat_v2' or self.execution_mode!='diagnostic_only':
            raise ValueError('New training reward requires independent construct validation')

    @classmethod
    def from_dict(cls,value):
        if set(value)!=set(cls.__dataclass_fields__):raise ValueError('Executed reward configuration must be complete and exact')
        return cls(**value)

    def to_dict(self):return asdict(self)

def aggregate_diagnostic(config,recorded_config,validity,answer_consistency,distortion,step_cost):
    if config.to_dict()!=recorded_config:raise ValueError('Executed coefficients differ from recorded configuration')
    validate_validity(validity)
    raw={'semantic':validity['score'],'answer_consistency':answer_consistency,'distortion':distortion,'cost':step_cost}
    for key,value in raw.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<=value<=1:
            raise ValueError('Missing, ambiguous or out-of-range component: '+key)
    eligible=consistency_eligible(validity)
    w=RewardWeights(**{k:getattr(config,k) for k in RewardWeights.__dataclass_fields__})
    result=validation_reward(SemanticResult(validity['verified'],validity['score'],validity['source'],validity['status']),
             answer_consistency if eligible else 0.,distortion,step_cost,w)
    return {**result,'schema_version':'reward_observation_v2','executed_config':config.to_dict(),
        'raw_components':raw,'consistency_eligible':eligible,'validity_status':validity['status'],
        'saturated_low':result['reward_unclipped'] < -1.,'saturated_high':result['reward_unclipped'] > 1.5,
        'training_eligible':False,'scope':'compatibility diagnostic; independent construct validation absent'}

def saturation_and_redundancy(rows):
    if not rows:raise ValueError('No observations')
    names=['semantic','answer_consistency','distortion','cost','invalid_penalty']
    values=np.array([[r[k] for k in names] for r in rows],dtype=float)
    if not np.isfinite(values).all():raise ValueError('Nonfinite instrumentation')
    correlations={}
    for i,a in enumerate(names):
        for j,b in enumerate(names[:i]):
            correlations[a+':'+b]=None if np.std(values[:,i])==0 or np.std(values[:,j])==0 else float(np.corrcoef(values[:,i],values[:,j])[0,1])
    return {'n':len(rows),'at_lower_bound':sum(r['reward']==-1 for r in rows),
        'at_upper_bound':sum(r['reward']==1.5 for r in rows),'component_correlations':correlations,
        'constant_components':[names[i] for i in range(len(names)) if np.std(values[:,i])==0]}
