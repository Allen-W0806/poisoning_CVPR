"""Explicit Phase5 Generator-RL integration; no controller-RL substitution."""
from .diffusion_policy_v2 import ContentBoundDiffusionPolicy
from .diffusion_rl_trainer_v2 import GuardedDiffusionRLTrainer
from .reward_v2 import RewardConfigV2,aggregate_diagnostic
from ..evaluation.semantic_validation_v2 import decision_v2

def make_backend(editor,config):
    policy=ContentBoundDiffusionPolicy(editor,config)
    return policy,GuardedDiffusionRLTrainer(policy)

def diagnostic_feedback(config,recorded_config,original,edited,record,consistency,judgment,quality,step_cost):
    if not isinstance(config,RewardConfigV2):raise ValueError('Explicit executed reward configuration required')
    validity=decision_v2(original,edited,record,consistency,judgment)
    return aggregate_diagnostic(config,recorded_config,validity,consistency['score'],quality['distortion'],step_cost)

def require_training_reward(receipt):
    if receipt.get('training_eligible') is not True:raise RuntimeError('Independent reward-construct validation is absent; training disabled')
    raise RuntimeError('No Phase5 training reward has been validated or released')
