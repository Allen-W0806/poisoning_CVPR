"""Strict, versioned benign reward bookkeeping; scientific release stays denied.

The five coefficients, component/clipping arithmetic and fixed normalization are
unchanged. Independent semantic/instruction validity is not inferred from pixels
or model agreement. CPU fixtures can exercise labeled cases without creating
human labels for real images.
"""
import math
from dataclasses import asdict
from .reward import RewardWeights, validation_reward
from ..editing.semantic_validator import SemanticResult


def _unit(name, value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Missing, nonfinite or out-of-range component: " + name)
    return float(value)


def feedback(config, recorded_config, semantic, answer_consistency, distortion, step_cost,
             *, resource_measurements=None):
    weights = RewardWeights.from_config(config)
    recorded_weights = RewardWeights.from_config(recorded_config)
    if config != recorded_config or asdict(weights) != asdict(recorded_weights):
        raise ValueError("Executed/recorded reward configuration mismatch")
    required = {'verified', 'score', 'verifier', 'reason'}
    if not isinstance(semantic, dict) or not required <= set(semantic):
        raise ValueError("Incomplete semantic observation")
    if type(semantic['verified']) is not bool or type(semantic.get('provisionally_valid', False)) is not bool:
        raise ValueError("Ambiguous semantic flags")
    if not isinstance(semantic['verifier'], str) or not semantic['verifier']:
        raise ValueError("Semantic source required")
    if semantic['verified'] and semantic.get('provisionally_valid', False):
        raise ValueError("Verified/provisional semantic flags conflict")
    s = _unit('semantic', semantic['score'])
    a = _unit('answer_consistency', answer_consistency)
    d = _unit('distortion', distortion)
    c = _unit('cost_proxy', step_cost)
    if semantic['verified'] and s not in (0., 1.):
        raise ValueError("Verified semantic labels must be binary")
    eligible = (semantic['verified'] and s == 1.) or semantic.get('provisionally_valid', False)
    if semantic.get('provisionally_valid', False) and s != .5:
        raise ValueError("Unexpected provisional compatibility score")
    effective_a = a if eligible else 0.
    result = validation_reward(SemanticResult(semantic['verified'], s, semantic['verifier'], semantic['reason']),
                               effective_a, d, c, weights)
    resources = dict(resource_measurements or {})
    for key, value in resources.items():
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0):
            raise ValueError('Malformed measured resource: ' + key)
    return {**result, 'schema_version': 'benign_reward_bookkeeping_v3',
            'raw_components': {'semantic': s, 'answer_consistency': a, 'distortion': d, 'cost_proxy': c},
            'effective_components': {'semantic': s, 'answer_consistency': effective_a, 'distortion': d, 'cost_proxy': c},
            'executed_weights': asdict(weights), 'recorded_weights': asdict(recorded_weights),
            'executed_yaml_config': dict(config), 'consistency_eligible': eligible,
            'normalization': 'fixed_reference_baseline0_scale1_no_batch_statistics',
            'component_units': {'semantic': 'legacy_validity_proxy_0_1', 'answer_consistency': 'fraction_0_1',
                                'distortion': 'legacy_pair_distortion_0_1', 'cost_proxy': 'denoising_steps_div_30'},
            'resource_measurements': resources,
            'saturated_low': result['reward_unclipped'] < -1., 'saturated_high': result['reward_unclipped'] > 1.5,
            'training_eligible': False, 'semantic_construct_validated': False,
            'scope': 'strict benign diagnostic bookkeeping; not a released scientific training reward'}


def caller_feedback(config, semantic, consistency, quality, steps, generation_seconds=None):
    """One caller path credits both verified validity and provisional agreement."""
    if not isinstance(consistency, dict) or 'score' not in consistency:
        raise ValueError('Missing answer-consistency score')
    if not isinstance(quality, dict) or 'distortion' not in quality:
        raise ValueError('Missing image-distortion measurement')
    if isinstance(steps, bool) or not isinstance(steps, int) or not 0 < steps <= 30:
        raise ValueError('Invalid denoising-step proxy count')
    return feedback(config, dict(config), semantic, consistency['score'], quality['distortion'], steps / 30.,
                    resource_measurements={'generation_wall_seconds': generation_seconds,
                                           'allocated_gpu_seconds': None})


def require_episode_reward_release(episodes):
    """Do not let corrected diagnostics silently enter historical train callers."""
    from .phase5_backend import require_training_reward
    for trajectory, _, _ in episodes:
        receipt = trajectory.get('reward', {})
        if receipt.get('schema_version') == 'benign_reward_bookkeeping_v3':
            require_training_reward(receipt)
