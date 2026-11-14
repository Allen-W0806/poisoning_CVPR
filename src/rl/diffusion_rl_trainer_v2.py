"""Phase5 batch guards; the validated DDPO update remains the original backend."""
from .diffusion_rl_trainer import DiffusionRLTrainer
import re

def validate_unique_batch(episodes):
    if not episodes:raise ValueError('Empty logical source batch')
    for key in ['trajectory_id','sample_id','image_id']:
        values=[str(t[key]) for t,c,r in episodes]
        if len(values)!=len(set(values)):raise ValueError('Duplicate within logical batch: '+key)
    hashes=[]
    for t,c,r in episodes:
        h=t.get('source_image_sha256')
        if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h) or c.get('source_image_sha256')!=h:
            raise ValueError('Versioned trajectory/condition content identity required')
        hashes.append(h)
    if len(hashes)!=len(set(hashes)):raise ValueError('Duplicate source content in logical batch')
    return True

class GuardedDiffusionRLTrainer(DiffusionRLTrainer):
    def update(self,episodes):
        validate_unique_batch(episodes)
        from .phase5_backend import require_training_reward
        for t,c,r in episodes:require_training_reward(t.get('reward_receipt',{}))
        return super().update(episodes)
