"""Add verified source-content provenance without changing diffusion mathematics."""
import re
from .diffusion_policy import DiffusionPolicy

class ContentBoundDiffusionPolicy(DiffusionPolicy):
    def condition(self,record,instruction=None):
        h=record.get('image_sha256')
        if not isinstance(h,str) or not re.fullmatch('[0-9a-f]{64}',h):raise ValueError('Verified dataset content hash required')
        value=super().condition(record) if instruction is None else super().condition(record,instruction)
        return {**value,'source_image_sha256':h}

    def sample(self,condition,seed,trajectory_id):
        image,trajectory=super().sample(condition,seed,trajectory_id)
        trajectory['source_image_sha256']=condition['source_image_sha256']
        return image,trajectory
