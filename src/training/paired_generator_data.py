"""Load explicit paired normal edits; forbid the old implicit +2% target.

This CPU data adapter is preparation, not evidence of validated SFT training.
"""
from pathlib import Path
from src.data.p0_protocol import sha256


def load_training_pair(row,split,root,prepare_image):
    if split!='train':raise ValueError('Updates require the training split')
    from PIL import Image
    base=Path(root).resolve();images=[]
    for key in ('input','target'):
        path=(base/row[key]).resolve()
        if not path.is_relative_to(base) or sha256(path)!=row[key+'_SHA256']:raise ValueError('Explicit pair content changed')
        with Image.open(path) as image:images.append(prepare_image(image.convert('RGB')))
    if images[0].size!=images[1].size:raise ValueError('Source/target preprocessing geometry differs')
    if not row['instruction'].strip():raise ValueError('Missing clean edit instruction')
    return {'source':images[0],'target':images[1],'instruction':row['instruction'],'sample_id':row['sample_id']}
