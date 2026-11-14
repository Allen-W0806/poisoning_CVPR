"""Constructed harmless file-integrity cases; never called natural-image truth."""
import hashlib
from pathlib import Path
from PIL import Image

def create_cases(root,split,count):
    if split not in {'train','gate'}:raise ValueError('Engineering fixture split')
    folder=Path(root)/split;folder.mkdir(parents=True,exist_ok=False)
    cases=[];oracle={}
    for i in range(count):
        token=hashlib.sha256(f'{split}_{i}'.encode()).digest()
        image=Image.new('RGB',(16,16),tuple(token[:3]));image.putpixel((0,0),tuple(token[3:6]))
        path=folder/f'{i:04d}.png';image.save(path)
        original_sha=hashlib.sha256(path.read_bytes()).hexdigest()
        valid=i%2==1
        if not valid:
            image.putpixel((8,8),tuple(token[6:9]));image.save(path)
        case={'case_id':f'{split}_{i:04d}','split':split,'image_path':path.relative_to(root).as_posix(),
            'expected_sha256':original_sha,'expected_size':[16,16],'uncertainty':1. if not valid else 0.,
            'scope':'constructed harmless integrity oracle'}
        cases.append(case);oracle[case['case_id']]=valid
    return cases,oracle
