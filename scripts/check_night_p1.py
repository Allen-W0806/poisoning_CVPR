"""CPU audit of explicit real-photo pairs; never launches a GPU experiment."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);a=parser.parse_args()
    run=Path(a.campaign).resolve();manifest=json.loads((run/'p1_preparation/paired_manifest.json').read_text())
    cfg=json.loads((run/'protocol.json').read_text());seen_ids=set();seen_orig=set();counts={}
    from PIL import Image
    for split,rows in manifest['splits'].items():
        if len(rows)!=cfg['P1_data_sizes'][split]:raise ValueError('P1 split count')
        counts[split]=len(rows)
        for row in rows:
            if row['image_id'] in seen_ids or row['original_SHA256'] in seen_orig:raise ValueError('P1 source leakage')
            seen_ids.add(row['image_id']);seen_orig.add(row['original_SHA256']);sizes=[]
            for key in ('input','target'):
                path=(run/row[key]).resolve()
                if not path.is_relative_to(run) or sha256(path)!=row[key+'_SHA256']:raise ValueError('Pair path/hash mismatch')
                with Image.open(path) as image:sizes.append(image.size)
            if sizes[0]!=sizes[1] or max(sizes[0])>512:raise ValueError('Paired geometry mismatch')
    result={'status':'PASS_CPU_PAIRED_DATA_CHECK','split_sizes':counts,'pair_manifest_SHA256':sha256(run/'p1_preparation/paired_manifest.json'),
       'task_families':{'train_dev_test':'controlled exposure restoration','transfer':'controlled deblurring'},
       'GPU_runs':'NOT_RUN','GPU_gate':'BLOCKED_FROZEN_QUALITY_AND_512_TRAINING_NUMERICS',
       'shared_arms':manifest['arms'],'same_pairs_and_seeds':True,
       'remaining_before_P1':['Frozen fidelity/structure quality gate','512px training and exact resume numerical gate','explicit-pair SFT GPU validation','common clean reward and task-success annotation protocol','equal-budget controller harness integration and learned transfer validation']}
    (run/'p1_preparation/CPU_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
