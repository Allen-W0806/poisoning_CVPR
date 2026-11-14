"""Question-specific visually screened edits, frozen Qwen; no learning."""
import csv
import argparse
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.models.qwen_loader import QwenVQA
from src.utils.seed import seed_everything
from src.data.dataset import sha256
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
from validate_benign_models import model_hash,dump


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='reviewed_edits_20261028_v1')
    args=parser.parse_args()
    run=ROOT/'results'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    source=ROOT/'results/benign_models_20261028_v1'
    manifest=json.loads((source/'evaluation_manifest.json').read_text())
    review=json.loads((ROOT/'results/editor_visual_review_20261028.json').read_text())
    refs={r['sample_id']:r for r in manifest['records']}
    accepted=[]
    for r in review['reviews']:
        if r['label_consistent'] is not True:continue
        repeat=0 if r['seed']==42 else 2
        path=source/f"editor_{r['image_index']}_seed{r['seed']}_repeat{repeat}.png"
        accepted.append({**r,'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'record':refs[r['sample_id']]})
    cfg=json.loads((ROOT/'configs/gradient_feedback_validation_v1.json').read_text());ec=cfg['evaluation']
    dump(run/'manifest.json',{'review_sha256':sha256(ROOT/'results/editor_visual_review_20261028.json'),
        'source_manifest_sha256':sha256(source/'evaluation_manifest.json'),'conditions':accepted,
        'seeds':cfg['seeds'],'model_revision':ec['model_revision'],'scope':'three accepted edits from two source images; descriptive only'})
    started=time.perf_counter();model=QwenVQA(revision=ec['model_revision'],device='cuda',max_new_tokens=32,seed=42)
    before=model_hash(model.model);rows=[]
    for seed in cfg['seeds']:
        seed_everything(seed)
        for r in accepted:
            with Image.open(ROOT/r['path']) as im:image=im.convert('RGB')
            torch.cuda.synchronize();t=time.perf_counter();raw=model.answer([image],[r['record']['question']])[0];torch.cuda.synchronize()
            score=score_answer(raw,r['record'])
            rows.append({'sample_id':r['sample_id'],'seed':seed,'edit_seed':r['seed'],'condition':'visually_reviewed_benign_edit',
                'image_path':r['path'],'image_sha256':r['sha256'],'question':r['record']['question'],'raw_answer':raw,
                'consensus_score':score['score'],'exact_any_reference':float(normalize_answer(raw) in {normalize_answer(x) for x in r['record']['answers']}),
                'latency_seconds':time.perf_counter()-t,'valid':bool(raw.strip()),'model_revision':ec['model_revision']})
    after=model_hash(model.model)
    with (run/'benign_vlm_evaluation.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    check={'parameter_hash_invariant':before==after,'before_sha256':before,'after_sha256':after,
        'all_frozen':all(not p.requires_grad for p in model.model.parameters()),'samples':len(rows),'unique_source_images':2,
        'raw_answer_seed_replay_exact':all(len({x['raw_answer'] for x in rows if x['sample_id']==r['sample_id'] and x['edit_seed']==r['seed']})==1 for r in accepted),
        'seconds':time.perf_counter()-started,'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
    dump(run/'checks.json',check);print(json.dumps(check),flush=True)


if __name__=='__main__':main()
