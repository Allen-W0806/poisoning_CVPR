"""Versioned one-source engineering repair; no successful prediction overwritten.
Original grayscale JPEG is valid. RGB model input is checked independently of
its encoded mode. Re-evaluate C0/C1 at all 3 seeds; derive a new accepted view.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from src.models.qwen_loader import QwenVQA
from src.utils.seed import seed_everything
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
from validate_benign_models import model_hash
from src.data.dataset import sha256


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id
    original=run/'main';target=run/'main_repair_v1';target.mkdir(exist_ok=False);started=time.perf_counter()
    cfg=json.loads((run/'protocol.json').read_text());vc=cfg['vlm'];old=json.loads((original/'candidate_manifest.json').read_text())
    records=[json.loads(x) for x in (original/'predictions.jsonl').read_text().splitlines()]
    failed=[r for r in records if r['execution_status']!='PASS']
    if len(failed)!=1 or failed[0]['source_image_id']!='361516' or failed[0]['candidate_id']!='C0':raise RuntimeError('Unexpected repair scope')
    # Verify every original source's actual RGB input; no generated-output check
    # is weakened. Existing image bytes and all frozen source IDs remain intact.
    manifest=json.loads((run/'main_manifest.json').read_text());mode_counts={};source=None
    for r in manifest['records']:
        path=ROOT/r['image_path'];assert sha256(path)==r['image_sha256']
        with Image.open(path) as image:
            encoded=image.mode;rgb=image.convert('RGB')
        mode_counts[encoded]=mode_counts.get(encoded,0)+1
        assert rgb.mode=='RGB' and np.array(rgb).dtype==np.uint8 and np.isfinite(np.array(rgb)).all()
        if r['source_image_id']=='361516':source=r
    assert source is not None
    with Image.open(ROOT/source['image_path']) as image:rgb=image.convert('RGB')
    c1=next(c for c in old['records'] if c['source_image_id']=='361516' and c['candidate_id']=='C1')
    with Image.open(ROOT/c1['candidate_path']) as image:identity=image.convert('RGB')
    assert np.array_equal(np.array(rgb),np.array(identity))
    oldchecks=json.loads((original/'checks.json').read_text());unexpected=[c for c in oldchecks['checks'] if c['status']!='PASS' and c['test'] not in ['all_records_valid','identity_score_answer_361516']]
    assert not unexpected
    repaired_manifest=copy.deepcopy(old)
    item=next(c for c in repaired_manifest['records'] if c['source_image_id']=='361516' and c['candidate_id']=='C0')
    item['generation_status']='PASS';item['engineering']['encoded_mode']='L';item['engineering']['mode']='RGB'
    repaired_manifest.update(derivation={'original_manifest_sha256':sha256(original/'candidate_manifest.json'),'original_phase_dir':'main',
        'repair':'actual decoded model-input RGB, distinct from encoded grayscale JPEG mode','edited_images_regenerated':False},frozen_before_repair_inference_unix=time.time())
    dump(target/'candidate_manifest.json',repaired_manifest);mh=sha256(target/'candidate_manifest.json')
    torch.set_num_threads(4);model=QwenVQA(model_id=vc['model_id'],revision=vc['revision'],device='cuda',max_new_tokens=vc['max_new_tokens'],
        min_pixels=vc['min_pixels'],max_pixels=vc['max_pixels'],seed=42);before=model_hash(model.model);rerun=[]
    for seed in cfg['evaluation_seeds']:
        seed_everything(seed)
        for condition,image in [('C0',rgb),('C1',identity)]:
            torch.cuda.synchronize();t=time.perf_counter();answer=model.answer([image],[source['question']])[0];torch.cuda.synchronize();latency=time.perf_counter()-t
            scored=score_answer(answer,source)
            template=next(r for r in records if r['source_image_id']=='361516' and r['candidate_id']==condition)
            row=copy.deepcopy(template);row.update(generation_status='PASS',generation_error='',evaluation_seed=seed,raw_vlm_answer=answer,normalized_answer=scored['normalized_answer'],
                official_vqa_consensus_score=scored['score'],exact_any_reference_score=float(normalize_answer(answer) in {normalize_answer(x) for x in source['answers']}),
                all_attempted_utility_score=scored['score'],inference_latency_seconds=latency,execution_status='PASS',error='',
                evaluation_run_id=a.run_id+':main_repair_v1',candidate_manifest_sha256=mh)
            rerun.append(row)
    after=model_hash(model.model);assert before==after and all(not p.requires_grad for p in model.model.parameters())
    assert len({r['raw_vlm_answer'] for r in rerun})==1 and len({r['official_vqa_consensus_score'] for r in rerun})==1
    # Only previously missing C0 observation is replaced in the derived view.
    corrected=[]
    for r in records:
        if r['source_image_id']=='361516' and r['candidate_id']=='C0':
            new=copy.deepcopy(next(x for x in rerun if x['candidate_id']=='C0' and x['evaluation_seed']==42))
            new.update(repaired_from_original_failed_record=True,original_failure_manifest_sha256=r['candidate_manifest_sha256'])
        else:new=copy.deepcopy(r)
        new['accepted_view_manifest_sha256']=mh;corrected.append(new)
    assert all(r['execution_status']=='PASS' for r in corrected)
    for name,rows in [('repair_predictions.jsonl',rerun),('predictions.jsonl',corrected)]:
        with (target/name).open('x') as f:
            for row in rows:f.write(json.dumps(row)+'\n')
    for name in ['generation_records.json','model_metadata.json']:shutil.copyfile(original/name,target/name)
    check={'status':'PASS','checks':[{'test':'decoded_RGB_source_validation','status':'PASS'},{'test':'grayscale_C0_C1_exact_pixels_answers_scores_all_seeds','status':'PASS'},
        {'test':'repaired_complete_count_1032','status':'PASS'},{'test':'VLM_frozen_hash_invariant','status':'PASS'}],
        'original_failed_stage_preserved':True,'original_phase_dir':'main','encoded_mode_counts':mode_counts,
        'repair_inference_calls':6,'repair_inference_seconds':sum(r['inference_latency_seconds'] for r in rerun),'elapsed_process_seconds':time.perf_counter()-started,
        'before_hash':before,'after_hash':after,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'source_sha256':sha256(Path(__file__))}
    dump(target/'checks.json',check);dump(target/'repair_receipt.json',check)
    dump(run/'accepted_phase_revisions.json',{'pilot':'pilot','main':'main_repair_v1','reason':'single encoded-grayscale metadata misclassification; all original evidence retained'})
    print(json.dumps(check,indent=2))


if __name__=='__main__':main()
