"""Frozen IP2Pix/Qwen matched inference. No optimizer, reward fitting or search."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from PIL import Image,ImageEnhance
from skimage.metrics import structural_similarity
from src.data.dataset import sha256
from src.editing.pretrained_editor import PretrainedEditor
from src.editing.editor_config import PretrainedEditorConfig
from src.models.qwen_loader import QwenVQA
from src.utils.seed import seed_everything
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
from validate_benign_models import model_hash


def dump(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['pilot','main'],required=True);a=p.parse_args()
    run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());manifest_path=run/f'{a.phase}_manifest.json'
    source=json.loads(manifest_path.read_text());phase=run/a.phase;phase.mkdir(exist_ok=False)
    (phase/'images').mkdir();started=time.perf_counter();start_unix=time.time();checks=[];candidates=[];rows=[];generation=[]
    def check(name,ok,**kw):
        check={'test':name,'status':'PASS' if bool(ok) else 'FAIL',**kw};checks.append(check);print(json.dumps(check),flush=True)
    def live(stage,**details):
        gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        value={'stage':stage,'elapsed_seconds':time.perf_counter()-started,'records':len(rows),'candidates':len(candidates),'gpu':gpu,**details}
        path=phase/'live.json';temp=phase/'live.partial';temp.write_text(json.dumps(value,indent=2));temp.replace(path)
    try:
        if a.phase=='main':
            prior=json.loads((run/'pilot/checks.json').read_text())
            if prior['status']!='PASS':raise RuntimeError('Pilot engineering gate did not pass')
        check('source_manifest_frozen',sha256(manifest_path)==json.loads((run/'data_checks.json').read_text())['manifest_sha256'][a.phase])
        torch.set_num_threads(4);seed_everything(42)
        if not torch.cuda.is_available():raise RuntimeError('Slurm GPU required')
        g=torch.Generator(device='cuda').manual_seed(42);x=torch.randn(16,16,generator=g,device='cuda')
        check('seeded_CUDA_witness',torch.isfinite(x@x).all(),device=torch.cuda.get_device_name())
        ec=cfg['editor'];config=PretrainedEditorConfig(**{k:v for k,v in ec.items() if k!='instruction'})
        live('loading frozen editor');editor=PretrainedEditor(config,device='cuda')
        modules={'UNet':editor.pipe.unet,'VAE':editor.pipe.vae,'text_encoder':editor.pipe.text_encoder}
        oldhash={k:model_hash(m) for k,m in modules.items()}
        check('editor_all_weights_frozen',all(not p.requires_grad for m in modules.values() for p in m.parameters()))
        for r in source['records']:
            source_path=ROOT/r['image_path']
            if sha256(source_path)!=r['image_sha256']:raise ValueError('Source checksum changed')
            with Image.open(source_path) as image:original=image.convert('RGB')
            prepared=editor.prepare_image(original);images={'C0':original,'C1':original.copy(),
                'C2':ImageEnhance.Brightness(original).enhance(1.03),'C3':ImageEnhance.Contrast(original).enhance(1.03),'C5':prepared}
            output_path=phase/'images'/f"{r['source_image_id']}_C4.png";gen_status='PASS';error='';metadata={}
            try:
                result=editor.edit(original,ec['instruction'],r['source_image_id'],seed=ec['seed'],output_path=output_path)
                images['C4']=result.image;metadata=result.metadata
                gen_record={'source_image_id':r['source_image_id'],'status':'PASS','metadata':metadata}
                if r['repeat_case']:
                    repeat_path=phase/'images'/f"{r['source_image_id']}_C4_repeat.png"
                    repeated=editor.edit(original,ec['instruction'],r['source_image_id'],seed=ec['seed'],output_path=repeat_path)
                    identical=np.array_equal(np.array(result.image),np.array(repeated.image))
                    check('editor_same_seed_repeat_'+r['source_image_id'],identical)
                    gen_record.update(repeat_generation_seconds=repeated.metadata['generation_seconds'],repeat_sha256=sha256(repeat_path),same_seed_pixel_equal=bool(identical))
                generation.append(gen_record)
            except Exception as e:
                gen_status='FAIL';error=repr(e);generation.append({'source_image_id':r['source_image_id'],'status':'FAIL','error':error})
            for condition in cfg['primary_conditions']+cfg['auxiliary_conditions']:
                image=images.get(condition);path=source_path if condition=='C0' else phase/'images'/f"{r['source_image_id']}_{condition}.png"
                valid=image is not None
                info={};diagnostics={}
                if valid:
                    if condition not in ['C0','C4']:image.save(path)
                    with Image.open(path) as opened:opened.load();check_mode=opened.mode;size=opened.size
                    arr=np.array(image);valid=bool(arr.dtype==np.uint8 and np.isfinite(arr).all() and image.mode=='RGB' and (condition=='C0' or check_mode=='RGB'))
                    if condition in ['C4','C5']:valid=valid and size==(ec['image_resolution'],ec['image_resolution'])
                    else:valid=valid and size==original.size
                    info={'image_sha256':sha256(path),'pixel_sha256':hashlib.sha256(arr.tobytes()).hexdigest(),'size':list(size),'mode':image.mode,'encoded_mode':check_mode}
                    if condition in ['C2','C3','C4']:
                        ref=prepared if condition=='C4' else original;refarr=np.array(ref)
                        ssim=float(structural_similarity(refarr,arr,channel_axis=2,data_range=255))
                        lum=float(np.array(image.convert('L')).mean()-np.array(ref.convert('L')).mean())
                        diagnostics={'SSIM':ssim,'mean_luminance_difference':lum,'RGB_MAE_255':float(np.abs(arr.astype(float)-refarr.astype(float)).mean()),
                            'risk_low_SSIM':ssim<cfg['risk_flags_only']['SSIM_below'],
                            'risk_large_luminance_change':abs(lum)>cfg['risk_flags_only']['absolute_mean_luminance_change_above'],
                            'risk_darkening':condition in ['C2','C4'] and lum<0,
                            'interpretation':'screening flags only; no semantic certification'}
                candidates.append({**r,'candidate_id':condition,'transformation_type':cfg['conditions'][condition],
                    'candidate_path':str(path.relative_to(ROOT)),'candidate_image_hash':info.get('image_sha256'),
                    'original_image_hash':r['image_sha256'],'editor_seed':ec['seed'] if condition=='C4' else None,
                    'generation_status':'PASS' if valid else 'FAIL','generation_error':error if condition=='C4' else '',
                    'semantic_review_status':'PASS' if condition in ['C0','C1'] else 'NOT VALIDATED',
                    'semantic_evidence':'original annotation / exact identity' if condition in ['C0','C1'] else 'independent review pending',
                    'engineering':info,'diagnostics':diagnostics})
            identity=next(x for x in candidates[-6:] if x['candidate_id']=='C1')
            c0=next(x for x in candidates[-6:] if x['candidate_id']=='C0')
            check('identity_pixel_exact_'+r['source_image_id'],identity['engineering']['pixel_sha256']==c0['engineering']['pixel_sha256'])
            if r['source_index']%8==0:live('generating frozen candidates',sources_completed=r['source_index']+1)
        for name,module in modules.items():
            new=model_hash(module);check('editor_frozen_hash_'+name,new==oldhash[name],before=oldhash[name],after=new)
        dump(phase/'generation_records.json',generation)
        dump(phase/'candidate_manifest.json',{'source_manifest_sha256':sha256(manifest_path),'protocol_sha256':sha256(run/'protocol.json'),
            'frozen_before_VQA_unix':time.time(),'records':candidates})
        candidate_manifest_hash=sha256(phase/'candidate_manifest.json')
        del editor,modules,module;gc.collect();torch.cuda.empty_cache()
        vc=cfg['vlm'];live('loading frozen Qwen')
        model=QwenVQA(model_id=vc['model_id'],revision=vc['revision'],device='cuda',max_new_tokens=vc['max_new_tokens'],
                      min_pixels=vc['min_pixels'],max_pixels=vc['max_pixels'],seed=42)
        vlm_before=model_hash(model.model);check('Qwen_frozen',all(not p.requires_grad for p in model.model.parameters()))
        with (phase/'predictions.jsonl').open('x',encoding='utf-8') as log:
            for seed in cfg['evaluation_seeds']:
                seed_everything(seed)
                for r in source['records']:
                    if seed!=42 and not r['repeat_case']:continue
                    selected=[c for c in candidates if c['source_image_id']==r['source_image_id']]
                    shift=r['source_index']%len(selected);selected=selected[shift:]+selected[:shift]
                    for c in selected:
                        raw_answer='';error='';status='PASS';vqa_score=None;exact=None;normalized='';latency=None
                        if c['generation_status']=='PASS':
                            torch.cuda.synchronize();t=time.perf_counter()
                            try:
                                if sha256(ROOT/c['candidate_path'])!=c['candidate_image_hash']:raise ValueError('Candidate checksum changed')
                                with Image.open(ROOT/c['candidate_path']) as im:image=im.convert('RGB')
                                raw_answer=model.answer([image],[r['question']])[0]
                                torch.cuda.synchronize();latency=time.perf_counter()-t
                                scoring=score_answer(raw_answer,r);vqa_score=scoring['score'];normalized=scoring['normalized_answer']
                                exact=float(normalize_answer(raw_answer) in {normalize_answer(x) for x in r['answers']})
                            except Exception as e:status='FAIL';error=repr(e)
                        else:status='FAIL';error='generation_failed: '+c['generation_error']
                        row={k:v for k,v in c.items() if k not in ['engineering','diagnostics','image_path','image_sha256','answers']}
                        row.update(evaluation_seed=seed,raw_vlm_answer=raw_answer,normalized_answer=normalized,official_vqa_consensus_score=vqa_score,
                            exact_any_reference_score=exact,all_attempted_utility_score=vqa_score if vqa_score is not None else 0.,
                            inference_latency_seconds=latency,execution_status=status,error=error,model_revision=vc['revision'],evaluation_run_id=a.run_id,
                            phase=a.phase,reference_answers=r['answers'],candidate_manifest_sha256=candidate_manifest_hash)
                        rows.append(row);log.write(json.dumps(row,allow_nan=False)+'\n');log.flush()
                    if r['source_index']%8==0:live('frozen matched Qwen inference',evaluation_seed=seed,source=r['source_index'])
                print(f'{a.phase} seed={seed} completed records={len(rows)}',flush=True)
        vlm_after=model_hash(model.model);check('Qwen_weight_hash_invariant',vlm_before==vlm_after,before=vlm_before,after=vlm_after)
        expected=len(source['records'])*6+cfg['repeat_sources_per_phase']*6*2
        check('prediction_count_reconciled',len(rows)==expected,expected=expected,actual=len(rows))
        primary=[r for r in rows if r['evaluation_seed']==42]
        check('all_records_valid',all(r['execution_status']=='PASS' for r in rows),invalid=sum(r['execution_status']!='PASS' for r in rows))
        for r in source['records']:
            rr={x['candidate_id']:x for x in primary if x['source_image_id']==r['source_image_id']}
            check('identity_score_answer_'+r['source_image_id'],rr['C0']['raw_vlm_answer']==rr['C1']['raw_vlm_answer'] and rr['C0']['official_vqa_consensus_score']==rr['C1']['official_vqa_consensus_score'])
        repeat_checks=[]
        for r in primary:
            others=[x for x in rows if x['source_image_id']==r['source_image_id'] and x['candidate_id']==r['candidate_id']]
            if len(others)>1:repeat_checks.append(len({x['raw_vlm_answer'] for x in others})==1 and len({x['official_vqa_consensus_score'] for x in others})==1)
        check('selected_seed_repeat_exact',all(repeat_checks),repeated_candidate_cases=len(repeat_checks))
        dump(phase/'model_metadata.json',model.metadata())
    except Exception as e:check('stage_execution_error',False,error=repr(e),traceback=traceback.format_exc())
    finally:
        status='PASS' if checks and all(c['status']=='PASS' for c in checks) else 'FAIL'
        dump(phase/'checks.json',{'status':status,'checks':checks,'elapsed_process_seconds':time.perf_counter()-started,
            'start_unix':start_unix,'finish_unix':time.time(),'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'source_sha256':sha256(Path(__file__)),
            'generation_calls':len(generation)+sum('repeat_generation_seconds' in r for r in generation),'VQA_records':len(rows),
            'generation_seconds':sum(r.get('metadata',{}).get('generation_seconds',0)+r.get('repeat_generation_seconds',0) for r in generation),
            'inference_seconds':sum(r['inference_latency_seconds'] or 0 for r in rows)})
        live('complete' if status=='PASS' else 'failed')
    return 0 if status=='PASS' else 1


if __name__=='__main__':raise SystemExit(main())
