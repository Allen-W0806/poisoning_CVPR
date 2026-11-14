"""Frozen Qwen VQA and benign IP2Pix inference, exclusive evidence directory.

Run only on an allocated GPU. Cached official data, no dataset/model download,
no optimizer, no LoRA attachment, no security-metric optimization.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import numpy as np
import torch
from PIL import Image
from src.data.dataset import sha256
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer,reference_metadata
from src.models.qwen_loader import QwenVQA
from src.editing.editor_config import PretrainedEditorConfig
from src.editing.pretrained_editor import PretrainedEditor
from src.utils.seed import seed_everything


def dump(path,value):
    with path.open('x',encoding='utf-8') as f: json.dump(value,f,indent=2,allow_nan=False)


def snapshot(path,value):
    # This one file is explicitly live state, not immutable evidence.
    temp=path.with_suffix('.partial'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)); temp.replace(path)


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def model_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        h.update(name.encode()); h.update(str(p.dtype).encode()); h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--editor-only',action='store_true');args=p.parse_args()
    run=ROOT/'results'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((ROOT/'configs/gradient_feedback_validation_v1.json').read_text()); ec=cfg['evaluation']
    dump(run/'protocol.json',cfg);started=time.perf_counter();checks=[];rows=[];edits=[];param_rows=[]
    def check(name,ok,**details):
        value={'test':name,'passed':bool(ok),**details}; checks.append(value); print(json.dumps(value),flush=True)
    def live(phase,**details):
        try: gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        except OSError:gpu='unavailable'
        snapshot(run/'live.json',{'phase':phase,'seconds':time.perf_counter()-started,'completed_vqa':len(rows),'checks':checks,'gpu':gpu,**details})
    try:
        check('cuda_available',torch.cuda.is_available())
        if not torch.cuda.is_available():raise RuntimeError('GPU required for bounded frozen evaluation')
        torch.set_num_threads(4)
        kernel_gen=torch.Generator(device='cuda').manual_seed(42); witness=torch.randn(16,16,generator=kernel_gen,device='cuda')
        check('seeded_CUDA_kernel_witness',torch.isfinite(witness@witness).all(),device=torch.cuda.get_device_name())
        base=ROOT/'data/processed/vqav2_seed42_v1'; provenance=json.loads((base/'provenance.json').read_text())
        splits={s:json.loads((base/f'{s}.json').read_text()) for s in ['train','validation','test']}
        for s in splits:
            if sha256(base/f'{s}.json')!=provenance['split_hashes'][s]:raise ValueError('Dataset split manifest changed: '+s)
        source=splits['validation']; source=list(source);random.Random(ec['selection_seed']).shuffle(source)
        selected=[];seen_ids=set();seen_hashes=set();excluded=[]
        foreign_ids={r['image_id'] for s in ['train','test'] for r in splits[s]}
        foreign_hashes={r['image_sha256'] for s in ['train','test'] for r in splits[s]}
        for r in source:
            if r['image_id'] in seen_ids or r['image_sha256'] in seen_hashes:
                excluded.append({'sample_id':r['sample_id'],'reason':'duplicate image ID or content'});continue
            if r['image_id'] in foreign_ids or r['image_sha256'] in foreign_hashes:raise ValueError('Selected evaluation image overlaps other split')
            path=ROOT/r['image_path']
            if sha256(path)!=r['image_sha256']:raise ValueError('Image content mismatch: '+r['sample_id'])
            if len(r['answers'])!=10:raise ValueError('Missing original annotators')
            with Image.open(path) as image:image.verify()
            selected.append(r);seen_ids.add(r['image_id']);seen_hashes.add(r['image_sha256'])
            if len(selected)==ec['samples']:break
        if len(selected)!=160:raise ValueError('Insufficient cached unique samples')
        # Balance the KNOWN answer category across diagnostic batches. Residual
        # difficulty is not controlled and these batches are not GRPO candidates.
        batches=[[] for _ in range(10)]
        for category in sorted({r['answer_type'] for r in selected}):
            for r in [r for r in selected if r['answer_type']==category]:
                b=min((i for i in range(10) if len(batches[i])<16),key=lambda i:(sum(x['answer_type']==category for x in batches[i]),len(batches[i]),i))
                batches[b].append(r)
        check('manifest_unique_160_images',len(seen_ids)==160 and len(seen_hashes)==160)
        check('split_no_observed_leakage',not seen_ids & foreign_ids and not seen_hashes & foreign_hashes)
        manifest_records=[]
        for seed in cfg['seeds']:
            for b,records in enumerate(batches):
                for r in records:manifest_records.append({**r,'group_id':f'diagnostic_batch_{b:02d}','batch_id':b,'condition':'original',
                    'evaluation_seed':seed,'model_id':ec['model_id'],'model_revision':ec['model_revision'],'inference_microbatch':1})
        manifest={'schema':'benign_vqa_manifest_v1','created_unix':time.time(),'protocol_sha256':sha256(ROOT/'configs/gradient_feedback_validation_v1.json'),
            'dataset':provenance,'license':{'annotations':'CC BY 4.0','annotation_terms':'https://visualqa.org/terms.html',
            'images':'COCO Flickr images: per-image original licenses; see COCO terms. Cached official research images, not redistributed.',
            'image_terms':'https://cocodataset.org/#termsofuse'},'normalization':reference_metadata(),
            'selection':'seeded shuffle, deduplicate image ID and SHA256, take 160; balance answer_type across batches',
            'prior_exposure':'existing development split; previously evaluated. Reproducibility diagnostic, not new held-out efficacy evidence.',
            'model_pretraining_contamination':'unknown; cannot audit target pretraining data',
            'group_warning':'same examples across seeds; disjoint images within each seed. Different batches are not matched GRPO candidates.',
            'excluded':excluded,'records':manifest_records}
        dump(run/'evaluation_manifest.json',manifest); dump(run/'manifest_sha256.json',{'sha256':sha256(run/'evaluation_manifest.json')})
        if not args.editor_only:
            live('loading frozen Qwen')
            model=QwenVQA(model_id=ec['model_id'],revision=ec['model_revision'],device='cuda',max_new_tokens=ec['max_new_tokens'],
                          min_pixels=ec['min_pixels'],max_pixels=ec['max_pixels'],seed=cfg['seeds'][0],local_files_only=True)
            before=model_hash(model.model);check('Qwen_all_frozen',all(not p.requires_grad for p in model.model.parameters()))
            with (run/'predictions.jsonl').open('x',encoding='utf-8') as evidence:
                for seed in cfg['seeds']:
                    seed_everything(seed)
                    for b,records in enumerate(batches):
                        for r in records:
                            torch.cuda.synchronize();t=time.perf_counter();raw='';error='';valid=True
                            try:
                                with Image.open(ROOT/r['image_path']) as im:image=im.convert('RGB')
                                raw=model.answer([image],[r['question']])[0]
                                valid=bool(raw.strip()) and len(raw.strip())<500
                            except Exception as e:
                                error=repr(e);valid=False
                            torch.cuda.synchronize();latency=time.perf_counter()-t
                            scoring=score_answer(raw,r) if valid else {'score':0.,'normalized_answer':'','raw_answer':raw}
                            exact=float(normalize_answer(raw) in {normalize_answer(x) for x in r['answers']}) if valid else 0.
                            row={'sample_id':r['sample_id'],'image_id':r['image_id'],'image_sha256':r['image_sha256'],'batch_id':b,
                                'group_id':f'diagnostic_batch_{b:02d}','seed':seed,'condition':'original','category':r['answer_type'],
                                'question':r['question'],'raw_answer':raw,'normalized_answer':scoring['normalized_answer'],
                                'consensus_score':scoring['score'],'exact_any_reference':exact,'valid':valid,'error':error,'latency_seconds':latency,
                                'model_id':ec['model_id'],'model_revision':ec['model_revision'],'manifest_sha256':sha256(run/'evaluation_manifest.json')}
                            rows.append(row); evidence.write(json.dumps({**row,'references':r['answers']})+'\n');evidence.flush()
                        live('frozen Qwen original images',seed=seed,batch=b)
                        print(f'VQA seed={seed} batch={b} total={len(rows)}',flush=True)
            after=model_hash(model.model);check('Qwen_parameter_hash_invariant',before==after,before=before,after=after)
            param_rows.append({'model':'Qwen3-VL-2B-Instruct','parameter':'ALL','trainable':False,'optimizer_owned':False,'before_sha256':before,'after_sha256':after,'changed':before!=after})
            dump(run/'qwen_metadata.json',model.metadata())
            check('VQA_record_count',len(rows)==480);check('VQA_no_invalid_records',all(r['valid'] for r in rows))
            consistency=[]
            for r in selected:
                rr=[x for x in rows if x['sample_id']==r['sample_id']]
                consistency.append(len({x['raw_answer'] for x in rr})==1)
            check('Qwen_seed_replay_exact',all(consistency),exact_replay_examples=sum(consistency),total=len(consistency))
            del model;import gc;gc.collect();torch.cuda.empty_cache()
        live('loading frozen InstructPix2Pix')
        config=PretrainedEditorConfig(revision=ec['editor_revision'],variant='fp16',image_resolution=ec['editor_resolution'],num_inference_steps=ec['editor_steps'])
        editor=PretrainedEditor(config,device='cuda')
        modules={'unet':editor.pipe.unet,'vae':editor.pipe.vae,'text_encoder':editor.pipe.text_encoder}
        before={n:model_hash(m) for n,m in modules.items()}
        check('editor_all_parameters_frozen',all(not p.requires_grad for m in modules.values() for p in m.parameters()))
        check('editor_no_LoRA_loaded',not any('lora_' in n for n,_ in editor.pipe.unet.named_parameters()))
        selected_editor=[r for r in selected if r['answer_type']=='yes/no' and max(Counter(normalize_answer(x) for x in r['answers']).values())>=8][:3]
        instruction='Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.'
        from PIL import ImageDraw
        sheet=Image.new('RGB',(768,3*290),'white');draw=ImageDraw.Draw(sheet)
        for i,r in enumerate(selected_editor):
            with Image.open(ROOT/r['image_path']) as im:source=im.convert('RGB')
            source_prepared=editor.prepare_image(source);source_prepared.save(run/f'editor_{i}_source.png'); hashes=[];images=[]
            for repeat,seed in enumerate([42,42,123]):
                path=run/f'editor_{i}_seed{seed}_repeat{repeat}.png'
                result=editor.edit(source,instruction,r['image_id'],seed=seed,output_path=path)
                a=np.array(result.image);hashes.append(hashlib.sha256(a.tobytes()).hexdigest());images.append(result.image)
                row={'sample_id':r['sample_id'],'question':r['question'],'reference_answers':r['answers'],'seed':seed,'repeat':repeat,
                     'output_path':str(path),'output_sha256':sha256(path),'pixel_sha256':hashes[-1],
                     'size':list(result.image.size),'uint8_valid':bool(a.dtype==np.uint8 and np.isfinite(a).all()),
                     'input_mean_luminance':float(np.array(source_prepared.convert('L')).mean()),'output_mean_luminance':float(np.array(result.image.convert('L')).mean()),
                     'semantic_status':'PENDING_VISUAL_REVIEW_NOT_INCLUDED_IN_VQA','metadata':result.metadata}
                edits.append(row)
            check(f'editor_seed_replay_image_{i}',hashes[0]==hashes[1],same_seed_equal=hashes[0]==hashes[1],different_seed_equal=hashes[0]==hashes[2])
            check(f'editor_valid_dimensions_{i}',all(image.size==(256,256) for image in images))
            for j,image in enumerate([source_prepared,images[0],images[2]]):sheet.paste(image,(j*256,i*290+30))
            draw.text((4,i*290+2),f'{i} original | seed42 | seed123; ID {r["sample_id"]}',fill='black')
            live('frozen benign editor',completed_sources=i+1)
        sheet.save(run/'editor_contact_sheet.png')
        for n,m in modules.items():
            after=model_hash(m);check('editor_'+n+'_parameter_hash_invariant',before[n]==after,before=before[n],after=after)
            param_rows.append({'model':'InstructPix2Pix','parameter':n+'.ALL','trainable':False,'optimizer_owned':False,'before_sha256':before[n],'after_sha256':after,'changed':before[n]!=after})
        check('editor_outputs_valid',all(r['uint8_valid'] for r in edits),outputs=len(edits))
    except Exception as e:
        check('model_execution_error',False,error=repr(e),traceback=traceback.format_exc())
    finally:
        write_csv(run/'benign_vlm_evaluation.csv',rows);write_csv(run/'parameter_update_audit.csv',param_rows)
        dump(run/'editor_checks.json',edits);dump(run/'model_checks.json',{'passed':all(x['passed'] for x in checks),'checks':checks,'seconds':time.perf_counter()-started,
            'torch':torch.__version__,'python':sys.version,'slurm_job_id':os.environ.get('SLURM_JOB_ID'),'sources':{str(p.relative_to(ROOT)):sha256(p) for p in [Path(__file__),ROOT/'src/models/qwen_loader.py',ROOT/'src/editing/pretrained_editor.py',ROOT/'src/evaluation/vqa_official_compatible.py']}})
        live('complete' if all(x['passed'] for x in checks) else 'failed')
    return 0 if all(x['passed'] for x in checks) else 1


if __name__=='__main__':raise SystemExit(main())
