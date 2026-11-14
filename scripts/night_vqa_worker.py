"""Gated normal-VQA worker; immutable recipe, complete checkpoints, no attack loop."""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256,metric_summary
from src.operations.night_budget import finite_update


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--campaign',required=True);parser.add_argument('--stage',required=True)
    parser.add_argument('--attempt',type=int,default=0);parser.add_argument('--resume');a=parser.parse_args()
    run=Path(a.campaign).resolve();assets=run.parents[1];cfg=json.loads((run/'protocol.json').read_text())
    receipt=json.loads((run/'prepare_receipt.json').read_text());manifest=json.loads((run/'manifest.json').read_text())
    if receipt['status']!='PASS_DATA_PREPARED' or receipt['manifest_SHA256']!=sha256(run/'manifest.json') or receipt['protocol_SHA256']!=sha256(run/'protocol.json'):
        raise ValueError('Data/protocol lock mismatch')
    if a.stage not in cfg['stage_wall_seconds']:raise ValueError('Stage not allowlisted')
    if a.stage!='canary' and json.loads((run/'stage_success/canary.json').read_text())['status']!='PASS':raise ValueError('New canary gate not passed')
    output=run/'stages'/a.stage/f'attempt_{a.attempt}';output.mkdir(parents=True,exist_ok=False)
    started=time.time();job=os.getenv('SLURM_JOB_ID');state={'stage':a.stage,'attempt':a.attempt,'job_id':job,'status':'STARTED','started_UTC':datetime.now(timezone.utc).isoformat(),'PID':os.getpid(),'actual_optimizer_calls':0}
    import numpy as np
    import torch
    from PIL import Image
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from peft import PeftModel
    from src.models.qwen_loader import QwenVQA
    from src.training.clean_vlm_pipeline import CleanVLMTrainer,attach_language_lora,parameter_hash
    from src.evaluation.vqa_official_compatible import score_answer
    from src.mini_pilot.engine import cpu_copy,compare_trees
    if not job or not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError('Single explicit Slurm GPU required')
    seed=42 if a.stage in ('canary','frozen') else int(a.stage.removeprefix('seed'))
    torch.set_num_threads(4);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    cap=cfg['stage_wall_seconds'][a.stage]-90
    def live(phase,**kw):
        if time.time()-started>cap:raise TimeoutError('Prespecified stage wall cap; no blind continuation')
        record={'phase':phase,'updated_unix':time.time(),'elapsed_seconds':time.time()-started,'job_id':job,
                'stage':a.stage,'attempt':a.attempt,'GPU_allocated_bytes':torch.cuda.memory_allocated(),
                'GPU_reserved_bytes':torch.cuda.memory_reserved(),**kw}
        put(output/'live.json',record);put(run/'stage_live'/f'{a.stage}.json',record)
        print(json.dumps({'progress':record}),flush=True)
    def emit_update(row,label):
        if not finite_update(row):raise FloatingPointError('Nonfinite actual clean optimizer update')
        state['actual_optimizer_calls']+=1;row={'label':label,'UTC':datetime.now(timezone.utc).isoformat(),**row}
        with (output/'training.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        live('training',retained_step=row['retained_step'],loss=row['loss'],actual_optimizer_calls=state['actual_optimizer_calls'])
    kernel=torch.randn(16,16,device='cuda',generator=torch.Generator(device='cuda').manual_seed(seed))
    if not torch.isfinite(kernel@kernel).all():raise RuntimeError('CUDA kernel witness failed')
    state['GPU']={'name':torch.cuda.get_device_name(),'total_bytes':torch.cuda.get_device_properties(0).total_memory,
                  'initial_free_bytes':torch.cuda.mem_get_info()[0],'physical_job_GPU':os.getenv('SLURM_JOB_GPUS')}
    def load():
        live('load_pinned_Qwen')
        return QwenVQA(model_id=cfg['model_id'],revision=cfg['model_revision'],device='cuda',seed=seed,
                       min_pixels=4096,max_pixels=cfg['max_pixels'],max_new_tokens=cfg['max_new_tokens'])
    qwen=trainer=None
    def evaluate(rows,label):
        qwen.model.eval();qwen.model.config.use_cache=True;predictions=[]
        with sdpa_kernel(SDPBackend.MATH):
            for index,row in enumerate(rows):
                if index%8==0:live('evaluate_'+label,evaluated=index,total=len(rows))
                with Image.open(assets/row['image_path']) as image:image=image.convert('RGB')
                answer=qwen.answer([image],[row['question']])[0];score=score_answer(answer,row)
                record={'stage':a.stage,'seed':seed,'label':label,'sample_id':row['sample_id'],'image_id':row['image_id'],
                        'source_SHA256':row['image_sha256'],'answer_type':row['answer_type'],'prediction':answer,
                        'normalized_answer':score['normalized_answer'],'score':score['score']}
                predictions.append(record)
                with (output/'predictions.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
        qwen.model.config.use_cache=False
        return predictions,metric_summary([r['score'] for r in predictions],cfg['analysis_seed'],cfg['bootstrap_resamples'])
    try:
        qwen=load()
        if a.stage=='frozen':
            results={}
            for split in ('dev','test','transfer'):
                _,results[split]=evaluate(manifest['splits'][split],'frozen_'+split)
            state.update(status='PASS',metrics=results,optimizer_steps=0,scope='shared frozen greedy baseline; no training cost; common evaluation sources')
        else:
            qwen=attach_language_lora(qwen,cfg['vlm_rank'],cfg['vlm_alpha'])
            protocol={k:cfg[k] for k in ('vlm_rank','vlm_alpha','vlm_learning_rate','vlm_accumulation','vlm_max_grad_norm','analysis_seed','bootstrap_resamples')}
            protocol.update(seed=seed,model_revision=cfg['model_revision'],max_pixels=cfg['max_pixels'],max_new_tokens=cfg['max_new_tokens'])
            train=manifest['splits']['train'];stream=[]
            for epoch in range((cfg['updates']*cfg['vlm_accumulation']+len(train)-1)//len(train)):
                indices=list(range(len(train)));random.Random(seed+epoch*1000003).shuffle(indices);stream+=indices
            import hashlib
            protocol['stream_SHA256']=hashlib.sha256(json.dumps(stream).encode()).hexdigest()
            trainer=CleanVLMTrainer(qwen,assets,protocol,receipt['manifest_SHA256'])
            frozen_hash=parameter_hash((n,p) for n,p in qwen.model.named_parameters() if not p.requires_grad)
            state['frozen_base_SHA256']=frozen_hash
            history=[]
            if a.resume:
                pointer=json.loads(Path(a.resume).read_text());trainer.restore(Path(pointer['path']));history=pointer.get('dev_history',[])
                if pointer['manifest_SHA256']!=receipt['manifest_SHA256'] or trainer.step!=pointer['step']:raise ValueError('Resume cursor/data mismatch')
            def batch(k):
                start=k*cfg['vlm_accumulation'];return [train[i] for i in stream[start:start+cfg['vlm_accumulation']]]
            def save(label):
                directory=output/'checkpoints'/label;trainer.save(directory)
                pointer={'path':str(directory),'step':trainer.step,'data_cursor':trainer.step*cfg['vlm_accumulation'],
                         'manifest_SHA256':receipt['manifest_SHA256'],'protocol_SHA256':sha256(run/'protocol.json'),
                         'state_SHA256':sha256(directory/'training_state.pt'),'dev_history':history}
                put(run/'latest_checkpoints'/f'{a.stage}.json',pointer)
                return directory
            if a.stage=='canary':
                _,before=evaluate(manifest['splits']['dev'][:8],'canary_dev_before')
                emit_update(trainer.update(batch(0)),'canary_first')
                checkpoint=save('step1')
                emit_update(trainer.update(batch(1)),'canary_reference_next')
                expected={'state':trainer.state(),'gradients':cpu_copy(trainer.raw_gradients)}
                fresh=load();fresh.model=PeftModel.from_pretrained(fresh.model,checkpoint/'adapter',is_trainable=True)
                saved=torch.load(checkpoint/'training_state.pt',map_location='cpu',weights_only=False)
                loaded_params={n:p.detach().cpu() for n,p in fresh.model.named_parameters() if p.requires_grad}
                if not compare_trees(saved['params'],loaded_params)['exact']:raise AssertionError('Public adapter load differs before full-state restoration')
                qwen=fresh;trainer=CleanVLMTrainer(qwen,assets,protocol,receipt['manifest_SHA256']);trainer.restore(checkpoint)
                emit_update(trainer.update(batch(1)),'canary_fresh_object_next')
                actual={'state':trainer.state(),'gradients':cpu_copy(trainer.raw_gradients)}
                comparison=compare_trees(expected,actual);put(output/'resume_comparison.json',comparison)
                torch.save(expected,output/'reference_next.pt');torch.save(actual,output/'resumed_next.pt')
                if not comparison['exact']:raise AssertionError('Canary fresh-object next update mismatch')
                _,after=evaluate(manifest['splits']['dev'][:8],'canary_dev_after')
                final=save('canary_final')
                if parameter_hash((n,p) for n,p in qwen.model.named_parameters() if not p.requires_grad)!=frozen_hash:raise AssertionError('Canary base weight mutation')
                state.update(status='PASS',resume_exact=True,public_adapter_load_exact=True,dev_before=before,dev_after=after,
                             retained_steps=trainer.step,checkpoint=str(final),profile_seconds=time.time()-started)
            else:
                while trainer.step<cfg['updates']:
                    k=trainer.step;emit_update(trainer.update(batch(k)),'formal_clean_SFT')
                    if trainer.step in cfg['dev_steps']:
                        _,stats=evaluate(manifest['splits']['dev'],f'dev_step{trainer.step}')
                        directory=output/'checkpoints'/f'step{trainer.step}'
                        history.append({'step':trainer.step,'score':stats['mean'],'path':str(directory)})
                    if trainer.step%cfg['checkpoint_every']==0 or trainer.step==cfg['updates']:save(f'step{trainer.step}')
                if not history:raise AssertionError('No prespecified dev selection observations')
                selected=max(history,key=lambda r:(r['score'],-r['step']));trainer.restore(Path(selected['path']))
                if parameter_hash((n,p) for n,p in qwen.model.named_parameters() if not p.requires_grad)!=frozen_hash:raise AssertionError('Frozen base mutation')
                results={}
                for split in ('test','transfer'):
                    _,results[split]=evaluate(manifest['splits'][split],'selected_'+split)
                selected_path=output/'selected_checkpoint';trainer.save(selected_path)
                state.update(status='PASS',seed=seed,attempted_training_steps=cfg['updates'],retained_step=trainer.step,
                    dev_history=history,selected_by='maximum dev score; earliest step on tie; fixed full512 attempted',
                    metrics=results,checkpoint=str(selected_path),checkpoint_SHA256=sha256(selected_path/'training_state.pt'))
        live('completed')
        state['finished_UTC']=datetime.now(timezone.utc).isoformat();state['elapsed_seconds']=time.time()-started
        state['peak_GPU_allocated_bytes']=torch.cuda.max_memory_allocated()
        put(output/'metrics.json',state)
        put(run/'stage_success'/f'{a.stage}.json',{'status':'PASS','stage':a.stage,'attempt':a.attempt,'metrics_path':str(output/'metrics.json'),'metrics_SHA256':sha256(output/'metrics.json')})
    except Exception:
        trace=traceback.format_exc();state.update(status='FAILED',trace=trace,elapsed_seconds=time.time()-started)
        (output/'failure.txt').write_text(trace);put(output/'metrics.json',state);print(trace,flush=True);raise
    finally:
        put(output/'status.json',state)
        del trainer,qwen;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':main()
