"""Bounded original-VQA language-attention LoRA engineering validation; no RL."""
import argparse,copy,gc,hashlib,importlib.metadata,json,os,random,subprocess,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn.attention import sdpa_kernel,SDPBackend
from peft import LoraConfig,get_peft_model
from src.models.qwen_loader import QwenVQA
from src.evaluation.vqa_official_compatible import score_answer
from src.utils.seed import seed_everything
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def tensorhash(items):
    h=hashlib.sha256()
    for n,p in items:
        h.update(n.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
def equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b
def rng():return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all(),'numpy':np.random.get_state(),'python':random.getstate()}
def restore_rng(r):torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);ap.add_argument('--phase',choices=['pilot','main'],required=True);ap.add_argument('--revision',default='');a=ap.parse_args()
    run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());data=json.loads((run/'clean_data_manifest.json').read_text());out=run/(f'clean_{a.phase}'+('_'+a.revision if a.revision else ''));out.mkdir(exist_ok=False)
    checks=[];updates=[];preds=[];start=time.perf_counter();started=time.time();steps=cfg[f'{a.phase}_steps'];qwen=None
    def check(name,ok,**kw):
        row={'test':name,'status':'PASS' if bool(ok) else 'FAIL',**kw};checks.append(row);print(json.dumps(row),flush=True)
        if not ok:raise AssertionError(name)
    def live(stage,**kw):
        gpu=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
        temp=out/'live.partial';temp.write_text(json.dumps({'stage':stage,'seconds':time.perf_counter()-start,'gpu':gpu,**kw},indent=2));temp.replace(out/'live.json')
    try:
        accepted=json.loads((run/'accepted_clean_phases.json').read_text()) if (run/'accepted_clean_phases.json').exists() else {}
        if a.phase=='main':check('pilot_gate',json.loads((run/accepted.get('pilot','clean_pilot')/'checks.json').read_text())['status']=='PASS')
        check('manifest_protocol_frozen',data['protocol_sha256']==sha(run/'protocol.json'))
        train=data['train'];evaluation=data['evaluation'];check('source_ID_disjoint',not {r['image_id'] for r in train}&{r['image_id'] for r in evaluation});check('content_disjoint',not {r['image_sha256'] for r in train}&{r['image_sha256'] for r in evaluation})
        from PIL import Image
        for r in train+evaluation:
            check('original_hash_'+r['sample_id'],sha(ROOT/r['image_path'])==r['image_sha256'])
        torch.set_num_threads(4);seed_everything(cfg['seed']);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        check('Slurm_CUDA',torch.cuda.is_available() and bool(os.environ.get('SLURM_JOB_ID')))
        live('load pinned clean target')
        qwen=QwenVQA(model_id=cfg['model_id'],revision=cfg['model_revision'],device='cuda',max_new_tokens=cfg['max_new_tokens'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'],seed=cfg['seed'])
        original_model=qwen.model
        lora=cfg['lora'];lc=LoraConfig(r=lora['r'],lora_alpha=lora['lora_alpha'],lora_dropout=lora['lora_dropout'],target_modules=lora['target_regex'],bias='none')
        model=get_peft_model(original_model,lc);qwen.model=model;model.config.use_cache=False
        for n,p in model.named_parameters():
            if p.requires_grad:p.data=p.data.float()
        params={n:p for n,p in model.named_parameters() if p.requires_grad}
        check('only_language_attention_LoRA',bool(params) and all('language_model.layers.' in n and 'lora_' in n and ('.q_proj.' in n or '.v_proj.' in n) for n in params))
        check('vision_frozen',all(not p.requires_grad for n,p in model.named_parameters() if '.visual.' in n))
        op=cfg['optimizer'];optimizer=torch.optim.AdamW(list(params.values()),lr=op['lr'],betas=tuple(op['betas']),eps=op['eps'],weight_decay=op['weight_decay']);scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
        check('optimizer_exact_ownership',{id(p) for g in optimizer.param_groups for p in g['params']}=={id(p) for p in params.values()})
        dump(out/'optimizer_groups.json',{'tensors':[{'name':n,'shape':list(p.shape),'dtype':str(p.dtype),'elements':p.numel(),'group':0} for n,p in params.items()],'elements':sum(p.numel() for p in params.values()),'optimizer':op,'lora':lora,'scheduler':cfg['scheduler']})
        frozen_before=tensorhash((n,p) for n,p in model.named_parameters() if not p.requires_grad);initial={n:p.detach().cpu().clone() for n,p in params.items()}
        def evaluate(label,rows):
            model.eval();model.config.use_cache=True
            with sdpa_kernel(SDPBackend.MATH):
                for j,r in enumerate(rows):
                    with Image.open(ROOT/r['image_path']) as im:im=im.convert('RGB')
                    torch.cuda.synchronize();t=time.perf_counter();answer=qwen.answer([im],[r['question']])[0];torch.cuda.synchronize()
                    score=score_answer(answer,r)
                    record={'stage':label,'sample_id':r['sample_id'],'image_id':r['image_id'],'image_sha256':r['image_sha256'],'answer_type':r['answer_type'],'raw_answer':answer,'normalized_answer':score['normalized_answer'],'score':score['score'],'latency_seconds':time.perf_counter()-t,'model_revision':cfg['model_revision']};preds.append(record)
                    with (out/'predictions.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
                    if j%8==0:live(label,evaluated=j+1)
            model.config.use_cache=False
        evalrows=evaluation[:8] if a.phase=='pilot' else evaluation
        evaluate('before',evalrows)
        processor=qwen.processor;accum=cfg['gradient_accumulation']
        def example(r):
            with Image.open(ROOT/r['image_path']) as im:im=im.convert('RGB')
            msgs=[{'role':'system','content':[{'type':'text','text':qwen.system_prompt}]},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':r['question']}]}]
            prompt=processor.apply_chat_template([msgs],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True)
            full=processor.apply_chat_template([msgs+[{'role':'assistant','content':[{'type':'text','text':r['training_answer']}]}]],tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',padding=True)
            n=prompt['input_ids'].shape[1]
            if not torch.equal(prompt['input_ids'],full['input_ids'][:,:n]):raise AssertionError('response-only prefix mismatch')
            labels=full['input_ids'].clone();labels[:,:n]=-100
            if not (labels!=-100).any():raise AssertionError('empty response labels')
            full['labels']=labels;return full.to('cuda')
        def step(k):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[];sampleids=[]
            with sdpa_kernel(SDPBackend.MATH):
                for j in range(accum):
                    r=train[(k*accum+j)%len(train)];batch=example(r);result=model(**batch);loss=result.loss/accum
                    if not loss.requires_grad or not torch.isfinite(loss):raise AssertionError('loss disconnected or nonfinite')
                    losses.append(float(loss.detach())*accum);sampleids.append(r['sample_id']);loss.backward()
            grads={n:p.grad.detach().cpu().clone() for n,p in params.items() if p.grad is not None}
            if len(grads)!=len(params) or not all(torch.isfinite(x).all() for x in grads.values()):raise AssertionError('missing/nonfinite gradient')
            norm=torch.nn.utils.clip_grad_norm_(list(params.values()),op['max_grad_norm']);norm=float(norm)
            if not np.isfinite(norm) or norm==0:raise AssertionError('zero/nonfinite gradient norm')
            before={n:p.detach().cpu().clone() for n,p in params.items()};optimizer.step();scheduler.step()
            if not all(torch.isfinite(v).all() for state in optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)):raise AssertionError('nonfinite optimizer')
            changes={n:float((p.detach().cpu()-before[n]).abs().max()) for n,p in params.items()}
            if not any(changes.values()):raise AssertionError('no parameter change')
            row={'step':k+1,'sample_ids':sampleids,'loss':float(np.mean(losses)),'grad_norm_before_clip':norm,'grad_norm_cap':op['max_grad_norm'],'changed_tensors':sum(x>0 for x in changes.values()),'max_parameter_change':max(changes.values()),'finite_optimizer':True};updates.append(row)
            with (out/'updates.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            live('clean optimizer step',step=k+1,loss=row['loss'],grad_norm=norm)
            return grads,row
        def snapshot(k):return {'params':{n:p.detach().cpu().clone() for n,p in params.items()},'optimizer':copy.deepcopy(optimizer.state_dict()),'scheduler':copy.deepcopy(scheduler.state_dict()),'rng':rng(),'step':k,'protocol_sha256':sha(run/'protocol.json'),'manifest_sha256':sha(run/'clean_data_manifest.json'),'model_revision':cfg['model_revision']}
        def restore(state):
            for n,p in params.items():p.data.copy_(state['params'][n].to(p.device))
            # load_state_dict can alias the CPU step tensor; never consume the replay reference.
            optimizer.load_state_dict(copy.deepcopy(state['optimizer']));scheduler.load_state_dict(copy.deepcopy(state['scheduler']));restore_rng(state['rng'])
        checkpoint_at=1 if a.phase=='pilot' else 8
        for k in range(steps):
            if k==checkpoint_at:
                state=snapshot(k);checkpoint=out/'continuation_checkpoint.pt';torch.save(state,checkpoint);loaded=torch.load(checkpoint,map_location='cpu',weights_only=False)
                # Move optimizer tensors through load_state_dict onto actual parameter devices.
                restore(loaded);check('checkpoint_adapter_bitwise_restore',all(torch.equal(p.detach().cpu(),loaded['params'][n]) for n,p in params.items()))
                immutable_optimizer=copy.deepcopy(loaded['optimizer'])
                gradients,reference_row=step(k);expected=snapshot(k+1)
                check('checkpoint_optimizer_reference_not_mutated',equal(loaded['optimizer'],immutable_optimizer))
                restore(loaded);repeated_gradients,repeated_row=step(k);actual=snapshot(k+1)
                check('continuation_gradient_bitwise_equal',equal(gradients,repeated_gradients))
                check('continuation_adapter_bitwise_equal',equal(expected['params'],actual['params']))
                check('continuation_optimizer_bitwise_equal',equal(expected['optimizer'],actual['optimizer']))
                check('continuation_scheduler_equal',equal(expected['scheduler'],actual['scheduler']))
                check('continuation_CUDA_RNG_equal',equal(expected['rng']['cuda'],actual['rng']['cuda']))
                check('continuation_CPU_RNG_equal',equal(expected['rng']['torch'],actual['rng']['torch']))
                dump(out/'continuation_receipt.json',{'status':'PASS','checkpoint_step':k,'checkpoint_sha256':sha(checkpoint),'same_instance_disk_restore':True,'extra_replayed_optimizer_calls':1,'full_base_model_optimizer_resume':False})
            else:step(k)
        finalstate=snapshot(steps);torch.save(finalstate,out/'final_training_state.pt');model.save_pretrained(out/'adapter')
        check('frozen_base_hash',tensorhash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_before,before=frozen_before)
        change=[{'parameter':n,'shape':list(p.shape),'initial_sha256':tensorhash([(n,initial[n])]),'final_sha256':tensorhash([(n,p)]),'max_abs_change':float((p.detach().cpu()-initial[n]).abs().max())} for n,p in params.items()]
        check('all_adapter_tensors_changed_after_two_or_more_steps',all(r['max_abs_change']>0 for r in change))
        dump(out/'parameter_updates.json',change)
        evaluate('after',evalrows);evaluate('after_repeat',evalrows)
        after=[r for r in preds if r['stage']=='after'];repeat=[r for r in preds if r['stage']=='after_repeat'];check('posttraining_greedy_repeat',all(x['raw_answer']==y['raw_answer'] and x['score']==y['score'] for x,y in zip(after,repeat)))
        # Fresh object construction from pinned base plus public adapter files.
        expectedhash=tensorhash(params.items());live('fresh base plus adapter reload')
        from peft import PeftModel
        fresh=QwenVQA(model_id=cfg['model_id'],revision=cfg['model_revision'],device='cuda',max_new_tokens=cfg['max_new_tokens'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'],seed=cfg['seed'])
        fresh.model=PeftModel.from_pretrained(fresh.model,out/'adapter',is_trainable=True)
        freshparams={n:p for n,p in fresh.model.named_parameters() if p.requires_grad}
        check('fresh_object_adapter_reload_bitwise',tensorhash(freshparams.items())==expectedhash)
        freshoptimizer=torch.optim.AdamW(list(freshparams.values()),lr=op['lr'],betas=tuple(op['betas']),eps=op['eps'],weight_decay=op['weight_decay'])
        # AdamW has CPU step counters and CUDA moment tensors. Let the optimizer
        # restore each field to its required device rather than moving all to CUDA.
        saved=torch.load(out/'final_training_state.pt',map_location='cpu',weights_only=False);freshoptimizer.load_state_dict(copy.deepcopy(saved['optimizer']))
        check('fresh_object_optimizer_reload_bitwise',equal(optimizer.state_dict(),freshoptimizer.state_dict()))
        qwen=fresh;model=fresh.model;evaluate('after_reload',evalrows)
        reloadpreds=[r for r in preds if r['stage']=='after_reload'];check('fresh_object_prediction_reload_exact',all(x['raw_answer']==y['raw_answer'] and x['score']==y['score'] for x,y in zip(after,reloadpreds)))
        check('fresh_base_hash_matches',tensorhash((n,p) for n,p in model.named_parameters() if not p.requires_grad)==frozen_before)
        dump(out/'summary.json',{'status':'PASS','training_steps':steps,'actual_optimizer_calls_including_replay':len(updates),'trainable_tensors':len(params),'trainable_elements':sum(p.numel() for p in params.values()),'evaluated_unique_images':len(evalrows),'before_accuracy':float(np.mean([r['score'] for r in preds if r['stage']=='before'])),'after_accuracy':float(np.mean([r['score'] for r in after])),'frozen_base_sha256':frozen_before,'adapter_sha256':expectedhash,'peak_allocated_cuda_bytes':torch.cuda.max_memory_allocated(),'full_model_tuning':False,'strict_repeated_continuation':'PASS for clean language LoRA, fixed original examples, math SDPA','pretrained_diffusion_continuation':'NOT VALIDATED; previous failures unchanged'})
    except Exception:
        error=traceback.format_exc();(out/'failure.txt').write_text(error);print(error,flush=True);checks.append({'test':'worker_exception','status':'FAIL','error':error})
    finally:
        dump(out/'checks.json',{'status':'PASS' if checks and all(x['status']=='PASS' for x in checks) else 'FAIL','checks':checks,'elapsed_seconds':time.perf_counter()-start,'started_unix':started,'finished_unix':time.time(),'optimizer_calls':len(updates),'unique_training_steps':steps if checks and all(x['status']=='PASS' for x in checks) else None,'prediction_records':len(preds),'phase':a.phase,'training_kind':'clean original VQA language-attention LoRA only','versions':{p:importlib.metadata.version(p) for p in ['torch','transformers','peft','Pillow']}})
    if not checks or not all(x['status']=='PASS' for x in checks):raise SystemExit(1)
if __name__=='__main__':main()
