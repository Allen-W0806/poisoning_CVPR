"""Pretrained BF16 restart diagnostic with constant benign coefficient, never VLM reward."""
import argparse,copy,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,yaml
from PIL import Image
from torch.nn.attention import sdpa_kernel,SDPBackend
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.rl.diffusion_policy import DiffusionPolicy,INSTRUCTION
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.training.generator_adapter import adapter_state
from src.utils.seed import seed_everything
from src.rl.phase5_backend import require_training_reward
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b
def discrepancy(a,b):
    pairs=[(x,y) for x,y in zip(a,b) if x is not None and y is not None];den=sum(float(x.double().square().sum()) for x,y in pairs)
    return {'exact':same(a,b),'max_abs':max([float((x-y).abs().max()) for x,y in pairs] or [0.]),'relative_l2':(sum(float((x-y).double().square().sum()) for x,y in pairs)/(den+1e-300))**.5}
def frozen_hash(pipe):
    h=hashlib.sha256()
    for prefix,m in [('unet',pipe.unet),('vae',pipe.vae),('text',pipe.text_encoder)]:
        for n,p in m.named_parameters():
            if not p.requires_grad:h.update((prefix+n).encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());out=run/'diagnostic';out.mkdir(exist_ok=False);checks=[];started=time.time();comparisons=[];calls=0
    def check(n,b,**kw):row={'test':n,'status':'PASS' if b else 'FAIL',**kw};checks.append(row);print(json.dumps(row),flush=True)
    def live(stage):
        p=out/'live.json';p.write_text(json.dumps({'stage':stage,'elapsed_seconds':time.time()-started,'optimizer_calls':calls}))
    try:
        torch.set_num_threads(4);seed_everything(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
        for reward in [{},{'training_eligible':True}]:
            rejected=False
            try:require_training_reward(reward)
            except RuntimeError:rejected=True
            check('reward_guard_rejects_'+str(reward),rejected)
        assert torch.cuda.is_available() and os.environ.get('SLURM_JOB_ID')
        record=cfg['source_record'];path=ROOT/record['image_path'];assert sha(path)==record['image_sha256']
        with Image.open(path) as image:image=image.convert('RGB')
        source={'image':image,'sample_id':record['sample_id'],'image_id':record['image_id']}
        pc=yaml.safe_load((ROOT/'configs/phase3_generator_rl.yaml').read_text());pc.update(logical_batch_size=2,physical_micro_batch_size=1,gradient_accumulation_steps=2,num_inference_steps=2,gradient_checkpointing=True)
        dump(out/'effective_config.json',{'production_config':pc,'diagnostic_coefficient':1.,'instruction':INSTRUCTION,'no_VLM_loaded':True,'attack_metrics_used':False,'historical_scope':'No historical attack reward reused; this is an isolated numerical test, not replay of old reward gradients.'})
        def construct(disable_checkpoint=False):
            seed_everything(42);editor=load_pretrained_editor(ROOT,device='cuda');policy=DiffusionPolicy(editor,pc)
            if disable_checkpoint:policy.pipe.unet.disable_gradient_checkpointing()
            trainer=DiffusionRLTrainer(policy);return editor,policy,trainer
        def episodes(policy,condition,prefix):
            output=[];pixels=[];seeds=[]
            for i in range(2):
                seed=int(torch.randint(0,2**30,(1,)).item());im,t=policy.sample(condition,seed,f'{prefix}_{i}');seeds.append(seed);pixels.append(np.array(im));output.append((t,condition,1.))
            return output,pixels,seeds
        with sdpa_kernel(SDPBackend.MATH):
            live('load pretrained BF16 editor');editor,policy,trainer=construct();condition=policy.condition(source);frozen=frozen_hash(policy.pipe)
            warm,_,_=episodes(policy,condition,'warm');warm_result=trainer.update(warm);calls+=1;trainer.save(out/'before_next');dump(out/'warm_update.json',warm_result)
            live('reference stochastic continuation');ep,px,seeds=episodes(policy,condition,'next');torch.save({'episodes':[t for t,c,r in ep],'pixels':px,'seeds':seeds,'conditioning':{k:v.cpu() for k,v in condition.items() if isinstance(v,torch.Tensor)}},out/'reference_rollout.pt')
            receipt=trainer.update(ep);calls+=1;reference={'grads':copy.deepcopy(trainer.last_gradients),'adapter':adapter_state(policy.pipe.unet),'optimizer':copy.deepcopy(trainer.optimizer.state_dict()),'scheduler':copy.deepcopy(trainer.scheduler.state_dict()),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all()};torch.save(reference,out/'reference_after.pt');check('reference_frozen_base_unchanged',frozen_hash(policy.pipe)==frozen);trainer.save(out/'reference_after')
            reference_condition={k:v.detach().cpu() for k,v in condition.items() if isinstance(v,torch.Tensor)};reference_steps=[t['steps'] for t,c,r in ep]
            del trainer,policy,editor,condition,warm,ep;gc.collect();torch.cuda.empty_cache()
            for mode in ['matched_checkpoint_enabled','checkpoint_disabled_control']:
                live(mode);editor,policy,trainer=construct(mode=='checkpoint_disabled_control');trainer.restore(out/'before_next');state=torch.load(out/'before_next/training_state.pt',map_location='cpu',weights_only=False)
                restored=same(adapter_state(policy.pipe.unet),state['adapter']) and same(trainer.optimizer.state_dict(),state['optimizer']) and same(torch.get_rng_state(),state['rng_torch']) and same(torch.cuda.get_rng_state_all(),state['rng_cuda'])
                condition=policy.condition(source);ep,px2,seeds2=episodes(policy,condition,'next');forward=same(reference_steps,[t['steps'] for t,c,r in ep]);pixels=same(px,px2);conditioning=same(reference_condition,{k:v.detach().cpu() for k,v in condition.items() if isinstance(v,torch.Tensor)})
                update=trainer.update(ep);calls+=1;gradient=discrepancy(reference['grads'],trainer.last_gradients);adapter=same(reference['adapter'],adapter_state(policy.pipe.unet));optimizer=same(reference['optimizer'],trainer.optimizer.state_dict());scheduler=same(reference['scheduler'],trainer.scheduler.state_dict());rng=same(reference['rng_torch'],torch.get_rng_state()) and same(reference['rng_cuda'],torch.cuda.get_rng_state_all());frozen_ok=frozen_hash(policy.pipe)==frozen
                order=[('checkpoint_restore',restored),('conditioning',conditioning),('RNG_seed_draw',seeds==seeds2),('stochastic_forward',forward),('decoded_pixels',pixels),('backward_gradients',gradient['exact']),('optimizer_parameters',adapter),('optimizer_state',optimizer),('scheduler',scheduler),('post_step_RNG',rng)]
                result={'mode':mode,'earliest_divergence':next((n for n,b in order if not b),None),'comparisons':dict(order),'gradient_error':gradient,'frozen_parameters_unchanged':frozen_ok,'behavior_logp_replay_max_difference':update['behavior_logp_replay_max_difference'],'optimizer_call':calls,'interpretation':'checkpoint-disabled control deliberately changes execution; not the strict acceptance comparison' if mode.endswith('control') else 'locked strict fresh-object stochastic continuation'};comparisons.append(result);dump(out/(mode+'.json'),result);trainer.save(out/mode);torch.save({'gradients':trainer.last_gradients},out/(mode+'_gradients.pt'))
                if mode=='matched_checkpoint_enabled':
                    for n,b in order:check(n,b)
                    check('resumed_frozen_base_unchanged',frozen_ok)
                del trainer,policy,editor,condition,ep;gc.collect();torch.cuda.empty_cache()
        check('optimizer_budget',calls<=4)
        dump(out/'summary.json',{'status':'PASS' if all(x['status']=='PASS' for x in checks) else 'FAIL','comparisons':comparisons,'optimizer_calls':calls,'scope':'Actual pretrained IP2Pix BF16 / rank8 attention LoRA; two stochastic trajectories from one fixed benign original source, two diffusion steps; constant content-independent coefficient. No target VLM or attack reward.','historical_full_batch10_continuation':'NOT VALIDATED; old backward differences remain preserved; this smaller controlled test does not certify the historical run.','frozen_hash':frozen,'seconds':time.time()-started})
    except Exception:
        trace=traceback.format_exc();(out/'failure.txt').write_text(trace);print(trace,flush=True);check('diagnostic_exception',False,error=trace)
    finally:
        dump(out/'checks.json',{'status':'PASS' if checks and all(x['status']=='PASS' for x in checks) else 'FAIL','checks':checks,'optimizer_calls':calls,'elapsed_seconds':time.time()-started,'source_sha256':sha(Path(__file__))})
        dump(out/'artifact_inventory.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()]})
    if not checks or any(x['status']=='FAIL' for x in checks):raise SystemExit(1)
if __name__=='__main__':main()
