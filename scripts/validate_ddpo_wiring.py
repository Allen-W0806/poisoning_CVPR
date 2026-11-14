"""Actual DDPO trainer on a tiny synthetic UNet, then full IP2Pix ownership audit.
Only the randomly initialized tiny synthetic UNet's adapters receive updates.
The pretrained editor is instantiated for optimizer inventory with ZERO steps.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import types
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from diffusers import UNet2DConditionModel,DDIMScheduler
from src.training.generator_adapter import add_attention_lora,adapter_state
from src.rl.diffusion_trajectory import transition
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.rl.diffusion_policy import DiffusionPolicy
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.utils.seed import seed_everything


def dump(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def hashes(model):
    return {n:hashlib.sha256(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest() for n,p in model.named_parameters()}


def groups(trainer,model):
    output=[]
    for i,g in enumerate(trainer.optimizer.param_groups):
        ids={id(p) for p in g['params']}
        output.append({'index':i,'lr':g['lr'],'weight_decay':g['weight_decay'],'betas':g['betas'],'eps':g['eps'],
            'parameters':[{'name':n,'shape':list(p.shape),'numel':p.numel(),'dtype':str(p.dtype)} for n,p in model.named_parameters() if id(p) in ids]})
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args()
    run=ROOT/'results'/a.run_id;run.mkdir(parents=True,exist_ok=False);checks=[];started=time.perf_counter()
    def check(name,ok,**kw):
        checks.append({'test':name,'passed':bool(ok),**kw});print(json.dumps(checks[-1]),flush=True)
    try:
        seed_everything(42);torch.set_num_threads(4)
        unet=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,layers_per_block=1,
            block_out_channels=(32,),down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),
            cross_attention_dim=32,norm_num_groups=8,attention_head_dim=4).cuda()
        pipe=types.SimpleNamespace(unet=unet,vae=torch.nn.Identity(),text_encoder=torch.nn.Identity())
        metadata=add_attention_lora(pipe,rank=2,alpha=4)
        scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,steps_offset=1,set_alpha_to_one=False)
        scheduler.set_timesteps(2,device='cuda');pipe.scheduler=scheduler
        config={'physical_micro_batch_size':1,'gradient_accumulation_steps':2,'logical_batch_size':2,'learning_rate':1e-5,'eta':1.,'max_grad_norm':1.}
        policy=types.SimpleNamespace(pipe=pipe,config=config,editor=types.SimpleNamespace(device='cuda'),version=0,contract={'task':'tiny synthetic Gaussian terminal cosine','version':1})
        def predict(state,t,condition):return unet(state,t,encoder_hidden_states=condition['hidden']).sample
        policy.predict=predict;trainer=DiffusionRLTrainer(policy)
        inventory=groups(trainer,unet);dump(run/'tiny_optimizer_groups.json',inventory)
        owned={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
        check('tiny_production_optimizer_ownership',owned=={id(p) for p in unet.parameters() if p.requires_grad},trainable_parameters=metadata['trainable_parameters'])
        before={n:p.detach().cpu().clone() for n,p in unet.named_parameters()};before_hash=hashes(unet)
        receipts=[]
        @torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
        def episodes(seed):
            result=[]
            with torch.no_grad():
                for i in range(2):
                    g=torch.Generator(device='cuda').manual_seed(seed+i);state=torch.randn((1,4,8,8),generator=g,device='cuda')
                    condition={'sample_id':str(i),'image_id':str(i),'instruction':'benign synthetic cosine','hidden':torch.randn((1,3,32),generator=g,device='cuda')}
                    trajectory={'trajectory_id':f'{seed}_{i}','policy_version':policy.version,'contract':copy.deepcopy(policy.contract),
                        'sample_id':str(i),'image_id':str(i),'instruction':condition['instruction'],'steps':[]}
                    for t in scheduler.timesteps:
                        action,lp,_,_=transition(scheduler,predict(state,t,condition),t,state,noise=torch.randn(state.shape,generator=g,device='cuda'))
                        trajectory['steps'].append({'t':int(t),'state':state.cpu(),'action':action.cpu(),'logp':lp.cpu()});state=action
                    reward=float(state[0,0,0,0].cos());result.append((trajectory,condition,reward))
            return result
        for seed in [42,123]:receipts.append(trainer.update(episodes(seed)))
        dump(run/'tiny_production_updates.json',receipts)
        after_hash=hashes(unet);audit=[]
        for n,p in unet.named_parameters():
            delta=float((p.detach().cpu()-before[n]).abs().max())
            audit.append({'parameter':n,'trainable':p.requires_grad,'optimizer_owned':id(p) in owned,'max_abs_change':delta,
                'grad_norm':None if p.grad is None else float(p.grad.norm()),'gradient_finite':p.grad is None or bool(torch.isfinite(p.grad).all()),
                'before_sha256':before_hash[n],'after_sha256':after_hash[n]})
        dump(run/'tiny_parameter_updates.json',audit)
        check('tiny_frozen_parameters_invariant',all(r['before_sha256']==r['after_sha256'] for r in audit if not r['trainable']))
        check('tiny_LoRA_A_and_B_updated',any('lora_A' in r['parameter'] and r['max_abs_change']>0 for r in audit) and any('lora_B' in r['parameter'] and r['max_abs_change']>0 for r in audit))
        check('tiny_gradients_finite_nonzero',all(r['gradient_finite'] for r in audit) and all(r['gradient_norm']>0 for r in receipts))
        # Exact production checkpoint save/reload. No pretrained checkpoint touched.
        trainer.save(run/'tiny_checkpoint');saved=adapter_state(unet)
        ep=episodes(3407);expected_receipt=trainer.update(ep);expected=adapter_state(unet)
        trainer.restore(run/'tiny_checkpoint')
        check('tiny_checkpoint_reload_exact',all(torch.equal(v,saved[k]) for k,v in adapter_state(unet).items()))
        resumed_receipt=trainer.update(ep)
        check('tiny_checkpoint_fixed_trajectory_step_exact',all(torch.equal(v,expected[k]) for k,v in adapter_state(unet).items()),
            expected_gradient_norm=expected_receipt['gradient_norm'],resumed_gradient_norm=resumed_receipt['gradient_norm'])
        # Explicitly restore to second step for final evidence checkpoint.
        trainer.restore(run/'tiny_checkpoint')
        del trainer,policy,pipe,unet;import gc;gc.collect();torch.cuda.empty_cache()
        import yaml
        ec=yaml.safe_load((ROOT/'configs/phase3_generator_rl.yaml').read_text())
        editor=load_pretrained_editor(ROOT,device='cuda');policy=DiffusionPolicy(editor,ec);trainer=DiffusionRLTrainer(policy)
        full=groups(trainer,policy.pipe.unet);dump(run/'pretrained_IP2Pix_optimizer_groups_ZERO_STEPS.json',full)
        owned={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
        check('pretrained_optimizer_only_attention_LoRA',all('lora_' in n for n,p in policy.pipe.unet.named_parameters() if id(p) in owned),
            optimizer_steps=trainer.step,parameter_tensors=sum(len(g['parameters']) for g in full),parameter_elements=sum(p['numel'] for g in full for p in g['parameters']))
        check('pretrained_VAE_text_encoder_frozen',all(not p.requires_grad for module in [editor.pipe.vae,editor.pipe.text_encoder] for p in module.parameters()))
    except Exception as e:check('wiring_execution_error',False,error=repr(e),traceback=traceback.format_exc())
    finally:dump(run/'wiring_checks.json',{'checks':checks,'passed':all(x['passed'] for x in checks),'seconds':time.perf_counter()-started,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    return 0 if all(x['passed'] for x in checks) else 1


if __name__=='__main__':raise SystemExit(main())
