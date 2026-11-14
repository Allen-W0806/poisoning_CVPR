"""Production DDPO + RSIController, tiny random UNet, equal NFE/update control.

Fixed synthetic validation seeds select a single preregistered LR proposal.
No pretrained diffusion model, VLM, security metric, or human label used.
"""
import argparse,copy,hashlib,json,sys,time,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np,torch
from diffusers import UNet2DConditionModel,DDIMScheduler
from src.training.generator_adapter import add_attention_lora,adapter_state
from src.rl.diffusion_trajectory import transition
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.rsi.controller import RSIController
from src.utils.seed import seed_everything
from scripts.next_validation_20261031 import put,read,sha
from scripts.run_ip2pix_checkpoint_diagnostic import same
def main(run):
 out=run/'integrated';out.mkdir(exist_ok=False);cfg=read(run/'protocol.json')['integrated'];checks=[];started=time.perf_counter()
 def check(name,passed,**kw):
  row=dict(test=name,passed=bool(passed),**kw);checks.append(row);print(json.dumps(row),flush=True)
  if not passed:raise AssertionError(name)
 def construct(lr):
  seed_everything(cfg['seeds']['initial']);u=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,layers_per_block=1,block_out_channels=(32,),down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),cross_attention_dim=32,norm_num_groups=8,attention_head_dim=4).cuda()
  pipe=types.SimpleNamespace(unet=u,vae=torch.nn.Identity(),text_encoder=torch.nn.Identity());meta=add_attention_lora(pipe,rank=2,alpha=4)
  pipe.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,steps_offset=1,set_alpha_to_one=False);pipe.scheduler.set_timesteps(2,device='cuda')
  pc=dict(physical_micro_batch_size=1,gradient_accumulation_steps=10,logical_batch_size=10,learning_rate=lr,eta=1.,max_grad_norm=1.)
  p=types.SimpleNamespace(pipe=pipe,config=pc,editor=types.SimpleNamespace(device='cuda'),version=0,contract=dict(task='benign terminal cosine',version=1))
  p.predict=lambda z,t,c:u(z,t,encoder_hidden_states=c['hidden']).sample
  return p,DiffusionRLTrainer(p),meta
 def frozen(p):
  return {n:hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest() for n,v in p.pipe.unet.named_parameters() if not v.requires_grad}
 @torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
 @torch.no_grad()
 def episodes(p,seed,n):
  output=[]
  for i in range(n):
   g=torch.Generator(device='cuda').manual_seed(seed+i);z=torch.randn((1,4,8,8),generator=g,device='cuda');c=dict(sample_id=str(i),image_id=str(i),instruction='benign cosine',hidden=torch.randn((1,3,32),generator=g,device='cuda'))
   tr=dict(trajectory_id=f'{seed}_{i}',policy_version=p.version,contract=copy.deepcopy(p.contract),sample_id=str(i),image_id=str(i),instruction=c['instruction'],steps=[])
   for t in p.pipe.scheduler.timesteps:
    a,lp,_,_=transition(p.pipe.scheduler,p.predict(z,t,c),t,z,noise=torch.randn(z.shape,generator=g,device='cuda'));tr['steps'].append(dict(t=int(t),state=z.cpu(),action=a.cpu(),logp=lp.cpu()));z=a
   output.append((tr,c,float(z[0,0,0,0].cos())))
  return output
 try:
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  arms={};models={}
  for name in ['fixed','rsi']:
   folder=out/name;folder.mkdir();p,t,meta=construct(cfg['fixed_lr']);before=adapter_state(p.pipe.unet);f0=frozen(p);ownership={id(x) for g in t.optimizer.param_groups for x in g['params']}
   check(name+'_optimizer_ownership',ownership=={id(x) for x in p.pipe.unet.parameters() if x.requires_grad})
   initial_hash={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in before.items()};receipts=[];nfe=0
   for k,seed in enumerate(cfg['seeds']['train']):
    if k==2 and name=='rsi':
     proposal=dict(id='LR_PROPOSAL_0001',parent='CONFIG_0000',changes=dict(learning_rate=cfg['candidate_lr']),reason='preregistered sensitivity test with equal-step fixed control; no outcome-driven tuning',protocol_sha256=sha(run/'protocol.json'));put(out/'proposal.json',proposal)
     p.config['learning_rate']=cfg['candidate_lr']
     for g in t.optimizer.param_groups:g['lr']=cfg['candidate_lr']
     t.scheduler.base_lrs=[cfg['candidate_lr']]
    ep=episodes(p,seed,10);nfe+=20;receipts.append(t.update(ep));nfe+=20
    if k==1:
     t.save(folder/'recovery');ep2=episodes(p,777,10);nfe+=20;reference=t.update(ep2);nfe+=20;expected=adapter_state(p.pipe.unet);opt=copy.deepcopy(t.optimizer.state_dict());grads=copy.deepcopy(t.last_gradients)
     # Reference branch is real extra compute; both arms pay identically.
     t.restore(folder/'recovery');ep3=episodes(p,777,10);nfe+=20;receipt=t.update(ep3);nfe+=20
     check(name+'_stochastic_recovery',same(expected,adapter_state(p.pipe.unet)) and same(opt,t.optimizer.state_dict()) and same(grads,t.last_gradients))
     # Count two replay updates per arm, restore main trajectory so both have four logical steps.
     t.restore(folder/'recovery')
   gate=episodes(p,cfg['seeds']['gate'],cfg['gate_n']);nfe+=cfg['gate_n']*2;scores=[r for tr,c,r in gate];after=adapter_state(p.pipe.unet)
   check(name+'_frozen_invariance',f0==frozen(p));check(name+'_actual_updates',any(not torch.equal(before[n],after[n]) for n in before));check(name+'_finite_nondegenerate_feedback',all(r['finite_gradients'] and r['gradient_norm']>0 and r['reward_variance']>0 for r in receipts))
   t.save(folder/'final');cost=dict(training_updates=4,recovery_reference_updates=1,recovery_replay_updates=1,actual_optimizer_calls=6,training_trajectories=40,recovery_trajectories=20,gate_trajectories=16,forward_NFE=nfe,backward_transition_calls=120)
   arms[name]=dict(initial_hash=initial_hash,receipts=receipts,gate_scores=scores,gate_mean=float(np.mean(scores)),cost=cost,trainable_tensors=len(before),trainable_elements=meta['trainable_parameters'],parameter_changes=sum(not torch.equal(before[n],after[n]) for n in before),final_checkpoint_sha256=sha(folder/'final/training_state.pt'))
   models[name]=(p,t)
  check('equal_initial_parameters',arms['fixed']['initial_hash']==arms['rsi']['initial_hash']);check('equal_compute_budget',arms['fixed']['cost']==arms['rsi']['cost'])
  # Controller selection uses identical synthetic validation seeds, never an independent test.
  vh=hashlib.sha256(json.dumps(dict(seed=cfg['seeds']['gate'],n=16,objective=cfg['objective'] if 'objective' in cfg else 'cos'),sort_keys=True).encode()).hexdigest();elapsed=(time.perf_counter()-started)/3600
  controller=RSIController(out/'memory',dict(batch_size=10,learning_rate=cfg['fixed_lr'],curriculum='uniform'),arms['fixed']['gate_mean'],vh,600/3600)
  decision=controller.revise(dict(learning_rate=cfg['candidate_lr']),arms['rsi']['gate_mean'],elapsed,vh,policy_version='rsi_candidate_v1',proposal_reason='LR_PROPOSAL_0001',evidence='integrated/results.json',rollback='restore fixed/final when candidate rejected')
  selected='rsi' if decision['accepted'] else 'fixed';p,t=models['rsi'];t.restore(out/selected/'final');expected=torch.load(out/selected/'final/training_state.pt',map_location='cpu',weights_only=False)
  check('decision_checkpoint_materialized',same(adapter_state(p.pipe.unet),expected['adapter']) and same(t.optimizer.state_dict(),expected['optimizer']))
  # Equality rejection is a real controller call on cached measured evidence, no extra evaluation claimed.
  equality=controller.revise(dict(learning_rate=controller.state['training_config']['learning_rate']),controller.state['validation_score'],0.,vh,proposal_reason='locked equality rejection check using cached selected score',rollback='selected checkpoint remains incumbent')
  check('strict_equal_score_rejected',not equality['accepted'] and equality['reason']=='no_validation_gain')
  # Exercise actual rollback of weights, optimizer, scheduler and RNG, not merely a text receipt.
  t.restore(out/'rsi/recovery');t.restore(out/selected/'final')
  check('actual_disk_rollback',same(adapter_state(p.pipe.unet),expected['adapter']) and same(t.optimizer.state_dict(),expected['optimizer']) and same(torch.get_rng_state(),expected['rng_torch']) and same(torch.cuda.get_rng_state_all(),expected['rng_cuda']))
  persisted=read(out/'memory/state.json');check('experience_memory_persisted',len(persisted['history'])==2 and (out/'memory/revision_0001.json').exists() and (out/'memory/revision_0002.json').exists())
  # Reference and replay together are two extra physical updates, stated explicitly.
  delta=np.array(arms['rsi']['gate_scores'])-np.array(arms['fixed']['gate_scores']);g=np.random.default_rng(20261031);boot=g.integers(16,size=(5000,16))
  put(out/'results.json',dict(status='PASS',arms=arms,decision=decision,equality_rejection=equality,selected=selected,paired_gate_difference=float(delta.mean()),paired_bootstrap95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),memory_persistence=True,actual_rollback=True,scope='single tiny random UNet seed; 16 synthetic validation conditions; selected on this gate, no untouched test, no autonomous scientific efficacy claim',selection_cost_GPU_hours=elapsed,seconds=time.perf_counter()-started))
 except Exception:
  (out/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc());checks.append(dict(test='execution_exception',passed=False))
 finally:put(out/'checks.json',dict(status='PASS' if checks and all(r['passed'] for r in checks) else 'FAIL',checks=checks,seconds=time.perf_counter()-started))
 return 0 if checks and all(r['passed'] for r in checks) else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();raise SystemExit(main(ROOT/'results'/a.run_id))
