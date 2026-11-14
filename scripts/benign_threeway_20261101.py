"""Preregistered synthetic three-way comparison; no image/VLM/security rewards."""
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
from scripts.run_ip2pix_checkpoint_diagnostic import same
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(run):
 cfg=json.loads((run/'protocol.json').read_text());out=run/'threeway';out.mkdir(exist_ok=False);checks=[];results=[];clock=time.perf_counter()
 def check(n,b,**kw):
  checks.append(dict(test=n,passed=bool(b),**kw));print(json.dumps(checks[-1]),flush=True)
  if not b:raise AssertionError(n)
 def construct(seed):
  seed_everything(seed);u=UNet2DConditionModel(sample_size=8,in_channels=4,out_channels=4,layers_per_block=1,block_out_channels=(32,),down_block_types=('CrossAttnDownBlock2D',),up_block_types=('CrossAttnUpBlock2D',),cross_attention_dim=32,norm_num_groups=8,attention_head_dim=4).cuda()
  pipe=types.SimpleNamespace(unet=u,vae=torch.nn.Identity(),text_encoder=torch.nn.Identity());add_attention_lora(pipe,rank=2,alpha=4);pipe.scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,steps_offset=1,set_alpha_to_one=False);pipe.scheduler.set_timesteps(2,device='cuda')
  pc=dict(physical_micro_batch_size=1,gradient_accumulation_steps=10,logical_batch_size=10,learning_rate=1e-5,eta=1.,max_grad_norm=1.)
  p=types.SimpleNamespace(pipe=pipe,config=pc,editor=types.SimpleNamespace(device='cuda'),version=0,contract=dict(task='terminal cos(first latent coordinate)',version=1),NFE=0)
  def predict(z,t,c):p.NFE+=1;return u(z,t,encoder_hidden_states=c['hidden']).sample
  p.predict=predict;return p,DiffusionRLTrainer(p)
 @torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
 @torch.no_grad()
 def episodes(p,seed,n):
  rows=[]
  for i in range(n):
   g=torch.Generator(device='cuda').manual_seed(seed+i);z=torch.randn((1,4,8,8),generator=g,device='cuda');c=dict(sample_id=str(seed+i),image_id=str(seed+i),instruction='benign terminal cosine',hidden=torch.randn((1,3,32),generator=g,device='cuda'));tr=dict(trajectory_id=str(seed+i),sample_id=c['sample_id'],image_id=c['image_id'],instruction=c['instruction'],policy_version=p.version,contract=copy.deepcopy(p.contract),steps=[])
   for t in p.pipe.scheduler.timesteps:
    a,lp,_,_=transition(p.pipe.scheduler,p.predict(z,t,c),t,z,noise=torch.randn(z.shape,generator=g,device='cuda'));tr['steps'].append(dict(t=int(t),state=z.cpu(),action=a.cpu(),logp=lp.cpu()));z=a
   rows.append((tr,c,float(z[0,0,0,0].cos())))
  return rows
 def evaluate(p,seed,n):return [r for tr,c,r in episodes(p,seed,n)]
 def snapshot(p,t,path):
  t.save(path);v=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False);v['full_parameters']={n:x.detach().cpu().clone() for n,x in p.pipe.unet.named_parameters()};v['last_gradients']=copy.deepcopy(getattr(t,'last_gradients',None));return v
 try:
  torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  for seed in cfg['training_seeds']:
   reference=None;initial_reference=None;prefix_trace=None;initial_gate_reference=None;prefix_gate_reference=None
   for arm in ['fixed','adaptive_scheduler','rsi']:
    folder=out/f'seed{seed}'/arm;folder.mkdir(parents=True);torch.cuda.synchronize();beg=time.perf_counter();p,t=construct(seed);initial=adapter_state(p.pipe.unet);frozen={n:x.detach().cpu().clone() for n,x in p.pipe.unet.named_parameters() if not x.requires_grad};snap0=snapshot(p,t,folder/'initial');initial_gate=evaluate(p,100000,16);updates=[]
    for step,noise in enumerate([100,200]):updates.append(t.update(episodes(p,noise,10)))
    prefix=snapshot(p,t,folder/'prefix');prefix_gate=evaluate(p,100000,16)
    if reference is None:
     initial_reference=snap0;reference=prefix;prefix_trace=copy.deepcopy(updates);initial_gate_reference=initial_gate;prefix_gate_reference=prefix_gate
    check(f'seed{seed}_{arm}_initial_full_state_equal',same(snap0,initial_reference))
    check(f'seed{seed}_{arm}_preadaptation_full_state_equal',same(prefix,reference),fields=['all_parameters','optimizer_moments','scheduler','CPU_CUDA_NumPy_Python_RNG','version','consumed_trajectory_ids','contract'])
    check(f'seed{seed}_{arm}_preadaptation_gradients_updates_equal',all(same(a[k],b[k]) for a,b in zip(updates,prefix_trace) for k in ['loss','reward_mean','reward_variance','gradient_norm','individual_rewards','changes']))
    check(f'seed{seed}_{arm}_preadaptation_predictions_equal',same(initial_gate,initial_gate_reference) and same(prefix_gate,prefix_gate_reference))
    ordinary=None;controller=None;proposal=None;pre=float(np.mean(prefix_gate));startscore=float(np.mean(initial_gate));lr=1e-5
    if arm=='adaptive_scheduler':
     ordinary=torch.optim.lr_scheduler.ReduceLROnPlateau(t.optimizer,mode='max',factor=.5,patience=0,threshold=0.,threshold_mode='abs');ordinary.step(startscore);ordinary.step(pre);lr=t.optimizer.param_groups[0]['lr']
    if arm=='rsi':
     # This is a modest controller test, not a claim of a novel autonomous proposer.
     lr=5e-6 if pre<=startscore else 2e-5;vh=hashlib.sha256(json.dumps(dict(seed=100000,n=16,objective=p.contract)).encode()).hexdigest();controller=RSIController(folder/'memory',dict(batch_size=10,learning_rate=1e-5,curriculum='uniform'),pre,vh,cfg['gpu_cap_seconds']/3600)
     proposal=dict(id='CONFIG_0001',parent='CONFIG_0000',changes=dict(learning_rate=lr),reason='predeclared cooldown on no gain, else LR sensitivity increase',evidence=dict(initial_mean=startscore,prefix_mean=pre),protocol_sha256=sha(run/'protocol.json'));put(folder/'proposal.json',proposal)
    p.config['learning_rate']=lr;t.scheduler.base_lrs=[lr]
    for g in t.optimizer.param_groups:g['lr']=lr
    for noise in [300,400]:updates.append(t.update(episodes(p,noise,10)))
    post=evaluate(p,100000,16);candidate=snapshot(p,t,folder/'candidate');decision=None
    if controller:
     decision=controller.revise(dict(learning_rate=lr),float(np.mean(post)),(time.perf_counter()-beg)/3600,vh,proposal_reason=proposal['reason'],policy_version='candidate_v1',evidence=str(folder.relative_to(run)),rollback='own preadaptation checkpoint, never another control branch')
     if not decision['accepted']:t.restore(folder/'prefix')
     p.config['learning_rate']=controller.state['training_config']['learning_rate'];check(f'seed{seed}_selected_config_optimizer_agree',p.config['learning_rate']==t.optimizer.param_groups[0]['lr'])
     rejection=controller.revise(dict(learning_rate=controller.state['training_config']['learning_rate']),controller.state['validation_score'],0.,vh,proposal_reason='cached-equality rejection witness')
     check(f'seed{seed}_memory_and_equality_rejection',not rejection['accepted'] and len(json.loads((folder/'memory/state.json').read_text())['history'])==2)
    selected=snapshot(p,t,folder/'selected');t.restore(folder/'initial');t.restore(folder/'selected');check(f'seed{seed}_{arm}_disk_rollback_exact',same(adapter_state(p.pipe.unet),selected['adapter']) and same(t.optimizer.state_dict(),selected['optimizer']) and same(torch.get_rng_state(),selected['rng_torch']) and same(torch.cuda.get_rng_state_all(),selected['rng_cuda']))
    # Test conditions are first exposed after final selection. Distinct from all training/validation noise IDs.
    test=evaluate(p,200000,32);check(f'seed{seed}_{arm}_frozen_parameters',all(torch.equal(v,dict(p.pipe.unet.named_parameters())[n].detach().cpu()) for n,v in frozen.items()));changed=sum(not torch.equal(v,adapter_state(p.pipe.unet)[n]) for n,v in initial.items());check(f'seed{seed}_{arm}_updates_and_finite_feedback',changed>0 and all(x['finite_gradients'] and x['gradient_norm']>0 and x['reward_variance']>0 for x in updates));check(f'seed{seed}_{arm}_actual_equal_forward_budget',p.NFE==320,NFE=p.NFE)
    torch.cuda.synchronize();seconds=time.perf_counter()-beg
    row=dict(seed=seed,arm=arm,initial_validation=initial_gate,preadaptation_validation=prefix_gate,postadaptation_validation=post,selected_test=test,test_mean=float(np.mean(test)),candidate_lr=lr,retained_lr=p.config['learning_rate'],decision=decision,scheduler_state=None if ordinary is None else ordinary.state_dict(),optimizer_calls=4,backward_transition_calls=sum(r['backward_transition_calls'] for r in updates),forward_NFE=p.NFE,validation_queries=48,test_queries=32,updates=updates,updated_adapter_tensors=changed,GPU_held_seconds_measured=seconds,GPU_count=1,selected_checkpoint_sha256=sha(folder/'selected/training_state.pt'),prefix_checkpoint_sha256=sha(folder/'prefix/training_state.pt'))
    results.append(row);put(folder/'receipt.json',row);del p,t,initial,frozen,snap0,prefix,candidate,selected;import gc;gc.collect();torch.cuda.empty_cache()
  check('all_branches_identical_eval_and_update_budget',len({(r['optimizer_calls'],r['backward_transition_calls'],r['forward_NFE'],r['validation_queries'],r['test_queries']) for r in results})==1)
  comparisons=[];g=np.random.default_rng(20261101)
  for other in ['fixed','adaptive_scheduler']:
   deltas=np.array([next(r for r in results if r['seed']==s and r['arm']=='rsi')['selected_test'] for s in cfg['training_seeds']])-np.array([next(r for r in results if r['seed']==s and r['arm']==other)['selected_test'] for s in cfg['training_seeds']]);seedidx=g.integers(3,size=(5000,3));condidx=g.integers(32,size=(5000,32));boot=np.array([deltas[si][:,ci].mean() for si,ci in zip(seedidx,condidx)])
   comparisons.append(dict(comparison='rsi_minus_'+other,mean=float(deltas.mean()),per_seed_means=deltas.mean(1).tolist(),crossed_seed_condition_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),independent_training_seeds=3,shared_test_conditions=32,scope='synthetic held-out noise, conditional on tiny architecture; only three training seeds, exploratory'))
  put(out/'results.json',dict(status='PASS',branches=results,comparisons=comparisons,checks=checks,total_component_GPU_held_seconds=sum(r['GPU_held_seconds_measured'] for r in results),total_wall_seconds=time.perf_counter()-clock,selection='RSI compares only with its own prefix; no borrowed fixed-control outcome/checkpoint',no_security_metrics=True))
 except Exception:(out/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc());checks.append(dict(test='execution_exception',passed=False))
 finally:put(out/'checks.json',dict(status='PASS' if checks and all(x['passed'] for x in checks) else 'FAIL',checks=checks,seconds=time.perf_counter()-clock))
 return 0 if checks and all(x['passed'] for x in checks) else 1
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();raise SystemExit(main(ROOT/'results'/a.run_id))
