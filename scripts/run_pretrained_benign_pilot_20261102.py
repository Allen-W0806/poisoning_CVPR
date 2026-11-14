"""Single-seed pretrained IP2Pix brightness engineering fork experiment.

This explicitly isolated pixel-task proxy does NOT replace/release the original
scientific reward. No Qwen, semantic labels, ASR or stealth/cost optimization.
One common DDPO prefix is physically computed once and forked into three arms.
"""
import argparse,copy,hashlib,json,math,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main(run):
 import numpy as np,torch
 from PIL import Image
 from src.models.pretrained_editor_loader import load_pretrained_editor
 from src.rl.diffusion_policy import DiffusionPolicy
 from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
 from src.rl.phase5_backend import require_training_reward
 from src.rl.runtime_checkpoint_metadata import save_runtime_metadata,restore_runtime_metadata
 from src.training.generator_adapter import adapter_state
 from src.rsi.controller import RSIController
 from src.utils.seed import seed_everything
 from scripts.run_ip2pix_checkpoint_diagnostic import same,frozen_hash
 cfg=json.loads((run/'protocol.json').read_text());data=json.loads((run/'source_manifest.json').read_text());checks=[];branches=[];clock=time.perf_counter();predictions=[]
 def check(n,ok,**kw):
  checks.append(dict(test=n,passed=bool(ok),**kw));print(json.dumps(checks[-1]),flush=True)
  if not ok:raise AssertionError(n)
 try:
  assert os.getenv('SLURM_JOB_ID') and torch.cuda.device_count()==1
  for n,h in cfg['code_sha256'].items():assert sha(ROOT/n)==h,n
  seed_everything(cfg['seed']);torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  editor=load_pretrained_editor(ROOT,device='cuda');p=DiffusionPolicy(editor,cfg['diffusion_config']);t=DiffusionRLTrainer(p);p.contract['pilot_task']=cfg['engineering_proxy'];p.contract['source_manifest_sha256']=sha(run/'source_manifest.json')
  def save(directory,outer=None):
   t.save(directory);save_runtime_metadata(t,directory,outer)
  def restore(directory,outer=None):
   t.restore(directory);restore_runtime_metadata(t,directory,outer)
  check('actual_pretrained_generator',sum(x.numel() for x in p.pipe.unet.parameters())>800_000_000)
  trainable={n:v for n,v in p.pipe.unet.named_parameters() if v.requires_grad};check('one_shared_generator_policy_parameter_set',len(trainable)==256 and {id(v) for v in trainable.values()}=={id(v) for g in t.optimizer.param_groups for v in g['params']})
  check('source_ID_and_content_separation',all(not {x['image_id'] for x in data[a]}&{x['image_id'] for x in data[b]} and not {x['image_sha256'] for x in data[a]}&{x['image_sha256'] for x in data[b]} for a,b in [('train','validation'),('train','assessment'),('validation','assessment')]))
  for rows in data.values():
   for r in rows:assert sha(ROOT/r['image_path'])==r['image_sha256']
  conditions={}
  for rows in data.values():
   for r in rows:
    with Image.open(ROOT/r['image_path']) as im:conditions[r['sample_id']]=p.condition({**r,'image':im.convert('RGB')})
  def metric(source,edited):
   a=np.asarray(source.convert('RGB'),dtype=np.float64)/255.;b=np.asarray(edited.convert('RGB'),dtype=np.float64)/255.
   luminance=lambda x:float((x*np.array([.2126,.7152,.0722])).sum(axis=2).mean())
   before,after=luminance(a),luminance(b);target=min(1.,1.03*before);reward=abs(before-target)-abs(after-target)
   assert -1<=reward<=1 and math.isfinite(reward)
   return dict(reward=reward,source_mean_luminance=before,edited_mean_luminance=after,target_mean_luminance=target,exact_noop=bool(np.array_equal(a,b)),semantic_validity='UNREVIEWED',instruction_success=None)
  def collect(rows,label,train):
   episodes=[];scores=[];directory=run/label;directory.mkdir(parents=True,exist_ok=False)
   for i,r in enumerate(rows):
    c=conditions[r['sample_id']];seed=cfg['seed']*100000+(10000 if train else 20000)+i
    image,tr=p.sample(c,seed,f'{label}_{i}');image.save(directory/f'{i}.png');m=metric(c['source'],image);scores.append(m['reward'])
    predictions.append(dict(phase=label,sample_id=r['sample_id'],source_image_id=r['image_id'],seed=seed,image_sha256=sha(directory/f'{i}.png'),**m))
    if train:episodes.append((tr,c,m['reward']))
   return episodes,scores
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   frozen=frozen_hash(p.pipe);initial_adapter=adapter_state(p.pipe.unet);save(run/'shared_initial');initial_state=torch.load(run/'shared_initial/training_state.pt',map_location='cpu',weights_only=False)
   _,initial_val=collect(data['validation'],'shared_initial_validation',False)
   ep,_=collect(data['train'][:10],'shared_prefix_training',True);prefix_update=t.update(ep);save(run/'shared_prefix');prefix_state=torch.load(run/'shared_prefix/training_state.pt',map_location='cpu',weights_only=False)
   _,prefix_val=collect(data['validation'],'shared_prefix_validation',False);shared_nfe=p.unet_calls
   put(run/'shared_prefix_receipt.json',dict(initial_validation=initial_val,prefix_validation=prefix_val,update=prefix_update,physical_optimizer_calls=1,forward_UNet_NFE=shared_nfe,seconds=time.perf_counter()-clock,accounting='One common prefix physically executed once; shared cost is not billed once per arm'))
   for arm in cfg['arms']:
    folder=run/arm;folder.mkdir();torch.cuda.synchronize();start=time.perf_counter();start_nfe=p.unet_calls
    source=run/('shared_initial' if arm=='frozen' else 'shared_prefix');restore(source);p.config['learning_rate']=cfg['diffusion_config']['learning_rate'];save(folder/'start')
    state=torch.load(folder/'start/training_state.pt',map_location='cpu',weights_only=False);check(arm+'_fork_state_exact',same(state,initial_state if arm=='frozen' else prefix_state))
    lr=cfg['diffusion_config']['learning_rate'];controller=None;scheduler=None;decision=None;updates=[];diagnosis=None
    if arm=='ordinary_scheduler':
     scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(t.optimizer,mode='max',factor=.5,patience=0,threshold=0.,threshold_mode='abs');scheduler.step(float(np.mean(initial_val)));scheduler.step(float(np.mean(prefix_val)));lr=t.optimizer.param_groups[0]['lr']
    elif arm=='rsi':
     diagnosis=dict(observed_prefix_change=float(np.mean(prefix_val)-np.mean(initial_val)),source='own shared-prefix validation evidence',scientific_efficacy='NOT ESTABLISHED')
     lr*=2. if diagnosis['observed_prefix_change']>0 else .5
     controller=RSIController(folder/'memory',dict(batch_size=10,learning_rate=cfg['diffusion_config']['learning_rate'],curriculum='uniform'),float(np.mean(prefix_val)),sha(run/'source_manifest.json'),cfg['GPU_cap_seconds']/3600.)
     put(folder/'proposal.json',dict(diagnosis=diagnosis,change={'learning_rate':lr},validation_source_hash=sha(run/'source_manifest.json'),data_scope='development-only, no locked final test',reason='Bounded LR revision from own validation gain; not a second generation network'))
    if arm!='frozen':
     t.optimizer.param_groups[0]['lr']=lr;t.scheduler.base_lrs=[lr];p.config['learning_rate']=lr
     ep,_=collect(data['train'][10:20],arm+'/candidate_training',True);updates.append(t.update(ep))
    _,candidate_val=collect(data['validation'],arm+'/candidate_validation',False);save(folder/'candidate',scheduler)
    if controller:
     torch.cuda.synchronize();candidate_cost=(time.perf_counter()-start)/3600.
     decision=controller.revise({'learning_rate':lr},float(np.mean(candidate_val)),candidate_cost,sha(run/'source_manifest.json'),policy_version='candidate_step2',proposal_reason='Own validation-based bounded LR diagnosis',evidence=str(folder/'proposal.json'),rollback='its exact shared-prefix fork, not another arm')
     if not decision['accepted']:restore(run/'shared_prefix')
     p.config['learning_rate']=controller.state['training_config']['learning_rate']
    save(folder/'selected',scheduler);selected=torch.load(folder/'selected/training_state.pt',map_location='cpu',weights_only=False);restore(folder/'selected',scheduler)
    check(arm+'_active_LR_matches_optimizer',p.config['learning_rate']==t.optimizer.param_groups[0]['lr'] and t.scheduler.base_lrs==[t.optimizer.param_groups[0]['lr']])
    check(arm+'_selected_reload_exact',same(adapter_state(p.pipe.unet),selected['adapter']) and same(t.optimizer.state_dict(),selected['optimizer']) and same(t.scheduler.state_dict(),selected['scheduler']) and same(torch.get_rng_state(),selected['rng_torch']) and same(torch.cuda.get_rng_state_all(),selected['rng_cuda']))
    _,assessment=collect(data['assessment'],arm+'/selected_assessment',False);torch.cuda.synchronize()
    row=dict(arm=arm,seed=cfg['seed'],candidate_validation_mean=float(np.mean(candidate_val)),assessment_mean=float(np.mean(assessment)),assessment_scores=assessment,optimizer_calls_incremental=len(updates),shared_prefix_optimizer_calls=0 if arm=='frozen' else 1,attempted_updates=0 if arm=='frozen' else 2,retained_checkpoint_step=selected['step'],incremental_forward_UNet_NFE=p.unet_calls-start_nfe,incremental_GPU_held_seconds=time.perf_counter()-start,actual_arm_validation_queries=sum(x['phase']==arm+'/candidate_validation' for x in predictions),actual_arm_assessment_queries=sum(x['phase']==arm+'/selected_assessment' for x in predictions),candidate_lr=lr,retained_lr=t.optimizer.param_groups[0]['lr'],decision=decision,updates=updates,selected_checkpoint_sha256=sha(folder/'selected/training_state.pt'),clean_accuracy=None,ASR=None,semantic_validity='UNREVIEWED',scientific_effectiveness='NOT ESTABLISHED')
    branches.append(row);put(folder/'receipt.json',row)
   check('matched_incremental_queries',all(x['actual_arm_validation_queries']==4 and x['actual_arm_assessment_queries']==4 for x in branches))
   check('matched_attempted_training_updates',all(x['attempted_updates']==2 for x in branches if x['arm']!='frozen'))
   check('frozen_base_invariant',frozen_hash(p.pipe)==frozen)
   check('valid_finite_real_updates',prefix_update['finite_gradients'] and all(x['finite_gradients'] and x['changes']['max_abs_change']>0 for b in branches for x in b['updates']))
   try:require_training_reward({'training_eligible':True})
   except RuntimeError:guard=True
   else:guard=False
   check('scientific_reward_guard_still_denies',guard)
  put(run/'results.json',dict(status='PASS_ENGINEERING_ONLY',branches=branches,shared_prefix_physical_optimizer_calls=1,shared_forward_UNet_NFE=shared_nfe,total_forward_UNet_NFE=p.unet_calls,physical_optimizer_calls=1+sum(x['optimizer_calls_incremental'] for x in branches),seconds=time.perf_counter()-clock,scope='one-seed pretrained brightness proxy on exposed development images; no semantic/generalization/RSI superiority or ASR claim'))
  put(run/'prediction_observations.json',predictions)
 except Exception:(run/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc());checks.append(dict(test='exception',passed=False))
 finally:put(run/'checks.json',dict(status='PASS' if checks and all(x['passed'] for x in checks) else 'FAIL',checks=checks,diagnostic_complete=True,scientific_release=False,seconds=time.perf_counter()-clock))
 put(run/'artifact_inventory.json',dict(files=[dict(path=str(q.relative_to(run)),sha256=sha(q),bytes=q.stat().st_size) for q in run.rglob('*') if q.is_file() and q.name!='artifact_inventory.json']))
 return 0 if checks and all(x['passed'] for x in checks) else 1

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();raise SystemExit(main(ROOT/'results'/a.run_id))
