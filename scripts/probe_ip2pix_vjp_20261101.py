"""Isolated local VJP localization; constant coefficient, never VLM/security reward.
Instrumentation changes execution and cannot certify the ordinary full-graph path.
"""
import argparse,copy,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from PIL import Image
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.rl.diffusion_policy import DiffusionPolicy
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.training.generator_adapter import adapter_state
from src.utils.seed import seed_everything
from scripts.run_ip2pix_checkpoint_diagnostic import same,frozen_hash
def put(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(run):
 out=run/'vjp_probe';out.mkdir(exist_ok=False);checks=[];handles=[];started=time.perf_counter()
 def check(n,b,**kw):
  row=dict(test=n,passed=bool(b),**kw);checks.append(row);print(json.dumps(row),flush=True)
  if not b:raise AssertionError(n)
 try:
  cfg=json.loads((run/'protocol.json').read_text());audit=json.loads((run/'cpu_verification.json').read_text());parent=ROOT/audit['failed_parent'];effective=json.loads((parent/'effective_config.json').read_text())['production_config'];target=audit['first_different_parameter']['name'];torch.set_num_threads(4);seed_everything(42);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
  check('Slurm_one_CUDA_GPU',bool(os.getenv('SLURM_JOB_ID')) and torch.cuda.device_count()==1)
  editor=load_pretrained_editor(ROOT,device='cuda');policy=DiffusionPolicy(editor,effective);trainer=DiffusionRLTrainer(policy);trainer.restore(parent/'before_next');f0=frozen_hash(policy.pipe);before=adapter_state(policy.pipe.unet)
  names={n:p for n,p in policy.pipe.unet.named_parameters() if p.requires_grad};ordered=list(names);check('historical_gradient_name_mapping',target in names and ordered==audit['optimizer_parameter_order'])
  prefix=target.split('.lora_')[0];module_names=[prefix+'.lora_A.default',prefix+'.lora_B.default'];captured={n:[] for n in module_names};parameter_contributions={n:[] for n in module_names}
  for n in module_names:
   m=policy.pipe.unet.get_submodule(n)
   def forward_hook(module,args,output,name=n):
    if not output.requires_grad:return
    x=args[0].detach().cpu().clone();w=module.weight.detach().cpu().clone()
    def gradient_hook(g):captured[name].append(dict(input=x,weight=w,upstream_gradient=g.detach().cpu().clone()));return None
    output.register_hook(gradient_hook)
   handles.append(m.register_forward_hook(forward_hook));handles.append(m.weight.register_hook(lambda g,name=n:parameter_contributions[name].append(g.detach().cpu().clone())))
  rec=json.loads((ROOT/'results/replication_monitor_20261029_v1_ip2pix_a0/protocol.json').read_text())['source_record']
  with Image.open(ROOT/rec['image_path']) as im:im=im.convert('RGB')
  with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
   condition=policy.condition(dict(image=im,sample_id=rec['sample_id'],image_id=rec['image_id']));roll=torch.load(parent/'reference_rollout.pt',map_location='cpu',weights_only=False)
   check('conditioning_exact',same({k:v.detach().cpu() for k,v in condition.items() if isinstance(v,torch.Tensor)},roll['conditioning']))
   episodes=[(t,condition,1.) for t in roll['episodes']];receipt=trainer.update(episodes)
  check('real_adapter_update_constant_diagnostic',receipt['finite_gradients'] and receipt['trainable_max_change']>0)
  check('frozen_pretrained_parameters_invariant',frozen_hash(policy.pipe)==f0)
  trainer.save(out/'after_diagnostic_update')
  torch.save(dict(gradients=trainer.last_gradients),out/'full_graph_postclip_gradients.pt')
  for h in handles:h.remove()
  handles=[];torch.save(dict(activations=captured,parameter_backward_contributions=parameter_contributions),out/'local_vjp_inputs.pt');details=[]
  for name,rows in captured.items():
   check('captured_actual_linear_inputs_'+name,bool(rows))
   for i,row in enumerate(rows):
    x=row['input'].cuda();g=row['upstream_gradient'].cuda();w=row['weight'];est=[]
    with torch.autocast('cuda',enabled=False):
     for repeat in range(3):
      leaf=w.cuda().clone().requires_grad_(True);y=torch.nn.functional.linear(x.float(),leaf.float());grad=torch.autograd.grad(y,leaf,grad_outputs=g.float())[0];est.append(grad.detach().cpu());del y,grad,leaf
    exact=same(est[0],est[1]) and same(est[0],est[2]);reference=row['upstream_gradient'].double().reshape(-1,g.shape[-1]).T@row['input'].double().reshape(-1,x.shape[-1]);error=float((est[0].double()-reference).norm()/(reference.norm()+1e-300));details.append(dict(module=name,backward_capture_index=i,input_shape=list(x.shape),weight_shape=list(w.shape),GPU_repeat_bitwise=exact,finite=all(bool(torch.isfinite(v).all()) for v in est),FP64_CPU_relative_error_descriptive=error));torch.save(dict(repeated_GPU_gradients=est,FP64_CPU_reference=reference),out/(name.replace('.','_')+f'_call{i}.pt'))
  put(out/'local_vjp_comparisons.json',details)
  check('local_VJP_exact_repeats',all(r['GPU_repeat_bitwise'] for r in details))
  check('local_VJP_finite',all(r['finite'] for r in details))
  put(out/'summary.json',dict(status='PASS_LOCAL_VJP_ONLY',selected_parameter=target,selected_module=prefix,captures={n:len(v) for n,v in captured.items()},parameter_hook_contributions={n:len(v) for n,v in parameter_contributions.items()},isolated_VJP_calls=len(details)*3,optimizer_updates=1,update=receipt,maximum_FP64_CPU_relative_error=max(r['FP64_CPU_relative_error_descriptive'] for r in details),earliest_local_repeat_mismatch=next((r for r in details if not r['GPU_repeat_bitwise']),None),ordinary_full_graph_reproducibility='NOT VALIDATED; captures synchronize and alter lifetimes; local PASS cannot close historical failures',CPU_FP64_reference='analytical g.T @ x; descriptive floating-point comparison, not a changed acceptance threshold',device=torch.cuda.get_device_name(),seconds=time.perf_counter()-started))
 except Exception:(out/'failure.txt').write_text(traceback.format_exc());print(traceback.format_exc());checks.append(dict(test='probe_exception',passed=False))
 finally:
  for h in handles:h.remove()
  put(out/'checks.json',dict(status='PASS' if checks and all(r['passed'] for r in checks) else 'FAIL',checks=checks,seconds=time.perf_counter()-started))
 return 0 if checks and all(r['passed'] for r in checks) else 1
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();raise SystemExit(main(ROOT/'results'/a.run_id))
