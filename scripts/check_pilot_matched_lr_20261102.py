"""CPU paired numerical control on already completed same-LR pilot candidates."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def matched_control_exact(initial_exact,images_exact,adapters_exact,moments_exact,rng_exact):
 return all([initial_exact,images_exact,adapters_exact,moments_exact,rng_exact])
def main():
 import torch
 from scripts.run_ip2pix_checkpoint_diagnostic import same,discrepancy
 run=ROOT/'results/execution_remediation_20261102_v1';out=run/'matched_lr_control';out.mkdir(exist_ok=False);pilot=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';torch.set_num_threads(2)
 results=json.loads((pilot/'results.json').read_text());rows={x['arm']:x for x in results['branches']};names=['ordinary_scheduler','rsi'];assert rows[names[0]]['candidate_lr']==rows[names[1]]['candidate_lr']
 starts=[torch.load(pilot/n/'start/training_state.pt',map_location='cpu',weights_only=False) for n in names];states=[torch.load(pilot/n/'candidate/training_state.pt',map_location='cpu',weights_only=False) for n in names]
 images=[dict(index=i,equal=sha(pilot/names[0]/'candidate_training'/f'{i}.png')==sha(pilot/names[1]/'candidate_training'/f'{i}.png')) for i in range(10)]
 adapters={k:torch.equal(states[0]['adapter'][k],states[1]['adapter'][k]) for k in states[0]['adapter']};moments={}
 for key in ['exp_avg','exp_avg_sq']:
  count=sum(not torch.equal(states[0]['optimizer']['state'][i][key],states[1]['optimizer']['state'][i][key]) for i in states[0]['optimizer']['state']);moments[key]=count
 initial_exact=same(*starts);images_exact=all(x['equal'] for x in images);rng_exact=all(same(states[0][k],states[1][k]) for k in ['rng_torch','rng_cuda','rng_numpy','rng_python'])
 result=dict(status='PASS_EXACT_MATCHED_LR_CONTROL' if matched_control_exact(initial_exact,images_exact,all(adapters.values()),not any(moments.values()),rng_exact) else 'NUMERICAL_MISMATCH_PRESERVED',job_id=os.getenv('SLURM_JOB_ID'),initial_fork_states_exact=same(*starts),candidate_LRs=[rows[n]['candidate_lr'] for n in names],generated_training_images=images,all_training_images_exact=all(x['equal'] for x in images),different_adapter_tensors=sum(not x for x in adapters.values()),first_different_adapter=next((k for k,v in adapters.items() if not v),None),adapter_error=discrepancy(list(states[0]['adapter'].values()),list(states[1]['adapter'].values())),different_optimizer_moments=moments,losses=[rows[n]['updates'][0]['loss'] for n in names],gradient_norms=[rows[n]['updates'][0]['gradient_norm'] for n in names],candidate_validation_means=[rows[n]['candidate_validation_mean'] for n in names],RNG_after_exact=all(same(states[0][k],states[1][k]) for k in ['rng_torch','rng_cuda','rng_numpy','rng_python']),interpretation='A candidate difference at equal LR is numerical/control evidence, not RSI contribution. First unobserved raw/forward CUDA primitive remains unresolved.',engineering_executability='PASS; independent scope',scientific_RSI_superiority='NOT ESTABLISHED',new_GPU_seconds=0,new_model_calls=0)
 (out/'summary.json').write_text(json.dumps(result,indent=2));cmd=[sys.executable,'-B','-m','unittest','tests.test_execution_status_dashboard','-v'];p=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT);(out/'status_dashboard_regressions.log').write_text(p.stdout+p.stderr);assert p.returncode==0
 print(json.dumps(result),flush=True);return 0
if __name__=='__main__':raise SystemExit(main())
