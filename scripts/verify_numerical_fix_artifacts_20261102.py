"""Partial-aware CPU verification. Missing resume evidence is never a PASS."""
import json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put
def main():
 import torch
 torch.set_num_threads(2);out=ROOT/'results/numerical_fix_20261102_v1/verification';a=load(out/'raw0/update0.pt');b=load(out/'raw1/update0.pt');checks={};missing=[]
 checks['matched_raw_starts']=compare(a['before'],b['before'])
 for k in ['raw','postclip','runtime_total_norm']:checks['raw_pair_'+k]=compare(a['capture'][k],b['capture'][k])
 for k in ['adapter','optimizer','scheduler','rng_torch','rng_cuda','rng_numpy','rng_python']:checks['raw_pair_after_'+k]=compare(a['after'][k],b['after'][k])
 for phase in ['before','after1','after2']:
  x=load(out/'ordinary0'/phase/'training_state.pt');y=load(out/'ordinary1'/phase/'training_state.pt');checks['ordinary_'+phase]=compare(x,y)
  if phase!='before':
   checks[phase+'_adapter']=compare(x['adapter'],y['adapter'])
   for family in ['exp_avg','exp_avg_sq']:checks[phase+'_'+family]=compare({k:v[family] for k,v in x['optimizer']['state'].items()},{k:v[family] for k,v in y['optimizer']['state'].items()})
 one=load(out/'ordinary0/after1/training_state.pt');checks['raw_capture_vs_ordinary0_update1']=compare(a['after'],{k:one[k] for k in a['after']})
 for first,other,label in [('ordinary0','ordinary1','ordinary'),('ordinary0','resume','resumed')]:
  for filename in ['next_trajectory.pt','postclip.pt','after2/training_state.pt']:
   pa,pb=out/first/filename,out/other/filename;key=label+'_'+filename
   if pa.exists() and pb.exists():checks[key]=compare(load(pa),load(pb))
   else:missing.append(dict(test=key,status='NOT VERIFIED / TIMEOUT',missing=[str(p.relative_to(out)) for p in [pa,pb] if not p.exists()]))
 receipts={};guards={}
 for mode,expected in [('ordinary0',2),('ordinary1',2),('resume',1)]:
  p=out/mode/'receipt.json'
  if not p.exists():missing.append(dict(test=mode+'_actual_update_receipt',status='NOT VERIFIED / TIMEOUT'));continue
  rec=json.loads(p.read_text());assert len(rec['updates'])==expected
  assert all(rec[k] for k in ['ownership_exact','frozen_base_unchanged','unmodified_canonical_update','original_guard_unchanged'])
  assert all(type(v[k]) in (float,int) and math.isfinite(v[k]) for v in rec['updates'] for k in ['loss','gradient_norm']);receipts[mode]=rec
 if 'ordinary0' in receipts and 'ordinary1' in receipts:checks['ordinary_recorded_loss_norm']=compare([torch.tensor([v['loss'],v['gradient_norm']],dtype=torch.float64) for v in receipts['ordinary0']['updates']],[torch.tensor([v['loss'],v['gradient_norm']],dtype=torch.float64) for v in receipts['ordinary1']['updates']])
 for mode in ['raw0','raw1','ordinary0','ordinary1','resume']:
  p=out/mode/'execution_guard.json'
  if p.exists():guards[mode]=json.loads(p.read_text());assert guards[mode]['actual_environment']=='1' and not guards[mode]['CUDA_initialized_before']
  else:missing.append(dict(test=mode+'_guard_receipt',status='NOT VERIFIED / TIMEOUT'))
 complete=not missing;exact=all(v['exact'] for v in checks.values());tolerance=all(v['tolerance_pass'] for v in checks.values());verified=complete and exact and tolerance
 result=dict(status='FIX_VERIFIED' if verified else 'ROOT_CAUSE_PARTIALLY_LOCALIZED',candidate_status='VERIFIED_SCOPED' if verified else 'FAILED_ORDINARY_VERIFICATION',complete=complete,available_exact_pass=exact,available_tolerance_pass=tolerance,fix_verified=verified,checks=checks,missing=missing,receipts=receipts,execution_guards=guards,lowest_CUDA_primitive='NOT ESTABLISHED',actual_resumed_next_optimizer_update='COMPLETE' if (out/'resume/after2/training_state.pt').exists() else 'NOT COMPLETE / TIMEOUT',scope='Explicit bootstrap plus opt-in loader; raw pair passed but ordinary uninstrumented pair failed. No production training caller enabled.',scientific_loss_changed=False,scientific_reward_released=False,ASR='NOT TESTED')
 put(out/'mandatory_verification.json',result);print(json.dumps(dict(status=result['status'],candidate=result['candidate_status'],complete=complete,checks={k:dict(exact=v['exact'],tolerance=v['tolerance_pass'],max_abs=v['max_abs'],affected=v['affected_elements']) for k,v in checks.items() if k in ['raw_pair_raw','raw_pair_postclip','ordinary_before','after1_adapter','after1_exp_avg','after1_exp_avg_sq','after2_adapter','ordinary_postclip.pt','resumed_next_trajectory.pt']},missing=missing)))
if __name__=='__main__':main()
