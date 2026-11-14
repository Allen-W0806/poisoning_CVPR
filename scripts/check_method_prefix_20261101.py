"""Read saved historical preintervention tensors; never run training."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.run_ip2pix_checkpoint_diagnostic import same
RUN='method_code_audit_20261101T0510Z_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 run=ROOT/'results'/RUN;torch.set_num_threads(2);names={42:('EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL','EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI'),123:('EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL','EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI')};rows=[]
 for seed,(a,b) in names.items():
  ma=json.loads((ROOT/'experiments'/a/'metrics.json').read_text());mb=json.loads((ROOT/'experiments'/b/'metrics.json').read_text());order=all([o['sample_id'] for o in x['observations']]==[o['sample_id'] for o in y['observations']] and [o['seed'] for o in x['observations']]==[o['seed'] for o in y['observations']] for x,y in zip(ma['updates'][:4],mb['updates'][:4]))
  for step in [0,4]:
   pa=ROOT/'experiments'/a/f'checkpoints/step_{step:04d}/training_state.pt';pb=ROOT/'experiments'/b/f'checkpoints/step_{step:04d}/training_state.pt';x=torch.load(pa,map_location='cpu',weights_only=False);y=torch.load(pb,map_location='cpu',weights_only=False);keys=['adapter','optimizer','scheduler','step','version','contract','consumed','rng_torch','rng_cuda','rng_numpy','rng_python'];fields={k:(same(x[k],y[k]) if k in x and k in y else 'NOT_STORED') for k in keys};moments={k:sum(not same(x['optimizer']['state'][i].get(k),y['optimizer']['state'][i].get(k)) for i in x['optimizer']['state']) for k in ['exp_avg','exp_avg_sq']};rows.append(dict(seed=seed,step=step,baseline=a,rsi=b,fields_equal=fields,adapter_max_abs_difference=max(float((v-y['adapter'][n]).abs().max()) for n,v in x['adapter'].items()),different_optimizer_moment_tensors=moments,baseline_lr=x['optimizer']['param_groups'][0]['lr'],rsi_lr=y['optimizer']['param_groups'][0]['lr'],first_four_image_order_and_rollout_seed_draws_equal=order,source_sha256={str(pa.relative_to(ROOT)):sha(pa),str(pb.relative_to(ROOT)):sha(pb)}))
 result=dict(status='COMPLETE_HISTORICAL_DIFFERENCES_PRESERVED',rows=rows,scope='saved checkpoints step0 and step4 before LR intervention; matching RNG/order is not proof of equal backward execution')
 with (run/'historical_prefix_state_comparison.json').open('x') as f:json.dump(result,f,indent=2)
 for name in ['METHOD_IMPLEMENTATION_AUDIT.md','RL_RSI_CONTROLLED_VALIDATION.md']:
  p=ROOT/'reports'/name;backup=run/'audit/report_revision0'/name;backup.parent.mkdir(parents=True,exist_ok=True);backup.write_bytes(p.read_bytes());s=p.read_text(encoding='utf-8');s+='\n## Direct historical checkpoint/RNG follow-up\n\nA distinct read-only CPU task compared the preserved initial and pre-LR-step4 checkpoints directly. Evidence: results/'+RUN+'/historical_prefix_state_comparison.json. Matching data order and RNG do not imply matched model/optimizer trajectories.\n\n'+json.dumps(result,indent=2)+'\n';p.write_text(s,encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
