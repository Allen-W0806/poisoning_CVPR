"""Whitelist aggregates only. Does not copy raw evidence or alter report-sync task."""
import csv,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 s=json.loads((ROOT/'reports/NEXT_VALIDATION_STATUS_20261031.json').read_text(encoding='utf-8'));assert s['run_id']=='next_validation_20261101T0040Z_v1';assert s['reward_guard_intact'];assert s['resource_accounting']['consumed']<=3600
 out=ROOT/'reports/deidentified_next_validation_20261031';out.mkdir(exist_ok=True);rows=[]
 def add(scope,condition,metric,value):
  assert isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value)
  rows.append(dict(PROJECT_ID='RSI-POISONING',scope=scope,condition=condition,metric=metric,value=value))
 for key in ['prior','new','consumed','remaining','cap']:add('validation_gpu_budget','A100_seconds',key,s['resource_accounting'][key])
 for r in s['Qwen'].get('stats',[]):
  condition='base' if r['seed'] is None else 'clean_seed_'+str(r['seed'])
  for key in ['n','accuracy','paired_difference','changed_answers','mean_latency']:add('historical_development_only',condition,key,r[key])
  add('historical_development_only',condition,'paired_ci95_low',r['paired_ci95'][0]);add('historical_development_only',condition,'paired_ci95_high',r['paired_ci95'][1])
 for key in ['generated_SSIM_mean','generated_MAE_mean','n_sources']:add('geometry_aligned_image_audit','generated',key,s['image_quality'][key])
 for name,arm in s['integrated'].get('arms',{}).items():
  add('tiny_synthetic_validation',name,'gate_mean',arm['gate_mean'])
  for key,value in arm['cost'].items():add('tiny_synthetic_validation',name,key,value)
  add('tiny_synthetic_validation',name,'updated_tensors',arm['parameter_changes'])
 for key in ['paired_gate_difference']:add('tiny_synthetic_validation','rsi_minus_fixed',key,s['integrated'][key])
 for k,v in zip(['paired_ci95_low','paired_ci95_high'],s['integrated']['paired_bootstrap95']):add('tiny_synthetic_validation','rsi_minus_fixed',k,v)
 add('semantic_review','random_subset','selected_pairs',32);add('semantic_review','random_subset','genuine_reviews_received',0)
 csvpath=out/'NEXT_VALIDATION_AGGREGATES_20261031.csv'
 with csvpath.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 md=['# RSI-Poisoning: next validation decision','', 'PROJECT_ID = RSI-POISONING','',f"Evidence as of {s['updated_utc']}. **NO-GO remains.** Actual bounded CPU and one-GPU diagnostic work completed; this publication contains deidentified prose and aggregates only.",'','| Check | Status |','|---|---|']+[f"| {r['item']} | {r['status']} |" for r in s['completion_matrix']]
 md += ['','A random32 of160 historical image pairs are prepared for independent dual human review. Genuine reviews received0; no validity/changed/ambiguous frequencies can yet be estimated. Image SSIM is geometry-aligned; mean'+str(s['image_quality']['generated_SSIM_mean'])+'. LPIPS is missing because the calibrated implementation is unavailable, not substituted. Original, geometry-only and generated conditions remain distinct.','', 'The current paired BF16 diagnostic ordinary path PASS and instrumented path FAIL reverse the previous ordinary FAIL/instrumented PASS pattern. In the new failed path, the earliest mismatch is raw backward gradients, maxabs2.834317274391651e-5 and relativeL2 0.0001648017040489566; forward, restoration and RNG comparisons passed. Different allocations/nodes and instrumentation are confounded. One fixed repetition per path does not establish a fix or a unique cause. All negative evidence is retained.','', 'Existing clean Qwen language-attention LoRA checkpoints were evaluated without further optimization, using fixed greedy decoding and original references. Each checkpoint used two fresh model loads. Evaluation remains historical development-only because unused-source provenance cannot be certified. The aggregate CSV records paired bootstrap intervals. Changes on one common image across two seeds are not independent replication of an efficacy gain.','', 'The integrated benign test uses a tiny random UNet with the actual production DDPO score-function trainer, not GRPO. Each fixed/RSI arm received4 logical updates plus2 extra checkpoint-reference/replay updates,272 forward NFEs and120 backward transition calls. Actual LoRA updates, finite gradients, stochastic recovery, persistent revision memory, strict fixed acceptance/rejection and disk rollback were exercised. The accepted learning-rate proposal has gain'+str(s['integrated']['paired_gate_difference'])+' with95% interval'+str(s['integrated']['paired_bootstrap95'])+'; the interval crosses zero. This is not evidence of autonomous improvement, pretrained generator reliability or backdoor behavior.','',f"Cumulative validation GPU budget: {s['resource_accounting']['consumed']}/3600 A100 seconds; remaining{s['resource_accounting']['remaining']}. This is a validation subtotal, not lifetime project costs. Existing two-hour monitoring remains a single Slurm successor chain; no duplicate monitor was installed.",'', 'Next: obtain real independent semantic review and adjudication, certify a private evaluation split, and localize ordinary/instrumented numerical instability before pretrained generator learning. Generator reward-release guards remain active. No attack reward, poisoning optimization or ASR-driven learning was enabled.']
 report=out/'NEXT_VALIDATION_DECISION_20261031.md';report.write_text('\n'.join(md)+'\n',encoding='utf-8')
 # Explicit content boundary: no source IDs, raw questions/answers, paths, job IDs, model files or logs.
 for p in [report,csvpath]:
  text=p.read_text(encoding='utf-8');assert not any(x in text for x in ['/storage/','H:\\','ydk297','sample_id','question_id','image_path','raw_answer','88016'])
 receipt={p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in [report,csvpath]};(out/'LOCAL_PUBLICATION_AUDIT.json').write_text(json.dumps(dict(status='PASS',whitelisted_aggregate_rows=len(rows),files=receipt),indent=2));print(json.dumps(receipt))
if __name__=='__main__':main()
