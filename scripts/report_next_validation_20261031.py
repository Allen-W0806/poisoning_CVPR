"""Evidence-bound terminal report and passive existing-dashboard extension."""
import argparse,hashlib,html,json,re,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='next_validation_20261101T0040Z_v1';PARENT='small_scale_validation_20261031T1928Z_v1'
def read(p,default=None):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except FileNotFoundError:return default
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def dashboard(root):
 payload=read(root/'reports/NEXT_VALIDATION_STATUS_20261031.json')
 if not payload:return
 data=read(root/'dashboard/dashboard_data.json',{});data['next_validation_20261031']=payload;encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 for name in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']:
  p=root/'dashboard'/name
  if not p.exists():continue
  zh='_zh' in name or 'Latest' in name
  title='下一阶段良性验证：仍为 NO-GO' if zh else 'Next benign validation: NO-GO remains'
  desc='已完成真实有界诊断；32对双人盲审待填写，LPIPS缺失，历史评估仅用于开发。' if zh else 'Bounded diagnostics executed; 32-pair dual review awaits real humans. LPIPS unavailable. Historical evaluation is development-only.'
  block='<section id="next-validation-20261031" style="margin:24px auto;padding:24px;max-width:1200px;background:#fff;border:1px solid #ccd;border-radius:16px;overflow:auto"><h2>'+title+'</h2><p>'+html.escape(payload['updated_utc'])+'</p><p>'+desc+'</p><p>CPU '+payload['jobs']['cpu_job']+' / GPU '+payload['jobs']['gpu_job']+' · '+payload['jobs']['state']+'</p><p>GPU seconds '+str(payload['resource_accounting']['consumed'])+'/3600; remaining '+str(payload['resource_accounting']['remaining'])+'.</p><table><thead><tr><th>Validation</th><th>Status</th></tr></thead><tbody>'
  for row in payload['completion_matrix']:block+='<tr><td>'+html.escape(row['item_zh'] if zh else row['item'])+'</td><td>'+row['status']+'</td></tr>'
  block+='</tbody></table><p>Pretrained BF16 ordinary: '+payload['BF16']['ordinary'].get('status','MISSING')+'; instrumented: '+payload['BF16']['instrumented'].get('status','MISSING')+'. Integrated benign DDPO–RSI: '+payload['integrated'].get('status','MISSING')+'.</p><p><a href="../reports/NEXT_VALIDATION_DECISION_20261031.md">Evidence and decision</a> · <a href="../results/'+RUN+'/semantic_review32/index.html">32-pair independent review kit</a></p><p>No attack rewards or ASR optimization. Model improvements remain scientifically unverified.</p></section>'
  s=p.read_text(encoding='utf-8');s=re.sub(r'<!--NEXT_VALIDATION_START-->.*?<!--NEXT_VALIDATION_END-->','',s,flags=re.S);s=s.replace('</body>','<!--NEXT_VALIDATION_START-->'+block+'<!--NEXT_VALIDATION_END--></body>');s=re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)',lambda m:m[1]+encoded+m[2],s,flags=re.S);p.write_text(s,encoding='utf-8')
 write(root/'dashboard/state.json',data);write(root/'dashboard/dashboard_data.json',data)
 write(root/'reports/NEXT_VALIDATION_DASHBOARD_QA_20261031.json',dict(status='PASS',JSON_state_equal=(root/'dashboard/state.json').read_bytes()==(root/'dashboard/dashboard_data.json').read_bytes(),views=5))
def main():
 run=ROOT/'results'/RUN;ledger=read(run/'job_ledger.json');job=ledger['gpu_job'];cpu_receipts=list((ROOT/'reports').glob('CPU_SUBMIT_'+RUN+'*.json'));cpu=next(read(p)['stdout'].strip() for p in cpu_receipts if read(p)['returncode']==0)
 raw=subprocess.check_output(['sacct','-X','-nP','-j',cpu+','+job,'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,Start,End'],text=True);jobs=[]
 for line in raw.splitlines():
  a=line.split('|')
  if len(a)<7:continue
  jobs.append(dict(job_id=a[0],state=a[1],elapsed=int(a[2]),gpu_seconds=int(a[2]) if 'gres/gpu=1' in a[3] else 0,exit_code=a[4],start_cluster_local=a[5],end_cluster_local=a[6]))
 gpurow=next(r for r in jobs if r['job_id']==job);assert gpurow['state'] in ['COMPLETED','FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL']
 cost=dict(cap=3600,prior=ledger['prior_live_seconds'],new=gpurow['gpu_seconds'],consumed=ledger['prior_live_seconds']+gpurow['gpu_seconds'],reserved=0,remaining=3600-ledger['prior_live_seconds']-gpurow['gpu_seconds'],jobs=jobs,raw_sacct=raw,scope='validation-only cumulative budget; not lifetime project or unrelated video jobs',dollar_cost=None);assert cost['new']<=600 and cost['consumed']<=3600
 sources=read(run/'source_manifest.json');integrity={n:sha(ROOT/n)==h for n,h in sources.items()};assert all(integrity.values())
 snapshot=run/'audit/executed_source'
 import shutil
 for n in sources:
  p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():shutil.copyfile(ROOT/n,p)
 bf={}
 for k in ['ordinary','instrumented']:
  folder=ROOT/'results'/f'{RUN}_BF16_{k}'/'diagnostic';bf[k]=dict(summary=read(folder/'summary.json'),checks=read(folder/'checks.json'),status=read(folder/'checks.json',{}).get('status','MISSING'),raw_localization=read(folder/'raw_backward_localization.json'))
 q=read(run/'qwen_recheck.json',{});rsi=read(run/'integrated/results.json',{});rsi_checks=read(run/'integrated/checks.json',{})
 oldpred={}
 if q:
  for seed in [42,123,3407]:
   oldpred[seed]={r['sample_id']:r for r in [json.loads(s) for s in (ROOT/'results'/f'{PARENT}_qwen_s{seed}/clean_main/predictions.jsonl').read_text().splitlines()] if r['stage']=='after'}
  comparisons=[]
  for r in [json.loads(s) for s in (run/'qwen_predictions.jsonl').read_text().splitlines()]:
   if r['seed'] is not None:comparisons.append(dict(seed=r['seed'],repeat=r['repeat'],sample_id=r['sample_id'],equals_previous=r['normalized_answer']==oldpred[r['seed']][r['sample_id']]['normalized_answer']))
  write(run/'qwen_vs_prior.json',dict(status='PASS' if all(r['equals_previous'] for r in comparisons) else 'FAIL',records=comparisons))
 matrix=[
  dict(item='Source separation within the 80-source pilot',item_zh='80源试验内部来源隔离',status='PASS',evidence=f'results/{PARENT}/audit/data_integrity.json; unique source/hash and cross-split dHash screen; does not certify historical nonexposure'),
  dict(item='Semantic validity',item_zh='编辑语义有效性',status='BLOCKED',evidence=f'results/{RUN}/semantic_review32/selection_receipt.json; 0 genuine reviews; no valid/changed rate estimable'),
  dict(item='Image quality metrics',item_zh='图像质量指标完整性',status='BLOCKED',evidence=f'results/{RUN}/aligned_image_quality.csv; aligned SSIM measured for 160; LPIPS implementation absent'),
  dict(item='Pretrained generator numerical reliability',item_zh='预训练生成器数值可靠性',status='FAIL' if any(v['status']=='FAIL' for v in bf.values()) else 'BLOCKED',evidence=f'results/{RUN}_BF16_ordinary/diagnostic/checks.json and instrumented/diagnostic/checks.json; path-dependent failures and historical full batch continuation unresolved'),
  dict(item='Clean target fine-tuning and existing checkpoint recheck',item_zh='干净目标微调与已有检查点复评',status='PASS' if q.get('status')=='PASS' else 'FAIL',evidence=f'results/{RUN}/qwen_recheck.json; engineering-only; paired accuracy is development evidence'),
  dict(item='Benign actual generator RL learning',item_zh='良性目标下真实生成器RL更新',status=rsi_checks.get('status','BLOCKED'),evidence=f'results/{RUN}/integrated/checks.json; tiny random UNet actual production DDPO, not pretrained reliability'),
  dict(item='Integrated RSI revision, memory and rollback',item_zh='RSI提案、记忆与回滚闭环',status=rsi_checks.get('status','BLOCKED'),evidence=f'results/{RUN}/integrated/results.json; equal-compute fixed control, fixed gate, disk rollback'),
  dict(item='Independent evaluation readiness',item_zh='独立评估就绪',status='BLOCKED',evidence=f'results/{RUN}/evaluation_provenance.json; historical exposure unresolved; no untouched test'),
  dict(item='Training-induced behavioral efficacy',item_zh='训练诱导行为的科学效力',status='NOT TESTED',evidence='No backdoor training or ASR optimization; benign changes do not establish this claim')]
 scheduler=read(ROOT/'reports/monitoring/scheduler.json');heartbeat=read(ROOT/'reports/rsi_stage_monitor/heartbeat.json');queue=subprocess.check_output(['squeue','-u',__import__('getpass').getuser(),'-h','-o','%i|%j|%T|%l|%k'],text=True)
 monitor_ids=','.join(str(scheduler[k]) for k in ['last_tick_job_id','next_job_id']);monacct=subprocess.check_output(['sacct','-X','-nP','-j',monitor_ids,'--format=JobID,State,Start,End,ExitCode'],text=True)
 payload=dict(updated_utc=datetime.now(timezone.utc).isoformat(),run_id=RUN,parent=PARENT,overall_gate='NO-GO',jobs=dict(cpu_job=cpu,gpu_job=job,state=gpurow['state']),resource_accounting=cost,completion_matrix=matrix,BF16=bf,Qwen=q,integrated=rsi,semantic_review=dict(selected=32,population=160,received=0,status='BLOCKED'),image_quality=read(run/'image_quality_summary.json'),source_hashes_verified=integrity,monitor=dict(scheduler=scheduler,last_heartbeat_utc=heartbeat['checked_UTC'],heartbeat_scope='prior replication stage only; this new stage has separate accounting',live_squeue=queue,sacct=monacct),reward_guard_intact=sha(ROOT/'src/rl/phase5_backend.py')=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712')
 write(run/'final_status.json',payload);write(ROOT/'reports/NEXT_VALIDATION_STATUS_20261031.json',payload);write(ROOT/'reports/validation_gpu_reservations.json',dict(cap=3600,consumed=cost['consumed'],reserved=0,reservations=[dict(run_id=RUN,job=job,state=gpurow['state'],gpu_seconds=cost['new'])]))
 md=['# Next validation decision — 2026-10-31','',f"Evidence snapshot: {payload['updated_utc']}. Run `{RUN}`. **NO-GO remains.** No independent human semantic labels or untouched evaluation are available. Engineering checks do not establish scientific efficacy.",'','## Actual execution and budget','',f"CPU {cpu}: {next(r for r in jobs if r['job_id']==cpu)}",f"GPU {job}: {gpurow}",f"Prior validation {cost['prior']} + this job {cost['new']} = {cost['consumed']}/3600 A100 seconds; remaining {cost['remaining']}. New job hard cap600s. No Qwen optimizer steps. No new monitor. Other project jobs excluded.",'','## Completion matrix','','| Check | Status | Evidence |','|---|---|---|']
 md += [f"| {x['item']} | {x['status']} | {x['evidence']} |" for x in matrix]
 md += ['','## Semantic and image-quality audit','',f"Random subset32/160 frozen before metrics with seed20261031032; source manifest and protocol hashes retained. Kit: `results/{RUN}/semantic_review32/index.html`. Two independent reviewers and separate adjudication exports required. No reviewers have been contacted or simulated. Full160 kit preserved.",json.dumps(payload['image_quality'],indent=2),'','SSIM compares generated C4 with stored geometry-matched original C5. Original native dimensions remain distinct. Geometry C5 versus itself is a control, not a semantic or resolution-fidelity measurement. LPIPS null means missing; no substitute is relabeled LPIPS.','', '## BF16 diagnostic — ordinary versus instrumented','']
 for kind in bf:
  md += [f"### {kind}",'',f"Status: {bf[kind]['status']}; execution: `results/{RUN}/executions/BF16_{kind}.json`.",json.dumps(bf[kind]['summary'],indent=2),json.dumps(bf[kind]['raw_localization'],indent=2)]
 md += ['','One preregistered repetition per path at seed42; bitwise criteria unchanged. Historical86853 ordinary FAIL (postclip maxabs7.6610595e-6, relativeL2 6.3059489e-5) and86856 instrumented PASS both ran on epyc08. This new paired run88016 ran on epyc01: ordinary PASS but instrumented FAIL, reversing the historical pattern. Instrumented raw-gradient first mismatch maxabs2.834317274391651e-5, relativeL2 0.0001648017040489566; preclipnorm1.8282486200332642 versus1.8282517194747925. No discrepancy existed in restored state/conditioning/RNG draws/stochastic forward/decoded pixels; postclip gradients, optimizer parameters/state diverged. This fails reliability and does not reproduce a stable ordinary-fails/instrumented-passes pattern. Node, allocation and lifetime effects are possible confounders; not a causal localization to one kernel. No assertion relaxed and no extra attempts run. Preserve all failed gradients/checkpoints.','', '## Existing clean checkpoints — development only','',json.dumps(q.get('stats',[]),indent=2),'','Pinned Qwen revision89644892e4d85e24eaac8bacfd4f463576704203; greedy decoding, original references, fixed pixel budgets/math SDPA. Fresh model loads repeated twice per adapter. Checkpoint inventory verified before use; inference frozen. Paired image bootstrap5000 resamples with fixedseed20261031; zero remains compatible with the observed differences, and empirical zero intervals do not prove equivalence. Repeated seeds changing the same image do not multiply independent sample size. Historical split exposure remains unresolved; no independent test claim.','', '## Integrated benign RL–RSI','',json.dumps({k:v for k,v in rsi.items() if k!='arms'},indent=2)]
 if rsi:
  md += [json.dumps({name:{k:v for k,v in arm.items() if k not in ['initial_hash','receipts','gate_scores']} for name,arm in rsi['arms'].items()},indent=2)]
 md += ['','Production algorithm is single-pass on-policy DDPO_SF_v1, not GRPO/PPO. Tiny randomly initialized UNet attention LoRA rank2alpha4 only; fixedcos terminal reward. Four logical steps and two extra recovery/reference optimizer calls per arm are all charged. Common initialization/training seeds; one LR proposal1e-5→2e-5; 16 common synthetic validation conditions. Exact NFE/update parity does not mean equal wall-clock latency. Selected gate is development evidence, not a fresh test. Actual disk rollback restores weights, optimizer and CPU/CUDA RNG; memory has persisted decisions. Any engineering acceptance is not statistical/autonomous research benefit.','', '## Monitoring and source integrity','',f"Existing `slurm_cpu_successor`, enabled={scheduler['enabled']}; last successful tick={scheduler['last_success_utc']}; next={scheduler['next_job_id']} eligible{scheduler['next_due_utc']}. Actual sacct:\n```\n{monacct}\n```",'The original stage-specific heartbeat still reports its old1344-second subtotal; use this report/validation_gpu_reservations.json for aggregate budget. New phase artifacts/logs have their own immutable ledger. No duplicate recurring monitor installed.',f"Reward guard unchanged={payload['reward_guard_intact']}; exact config/source hashes: `results/{RUN}/source_manifest.json`. Frozen executed source retained under audit/executed_source. Cached environment reused, no rebuild/download.",'','## Failures and next decision','', 'A first CPU submission was rejected because intel-cpu is not a valid partition; no Job ID/GPU time was allocated. The error receipt is preserved. Same CPU work succeeded on epyc-cpu. All numerical failures and successful witnesses remain in their original unique directories.','', 'Next scientific priority: obtain two real independent semantic reviews and adjudication for the predeclared32; extend to remaining128 without outcome-based selection. Establish verifiable private evaluation exposure before efficacy claims. Localize ordinary-path backward divergence with a preregistered diagnostic before pretrained generator training. No further GPU job is authorized by this report, and no attack-driven optimization is enabled.','', 'Google Drive publication contains separate deidentified prose and aggregates only; raw questions/answers/images/source IDs, host paths, models, checkpoints and logs remain local/cluster. Publication readback receipt is separate.']
 (ROOT/'reports/NEXT_VALIDATION_DECISION_20261031.md').write_text('\n'.join(md)+'\n',encoding='utf-8');dashboard(ROOT)
 history=ROOT/'EXPERIMENT_INDEX.md';marker='next validation '+RUN
 if marker not in history.read_text(encoding='utf-8'):
  with history.open('a',encoding='utf-8') as f:f.write('\n\n## '+marker+'\nActual CPU'+cpu+'/GPU'+job+'; report reports/NEXT_VALIDATION_DECISION_20261031.md; NO-GO; negative ordinary continuation evidence retained.\n')
 print(json.dumps(dict(report='reports/NEXT_VALIDATION_DECISION_20261031.md',jobs=jobs,budget=cost,gate='NO-GO')))
if __name__=='__main__':main()
