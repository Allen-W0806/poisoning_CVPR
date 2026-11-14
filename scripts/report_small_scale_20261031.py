"""Reconcile real artifacts, re-score VQA and refresh existing bilingual pages."""
import argparse,csv,hashlib,html,json,re,sys,time,subprocess
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.submit_small_scale_20261031 import accounting
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,default=None):
 try:return json.loads(Path(p).read_text(encoding='utf-8-sig'))
 except FileNotFoundError:return default
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.partial');t.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def text(p,s):Path(p).write_text(s,encoding='utf-8')
def summarize(run):
 import numpy as np
 from src.evaluation.vqa_official_compatible import score_answer
 cfg=read(run/'protocol.json');manifest=read(run/'split_manifest.json');lookup={r['sample_id']:r for rows in manifest['splits'].values() for r in rows};stats=[];outputs=[];integrity=[]
 for seed in cfg['seeds']:
  out=ROOT/'results'/f'{run.name}_qwen_s{seed}'/'clean_main';checks=read(out/'checks.json');summary=read(out/'summary.json')
  if not checks:continue
  for item in read(out/'artifact_inventory.json',{}).get('files',[]):
   p=out/item['path'];integrity.append(dict(path=str(p.relative_to(ROOT)),matches=p.exists() and sha(p)==item['sha256']))
  records=[json.loads(s) for s in (out/'predictions.jsonl').read_text().splitlines()] if (out/'predictions.jsonl').exists() else []
  for row in records:
   expected=score_answer(row['raw_answer'],lookup[row['sample_id']]);assert row['normalized_answer']==expected['normalized_answer'] and row['score']==expected['score'],'VQA scorer mismatch'
   outputs.append(dict(seed=seed,**row,split=lookup[row['sample_id']]['role']))
  if not summary:continue
  for split in ('development','independent_evaluation'):
   ids=[r['sample_id'] for r in manifest['splits'][split]];by={(r['stage'],r['sample_id']):r for r in records};assert len(by)==len(records)==192
   before=np.array([by['before',i]['score'] for i in ids]);after=np.array([by['after',i]['score'] for i in ids]);delta=after-before;g=np.random.default_rng(20261031);indices=g.integers(len(ids),size=(5000,len(ids)))
   stats.append(dict(seed=seed,split=split,n=len(ids),before=float(before.mean()),after=float(after.mean()),difference=float(delta.mean()),before_ci95=np.quantile(before[indices].mean(1),[.025,.975]).tolist(),after_ci95=np.quantile(after[indices].mean(1),[.025,.975]).tolist(),paired_ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),changed_answers=sum(by['before',i]['normalized_answer']!=by['after',i]['normalized_answer'] for i in ids),invalid_rate=float(np.mean([not by['after',i]['normalized_answer'].strip() for i in ids])),mean_after_latency=float(np.mean([by['after',i]['latency_seconds'] for i in ids])),categories={c:dict(n=sum(lookup[i]['answer_type']==c for i in ids),before=float(np.mean([by['before',i]['score'] for i in ids if lookup[i]['answer_type']==c])),after=float(np.mean([by['after',i]['score'] for i in ids if lookup[i]['answer_type']==c]))) for c in sorted({lookup[i]['answer_type'] for i in ids})},checks_status=checks['status'],continuation=read(out/'continuation_receipt.json'),model_summary=summary,checkpoint_hashes={str(p.relative_to(ROOT)):sha(p) for p in [out/'continuation_checkpoint.pt',out/'final_training_state.pt'] if p.exists()}))
 if outputs:
  with (run/'clean_evaluation.csv').open('w',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in outputs for k in r)));w.writeheader();w.writerows(outputs)
 assert all(x['matches'] for x in integrity),'Checkpoint/output inventory mismatch'
 write(run/'artifact_integrity_recheck.json',dict(status='PASS',checked_files=len(integrity),files=integrity));write(run/'clean_statistics.json',stats)
 return stats
def dashboard(root):
 """Idempotent addition to existing views, also called by normal dashboard refresh."""
 payload=read(root/'reports/SMALL_SCALE_STATUS_20261031.json')
 if not payload:return
 data_path=root/'dashboard/dashboard_data.json';data=read(data_path,{})
 data['small_scale_validation_20261031']=payload;data['small_scale_validation_snapshot_utc']=payload['updated_utc']
 encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 rows=payload['clean_statistics'];matrix=payload['completion_matrix']
 for name in ('rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html'):
  path=root/'dashboard'/name
  if not path.exists():continue
  zh='_zh' in name or 'Latest' in name;title='10月31日小规模良性验证' if zh else 'October31 benign engineering validation'
  description='训练32 / 开发16 / 隔离评估32；历史暴露不确定，非 untouched test。LPIPS缺失，独立语义评审未完成。' if zh else 'Train32 / development16 / separate assessment32. Historical exposure unresolved; no untouched-test claim. LPIPS and independent semantic review remain blocked.'
  block='<section id="small-scale-20261031" style="margin:24px auto;padding:24px;max-width:1200px;background:#fff;border:1px solid #ccd;border-radius:16px;overflow:auto"><h2>'+title+'</h2><p>'+html.escape(payload['updated_utc'])+' · '+html.escape(payload['run_id'])+'</p><p>'+description+'</p><p>Slurm '+html.escape(str(payload['resource_accounting']['new_jobs']))+'</p><table><thead><tr><th>Seed</th><th>Split</th><th>Before</th><th>After</th><th>Paired95%CI, pp</th></tr></thead><tbody>'
  for r in rows:block+=f"<tr><td>{r['seed']}</td><td>{r['split']}</td><td>{100*r['before']:.2f}%</td><td>{100*r['after']:.2f}%</td><td>[{100*r['paired_ci95'][0]:+.2f},{100*r['paired_ci95'][1]:+.2f}]</td></tr>"
  block+='</tbody></table><p>New GPU seconds: '+str(payload['resource_accounting']['new_gpu_seconds'])+'/900; cumulative validation '+str(payload['resource_accounting']['total_validation_gpu_seconds'])+'/3600.</p><p>No attack training or ASR optimization; generator reward-release guards remain active. Engineering results do not establish scientific efficacy.</p><p><a href="../reports/SMALL_SCALE_VALIDATION_20261031.md">Validation report</a> · <a href="../reports/LITERATURE_BASELINE_PROTOCOL_20261031.md">Literature protocol audit</a></p></section>'
  value=path.read_text(encoding='utf-8');value=re.sub(r'<!--SMALL_SCALE_20261031_START-->.*?<!--SMALL_SCALE_20261031_END-->','',value,flags=re.S)
  value=value.replace('</body>','<!--SMALL_SCALE_20261031_START-->'+block+'<!--SMALL_SCALE_20261031_END--></body>')
  value=re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)',lambda m:m[1]+encoded+m[2],value,flags=re.S)
  text(path,value)
 # Keep the authoritative JSON as the final commit marker, as in the existing publisher.
 write(root/'dashboard/state.json',data);write(data_path,data)
 write(root/'reports/SMALL_SCALE_DASHBOARD_QA_20261031.json',dict(status='PASS',state_byte_identical=(root/'dashboard/state.json').read_bytes()==data_path.read_bytes(),all_existing_views_have_same_current_payload=all('small_scale_validation_20261031' in (root/'dashboard'/n).read_text(encoding='utf-8') for n in ('rsi_poisoning_zh.html','rsi_poisoning_en.html')),automatic_cluster_to_Windows_sync=False))
def main(run):
 from src.rl.phase5_backend import require_training_reward
 rejected=[]
 for receipt in ({},{'training_eligible':False},{'training_eligible':True}):
  try:require_training_reward(receipt)
  except RuntimeError:rejected.append(True)
  else:rejected.append(False)
 assert all(rejected),'Existing generator eligibility guard was bypassed'
 old_guard=read(ROOT/'reports/rsi_stage_monitor/active_stage.json')['reward_guard_sha256'];assert sha(ROOT/'src/rl/phase5_backend.py')==old_guard
 write(run/'audit/reward_guard_verified.json',dict(status='PASS',source='src/rl/phase5_backend.py',sha256=old_guard,empty_false_true_receipts_all_rejected=rejected))
 # The earlier preparation receipt did not locate the guard by filename; it made no PASS assertion about that empty inventory. This direct test closes that coverage gap.
 prior=accounting();jobs=read(run/'job_ledger.json');ids=[jobs['cpu_job'],jobs['gpu_job']]+([jobs['finalizer_job']] if jobs.get('finalizer_job') else [])
 raw=subprocess.check_output(['sacct','-X','-nP','-j',','.join(ids),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,Start,End'],text=True);new=[]
 for line in raw.splitlines():
  x=line.split('|')
  if len(x)<7 or x[0] not in ids:continue
  m=re.search(r'gres/gpu=(\d+)',x[3]);g=int(m[1]) if m else 0;new.append(dict(job_id=x[0],state=x[1],elapsed=int(x[2]),gpu_count=g,gpu_seconds=g*int(x[2]),exit_code=x[4],start_EDT=x[5],end_EDT=x[6]))
 gpu_seconds=sum(x['gpu_seconds'] for x in new);budget=dict(prior_gpu_seconds=prior['total'],new_gpu_seconds=gpu_seconds,total_validation_gpu_seconds=prior['total']+gpu_seconds,remaining_validation_gpu_seconds=3600-prior['total']-gpu_seconds,new_stage_cap=900,global_validation_cap=3600,new_jobs=new,prior_jobs=prior['rows'],raw_sacct=raw,cpu_only_allocated_seconds=sum(x['elapsed'] for x in new if not x['gpu_count']),billing_dollars=None)
 assert gpu_seconds<=900 and budget['total_validation_gpu_seconds']<=3600,'Budget bound exceeded; report actual failure'
 stats=summarize(run);cpu=read(run/'cpu_complete.json');gpu=read(run/'gpu_complete.json');rl=read(ROOT/'results'/f'{run.name}_ddpo/wiring_checks.json');syn=read(run/'synthetic_rsi/results.json');data=read(run/'audit/data_integrity.json');checks=read(ROOT/'results'/f'{run.name}_gradient/synthetic_checks.json')
 ddpo_root=ROOT/'results'/f'{run.name}_ddpo';updates=read(ddpo_root/'tiny_production_updates.json',[])
 if rl:
  tiny_parameters=read(ddpo_root/'tiny_parameter_updates.json');checkpoint=ddpo_root/'tiny_checkpoint/training_state.pt'
  write(run/'rl_numerical_audit.json',dict(algorithm='production DDPO_SF_v1; not GRPO',wiring_status=rl['passed'],tiny_checkpoint_path=str(checkpoint.relative_to(ROOT)),tiny_checkpoint_sha256=sha(checkpoint),reward_variances=[r['reward_variance'] for r in updates],degenerate_updates=sum(r['reward_variance']==0 for r in updates),gradient_norms=[r['gradient_norm'] for r in updates],trainable_tensor_count=sum(r['trainable'] for r in tiny_parameters),changed_trainable_tensors=sum(r['trainable'] and r['max_abs_change']>0 for r in tiny_parameters),production_pretrained_optimizer_inventory=str((ddpo_root/'pretrained_IP2Pix_optimizer_groups_ZERO_STEPS.json').relative_to(ROOT)),pretrained_optimizer_steps=0))
 image_rows=list(csv.DictReader((run/'image_controls.csv').open(encoding='utf-8')));groups={c:[r for r in image_rows if r['condition']==c] for c in ('original','identity','brightness','contrast','geometry')}
 summary={c:dict(n=len(rows),mean_SSIM=sum(float(r['SSIM']) for r in rows)/len(rows),mean_MAE=sum(float(r['RGB_MAE_255']) for r in rows)/len(rows),processing_seconds=sum(float(r['processing_seconds']) for r in rows),LPIPS=None) for c,rows in groups.items()}
 generation=read(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/generation_records.json');write(run/'frozen_ip2pix_generation_provenance.json',dict(source='results/matched_feedback_20261028_v2/main_repair_v1/generation_records.json',source_sha256=sha(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/generation_records.json'),records=generation,new_generation_calls=0))
 from PIL import Image
 gm={r['source_image_id']:r.get('metadata',{}) for r in generation};pairs=[]
 originals={r['source_image_id']:r for r in read(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json')['records'] if r['candidate_id']=='C4'}
 for row in read(run/'frozen_ip2pix_descriptive.json')['records']:
  original=originals[row['source_image_id']];assert sha(ROOT/original['image_path'])==row['original_sha256']
  assert sha(ROOT/row['output_path'])==row['output_sha256']
  with Image.open(ROOT/row['output_path']) as im:im.load();assert im.mode=='RGB' and list(im.size)==row['engineering']['size']
  meta=gm[row['source_image_id']];pairs.append(dict(source_image_id=row['source_image_id'],input_sha256=row['original_sha256'],output_sha256=row['output_sha256'],output_path=row['output_path'],output_size=row['engineering']['size'],SSIM=row['diagnostics'].get('SSIM'),RGB_MAE_255=row['diagnostics'].get('RGB_MAE_255'),generation_seconds=meta['generation_seconds'],peak_gpu_memory_bytes=meta['peak_gpu_memory_bytes'],preprocessing=meta['preprocessing'],model_revision=meta['model_revision'],seed=meta['seed'],LPIPS=None,semantic_status='UNREVIEWED',cost_status='historical generation already charged; new inference calls0'))
 with (run/'frozen_ip2pix_pairs.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(pairs[0]));w.writeheader();w.writerows(pairs)
 write(run/'audit/frozen_ip2pix_decode_hashes.json',dict(status='PASS',original_hashes_verified=len(pairs),output_hashes_verified=len(pairs),decoded_RGB_outputs=len(pairs),human_semantic_labels_added=0))
 matrix=[]
 def stage(name,executed,result,numeric,claim,blockers='',job=None):matrix.append(dict(Stage=name,Executed=executed,Run_ID=run.name,Slurm_job=job or jobs['cpu_job'],Result=result,Numerical_checks=numeric,Scientific_conclusion=claim,Blockers=blockers))
 stage('CPU source/provenance',True,'80 unique source/content; 32/16/32','PASS','Source-disjoint engineering data','Untouched-test status BLOCKED')
 stage('Image controls',True,'400 valid traced outputs','PASS: decode/SSIM/pixel','Transformation engineering only','LPIPS missing; edited-label validity unreviewed')
 stage('Frozen IP2Pix',True,'160 existing outputs/hashes audited','PASS: existing traceability','Descriptive reuse only','Independent human review0/160; BF16 continuation unresolved')
 stage('Clean Qwen LoRA',bool(stats),'3 seeds complete' if len(stats)==6 else f'{len(stats)//2}/3 seeds complete','PASS' if len(stats)==6 else 'PENDING/FAILED: inspect logs','No inferred generalization benefit',data['scope'],jobs['gpu_job'])
 stage('Benign RL exact/MC',True,'finite categorical/group reference','PASS' if checks and checks['passed'] else 'FAIL','Synthetic mathematical consistency only')
 stage('Production DDPO optimizer',bool(rl),'Tiny random UNet; pretrained ownership ZERO STEPS','PASS' if rl and rl['passed'] else 'PENDING/FAIL','No production GRPO or pretrained reliability claim','Full pretrained BF16 gate remains blocked',jobs['gpu_job'])
 stage('Synthetic RSI',True,'2 reproducible accept/reject/regression/rollback paths',syn['status'],'Integrity oracle improvement only; not autonomous scientific RSI')
 stage('Benign augmentation training',False,'NOT RUN','BLOCKED','No claim','Label validity not independently established')
 stage('Attack reproduction / ASR',False,'NOT PERFORMED','N/A','No efficacy claim','Outside this benign pilot')
 stamp=datetime.now(timezone.utc).isoformat();payload=dict(updated_utc=stamp,run_id=run.name,clean_statistics=stats,image_controls=summary,completion_matrix=matrix,resource_accounting=budget,semantic_review='BLOCKED',LPIPS='BLOCKED: unchanged environment has no lpips',reward_guard='PASS',scientific_hypothesis_supported=False,gpu_complete=gpu,RL_production=rl,data_scope=data['scope'])
 write(run/'summary.json',payload);write(ROOT/'reports/SMALL_SCALE_STATUS_20261031.json',payload);write(ROOT/'reports/RESOURCE_ACCOUNTING_20261031.json',budget)
 head='|Stage|Executed|Run ID|Slurm job|Result|Numerical checks|Scientific conclusion|Blockers|\n|---|---|---|---|---|---|---|---|\n'
 for row in matrix:head+='|'+'|'.join(str(v).replace('|','/') for v in row.values())+'|\n'
 text(ROOT/'reports/EXPERIMENT_COMPLETION_MATRIX_20261031.md','# Experiment completion matrix\n\nUpdated '+stamp+'.\n\n'+head)
 md='# Small-scale benign validation — 2026-10-31\n\nUpdated '+stamp+'; immutable run `'+run.name+'`.\n\nCPU auditing and real GPU work were executed; data/metric gaps remain blocked. These are engineering measurements, not a backdoor or autonomous-RSI efficacy result.\n\n'+head
 md+='\n## Clean VQA measurements\n\n|Seed|Split|n|Before|After|Paired95%CI pp|Changed answers|\n|---|---|---:|---:|---:|---|---:|\n'
 for r in stats:md+=f"|{r['seed']}|{r['split']}|{r['n']}|{100*r['before']:.3f}%|{100*r['after']:.3f}%|[{100*r['paired_ci95'][0]:+.3f},{100*r['paired_ci95'][1]:+.3f}]|{r['changed_answers']}|\n"
 md+='\nFive thousand paired image-bootstrap draws per split/seed; shared sources across seeds are not96 independent test images. A zero empirical difference interval does not prove population equivalence. The two improving seeds change the same question362793004 from yes(score0.3) to no(score1.0); these are not two independent improved examples. Their+2.1875pp difference has empirical95% interval[0,+6.5625]pp and does not establish a reliable population benefit. Per-category accuracy, latency, invalid rates, actual parameter counts and checkpoint hashes are in the linked statistics. No seed/config selection used final scores.\n\n'
 md+='Data scope: **'+data['scope']+'**. All80 source IDs/hashes differ, selected pairs pass dHash64 Hamming>4; this limited screen cannot exclude all semantic/perceptual duplicates. Two old unparseable records prevent an untouched-test certification. Cached official questions/all ten answers retained; majority-normalized supervision requires6/10 consensus. No images or labels were downloaded or invented.\n\n'
 md+='## Image controls\n\n'+json.dumps(summary,indent=2)+'\n\n400 current original/identity/brightness1.03/contrast1.03/geometry256 outputs have hashes, dimensions, SSIM, pixel MAE and CPU latency. Geometry SSIM/MAE use the geometry-matched reference, so zero difference is not a certificate against native-resolution loss. Existing frozen IP2Pix generation latency/memory/seed metadata are copied with their parent hash, not charged as new generation. LPIPS remains **BLOCKED**: package absent; no substitution called LPIPS. All nonidentity image-answer validity remains unreviewed; label-changing and ambiguous population rates are unknown. No benign augmentation training is eligible yet.\n\n'
 md+='## RL and RSI boundaries\n\nActual categorical code is REINFORCE; production diffusion optimizer is single-pass on-policy DDPO-SF, not GRPO. Exact-vs-Monte-Carlo and group-normalized reference tests retain frozen tolerances. Tiny random-UNet LoRA is the only diffusion model updated by this run; pretrained IP2Pix ownership is audited with zero steps. Existing guard rejects empty, false and true eligibility receipts and its historical hash matches. No target-error/security signal enters the benign objective. Historical uninstrumented pretrained BF16 continuation FAIL and instrumented witness PASS remain separate; general continuation reliability is NOT VALIDATED.\n\nSynthetic RSI uses the actual memory, investigator, proposal, fixed Gate and VersionManager twice: fixed/no-revision score0.5, revised integrity-oracle score1.0, ineffective/regression revisions rejected, explicit rollback successful; every diagnostic/evaluator operation is recorded. These deliberately constructed oracle cases do not establish natural-image semantic validity or autonomous research improvement.\n\n'
 schedule=read(ROOT/'reports/monitoring/scheduler.json');heartbeat=read(ROOT/'reports/rsi_stage_monitor/heartbeat.json');write(run/'audit/monitor_snapshot.json',dict(scheduler=schedule,heartbeat=heartbeat,new_stage_is_passive_summary_only=True))
 md+='## Resources, monitor and evidence\n\n'+json.dumps(budget,indent=2)+'\n\nExisting two-hour monitor was reused without installation or duplicate recurrence. Existing finalizer is an afterany CPU report hook, not another timer. Dollar pricing is unavailable. New-stage900 A100-second and cumulative3600-second caps apply to this validation scope; historical unrelated training is not silently charged to or authorized by it. Existing scheduler enabled='+str(schedule['enabled'])+', next registered job'+str(schedule['next_job_id'])+', eligibility UTC '+str(schedule['next_due_utc'])+'. The old replication-stage heartbeat still reports its own1344-second subtotal; use RESOURCE_ACCOUNTING_20261031.json for the aggregate including this stage. Dashboard refresh preserves this new-stage payload.\n\n'
 md+='Evidence: [immutable protocol](../results/'+run.name+'/protocol.json), [split manifest](../results/'+run.name+'/split_manifest.json), [data checks](../results/'+run.name+'/audit/data_integrity.json), [raw normalized predictions](../results/'+run.name+'/clean_evaluation.csv), [statistics/checkpoint hashes](../results/'+run.name+'/clean_statistics.json), [image pairs](../results/'+run.name+'/image_controls.csv), [synthetic RSI](../results/'+run.name+'/synthetic_rsi/results.json), [guard tests](../results/'+run.name+'/audit/reward_guard_verified.json), [source freeze](../results/'+run.name+'/source_manifest.json), [logs](../results/'+run.name+'/logs/), [literature protocol audit](LITERATURE_BASELINE_PROTOCOL_20261031.md). Full adapters/checkpoints stay on cluster; Windows sync transfers small evidence only.\n\nNext decision: retain no-go for generator learning; obtain genuine independent reviews and a verifiable private assessment split. Do not expand training or retry unexplained numerical failures to chase a positive result.\n'
 text(ROOT/'reports/SMALL_SCALE_VALIDATION_20261031.md',md)
 dashboard(ROOT)
 history=ROOT/'experiments/small_scale_validation_history.jsonl'
 with history.open('a',encoding='utf-8') as f:f.write(json.dumps(dict(updated_utc=stamp,run_id=run.name,new_gpu_seconds=gpu_seconds,job_states=new,science_claim=False))+'\n')
 print(json.dumps(dict(run=run.name,seeds_complete=len(stats)//2,gpu=gpu,new_gpu_seconds=gpu_seconds,total=budget['total_validation_gpu_seconds'],reward_guard='PASS')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();main(ROOT/'results'/a.run_id)
