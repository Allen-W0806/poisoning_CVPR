"""Record completed CPU publication work; preserve the existing scheduler unchanged."""
import csv,hashlib,html,json,os,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'results/baseline_comparison_20261101T2255Z_v1'
NAMES=['BASELINE_REAL_IMAGE_RESULTS.md','BASELINE_REAL_IMAGE_RESULTS.csv',
       'BASELINE_REAL_IMAGE_RESULTS.tex','RL_RSI_ABLATION_RESULTS.md',
       'RL_RSI_ABLATION_RESULTS.csv','RL_RSI_ABLATION_RESULTS.tex',
       'PUBLISHED_BASELINE_PROTOCOL_MATRIX.md','CVPR_EXPERIMENTAL_RESULTS_DRAFT.tex']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
def cmd(args):
 p=subprocess.run(args,capture_output=True,text=True,check=True,env={**os.environ,'TZ':'UTC'},timeout=45)
 return dict(command=args,stdout=p.stdout,stderr=p.stderr,returncode=p.returncode)
def main():
 out=RUN/'closure';out.mkdir()
 before={n:sha(ROOT/n) for n in ['src/rl/phase5_backend.py','reports/monitoring/scheduler.json',
       'scripts/overnight_supervisor_20261101.py','reports/overnight_supervisor/state.json']}
 assert before['src/rl/phase5_backend.py']=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
 now=datetime.now(timezone.utc).isoformat()
 queue=cmd(['squeue','-u','ydk297','-o','%i|%j|%T|%S|%k'])
 jobs=cmd(['sacct','-X','-nP','-j','88209,88229,88230,88290,88311,88317',
           '--format=JobID,JobName,State,Start,End,ElapsedRaw,ExitCode,AllocTRES'])
 monitor=read(ROOT/'reports/monitoring/scheduler.json')
 assert monitor['enabled'] and monitor['installed'] and monitor['next_job_id']=='88311'
 assert sum('|poisoning-monitor|' in x for x in queue['stdout'].splitlines())==1
 from scripts.overnight_supervisor_20261101 import accounting
 budget=accounting(read(ROOT/'reports/overnight_supervisor/state.json'))
 assert budget['consumed']==2381 and budget['remaining']==1219 and budget['reserved']==0
 assert not budget['unreconciled_project_GPU_jobs']
 ledger=cmd(['sacct','-X','-nP','-u','ydk297','--starttime','2026-10-26',
             '--format=JobID,State,Start,End,ElapsedRaw,AllocTRES,WorkDir%240'])
 owned=[]
 for line in ledger['stdout'].splitlines():
  v=line.split('|')
  if len(v)<7 or v[6].rstrip('/')!=str(ROOT):continue
  match=re.search(r'gres/gpu=(\d+)',v[5]);g=int(match[1]) if match else 0
  if g:owned.append(dict(job_id=v[0],state=v[1],gpu_count=g,gpu_seconds=g*int(v[4])))
 put(out/'live_status.json',dict(checked_UTC=now,queue=queue,selected_accounting=jobs,
     monitor=monitor,supervisor_heartbeat=read(ROOT/'reports/overnight_supervisor/heartbeat.json'),
     validation_budget=budget,visible_canonical_GPU_jobs_since_20261026=owned,
     visible_canonical_GPU_seconds=sum(x['gpu_seconds'] for x in owned),
     visible_history_scope='WorkDir exact match only, not full lifetime cost'))
 # Validate references without relying on an absence of compiler warnings.
 texpaths=[ROOT/'reports'/n for n in NAMES if n.endswith('.tex')]
 tex='\n'.join(p.read_text() for p in texpaths)
 labels=re.findall(r'\\label\{([^}]+)\}',tex);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',tex)
 assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
 assert not re.findall(r'\\cite\w*\{',tex)
 proof=read(RUN/'latex_verification.json')
 assert proof['kind']=='success' and all(sha(ROOT/n)==h for n,h in proof['report_sha256'].items())
 disk=read(ROOT/'results/audit_remediation_20261101T0120Z_v1/audit/preadaptation_disk_verification.json')
 assert disk['status']=='PASS' and len(disk['comparisons'])==12 and all(x['bitwise_equal'] for x in disk['comparisons'])
 # Append scoped new evidence; do not rewrite historical audit rows.
 matrix=ROOT/'reports/EXPERIMENT_COMPLETION_MATRIX_20261031.md'
 old=matrix.read_text();(out/'completion_matrix_before.md').write_text(old)
 event='baseline_comparison_20261101T2255Z_v1'
 assert event not in old
 matrix.write_text(old+'\n## Completed saved-data comparison '+now+'\n\n'
  '|Stage|Run ID|Slurm|Engineering result|Scientific scope|Remaining blocker|\n'
  '|---|---|---|---|---|---|\n'
  '|Matched160 controls and aligned SSIM|'+event+'|88317|PASS:960 primary records;640 new CPU SSIM|Frozen original-reference agreement only|0 independent labels;development exposure;LPIPS null|\n'
  '|Tiny3-seed/3-arm statistics|'+event+'|88317 + bounded CPU supplement|PASS:saved-data CIs reproduce;12 disk-prefix comparisons reused|All3 contrasts include0;RSI superiority NOT ESTABLISHED|3 seeds;synthetic task|\n'
  '|Publication tables/Results|'+event+'|CPU only|PASS:8 deliverables;current CVPR style compile|No external baseline reproduced|Full-method scientific gate NO-GO|\n',encoding='utf-8')
 real=read(RUN/'real_image/summary.json');abl=read(RUN/'ablation/recomputed.json')
 record=dict(event=event,checked_UTC=now,job_id='88317',state='COMPLETED',exit_code='0:0',
             new_GPU_seconds=0,new_training_steps=0,new_VLM_calls=0,new_generation_calls=0,
             new_CPU_SSIM_measurements=640,unique_sources=160,primary_records=960,
             validation_budget={k:budget[k] for k in ['cap','consumed','reserved','remaining']},
             monitor_next_job_id=monitor['next_job_id'],monitor_next_due_UTC=monitor['next_due_utc'],
             monitor_last_success_UTC=monitor['last_success_utc'],scientific_gate='NO-GO',
             RSI_superiority='NOT ESTABLISHED',BF16='NOT FIXED',human_labels=0,
             reports={n:sha(ROOT/'reports'/n) for n in NAMES})
 put(out/'publication_receipt.json',record)
 for name in NAMES:
  p=ROOT/'reports'/name;(RUN/'publication'/name).write_bytes(p.read_bytes())
 with (ROOT/'experiments/overnight_supervisor_history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
 # Add one shared source and matching bilingual annotations. No timer or trainer mutation.
 shared=dict(**record,conditions=real['conditions'],arms=abl['aggregates'])
 put(ROOT/'dashboard/baseline_comparison_latest.json',shared)
 files=['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']
 archive=out/'dashboard_before';archive.mkdir()
 for name in files:
  p=ROOT/'dashboard'/name
  if not p.exists():continue
  prior=p.read_text();(archive/name).write_text(prior)
  zh='_zh' in name or 'Latest' in name
  title='最新匹配对照：仅工程与描述性结果' if zh else 'Latest matched comparisons: engineering and descriptive evidence'
  rows=''.join('<tr><td>'+x['condition']+'</td><td>'+f"{100*x['original_reference_agreement']:.4f}%"+'</td><td>'+str(x['changed_predictions_vs_C0'])+'</td><td>'+f"{x['SSIM_aligned256_mean']:.6f}"+'</td></tr>' for x in real['conditions'])
  body='<aside id="baseline-comparison" style="padding:20px;background:#e6f2e9;color:#162d24"><h2>'+title+'</h2><p>'+html.escape(now)+' | CPU88317 COMPLETED0:0 | 160 sources,960 primary records | GPU+0</p><table><tr><th>Condition</th><th>Original-reference agreement</th><th>Changed/C0</th><th>SSIM</th></tr>'+rows+'</table><p>RSI-Fixed=-1.380162e-6;95%CI[-1.131677e-5,8.262257e-6]. RSI superiority NOT ESTABLISHED;BF16 NOT FIXED;human labels0;scientific NO-GO.</p><p>GPU validation2381/3600s;remaining1219s. Existing monitor88311: '+html.escape(monitor['next_due_utc'])+'</p><p><a href="../reports/BASELINE_REAL_IMAGE_RESULTS.md">Natural-image controls</a> | <a href="../reports/RL_RSI_ABLATION_RESULTS.md">RL/RSI ablation</a> | <a href="../reports/PUBLISHED_BASELINE_PROTOCOL_MATRIX.md">Protocol matrix</a></p></aside>'
  prior=re.sub(r'<!--BASELINE_COMPARISON_START-->.*?<!--BASELINE_COMPARISON_END-->','',prior,flags=re.S)
  assert '<body' in prior
  p.write_text(re.sub(r'(<body[^>]*>)',lambda m:m[1]+'<!--BASELINE_COMPARISON_START-->'+body+'<!--BASELINE_COMPARISON_END-->',prior,count=1),encoding='utf-8')
 for name in ['state.json','dashboard_data.json']:
  p=ROOT/'dashboard'/name;d=read(p);d['baseline_comparison_latest']=shared;put(p,d)
 assert all(sha(ROOT/n)==h for n,h in before.items()),'Scheduler, supervisor or guard changed'
 put(out/'checks.json',dict(status='PASS_SCOPED_PUBLICATION',files=len(NAMES),
     references_resolved=sorted(set(refs)),duplicate_labels=False,
     disk_prefix_comparisons=12,compiler_kind=proof['kind'],
     guard_and_monitor_state_unchanged=True,new_GPU_seconds=0,new_periodic_jobs=0,
     total_visible_canonical_GPU_seconds=sum(x['gpu_seconds'] for x in owned)))
 print(json.dumps(record,ensure_ascii=False))
if __name__=='__main__':main()
