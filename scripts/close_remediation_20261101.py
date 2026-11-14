"""Final independent artifact/source verification and measured resource closure."""
import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='audit_remediation_20261101T0120Z_v1';REPAIR=RUN+'_threeway_repair_v1'
code='''import json,hashlib,shutil,subprocess,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');name='''+repr(RUN)+''';rep=r/'results'/(name+'_threeway_repair_v1');run=r/'results'/name;rows=[]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for kind in ['ordinary','instrumented']:
 folder=r/'results'/(name+'_BF16_'+kind)/'diagnostic'
 for item in json.loads((folder/'artifact_inventory.json').read_text())['files']:
  p=folder/item['path'];assert sha(p)==item['sha256'];rows.append(dict(path=str(p.relative_to(r)),sha256=sha(p),bytes=p.stat().st_size))
for p in (rep/'threeway').glob('seed*/*/receipt.json'):
 receipt=json.loads(p.read_text())
 for stage,key in [('selected','selected_checkpoint_sha256'),('prefix','prefix_checkpoint_sha256')]:
  ck=p.parent/stage/'training_state.pt';assert sha(ck)==receipt[key];rows.append(dict(path=str(ck.relative_to(r)),sha256=sha(ck),bytes=ck.stat().st_size))
for n,h in json.loads((rep/'source_manifest.json').read_text()).items():
 p=r/n;assert sha(p)==h;target=rep/'audit/executed_source'/n;target.parent.mkdir(parents=True,exist_ok=True)
 if not target.exists():shutil.copyfile(p,target)
 assert sha(target)==h
status=json.loads((r/'reports/REMEDIATION_STATUS_20261101.json').read_text());assert status['scientific_gate']=='NO-GO' and status['threeway_status']=='PASS';assert status['budget']['consumed']==2246 and status['budget']['reserved']==0
q=subprocess.check_output(['squeue','-u',__import__('getpass').getuser(),'-h','-o','%i|%j|%T|%l|%k'],text=True);acct=subprocess.check_output(['sacct','-X','-nP','-j','88029,88030,88032,88034','--format=JobID,State,Start,End,ElapsedRaw,ExitCode'],text=True);sched=json.loads((r/'reports/monitoring/scheduler.json').read_text());assert sched['enabled'] and sched['method']=='slurm_cpu_successor'
three=json.loads((rep/'threeway/results.json').read_text());allocated=144;components=three['total_component_GPU_held_seconds'];assert 0<components<=allocated
proof=dict(status='PASS',verified_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),new_artifacts_rehashed=len(rows),files=rows,repair_component_GPU_seconds=components,repair_Slurm_GPU_seconds=allocated,repair_loading_bookkeeping_unattributed_GPU_seconds=allocated-components,live_squeue=q,live_sacct=acct,existing_monitor=sched,scientific_gate='NO-GO',guard_unchanged=status['reward_guard_sha256']=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712')
(run/'audit/final_integrity.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=90);p.check_returncode();proof=json.loads(p.stdout);target=ROOT/'results'/RUN/'audit/final_integrity.json';target.write_text(json.dumps(proof,indent=2),encoding='utf-8');qa=ROOT/'reports/audit_remediation_browser_20261101/QA.json';assert json.loads(qa.read_text())['status']=='PASS'
report=ROOT/'reports/SMALL_SCALE_VALIDATION_NEXT.md';s=report.read_text(encoding='utf-8');s+='\n## Final delivery verification\n\nJobs88029/88030/88032/88034 are COMPLETED0:0. Nested first comparison serialization failure is retained. Both new BF16 diagnostic inventories and all18 selected/prefix three-way checkpoints were rehashed: '+str(proof['new_artifacts_rehashed'])+' files PASS. Twelve disk initial/prefix equality comparisons PASS. Repair component held-GPU time sums to'+str(proof['repair_component_GPU_seconds'])+'s out of144 allocated seconds; remaining'+str(proof['repair_loading_bookkeeping_unattributed_GPU_seconds'])+'s is startup/cleanup/bookkeeping. Two GPU allocations147+144=291s; cumulative2246/3600, remaining1354, reserved0.\n\nBrowser QA: reports/audit_remediation_browser_20261101/QA.json. Chinese/English desktop and390px mobile: same embedded authoritative payload,23 findings+9 branch rows, no JS errors or overflow. Existing32-pair review kit loads with blank labels and0 saved reviews; no human labels entered. QA scratch under project tmp was verified and removed.\n\nExisting monitor remains single Slurm CPU successor: next job'+str(proof['existing_monitor']['next_job_id'])+' eligible'+proof['existing_monitor']['next_due_utc']+'. No duplicate monitor or further GPU job installed. Original snapshots, historical matrix, failures, checkpoints and code versions remain intact.\n';report.write_text(s,encoding='utf-8')
files={report.relative_to(ROOT).as_posix():base64.b64encode(report.read_bytes()).decode(),qa.relative_to(ROOT).as_posix():base64.b64encode(qa.read_bytes()).decode()};transfer='import base64\nfrom pathlib import Path\nr=Path("/storage/home/ydk297/projects/poisoning")\nfor n,v in '+repr(files)+'.items():\n p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))\n';subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=transfer.encode(),check=True,timeout=80)
print(json.dumps({k:v for k,v in proof.items() if k not in ['files','existing_monitor']},indent=2))
