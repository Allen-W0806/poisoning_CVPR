"""Deploy one state-machine extension, verify the same owned Slurm entry point."""
import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
names=['scripts/overnight_supervisor_20261101.py','scripts/overnight_evidence_cpu_20261101.py','scripts/overnight_bf16_pair_20261101.py']
files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
hook="            from overnight_supervisor_20261101 import cycle as overnight_cycle\n            legacy['overnight_supervisor']=overnight_cycle()\n"
dhook="    if (Path(root)/'reports/overnight_supervisor/heartbeat.json').exists():\n        from scripts.overnight_supervisor_20261101 import dashboard as overnight_dashboard\n        overnight_dashboard(Path(root))\n"
code=r'''import base64,hashlib,json,subprocess,shutil,uuid,datetime
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');base=r/'reports/overnight_supervisor';base.mkdir(exist_ok=False);(base/'audit').mkdir();files=FILES
for n,v in files.items():
 p=r/n;assert not p.exists();p.write_bytes(base64.b64decode(v));compile(p.read_text(),str(p),'exec');d=base/'audit/source'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
for n in ['scripts/manage_monitor.py','src/progression/dashboard.py']:
 p=r/n;d=base/'audit/before'/n;d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
p=r/'scripts/manage_monitor.py';old=p.read_text();assert 'overnight_cycle' not in old and old.count('            return legacy')==1;p.write_text(old.replace('            return legacy',HOOK+'            return legacy'));compile(p.read_text(),str(p),'exec')
p=r/'src/progression/dashboard.py';old=p.read_text();assert 'overnight_dashboard' not in old and old.count('    return result')==1;p.write_text(old.replace('    return result',DHOOK+'    return result'));compile(p.read_text(),str(p),'exec')
schedule=json.loads((r/'reports/monitoring/scheduler.json').read_text());assert schedule['enabled'] and not schedule['submission_uncertain'];q=subprocess.check_output(['squeue','-h','-u','ydk297','-o','%i|%j|%k|%T'],text=True);owned=[l for l in q.splitlines() if '|poisoning-monitor|poisoning-monitor-v1:f916f24e0de6:' in l];assert len(owned)==1 and owned[0].startswith(schedule['next_job_id']+'|');slot=datetime.datetime.now(datetime.timezone.utc).isoformat();tag='poisoning-monitor-v1:f916f24e0de6:verify_overnight_'+uuid.uuid4().hex
cmd=['sbatch','--parsable','--partition=epyc-cpu','--account=allusers','--cpus-per-task=1','--mem=2G','--time=00:05:00','--job-name=poisoning-monitor','--comment='+tag,'--output='+str(base/'verify-%j.out'),'--error='+str(base/'verify-%j.err'),str(r/'reports/monitoring/monitor_job.sbatch'),slot]
reply=subprocess.run(cmd,capture_output=True,text=True);receipt=dict(command=cmd,returncode=reply.returncode,stdout=reply.stdout,stderr=reply.stderr,existing_successor=schedule['next_job_id'],source_sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in files},prior_required_reports={n:hashlib.sha256((r/'reports'/n).read_bytes()).hexdigest() for n in ['METHOD_IMPLEMENTATION_AUDIT.md','METRIC_AND_NORMALIZATION_AUDIT.md','METHOD_AUDIT_STATUS_20261101.json','METHOD_AUDIT_CLOSURE_20261101.json','AUDIT_DECISION_MATRIX.md','NEXT_VALIDATION_DECISION_20261031.md']},git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=r,text=True).strip(),cloud_initial_failure='Existing googledrive grant returns invalid_grant; rclone official binary installed inside project; no credentials printed or uploaded')
(base/'installation_receipt.json').write_text(json.dumps(receipt,indent=2));reply.check_returncode();print(json.dumps(receipt))
'''.replace('FILES',repr(files)).replace('DHOOK',repr(dhook)).replace('HOOK',repr(hook))
p=subprocess.run(['ssh','utc-login','python3','-B','-'],input=code.encode(),capture_output=True,timeout=90);print(p.stdout.decode());print(p.stderr.decode());p.check_returncode()
