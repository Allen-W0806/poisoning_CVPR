import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='audit_remediation_20261101T0120Z_v1'
names=['scripts/report_remediation_20261101.py','docs/CURRENT_METHOD_20261101.md','docs/METHOD.md','src/progression/dashboard.py'];package={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
code=r'''import base64,json,subprocess,shutil
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+r''';files='''+repr(package)+r'''
banner='\n> Historical infrastructure-stage description retained below. For the current diffusion DDPO, clean Qwen, evaluation and audit boundaries, use [CURRENT_METHOD_20261101.md](CURRENT_METHOD_20261101.md). This dated entry distinguishes implemented components from unvalidated scientific claims.\n'
hook="    if (Path(root)/'reports/REMEDIATION_STATUS_20261101.json').exists():\\n        from scripts.report_remediation_20261101 import dashboard as remediation_dashboard\\n        remediation_dashboard(Path(root))\\n".replace('\\n','\n')
for n,v in files.items():
 p=r/n;content=base64.b64decode(v);text=content.decode('utf-8').replace('\r\n','\n')
 if p.exists():
  old=p.read_text(encoding='utf-8')
  if n=='docs/METHOD.md':assert old==text.replace(banner,'')
  elif n=='src/progression/dashboard.py':assert old==text.replace(hook,'')
  else:assert p.read_bytes()==content
  backup=run/'audit/before'/n;backup.parent.mkdir(parents=True,exist_ok=True);assert not backup.exists();shutil.copyfile(p,backup)
 p.write_bytes(content)
ledger=json.loads((run/'job_ledger.json').read_text());repair=json.loads((r/'results'/(run.name+'_threeway_repair_v1')/'job_ledger.json').read_text());assert not (run/'report_submission.json').exists()
cmd=['sbatch','--parsable','--no-requeue','--dependency=afterany:'+repair['gpu_job']+':'+ledger['cpu_job'],'--partition=epyc-cpu','--account=allusers','--cpus-per-task=2','--mem=4G','--time=00:03:00','--job-name=poisoning-audit-report','--output='+str(run/'logs/report-%j.out'),'--error='+str(run/'logs/report-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/report_remediation_20261101.py']
p=subprocess.run(cmd,capture_output=True,text=True);(run/'report_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();ledger['report_job']=p.stdout.strip().split(';')[0];(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=80);print(p.stdout.decode(),end='');print(p.stderr.decode(),end='');p.check_returncode()
