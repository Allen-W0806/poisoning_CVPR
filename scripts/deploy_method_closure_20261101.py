"""One-shot CPU closure; extend only the existing passive dashboard renderer."""
import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN='method_code_audit_20261101T0510Z_v1'
files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in ['scripts/close_method_audit_20261101.py','scripts/manage_method_audit_20261101.py']}
hook="    if (Path(root)/'reports/METHOD_AUDIT_CLOSURE_20261101.json').exists():\n        from scripts.close_method_audit_20261101 import dashboard as method_dashboard\n        method_dashboard(Path(root))\n"
code='''import base64,json,subprocess,shutil
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+'''
files='''+repr(files)+'''
for n,v in files.items():
 p=r/n;value=base64.b64decode(v)
 if p.exists():assert p.read_bytes()==value
 else:p.write_bytes(value)
path=r/'src/progression/dashboard.py';old=path.read_text();hook='''+repr(hook)+'''
assert 'method_dashboard' not in old and old.count('    return result')==1
backup=run/'audit/dashboard_hook_before.py';assert not backup.exists();shutil.copyfile(path,backup);path.write_text(old.replace('    return result',hook+'    return result'));compile(path.read_text(),str(path),'exec')
ledger=json.loads((run/'job_ledger.json').read_text());assert 'closure_job' not in ledger
cmd=['sbatch','--parsable','--no-requeue','--partition=epyc-cpu','--account=allusers','--cpus-per-task=2','--mem=8G','--time=00:03:00','--job-name=poisoning-method-close','--output='+str(run/'logs/closure-%j.out'),'--error='+str(run/'logs/closure-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/close_method_audit_20261101.py']
p=subprocess.run(cmd,capture_output=True,text=True);(run/'closure_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();ledger['closure_job']=p.stdout.strip().split(';')[0];(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-B','-'],input=code.encode(),capture_output=True,timeout=90)
print(p.stdout.decode(),end='');print(p.stderr.decode(),end='');p.check_returncode()
