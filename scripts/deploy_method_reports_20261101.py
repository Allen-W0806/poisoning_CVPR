import base64,json,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='method_code_audit_20261101T0510Z_v1'
names=['scripts/report_method_audit_20261101.py','scripts/verify_method_contracts_20261101.py','results/'+RUN+'/audit/pdf_readback.json'];files={n:base64.b64encode((ROOT/n).read_bytes()).decode() for n in names}
code='''import base64,json,subprocess
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');run=r/'results'/'''+repr(RUN)+''';files='''+repr(files)+'''
for n,v in files.items():
 p=r/n;p.parent.mkdir(parents=True,exist_ok=True);content=base64.b64decode(v)
 if p.exists():assert p.read_bytes()==content
 else:p.write_bytes(content)
ledger=json.loads((run/'job_ledger.json').read_text());assert 'report_job' not in ledger
cmd=['sbatch','--parsable','--no-requeue','--dependency=afterany:'+ledger['gpu_job'],'--partition=epyc-cpu','--account=allusers','--cpus-per-task=2','--mem=8G','--time=00:05:00','--job-name=poisoning-method-report','--output='+str(run/'logs/report-%j.out'),'--error='+str(run/'logs/report-%j.err'),'--wrap=cd '+str(r)+' && OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 '+str(r/'.venv-editor/bin/python')+' -B scripts/report_method_audit_20261101.py']
p=subprocess.run(cmd,capture_output=True,text=True);(run/'report_submission.json').write_text(json.dumps(dict(command=cmd,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode();ledger['report_job']=p.stdout.strip().split(';')[0];(run/'job_ledger.json').write_text(json.dumps(ledger,indent=2));print(json.dumps(ledger))
'''
p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-B','-'],input=code.encode(),capture_output=True,timeout=90);print(p.stdout.decode(),end='');print(p.stderr.decode(),end='');p.check_returncode()
