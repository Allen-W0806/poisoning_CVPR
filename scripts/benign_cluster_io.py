"""Small transport for a single isolated benign audit, using existing SSH auth.

Never modifies canonical runtime source, shared environments, models, or data.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
RUN = 'benign_ddpo_audit_20261111_v1'
REMOTE_ROOT = '/storage/home/ydk297/projects/poisoning'
REMOTE = REMOTE_ROOT+'/experiments/'+RUN
PYTHON = REMOTE_ROOT+'/.venv-editor/bin/python'
LOCAL = ROOT/'experiments'/RUN


def remote(code, input_bytes=None, timeout=240):
    wrapper = 'exec(__import__("base64").b64decode("'+base64.b64encode(code.encode()).decode()+'"))'
    command = shlex.join([PYTHON,'-c',wrapper])
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','utc-login',command],
                          input=input_bytes,capture_output=True,timeout=timeout)


def invoke(code, input_bytes=None, timeout=240, log=None):
    result = remote(code,input_bytes,timeout)
    if log:
        (LOCAL/log).write_bytes(result.stdout+result.stderr)
    sys.stdout.buffer.write(result.stdout)
    sys.stderr.buffer.write(result.stderr)
    if result.returncode: raise SystemExit(result.returncode)
    return result.stdout


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('action', choices=['stage','cpu','submit','status','fetch','instructions','postrun_cpu','fetch_postrun','finalize'])
    args = parser.parse_args()
    if args.action == 'finalize':
        code=f'''import hashlib,importlib.metadata,json,subprocess
from pathlib import Path
root=Path({REMOTE_ROOT!r});run=Path({REMOTE!r});job=json.loads((run/'submission.json').read_text())['job_id']
source_files=['src/rl/diffusion_policy.py','src/rl/diffusion_trajectory.py','src/mini_pilot/engine.py','src/training/generator_adapter.py','train_lighting_v3.py']
result={{'packages':dict(sorted((d.metadata.get('Name','UNKNOWN'),d.version) for d in importlib.metadata.distributions())),
 'sacct':subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES,Start,End'],text=True),
 'squeue_own_job':'\\n'.join(line for line in subprocess.check_output(['squeue','-h','-u','ydk297','-o','%i %T'],text=True).splitlines() if line.split()[0]==job),
 'owning_source_sha256':{{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in source_files}},
 'run_GPU_scratch_exists':(root/'tmp/bda_20261111_v1').exists(),
 'CPU_scratch_exists':(run/'cpu/scratch').exists(),
 'postrun_CPU_scratch_exists':(run/'postrun_fix/cpu/scratch').exists()}}
print(json.dumps(result,indent=2))
'''
        result=remote(code)
        if result.returncode: sys.stderr.buffer.write(result.stderr);raise SystemExit(result.returncode)
        record=json.loads(result.stdout)
        initial=json.loads((LOCAL/'remote_inventory.json').read_text(encoding='utf-8-sig'))
        record['canonical_source_unchanged_since_inventory']=record['owning_source_sha256']==initial['owning_source_sha256']
        (LOCAL/'final_resource_receipt.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in record.items() if k not in ('packages','owning_source_sha256')},indent=2))
    elif args.action == 'instructions':
        invoke("from pathlib import Path\nfor p in [Path('/storage/AGENTS.md'),Path('/storage/home/ydk297/AGENTS.md'),Path('"+REMOTE_ROOT+"/AGENTS.md')]:\n if p.is_file(): print(str(p)+'\\n'+p.read_text())\n")
    elif args.action in ('stage','postrun_cpu'):
        target_run = REMOTE if args.action == 'stage' else REMOTE+'/postrun_fix'
        files = list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))
        files += [ROOT/'scripts'/name for name in ['run_benign_ddpo_minimal.py','test_benign_ddpo_audit.py',
                  'run_bf16_adjoints_20261102.py','numerical_boundaries_20261102.py','audit_benign_crop_posthoc.py']]
        files += [ROOT/'configs/benign_ddpo_minimal.json',ROOT/'reports/phase2_model_metadata.json']
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in files: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
        code = f'''import io,json,sys,tarfile,hashlib
from pathlib import Path
run=Path({target_run!r})
run.mkdir(parents=True,exist_ok=True)
if (run/'submission.json').exists(): raise ValueError('Cannot replace source after GPU submission')
code=run/'code'; code.mkdir(exist_ok=True)
data=sys.stdin.buffer.read()
archives=run/'source_archives'; archives.mkdir(exist_ok=True)
if list(code.rglob('*.py')):
 with tarfile.open(archives/'before_refresh.tar.gz',mode='w:gz') as old:
  for path in code.rglob('*'):
   if path.is_file(): old.add(path,arcname=path.relative_to(code).as_posix(),recursive=False)
(archives/(hashlib.sha256(data).hexdigest()+'.tar.gz')).write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive.getmembers():
  target=(code/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(code.resolve()): raise ValueError('Unsafe source archive')
  target.parent.mkdir(parents=True,exist_ok=True)
  target.write_bytes(archive.extractfile(member).read())
receipt={{'archive_sha256':hashlib.sha256(data).hexdigest(),'source_files':len(list(code.rglob('*.py'))),'source_root':str(code),'canonical_source_modified':False}}
(run/'source_upload.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt))
'''
        invoke(code,buffer.getvalue(),log='stage.log')
        if args.action == 'postrun_cpu':
            code = f'''import os,subprocess,sys
from pathlib import Path
run=Path({target_run!r})
environment=os.environ.copy(); environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
with (run/'cpu_tests.log').open('w') as log:
 result=subprocess.run([{PYTHON!r},str(run/'code/scripts/test_benign_ddpo_audit.py'),str(run/'cpu')],cwd=run/'code',env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=180)
print((run/'cpu_tests.log').read_text())
if result.returncode: sys.exit(result.returncode)
subprocess.run([{PYTHON!r},str(run/'code/scripts/audit_benign_crop_posthoc.py'),{REMOTE_ROOT!r},{REMOTE!r},str(run/'crop_audit.json')],cwd=run/'code',env=environment,check=True)
'''
            invoke(code,timeout=220,log='postrun_cpu.log')
    elif args.action == 'cpu':
        code = f'''import os,subprocess,sys
from pathlib import Path
run=Path({REMOTE!r})
environment=os.environ.copy()
environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='')
with (run/'cpu_tests.log').open('w') as log:
 result=subprocess.run([{PYTHON!r},str(run/'code/scripts/test_benign_ddpo_audit.py'),str(run/'cpu')],cwd=run/'code',env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=180)
print((run/'cpu_tests.log').read_text())
sys.exit(result.returncode)
'''
        invoke(code,timeout=220,log='cpu_tests_remote.log')
    elif args.action == 'submit':
        code = f'''import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
run=Path({REMOTE!r}); code=run/'code'; assets=Path({REMOTE_ROOT!r})
receipt=json.loads((run/'cpu/cpu_tests.json').read_text())
if not receipt['passed']: raise ValueError('CPU gate failed')
if (run/'submission.json').exists(): raise ValueError('One allocation only; submission already exists')
scratch=assets/'tmp/bda_20261111_v1'
command=[{PYTHON!r},str(code/'scripts/run_benign_ddpo_minimal.py'),'--assets-root',str(assets),'--output',str(run/'gpu'),'--config',str(code/'configs/benign_ddpo_minimal.json'),'--cpu-receipt',str(run/'cpu/cpu_tests.json')]
script='\\n'.join(['#!/bin/bash','#SBATCH --job-name=benign-ddpo-audit','#SBATCH --partition=epyc-gpu','#SBATCH --account=allusers','#SBATCH --gres=gpu:tesla_a100_80gb:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=48G','#SBATCH --time=00:20:00',f'#SBATCH --chdir={{code}}',f'#SBATCH --output={{run}}/slurm-%j.log',f'#SBATCH --error={{run}}/slurm-%j.log','set -euo pipefail','export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false','export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8','export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',f'mkdir -p {{scratch}}',f'export TMPDIR={{scratch}}',"trap "+shlex.quote("python3 -c "+shlex.quote("from pathlib import Path; import shutil; p=Path('"+str(scratch)+"').resolve(); assert p.parent==Path('"+str(assets/'tmp')+"').resolve() and p.name=='bda_20261111_v1'; shutil.rmtree(p)"))+" EXIT",'timeout --signal=TERM 1080s '+shlex.join(command),''])
(run/'run.sbatch').write_text(script)
result=subprocess.run(['sbatch','--parsable',str(run/'run.sbatch')],text=True,capture_output=True)
record={{'created_UTC':datetime.now(timezone.utc).isoformat(),'reservation_GPU_seconds':1200,'scope':'Current user request: bounded benign validation only; old campaign budgets not reopened','command':command,'stdout':result.stdout,'stderr':result.stderr,'returncode':result.returncode,'job_id':result.stdout.strip().split(';')[0] if result.returncode==0 else None,'script':script}}
(run/'submission.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
'''
        invoke(code,log='submission_remote.json')
    elif args.action == 'status':
        code = f'''import json,subprocess
from pathlib import Path
run=Path({REMOTE!r})
submission=json.loads((run/'submission.json').read_text()); job=submission['job_id']
print(subprocess.run(['sacct','-X','-nP','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES,Start,End'],text=True,capture_output=True).stdout)
path=run/'gpu/status.json'
if path.is_file():
 state=json.loads(path.read_text()); print(json.dumps({{'status':state['status'],'elapsed_seconds':state['elapsed_seconds'],'numerical_gates':{{k:v.get('passed') for k,v in state['stages'].items()}},'inner':state.get('inner'),'outer_methods_finished':list(state.get('outer',{{}}))}},indent=2))
logs=list(run.glob('slurm-*.log'))
if logs: print('\\n'.join(logs[0].read_text(errors='replace').splitlines()[-6:]))
'''
        invoke(code,log='latest_remote_status.log')
    elif args.action in ('fetch','fetch_postrun'):
        fetch_run = REMOTE if args.action == 'fetch' else REMOTE+'/postrun_fix'
        # Checkpoints and failing numerical witnesses are reproducibility evidence.
        code = f'''import io,sys,tarfile
from pathlib import Path
run=Path({fetch_run!r}); buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
 for path in run.rglob('*'):
  if path.is_file() and 'code' not in path.relative_to(run).parts: archive.add(path,arcname=path.relative_to(run).as_posix(),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''
        result = remote(code,timeout=240)
        if result.returncode:
            sys.stderr.buffer.write(result.stderr); raise SystemExit(result.returncode)
        target = LOCAL/'remote' if args.action == 'fetch' else LOCAL/'postrun_fix'
        target.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
            for member in archive.getmembers():
                path = (target/member.name).resolve()
                if not member.isfile() or not path.is_relative_to(target.resolve()): raise ValueError('Unsafe result archive')
                path.parent.mkdir(parents=True,exist_ok=True)
                incoming = archive.extractfile(member).read()
                if path.is_file() and path.read_bytes()!=incoming and path.suffix in ('.pt','.png'):
                    raise ValueError('Refusing to overwrite changed finalized evidence')
                path.write_bytes(incoming)
        print(json.dumps({'fetched_files':sum(p.is_file() for p in target.rglob('*')),
                          'retained_bytes':sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}))


if __name__ == '__main__': main()
