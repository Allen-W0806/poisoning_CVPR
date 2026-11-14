"""Isolated, bounded P0 transport. Never edits canonical runtime source or jobs."""
import argparse
import base64
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]
RUN='p0_foundations_20261111_v1'
ASSETS='/storage/home/ydk297/projects/poisoning'
REMOTE=ASSETS+'/experiments/'+RUN
PYTHON=ASSETS+'/.venv-editor/bin/python'
LOCAL=ROOT/'experiments'/RUN


def remote(code,data=None,timeout=240):
    expression='exec(__import__("base64").b64decode("'+base64.b64encode(code.encode()).decode()+'"))'
    return subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','utc-login',
                           shlex.join([PYTHON,'-c',expression])],input=data,capture_output=True,timeout=timeout)


def call(code,data=None,timeout=240,log=None):
    result=remote(code,data,timeout)
    if log: (LOCAL/log).write_bytes(result.stdout+result.stderr)
    sys.stdout.buffer.write(result.stdout);sys.stderr.buffer.write(result.stderr)
    if result.returncode: raise SystemExit(result.returncode)
    return result.stdout


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inventory','stage','prepare','cpu','submit','status','fetch','quality_followup','quality_status','finalize','cleanup']);a=p.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    if a.action=='cleanup':
        code=f'''import json
from pathlib import Path
run=Path({REMOTE!r}).resolve();removed=[];nonempty=[]
for candidate in [run/'tmp',run/'code/tmp']:
 path=candidate.resolve()
 if not path.is_relative_to(run) or path.name!='tmp':raise ValueError('Unsafe cleanup path')
 if path.exists():
  if any(path.iterdir()):nonempty.append(str(path))
  else:path.rmdir();removed.append(str(path))
print(json.dumps({{'removed_empty_run_tmp':removed,'nonempty_left_intact':nonempty}},indent=2))
'''
        call(code,log='remote_cleanup.json')
    elif a.action=='finalize':
        code=f'''import json,subprocess,hashlib,importlib.metadata,zipfile
from pathlib import Path
assets=Path({ASSETS!r});run=Path({REMOTE!r})
manifest=json.loads((run/'manifest.json').read_text());rows=[r for value in manifest['splits'].values() for r in value]
ids={{int(r['sample_id']) for r in rows}}
extract={{'annotations':[],'questions':[],'archive_SHA256':{{}}}}
for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
 path=assets/'data/raw/vqav2'/filename;extract['archive_SHA256'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
 with zipfile.ZipFile(path) as archive:
  name=next(n for n in archive.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
  extract[key]=[r for r in json.loads(archive.read(name))[key] if int(r['question_id']) in ids]
lookup={{str(r['question_id']):r for r in extract['annotations']}};questions={{str(r['question_id']):r for r in extract['questions']}}
for row in rows:
 ann=lookup[row['sample_id']];question=questions[row['sample_id']]
 assert [x['answer'] for x in ann['answers']]==row['answers'] and question['question']==row['question'] and str(ann['image_id'])==row['image_id']
extract['all_16_manifest_labels_match_raw_archives']=True
jobs=[json.loads((run/'submission.json').read_text())['job_id'],json.loads((run/'quality_followup/submission.json').read_text())['job_id']]
account=subprocess.check_output(['sacct','-X','-nP','-j',','.join(jobs),'--format=JobID,State,ElapsedRaw,AllocTRES,Start,End'],text=True)
active='\\n'.join(line for line in subprocess.check_output(['squeue','-h','-u','ydk297','-o','%i %T'],text=True).splitlines() if line.split()[0] in jobs)
record={{'sacct':account,'active_own_jobs':active,'original_annotation_extract':extract,
 'scratch_exists':{{name:(assets/'tmp'/name).exists() for name in ['p0f_20261111_v1','p0q_20261111_v1']}},
 'versions':dict(sorted((d.metadata.get('Name','UNKNOWN'),d.version) for d in importlib.metadata.distributions())),
 'canonical_source_SHA256':{{name:hashlib.sha256((assets/name).read_bytes()).hexdigest() for name in ['src/rl/diffusion_policy.py','src/rl/diffusion_trajectory.py','src/mini_pilot/engine.py','src/training/generator_adapter.py','train_lighting_v3.py']}}}}
print(json.dumps(record,indent=2))
'''
        result=remote(code)
        if result.returncode:sys.stderr.buffer.write(result.stderr);raise SystemExit(result.returncode)
        data=json.loads(result.stdout);extract=data.pop('original_annotation_extract')
        (LOCAL/'original_annotation_extract.json').write_text(json.dumps(extract,indent=2),encoding='utf-8')
        (LOCAL/'final_resource_receipt.json').write_text(json.dumps(data,indent=2),encoding='utf-8')
        print(json.dumps({k:v for k,v in data.items() if k not in ('versions','canonical_source_SHA256')},indent=2))
    elif a.action=='quality_followup':
        files=list((ROOT/'src').rglob('*.py'))+[ROOT/'tests/__init__.py',ROOT/'tests/test_native_preprocessing.py',ROOT/'scripts/run_p0_quality_followup.py']
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in files:archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
        code=f'''import io,json,os,sys,tarfile,subprocess,shlex,hashlib
from pathlib import Path
from datetime import datetime,timezone
parent=Path({REMOTE!r});run=parent/'quality_followup'
if run.exists():raise ValueError('Followup already reserved; no duplicate or overwrite')
job=json.loads((parent/'submission.json').read_text())['job_id']
account=subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobID,State,ElapsedRaw'],text=True).strip().split('|')
if account[1]!='COMPLETED' or int(account[2])+300>1200:raise ValueError('Prior job incomplete or cumulative P0 cap exceeded')
run.mkdir();code=run/'code';code.mkdir();data=sys.stdin.buffer.read();(run/'source.tar.gz').write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive:
  target=(code/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(code.resolve()):raise ValueError('Unsafe archive')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
protocol={{'manifest_SHA256':hashlib.sha256((parent/'manifest.json').read_bytes()).hexdigest(),
 'instruction':'Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.',
 'steps':30,'max_side':512,'data':'same 4 dev sources; test closed',
 'configs':[{{'name':'native_aspect_30','text_guidance':7.5,'image_guidance':1.5}},{{'name':'native_aspect_preserve_30','text_guidance':5.0,'image_guidance':3.0}}],
 'GPU_cap_seconds':300,'prior_GPU_seconds':int(account[2]),'P1_launched':False}}
(run/'protocol.json').write_text(json.dumps(protocol,indent=2))
environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
result=subprocess.run([{PYTHON!r},'-m','unittest','tests.test_native_preprocessing','-v'],cwd=code,env=environment,text=True,capture_output=True,timeout=60)
(run/'cpu.log').write_text(result.stdout+result.stderr)
if result.returncode:print(result.stdout+result.stderr);raise SystemExit(result.returncode)
(run/'cpu_receipt.json').write_text(json.dumps({{'passed':True,'tests':3,'source_archive_SHA256':hashlib.sha256(data).hexdigest()}}))
command=[{PYTHON!r},str(code/'scripts/run_p0_quality_followup.py'),'--assets-root',{ASSETS!r},'--run',str(run)]
scratch=Path({ASSETS!r})/'tmp/p0q_20261111_v1'
cleanup="from pathlib import Path; import shutil; p=Path('"+str(scratch)+"').resolve(); assert p.parent==Path('{ASSETS}/tmp').resolve() and p.name=='p0q_20261111_v1'; shutil.rmtree(p)"
script='\\n'.join(['#!/bin/bash','#SBATCH --job-name=p0-frozen-aspect','#SBATCH --partition=epyc-gpu','#SBATCH --account=allusers','#SBATCH --gres=gpu:tesla_a100_80gb:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=32G','#SBATCH --time=00:05:00',f'#SBATCH --chdir={{code}}',f'#SBATCH --output={{run}}/slurm-%j.log',f'#SBATCH --error={{run}}/slurm-%j.log','set -euo pipefail','export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false','export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8','export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1',f'mkdir -p {{scratch}}',f'export TMPDIR={{scratch}}','trap '+shlex.quote('python3 -c '+shlex.quote(cleanup))+' EXIT','timeout --signal=TERM 270s '+shlex.join(command),''])
(run/'run.sbatch').write_text(script)
reply=subprocess.run(['sbatch','--parsable',str(run/'run.sbatch')],text=True,capture_output=True)
record={{'created_UTC':datetime.now(timezone.utc).isoformat(),'command':command,'reservation_GPU_seconds':300,
 'job_id':reply.stdout.strip().split(';')[0] if reply.returncode==0 else None,'returncode':reply.returncode,'stdout':reply.stdout,'stderr':reply.stderr}}
(run/'submission.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
'''
        call(code,buffer.getvalue(),timeout=100,log='quality_submission_remote.json')
    elif a.action=='quality_status':
        code=f'''import json,subprocess
from pathlib import Path
run=Path({REMOTE!r})/'quality_followup';job=json.loads((run/'submission.json').read_text())['job_id']
print(subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES,Start,End'],text=True))
if (run/'status.json').exists():print((run/'status.json').read_text())
logs=list(run.glob('slurm-*.log'))
if logs:print('\\n'.join(logs[0].read_text(errors='replace').splitlines()[-2:]))
'''
        call(code,log='quality_latest_status.log')
    elif a.action=='inventory':
        code=f'''import importlib.metadata,json,subprocess,shutil
from pathlib import Path
root=Path({ASSETS!r})
data={{'versions':{{n:importlib.metadata.version(n) for n in ['torch','diffusers','peft','accelerate','transformers','Pillow','scikit-image']}},
'ledgers':{{str(p.relative_to(root)):p.read_text() for p in (root/'.aris/compute').glob('utc*.md')}},
'data_assets':[{{'path':str(p.relative_to(root)),'bytes':p.stat().st_size}} for p in (root/'data/raw/vqav2').glob('*.zip')],
'lpips_receipt':json.loads((root/'results/execution_remediation_20261102_v1/lpips_dependency_verification.json').read_text()),
'squeue':subprocess.check_output(['squeue','-u','ydk297','-o','%i %j %T %M %l %D %R'],text=True),
'disk':shutil.disk_usage(root)._asdict(),
'qwen_cache_present':(Path('/storage/home/ydk297/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct/snapshots/89644892e4d85e24eaac8bacfd4f463576704203')).is_dir()}}
print(json.dumps(data,indent=2))
'''
        result=remote(code)
        (LOCAL/'remote_inventory.json').write_bytes(result.stdout)
        if result.returncode: sys.stderr.buffer.write(result.stderr);raise SystemExit(result.returncode)
        data=json.loads(result.stdout);print(json.dumps({k:v for k,v in data.items() if k!='ledgers'},indent=2))
    elif a.action=='stage':
        files=list((ROOT/'src').rglob('*.py'))+list((ROOT/'tests').rglob('*.py'))
        files += [ROOT/'scripts'/n for n in ['run_p0_foundations.py','prepare_p0_manifest.py','test_p0_foundations.py',
                  'run_bf16_adjoints_20261102.py','numerical_boundaries_20261102.py','audit_benign_crop_posthoc.py','test_benign_ddpo_audit.py','run_benign_ddpo_minimal.py']]
        files += [ROOT/'configs/p0_foundations_20261111.json',ROOT/'configs/benign_ddpo_minimal.json',ROOT/'reports/phase2_model_metadata.json']
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in files: archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
        code=f'''import io,sys,tarfile,hashlib,json
from pathlib import Path
run=Path({REMOTE!r});run.mkdir(parents=True,exist_ok=True)
if (run/'submission.json').exists():raise ValueError('Source frozen after submission')
code=run/'code';code.mkdir(exist_ok=True)
data=sys.stdin.buffer.read();key=hashlib.sha256(data).hexdigest()
archives=run/'source_archives';archives.mkdir(exist_ok=True);(archives/(key+'.tar.gz')).write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive:
  target=(code/member.name).resolve()
  if not member.isfile() or not target.is_relative_to(code.resolve()):raise ValueError('Unsafe archive')
  target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
(run/'source_upload.json').write_text(json.dumps({{'archive_SHA256':key,'source_root':str(code)}}))
print(json.dumps({{'archive_SHA256':key,'source_root':str(code)}}))
'''
        call(code,buffer.getvalue(),log='stage.log')
    elif a.action in ('prepare','cpu'):
        script='prepare_p0_manifest.py' if a.action=='prepare' else 'test_p0_foundations.py'
        code=f'''import os,subprocess,sys
from pathlib import Path
run=Path({REMOTE!r});environment=os.environ.copy();environment.update(CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
command=[{PYTHON!r},str(run/'code/scripts/{script}'),'--assets-root',{ASSETS!r},'--run',str(run)]
with (run/'{a.action}.log').open('w') as log:
 result=subprocess.run(command,cwd=run/'code',env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=240)
print((run/'{a.action}.log').read_text());sys.exit(result.returncode)
'''
        call(code,timeout=280,log=a.action+'_remote.log')
    elif a.action=='submit':
        code=f'''import json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone
run=Path({REMOTE!r});code=run/'code'
if (run/'submission.json').exists():raise ValueError('No duplicate allocation')
if not json.loads((run/'cpu_tests.json').read_text())['passed']:raise ValueError('CPU gate failed')
if not (run/'manifest.json').is_file():raise ValueError('No frozen manifest')
scratch=Path({ASSETS!r})/'tmp/p0f_20261111_v1'
command=[{PYTHON!r},str(code/'scripts/run_p0_foundations.py'),'--assets-root',{ASSETS!r},'--run',str(run)]
cleanup="from pathlib import Path; import shutil; p=Path('"+str(scratch)+"').resolve(); assert p.parent==Path('{ASSETS}/tmp').resolve() and p.name=='p0f_20261111_v1'; shutil.rmtree(p)"
script='\\n'.join(['#!/bin/bash','#SBATCH --job-name=p0-clean-foundations','#SBATCH --partition=epyc-gpu','#SBATCH --account=allusers','#SBATCH --gres=gpu:tesla_a100_80gb:1','#SBATCH --cpus-per-task=4','#SBATCH --mem=48G','#SBATCH --time=00:20:00',f'#SBATCH --chdir={{code}}',f'#SBATCH --output={{run}}/slurm-%j.log',f'#SBATCH --error={{run}}/slurm-%j.log','set -euo pipefail','export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false','export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8','export HF_HOME=/storage/home/ydk297/.cache/huggingface HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1',f'mkdir -p {{scratch}}',f'export TMPDIR={{scratch}}','trap '+shlex.quote('python3 -c '+shlex.quote(cleanup))+' EXIT','timeout --signal=TERM 1080s '+shlex.join(command),''])
(run/'run.sbatch').write_text(script)
result=subprocess.run(['sbatch','--parsable',str(run/'run.sbatch')],text=True,capture_output=True)
record={{'created_UTC':datetime.now(timezone.utc).isoformat(),'reservation_GPU_seconds':1200,'scope':'P0 Frozen inference + clean VLM pipeline engineering only; no P1 RL training','command':command,'returncode':result.returncode,'stdout':result.stdout,'stderr':result.stderr,'job_id':result.stdout.strip().split(';')[0] if result.returncode==0 else None}}
(run/'submission.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))
'''
        call(code,log='submission_remote.json')
    elif a.action=='status':
        code=f'''import json,subprocess
from pathlib import Path
run=Path({REMOTE!r});job=json.loads((run/'submission.json').read_text())['job_id']
print(subprocess.check_output(['sacct','-X','-nP','-j',job,'--format=JobID,State,ElapsedRaw,AllocTRES,Start,End'],text=True))
if (run/'status.json').exists():
 state=json.loads((run/'status.json').read_text());print(json.dumps({{k:v for k,v in state.items() if k!='source_SHA256'}},indent=2))
logs=list(run.glob('slurm-*.log'))
if logs:print('\\n'.join(logs[0].read_text(errors='replace').splitlines()[-2:]))
'''
        call(code,log='latest_remote_status.log')
    elif a.action=='fetch':
        code=f'''import io,sys,tarfile
from pathlib import Path
run=Path({REMOTE!r});buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
 for path in run.rglob('*'):
  if path.is_file() and 'code' not in path.relative_to(run).parts:archive.add(path,arcname=path.relative_to(run).as_posix(),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''
        result=remote(code,timeout=240)
        if result.returncode:sys.stderr.buffer.write(result.stderr);raise SystemExit(result.returncode)
        target=LOCAL/'remote';target.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(result.stdout),mode='r:gz') as archive:
            for member in archive:
                path=(target/member.name).resolve()
                if not member.isfile() or not path.is_relative_to(target.resolve()):raise ValueError('Unsafe result archive')
                data=archive.extractfile(member).read()
                if path.is_file() and path.suffix in ('.pt','.png','.jpg') and path.read_bytes()!=data:raise ValueError('Changed final evidence')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        print(json.dumps({'files':sum(p.is_file() for p in target.rglob('*')),'bytes':sum(p.stat().st_size for p in target.rglob('*') if p.is_file())}))


if __name__=='__main__':main()
