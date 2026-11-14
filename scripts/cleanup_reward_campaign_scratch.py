"""Remove only verified run scratch after every recorded allocation is terminal.

Never remove data, model environments, weights, checkpoints or result evidence.
The default is a read-only plan. --execute rechecks all job/path conditions.
"""
import argparse
from datetime import datetime,timezone
import json
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import write_json
from scripts.submit_reward_campaign import accounting

TERMINAL={'COMPLETED','FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--campaign',required=True,type=Path)
    p.add_argument('--execute',action='store_true');args=p.parse_args();c=args.campaign.resolve()
    if sys.platform=='win32':raise RuntimeError('Use verified native PowerShell for Windows scratch removal')
    if not c.is_relative_to(ROOT/'outputs') or c==ROOT/'outputs':raise ValueError('Expected a run-specific canonical campaign')
    acc=accounting(c)
    if not acc['records'] or any(not r['settled'] for r in acc['records']):
        raise RuntimeError('GPU allocations still active or accounting ambiguous')
    paths=set()
    for receipt in list((c/'submissions').glob('*/submission.json'))+list((c/'cpu_submissions').glob('*/submission.json')):
        row=json.loads(receipt.read_text())
        if row.get('GPU_count')==0 and row.get('job_id'):
            job=str(row['job_id'])
            lines=subprocess.run(['sacct','-X','-nP','-j',job,'--format=JobID,State'],check=True,text=True,capture_output=True).stdout.splitlines()
            states=[l.split('|')[1].split()[0] for l in lines if l.split('|')[0]==job]
            if len(states)!=1 or states[0] not in TERMINAL:raise RuntimeError('CPU allocation still active/ambiguous')
        if row.get('scratch'):
            path=Path(row['scratch'])
            if path.name.startswith('rd_'):
                key=hashlib.sha256((c.name+':'+row['stage']).encode()).hexdigest()[:12] if row.get('stage') else None
                expected='rd_'+key if key else 'rd_scq_'+hashlib.sha256(c.name.encode()).hexdigest()[:10]
                if path.name!=expected:raise ValueError('Short scratch hash is not bound to this campaign/stage')
            paths.add(path)
    # These exact shared staging labels belong to this campaign, never generic
    # tmp or a different task. Missing directories are harmless.
    for suffix in ['external_build','delivery','cpu_tests','diagnostics','internal',
                   'shadowcast_smoke','shadowcast_poison','shadowcast_poison_pathfix','shadowcast_clean']:
        paths.add(ROOT/'tmp'/(c.name+'_'+suffix))
    tmp=(ROOT/'tmp').resolve()
    if tmp!=ROOT/'tmp':raise ValueError('Unexpected symlinked tmp root')
    verified=[]
    for path in sorted(paths):
        target=path.resolve()
        if not target.is_relative_to(tmp) or target==tmp or path.is_symlink():
            raise ValueError('Scratch target is not a direct verified child of project tmp')
        if target.parent!=tmp or not (target.name.startswith(c.name+'_') or target.name.startswith('rd_')):
            raise ValueError('Unexpected scratch label')
        if target.exists():
            if not target.is_dir():raise ValueError('Expected scratch directory')
            files=[f for f in target.rglob('*') if f.is_file() and not f.is_symlink()]
            verified.append({'path':str(target),'files':len(files),'bytes':sum(f.stat().st_size for f in files)})
    result={'UTC':datetime.now(timezone.utc).isoformat(),'mode':'EXECUTED' if args.execute else 'DRY_RUN',
        'all_recorded_allocations_terminal':True,'verified_targets':verified,
        'deleted_bytes':sum(r['bytes'] for r in verified) if args.execute else 0,
        'retained':'all models/environments/data/checkpoints/raw logs and result evidence outside these scratch directories'}
    if args.execute:
        for row in verified:
            literal=Path(row['path']);target=literal.resolve()
            # Repeat the absolute containment check immediately before deletion.
            if literal.is_symlink() or target!=literal or target.parent!=tmp or not target.is_relative_to(tmp):
                raise ValueError('Scratch path changed before removal')
            shutil.rmtree(target)
        write_json(c/'SCRATCH_CLEANUP_RECEIPT.json',result,exclusive=True)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
