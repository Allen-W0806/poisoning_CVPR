"""Reuse existing Slurm accounting guard for a separate 1-A100-hour stage."""
import argparse
import json
from pathlib import Path
import subprocess
from submit_phase2 import accounting
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['pilot','main'],required=True);a=p.parse_args()
    run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());registry=run/'job_ledger.jsonl'
    previous=[json.loads(x) for x in registry.read_text().splitlines()] if registry.exists() else []
    charged,states=accounting(previous)
    if any(x['phase']==a.phase for x in previous):raise RuntimeError('Duplicate phase submission forbidden; explicit versioned repair required')
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    observed={r[0]:r[1].split()[0] for r in states}
    if any(observed.get(p['job_id']) not in terminal for p in previous):raise RuntimeError('Prior stage job active/unconfirmed')
    if a.phase=='main' and json.loads((run/'pilot/checks.json').read_text())['status']!='PASS':raise RuntimeError('Pilot gate failed')
    seconds=cfg[f'{a.phase}_gpu_wall_seconds']
    if charged+seconds>cfg['gpu_budget_seconds']:raise RuntimeError('One A100-hour budget would be exceeded')
    logs=run/'logs';logs.mkdir(exist_ok=True)
    cmd=['sbatch','--parsable','--time',f'00:{seconds//60:02d}:00','--output',str(logs/f'{a.phase}-%j.out'),
        '--error',str(logs/f'{a.phase}-%j.err'),str(ROOT/'slurm/matched_feedback_validation.sbatch'),a.run_id,a.phase]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0]
    record={'job_id':job,'phase':a.phase,'reserved_seconds':seconds,'charged_before':charged,'command':cmd}
    with registry.open('a') as f:f.write(json.dumps(record)+'\n')
    print(json.dumps(record))


if __name__=='__main__':main()
