"""Separate stage one-A100-hour accounting, pilot gate, exclusive phases."""
import argparse,json,subprocess
from pathlib import Path
from submit_phase2 import accounting
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--phase',choices=['pilot','main'],required=True);p.add_argument('--revision',default='');a=p.parse_args();run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());ledger=run/'job_ledger.jsonl';phase=a.phase+('_'+a.revision if a.revision else '')
    old=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else [];charged,states=accounting(old)
    if any(x['phase']==phase for x in old):raise RuntimeError('Phase already submitted; use explicit versioned repair')
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'};observed={r[0]:r[1].split()[0] for r in states}
    if any(observed.get(r['job_id']) not in terminal for r in old):raise RuntimeError('Prior allocation not terminal')
    accepted=json.loads((run/'accepted_clean_phases.json').read_text()) if (run/'accepted_clean_phases.json').exists() else {}
    if a.phase=='main' and json.loads((run/accepted.get('pilot','clean_pilot')/'checks.json').read_text())['status']!='PASS':raise RuntimeError('Pilot failed')
    seconds=cfg[f'{a.phase}_gpu_wall_seconds']
    if charged+seconds>cfg['gpu_budget_seconds']:raise RuntimeError('Stage budget exceeded')
    cmd=['sbatch','--parsable','--time',f'00:{seconds//60:02d}:00','--output',str(run/'logs'/f'{phase}-%j.out'),'--error',str(run/'logs'/f'{phase}-%j.err'),str(ROOT/'slurm/clean_target_baseline.sbatch'),a.run_id,a.phase,a.revision]
    job=subprocess.check_output(cmd,text=True).strip().split(';')[0];r={'job_id':job,'phase':phase,'reserved_seconds':seconds,'charged_before':charged,'command':cmd}
    with ledger.open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r))
if __name__=='__main__':main()
