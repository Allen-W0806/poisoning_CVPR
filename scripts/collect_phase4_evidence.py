"""Terminal-only CPU audit and reproducible Phase4 evidence archive."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import torch
import _bootstrap
from _phase4 import phase4_run
from submit_phase2 import accounting
from src.rl.optimizer_validation import state_equal
from src.utils.logging import write_json

def load(path):return json.loads(Path(path).read_text())
def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resource-audit',action='store_true');parser.add_argument('--archive')
    args=parser.parse_args();root=_bootstrap.ROOT
    assert not (root/'reports/phase4_submission_uncertain.json').exists(),'Resolve unknown submission before final accounting'
    ledger=[json.loads(x) for x in (root/'reports/phase4_jobs.jsonl').read_text().splitlines()]
    charged,rows=accounting(ledger)
    terminal={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    states={r[0]:r[1].split()[0] for r in rows}
    assert all(states.get(j['job_id']) in terminal for j in ledger),'Cannot finalize active or unconfirmed jobs'
    assert charged<=14400
    assert sum(int(r[2]) for r in rows if any(j['job_id']==r[0] and j['task'].startswith('diagnostic') for j in ledger))<=2700
    if args.resource_audit:
        pointer=root/'reports/phase4_latest_resource_audit.json'
        if not pointer.exists():
            runs=[{'id':p.name,'slurm_job_id':load(p/'manifest.json').get('slurm_job_id')} for p in sorted((root/'experiments').glob('*phase4_*'))]
            with phase4_run('PH4_RESOURCE_AUDIT',{'seed':42,'cpu_only':True,'ledger':ledger},splits=()) as run:
                run.metrics({'slurm_allocation_rows':rows,'total_additional_gpu_seconds':charged,'run_job_associations':runs,
                    'scope':'Cost charged once per Slurm allocation. Child runs/revisions share their parent job; CPU audits create no additional GPU allocation.'})
                run.finish(summary='Immutable actual allocation accounting for every Phase4 GPU job, including failed attempts.')
                write_json(pointer,{'id':run.id,'path':str(run.path)})
        print(json.dumps({'gpu_seconds':charged,'resource_audit':load(pointer)}));return
    evidence=[]
    for path in sorted((root/'experiments').glob('*phase4_*')):
        status=load(path/'status.json')['status'];assert status not in {'RUNNING','SUBMITTED','CREATED'},path.name
        metrics=load(path/'metrics.json');checkpoints=[]
        if metrics.get('arm') in {'rl','rsi'}:
            for update in metrics.get('updates',[]):
                state=torch.load(path/f'checkpoints/step_{update["step"]:04d}/training_state.pt',map_location='cpu',weights_only=False)
                finite=all(torch.isfinite(x).all().item() for x in state['adapter'].values())
                optfinite=all(torch.isfinite(x).all().item() for v in state['optimizer']['state'].values() for x in v.values() if isinstance(x,torch.Tensor))
                assert finite and optfinite and state['step']==update['step'],path.name
                checkpoints.append({'step':state['step'],'adapter_finite':finite,'optimizer_finite':optfinite,'version':state['version'],
                    'learning_rates':[g['lr'] for g in state['optimizer']['param_groups']],'consumed_trajectories':len(state['consumed'])})
            if status=='PASS':assert len(metrics['updates'])==8 and len(metrics['evaluations'])==4
            assert load(path/'frozen_before.json')==load(path/'frozen_after.json') if (path/'frozen_after.json').exists() else status!='PASS'
        files=[{'path':str(p.relative_to(path)),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(path.rglob('*')) if p.is_file()]
        evidence.append({'id':path.name,'status':status,'manifest':load(path/'manifest.json'),'checkpoint_cpu_audits':checkpoints,'files':files})
    registry=[json.loads(x)['experiment_id'] for x in (root/'experiments/registry.jsonl').read_text().splitlines()]
    assert len(registry)==len(set(registry)),'Duplicate registry IDs'
    changed=subprocess.check_output(['git','diff','--name-only','534dd80a80d6f86a0b9d131e5b180e50576fccb3','HEAD'],text=True).splitlines()
    source={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in changed if (root/name).is_file()}
    report={'gpu_seconds':charged,'slurm':rows,'phase4_runs':evidence,'registry_unique':True,'registry_count':len(registry),
        'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'normalized_lf_source_sha256':source,
        'retained_experiment_bytes':sum(f['bytes'] for r in evidence for f in r['files']),
        'heldout_marker':load(root/'reports/phase2_test_opened.json'),
        'heldout_marker_sha256':digest(root/'reports/phase2_test_opened.json'),'locked_protocol_sha256':digest(root/'configs/phase2_vqa_protocol.json')}
    assert report['locked_protocol_sha256']=='8874954829b2f2584f82e7eec6ad73f6158a0648aa095191ea584eaeb450d927'
    write_json(root/'reports/phase4_evidence.json',report)
    if args.archive:
        target=Path(args.archive).resolve();assert target.parent==root/'tmp/phase4_20261027_1648'
        with tarfile.open(target,'w:gz',compresslevel=2) as archive:
            for r in evidence:archive.add(root/'experiments'/r['id'],arcname='experiments/'+r['id'])
            archive.add(root/'experiments/registry.jsonl',arcname='experiments/registry.jsonl')
            paths=set((root/'reports').glob('PHASE4*.md'))|set((root/'reports').glob('phase4*'))
            paths|={root/'reports/MONITORING_STATUS.json',root/'reports/MONITORING_LOG.md'}
            for p in sorted(paths):
                if p.is_file() and p.suffix!='.lock':archive.add(p,arcname='reports/'+p.name)
            monitor=load(root/'reports/MONITORING_STATUS.json')
            monitor_paths=[root/'reports/monitoring/state.json',root/'reports/monitoring/scheduler.json',
                Path(monitor['latest_json']),Path(monitor['latest_report'])]
            for p in monitor_paths:
                if p.is_file():archive.add(p,arcname=str(p.relative_to(root)))
        write_json(root/'reports/phase4_archive_receipt.json',{'path':str(target),'bytes':target.stat().st_size,'sha256':digest(target)})
    print(json.dumps({'runs':len(evidence),'gpu_seconds':charged,'retained_experiment_bytes':report['retained_experiment_bytes']}))

if __name__=='__main__':main()
