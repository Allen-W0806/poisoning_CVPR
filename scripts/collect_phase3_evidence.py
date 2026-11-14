"""Read-only scientific/accounting inventory; rolling output in reports/."""
import json
import subprocess
import hashlib
import _bootstrap
from src.utils.logging import write_json

def capture(args):
    r=subprocess.run(args,capture_output=True,text=True,timeout=60)
    return {'command':args,'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}

if __name__=='__main__':
    root=_bootstrap.ROOT;ledger=root/'reports/phase3_jobs.jsonl'
    jobs=[json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
    runs=[]
    for path in sorted((root/'experiments').glob('EXP_*phase3_*')):
        read=lambda name:json.loads((path/name).read_text()) if (path/name).is_file() else None
        checkpoint_audit={}
        if 'generator_rl_' in path.name:
            import torch
            from src.rl.optimizer_validation import tensor_changes
            initial=path/'checkpoints/step_0000/training_state.pt'
            steps=sorted((path/'checkpoints').glob('step_*/training_state.pt'))
            if initial.is_file() and len(steps)>1:
                before=torch.load(initial,map_location='cpu',weights_only=False)['adapter']
                after=torch.load(steps[-1],map_location='cpu',weights_only=False)['adapter']
                checkpoint_audit={'scope':'read-only CPU audit after terminal/update evidence; original metrics unchanged',
                    'adapter_changes':tensor_changes(before,after),'trainable_parameters':sum(v.numel() for v in after.values()),
                    'frozen_tensor_count':len(read('frozen_before.json') or {}),
                    'frozen_hashes_equal':read('frozen_before.json')==read('frozen_after.json') if (path/'frozen_after.json').exists() else None}
        runs.append({'id':path.name,'manifest':read('manifest.json'),'status':read('status.json'),'metrics':read('metrics.json'),
            'config_file_sha256':hashlib.sha256((path/'config.yaml').read_bytes()).hexdigest(),
            'checkpoint_audit':checkpoint_audit,
            'restart_diagnostic':read('restart_diagnostic.json'),'restart_inputs':read('restart_inputs.json'),
            'bytes':sum(p.stat().st_size for p in path.rglob('*') if p.is_file())})
    result={'jobs':jobs,'runs':runs,'accounting':capture(['sacct','-X','-j',','.join(j['job_id'] for j in jobs),'-nP',
        '--format=JobIDRaw,State,ElapsedRaw,ExitCode,AllocTRES']),
        'queue':capture(['squeue','-u','ydk297']), 'disk':capture(['df','-h',str(root)]),
        'heldout_open_marker':json.loads((root/'reports/phase2_test_opened.json').read_text()),
        'monitor':capture([str(root/'.venv/bin/python'),'scripts/manage_monitor.py','status'])}
    write_json(root/'reports/phase3_evidence.json',result)
    print(json.dumps({'runs':len(runs),'retained_bytes':sum(r['bytes'] for r in runs),'accounting':result['accounting']['stdout']},indent=2))
