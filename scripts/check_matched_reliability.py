"""CPU reconciliation of current source, safeguards, and fresh regression receipts."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='matched_feedback_20261028_v2');a=p.parse_args()
    run=ROOT/'results'/a.run_id;out=run/'audit';out.mkdir(exist_ok=True);checks=[]
    old=json.loads((ROOT/'results/gradient_feedback_summary_20261028/inventory_cluster.json').read_text())
    oldhash={r['path']:r['normalized_lf_sha256'] for r in old['source_inventory']};files=[]
    for name in oldhash:
        if not name.startswith('src/'):continue
        path=ROOT/name;actual=hashlib.sha256(path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
        files.append({'path':name,'prior_hash':oldhash[name],'current_hash':actual,'unchanged':actual==oldhash[name]})
        dest=out/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(path,dest)
    checks.append({'test':'production_source_matches_prior_audit','status':'PASS' if all(r['unchanged'] for r in files) else 'FAIL'})
    from src.rl.phase5_backend import require_training_reward
    for receipt in [{},{'training_eligible':False},{'training_eligible':True}]:
        rejected=False
        try:require_training_reward(receipt)
        except RuntimeError:rejected=True
        checks.append({'test':'training_reward_guard_'+str(receipt),'status':'PASS' if rejected else 'FAIL'})
    gradient=json.loads((ROOT/'results'/f'{a.run_id}_gradient/synthetic_checks.json').read_text())
    tiny_dir=ROOT/'results'/f'{a.run_id}_tiny_checkpoint';tiny=json.loads((tiny_dir/'wiring_checks.json').read_text())
    checks.append({'test':'fresh_exact_MC_gradient_regression','status':'PASS' if gradient['passed'] else 'FAIL'})
    checks.append({'test':'fresh_tiny_checkpoint_fixed_replay','status':'PASS' if tiny['passed'] else 'FAIL'})
    checkpoint=torch.load(tiny_dir/'tiny_checkpoint/training_state.pt',map_location='cpu',weights_only=False)
    checks.append({'test':'tiny_checkpoint_optimizer_scheduler_RNG_present','status':'PASS' if {'adapter','optimizer','scheduler','rng_torch','rng_cuda','rng_numpy','rng_python','step','version','consumed'}<=set(checkpoint) else 'FAIL'})
    versions={}
    for package in ['torch','diffusers','transformers','peft','numpy','Pillow']:
        versions[package]=importlib.metadata.version(package)
    result={'status':'PASS' if all(r['status']=='PASS' for r in checks) else 'FAIL','time_unix':time.time(),'checks':checks,'source_files':files,
        'versions':versions,'torch_runtime':torch.__version__,'gradient_diagnostics':gradient,'tiny_wiring':tiny,
        'checkpoint':{'path':str((tiny_dir/'tiny_checkpoint/training_state.pt').relative_to(ROOT)),'sha256':hashlib.sha256((tiny_dir/'tiny_checkpoint/training_state.pt').read_bytes()).hexdigest(),
            'step':checkpoint['step'],'version':checkpoint['version'],'adapter_tensors':len(checkpoint['adapter'])},
        'algorithms':{'categorical':'REINFORCE','production_diffusion':'single-pass on-policy DDPO score function, fixed coordinate/time scaling','production_GRPO':'NOT VALIDATED: not implemented',
            'isolated_group_reference':'tested synthetic reference only; different group-normalized surrogate'},
        'pretrained_model_training_gate':'NOT VALIDATED','BF16_full_pretrained_restart':'NOT VALIDATED; historical exact continuation failures retained',
        'no_reward_guard_bypass':True,'pretrained_optimizer_steps_this_stage':0,'scope':'Tiny random UNet and synthetic checks are not full pretrained diffusion RL reliability evidence.'}
    (out/'rl_reliability.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps({k:v for k,v in result.items() if k not in ['gradient_diagnostics','tiny_wiring','source_files']},indent=2))


if __name__=='__main__':main()
