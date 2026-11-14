"""Read-only source/checkpoint inventory; no model or experiment mutation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
FILES=[
 'src/rl/trainer.py','src/rl/policy.py','src/rl/cached_editor_trainer.py',
 'src/rl/diffusion_rl_trainer.py','src/rl/diffusion_rl_trainer_v2.py','src/rl/diffusion_policy.py','src/rl/diffusion_policy_v2.py',
 'src/rl/diffusion_trajectory.py','src/rl/reward.py','src/rl/reward_v2.py','src/rl/reward_normalization.py','src/rl/phase5_backend.py',
 'src/training/generator_trainer.py','src/training/generator_adapter.py','src/models/qwen_loader.py','src/models/pretrained_editor_loader.py',
 'src/editing/pretrained_editor.py','src/editing/editor_config.py','src/evaluation/vqa_official_compatible.py','src/evaluation/_vendor/official_vqa.py',
 'src/data/dataset.py','src/rsi/controller.py','src/rsi/memory.py','src/rsi/harness/harness_executor.py','src/rsi/harness/harness_spec.py',
 'src/rsi/harness/version_manager.py','src/progression/real_harness.py','configs/rl.yaml','configs/phase3_generator_rl.yaml',
 'configs/generator_lora_smoke.yaml','configs/phase6.json','configs/gradient_feedback_validation_v1.json','configs/gradient_feedback_validation_v2.json',
 'scripts/validate_policy_gradients.py','scripts/validate_benign_models.py','scripts/validate_ddpo_wiring.py','scripts/validate_group_reference.py',
 'scripts/validate_reviewed_edits.py','scripts/report_gradient_feedback.py','scripts/collect_gradient_feedback_inventory.py']


def main():
    tag=sys.argv[1];out=ROOT/'results/gradient_feedback_summary_20261028';out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name in FILES:
        path=ROOT/name
        if not path.exists():continue
        data=path.read_bytes();rows.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
            'normalized_lf_sha256':hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()})
        snapshot=out/'source_snapshot'/hashlib.sha256(data).hexdigest()/name
        rows[-1]['snapshot_path']=str(snapshot.relative_to(ROOT))
        snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():snapshot.write_bytes(data)
    def command(args):return subprocess.run(args,cwd=ROOT,capture_output=True,text=True).stdout.strip()
    checkpoints=sorted((ROOT/'experiments').rglob('training_state.pt'))
    packages={}
    for package in ['torch','numpy','scipy','matplotlib','transformers','diffusers','peft','Pillow']:
        try:packages[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:packages[package]=None
    result={'root':str(ROOT),'platform':platform.platform(),'python':sys.version,'executable':sys.executable,
        'git_commit':command(['git','rev-parse','HEAD']),'source_inventory':rows,'packages':packages,
        'existing_checkpoints':{'count':len(checkpoints),'bytes':sum(p.stat().st_size for p in checkpoints),
            'sample_paths':[str(p.relative_to(ROOT)) for p in checkpoints[:20]],'content_status':'inventoried only; not reloaded by this metadata collector'},
        'existing_experiment_count':len(list((ROOT/'experiments').glob('EXP_*'))),'disk':dict(zip(['total','used','free'],shutil.disk_usage(ROOT)))}
    (out/f'inventory_{tag}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({'checkpoints':len(checkpoints),'sources':len(rows),'packages':packages}))


if __name__=='__main__':main()
