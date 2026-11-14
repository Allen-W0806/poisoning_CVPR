"""Read-only inventory. Run on the existing login node; never initializes CUDA."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

root = Path('/storage/home/ydk297/projects/poisoning')
def command(args):
    result = subprocess.run(args, text=True, capture_output=True, timeout=20)
    return {"args": args, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
metadata = json.loads((root / 'reports/phase2_model_metadata.json').read_text())['timbrooks/instruct-pix2pix']
files = {name: (Path(metadata['path']) / name).is_file() for name in metadata['files'] if not name.endswith('README.md')}
source = root / 'data/processed/vqav2_content_v2/train.json'
records = json.loads(source.read_text())
entrypoints = ['src/rl/diffusion_policy.py', 'src/rl/diffusion_trajectory.py',
               'src/mini_pilot/engine.py', 'src/training/generator_adapter.py', 'train_lighting_v3.py']
result = {"hostname": platform.node(), "python": sys.version, "executable": sys.executable,
          "packages": {name: importlib.metadata.version(name) for name in
                       ['torch','diffusers','peft','accelerate','transformers','numpy','Pillow','scikit-image']},
          "disk": shutil.disk_usage(root)._asdict(), "model": metadata, "model_files_present": files,
          "development_pool_count": len(records), "development_sample_image_present": (root/records[0]['image_path']).is_file(),
          "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
          "historical_test_opened": False,
          "owning_source_sha256": {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in entrypoints},
          "compute_ledgers": {str(p.relative_to(root)): p.read_text() for p in (root/'.aris/compute').glob('*.md')},
          "git": command(['git','-C',str(root),'rev-parse','HEAD']),
          "squeue": command(['squeue','-u','ydk297','-o','%i %j %T %M %l %D %R']),
          "nodes": command(['sinfo','-N','-h','-p','epyc-gpu','-o','%N %t %G %C'])}
print(json.dumps(result, indent=2))
