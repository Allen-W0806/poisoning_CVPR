"""Separate Phase6 baseline covering every pre-existing experiment, including Phase5."""
import argparse
import json
from pathlib import Path
import _bootstrap
from src.data.dataset import sha256
from src.utils.logging import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args();root=_bootstrap.ROOT
    path=root/'reports/phase6_preservation_baseline.json'
    if a.record:
        files=[p for d in (root/'experiments').glob('EXP_*') if 'phase6' not in d.name for p in d.rglob('*') if p.is_file()]
        files += [p for p in (root/'reports').glob('PHASE[1-5]*.md')]
        files += [p for p in (root/'data/processed/vqav2_content_v2').glob('*.json')]
        with path.open('x') as f:json.dump({p.relative_to(root).as_posix():sha256(p) for p in files},f,indent=2)
        print(json.dumps({'baseline_files':len(files)}))
    else:
        old=json.loads(path.read_text());bad=[p for p,h in old.items() if not (root/p).is_file() or sha256(root/p)!=h]
        assert not bad,bad
        value={'prephase6_files_unchanged':len(old),'failures':[]};write_json(root/'reports/phase6_preservation_verified.json',value);print(json.dumps(value))
if __name__=='__main__':main()
