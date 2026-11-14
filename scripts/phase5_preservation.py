"""Prove pre-Phase5 evidence remains byte-identical; no scientific execution."""
import argparse
import hashlib
import json
from pathlib import Path
import _bootstrap

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def selected(root):
    paths=[]
    for folder in (root/'experiments').iterdir():
        if folder.is_dir() and 'phase5' not in folder.name:
            paths.extend(p for p in folder.rglob('*') if p.is_file())
    for p in (root/'reports').glob('*'):
        if p.is_file() and ('PHASE' in p.name or 'phase' in p.name or 'AUDIT' in p.name) and 'PHASE5' not in p.name and 'phase5' not in p.name:paths.append(p)
    for base in [root/'data/manifests/cluster/vqav2_seed42_v1',root/'data/processed/vqav2_seed42_v1']:
        if base.exists():paths.extend(p for p in base.rglob('*') if p.is_file())
    return {p.relative_to(root).as_posix():digest(p) for p in sorted(set(paths))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',action='store_true');a=p.parse_args();root=_bootstrap.ROOT
    out=root/'reports/phase5_preservation_baseline.json'
    if a.record:
        value=selected(root)
        with out.open('x') as f:json.dump(value,f,indent=2)
        print(json.dumps({'snapshot_files':len(value),'path':str(out)}))
    else:
        old=json.loads(out.read_text());bad=[name for name,h in old.items() if not (root/name).is_file() or digest(root/name)!=h]
        if bad:raise AssertionError(bad)
        reference=json.loads((root/'reports/phase4_evidence.json').read_text())
        for name,h in reference['normalized_lf_source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h,name
        result={'historical_files_unchanged':len(old),'protected_phase4_sources_unchanged':32,'historical_outputs_overwritten':False}
        out=root/'reports/phase5_preservation_verified.json';out.write_text(json.dumps(result,indent=2))
        print(json.dumps(result))

if __name__=='__main__':main()
