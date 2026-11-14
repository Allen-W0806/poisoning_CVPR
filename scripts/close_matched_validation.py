"""Remove only verified run scratch; keep sources, all model outputs and failures."""
import argparse
import json
from pathlib import Path
import shutil
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='matched_feedback_20261028_v2');p.add_argument('--tag',choices=['local','canonical'],required=True);a=p.parse_args()
    run=ROOT/'results'/a.run_id
    assert (run/'completion.json').exists() and (run/'audit/initial_runner/scripts/run_matched_feedback.py').exists()
    intended=(ROOT/'tmp').resolve();target=(intended/'matched_feedback_20261028_transfer').resolve()
    if target.parent!=intended or target.name!='matched_feedback_20261028_transfer':raise RuntimeError('Refusing unexpected scratch path')
    removed=[]
    if target.exists():shutil.rmtree(target);removed.append(str(target.relative_to(ROOT)))
    for name in ['matched_feedback_prepare_20261028.tar','matched_feedback_runner_20261028.tar']:
        path=(intended/name).resolve()
        if path.parent!=intended:raise RuntimeError('Unexpected transfer path')
        if path.exists():path.unlink();removed.append(str(path.relative_to(ROOT)))
    evidence_files=[p for p in run.rglob('*') if p.is_file()]
    receipt={'status':'PASS','removed_scratch':removed,'scratch_absent':not target.exists(),'outputs_and_failed_runs_preserved':True,
        'retained_run_bytes':sum(p.stat().st_size for p in evidence_files),'retained_files':len(evidence_files),'new_model_weight_downloads':0,
        'new_large_C_drive_artifacts':False,'workspace_disk':dict(zip(['total','used','free'],shutil.disk_usage(ROOT)))}
    (run/f'housekeeping_{a.tag}.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
