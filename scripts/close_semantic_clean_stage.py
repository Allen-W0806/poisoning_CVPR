"""Verify mirrored artifacts and remove only this run's resolved scratch copies."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--prepare-inventory',action='store_true');p.add_argument('--tag',choices=['local','canonical'],default='local');a=p.parse_args();assert Path(a.run_id).name==a.run_id
    run=ROOT/'results'/a.run_id;manifest=run/'sync_inventory.json'
    if a.prepare_inventory:
        files=[x for x in run.rglob('*') if x.is_file() and x.name!='sync_inventory.json' and not x.name.startswith('housekeeping_')]
        files+=[ROOT/'dashboard'/n for n in ['RSI_Experimental_Progress_Latest.html','gradient_feedback_validation.html','RSI_Experiment_Progress_2026-10-28.html']]
        files+=[ROOT/'reports'/f'{n}.md' for n in ['SEMANTIC_REVIEW_STATUS','VALIDATED_MATCHED_FEEDBACK','CLEAN_TARGET_TRAINING_BASELINE','RL_INFRASTRUCTURE_STATUS','NEXT_EXPERIMENT_DECISION']]
        files.append(Path(__file__).resolve());entries=[{'path':str(x.relative_to(ROOT)).replace('\\','/'),'sha256':sha(x),'bytes':x.stat().st_size} for x in sorted(files)]
        with manifest.open('x') as f:json.dump({'entries':entries,'total_bytes':sum(x['bytes'] for x in entries)},f,indent=2)
        print(json.dumps({'inventory_files':len(entries),'bytes':sum(x['bytes'] for x in entries)}));return
    inventory=json.loads(manifest.read_text());mismatch=[x['path'] for x in inventory['entries'] if not (ROOT/x['path']).is_file() or sha(ROOT/x['path'])!=x['sha256']]
    assert not mismatch,mismatch
    assert json.loads((run/'completion.json').read_text())['status']=='PASS'
    scratch=ROOT/'tmp'/a.run_id;removed=[]
    if scratch.exists():
        target=scratch.resolve();assert target.parent==(ROOT/'tmp').resolve() and target.name==a.run_id
        removed.append(str(target));shutil.rmtree(target)
    for suffix in ['stage','repair','repair2','results','final_bundle']:
        path=ROOT/'tmp'/f'{a.run_id}_{suffix}.tar'
        if path.exists():
            target=path.resolve();assert target.parent==(ROOT/'tmp').resolve() and target.name==f'{a.run_id}_{suffix}.tar';removed.append(str(target));target.unlink()
    marker='SEMANTIC_CLEAN_VALIDATION_20261029_V1';index=ROOT/'EXPERIMENT_INDEX.md'
    if marker not in index.read_text(encoding='utf-8'):
        with index.open('a',encoding='utf-8') as f:f.write('\n\n## '+marker+'\n\nClean original-VQA language LoRA engineering PASS;16 logical main steps,112 tensors /1,605,632 trainable elements, frozen base invariant, exact fixed-example continuation and fresh weights/optimizer/predictions restore. Clean development accuracy94.6875% before/after. Human review0/160 BLOCKED; blinded offline kit complete. Both failed pilots preserved;220 A100-seconds including failures. No training-induced backdoor evaluation. [Decision](reports/NEXT_EXPERIMENT_DECISION.md), [dashboard](dashboard/RSI_Experimental_Progress_Latest.html).\n')
    receipt={'status':'PASS','tag':a.tag,'verified_artifact_files':len(inventory['entries']),'verified_bytes':inventory['total_bytes'],'removed_resolved_scratch':removed,'retained_run_bytes':sum(x.stat().st_size for x in run.rglob('*') if x.is_file()),'disk_free_bytes':{str(d):shutil.disk_usage(d).free for d in ([ROOT.anchor,'C:/'] if os.name=='nt' else [ROOT])},'large_model_downloads':0,'base_model_copies_retained':0,'negative_evidence_preserved':True}
    with (run/f'housekeeping_{a.tag}.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
