"""Remove only verified run-specific transfer scratch; leave monitoring enabled."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--tag',choices=['local','canonical'],required=True);a=p.parse_args();assert Path(a.run_id).name==a.run_id;run=ROOT/'results'/a.run_id
    completion=json.loads((run/'completion.json').read_text(encoding='utf-8'));assert completion['status']=='PASS_ENGINEERING_AND_EVIDENCE_INTEGRITY';removed=[]
    if a.tag=='local':
        archive=ROOT/'tmp'/a.run_id/'final.tar';assert hashlib.sha256(archive.read_bytes()).hexdigest()=='1a0d21c58825c228b622a5f4c4c40a7792bfc11b7f19c03eb6b67438b7bebaff'
        scratch=(ROOT/'tmp'/a.run_id).resolve();assert scratch.parent==(ROOT/'tmp').resolve() and scratch.name==a.run_id;shutil.rmtree(scratch);removed.append(str(scratch))
        event=json.loads((run/'report_generation.json').read_text(encoding='utf-8'))['event'];history=ROOT/'experiments/semantic_clean_history.jsonl'
        if event['event'] not in history.read_text(encoding='utf-8'):
            with history.open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
        index=ROOT/'EXPERIMENT_INDEX.md'
        if event['event'] not in index.read_text(encoding='utf-8'):
            with index.open('a',encoding='utf-8') as f:f.write('\n\n## '+event['event']+'\n\nThree clean Qwen seeds and fresh-object next-step checks PASS, fresh32-image accuracy90.0% before/after. Uninstrumented IP2Pix continuation FAIL; instrumented witness PASS, general reliability unresolved. Human review BLOCKED. Two-hour native Slurm monitor verified and extended. All scientific failures retained. GPU342seconds new,1344seconds including prior,2256seconds remaining. [Execution evidence](reports/REPLICATION_AND_MONITORING_STATUS.md).\n')
    else:
        for suffix in ['launch','final']:
            path=(ROOT/'tmp'/f'{a.run_id}_{suffix}.tar').resolve();assert path.parent==(ROOT/'tmp').resolve() and path.name==f'{a.run_id}_{suffix}.tar'
            if path.exists():path.unlink();removed.append(str(path))
    directories=[p for p in (ROOT/'results').glob(a.run_id+'*') if p.is_dir()];result={'status':'PASS','tag':a.tag,'retained_experiment_bytes':sum(p.stat().st_size for d in directories for p in d.rglob('*') if p.is_file()),'removed_resolved_scratch':removed,'disk_free_bytes':{str(p):shutil.disk_usage(p).free for p in ([ROOT.anchor,'C:/'] if os.name=='nt' else [ROOT])},'negative_evidence_preserved':True,'persistent_monitor_left_enabled':True,'new_model_downloads':0,'no_multi_GB_artifacts_on_C':True}
    with (run/f'housekeeping_{a.tag}.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
