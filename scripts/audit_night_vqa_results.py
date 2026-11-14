"""Read-only final-cohort audit from predictions, training rows and Slurm receipts."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import validate_splits
from src.evaluation.vqa_official_compatible import score_answer


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    run=Path(run);base=run/'remote';cfg=json.loads((run/'protocol.json').read_text(encoding='utf-8'))
    snapshot=json.loads((run/'latest_snapshot.json').read_text(encoding='utf-8'));manifest_path=base/'manifest.json'
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'));receipt=json.loads((base/'prepare_receipt.json').read_text(encoding='utf-8'))
    assert digest(manifest_path)==receipt['manifest_SHA256']
    assert digest(base/'protocol.json')==receipt['protocol_SHA256']
    assert json.loads((base/'protocol.json').read_text(encoding='utf-8'))==cfg
    split_check=validate_splits(manifest['splits'],cfg['split_sizes'])
    lookup={row['sample_id']:row for rows in manifest['splits'].values() for row in rows}
    jobs=snapshot['jobs'];by_stage={r['stage']:r for r in jobs};checks={};predictions={}
    assert set(by_stage)=={'canary','frozen','seed42','seed123','seed3407'}
    assert all(j['state']=='COMPLETED' and j['result']['status']=='PASS' for j in jobs)
    assert sum(j['elapsed_seconds'] for j in jobs)==snapshot['budget']['actual_GPU_seconds']<=cfg['budget_GPU_seconds']
    intervals=sorted((j['start'],j['end']) for j in jobs)
    assert all(intervals[i][1]<=intervals[i+1][0] for i in range(len(intervals)-1))
    assert all(re.search(r'(?:^|,)gres/gpu=1(?:,|$)',j['AllocTRES']) for j in jobs)
    assert not snapshot['budget']['active_GPU_jobs']
    for stage in ['frozen','seed42','seed123','seed3407']:
        job=by_stage[stage];result=job['result'];folder=base/'stages'/stage/f'attempt_{job["attempt"]}'
        rows=[json.loads(line) for line in (folder/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
        verified=0
        for row in rows:
            original=lookup[row['sample_id']]
            assert row['source_SHA256']==original['image_sha256'] and str(row['image_id'])==str(original['image_id'])
            actual=score_answer(row['prediction'],original)
            assert abs(actual['score']-row['score'])<1e-12 and actual['normalized_answer']==row['normalized_answer']
            verified+=1
        predictions[stage]={}
        for split in ['test','transfer']:
            label=('frozen_' if stage=='frozen' else 'selected_')+split
            subset=[row for row in rows if row['label']==label]
            expected={r['sample_id'] for r in manifest['splits'][split]}
            assert len(subset)==len(expected) and {r['sample_id'] for r in subset}==expected
            value=statistics.mean(r['score'] for r in subset)
            assert abs(value-result['metrics'][split]['mean'])<1e-12
            predictions[stage][split]={r['sample_id']:r['score'] for r in subset}
        checked={'prediction_scores_recomputed':verified,'test_sources':512,'transfer_sources':128}
        if stage!='frozen':
            training=[json.loads(line) for line in (folder/'training.jsonl').read_text(encoding='utf-8').splitlines()]
            assert len(training)==cfg['updates']==result['actual_optimizer_calls']==result['attempted_training_steps']
            assert [r['retained_step'] for r in training]==list(range(1,cfg['updates']+1))
            for row in training:
                assert all(math.isfinite(row[k]) for k in ['loss','raw_gradient_norm','max_parameter_change'])
                assert row['raw_gradient_norm']>0 and row['max_parameter_change']>0
                assert row['actual_optimizer_calls']==1 and len(row['sample_ids'])==cfg['vlm_accumulation']
            actual_stream=[sid for row in training for sid in row['sample_ids']];train=manifest['splits']['train'];expected_stream=[]
            seed=int(stage[4:])
            for epoch in range(2):
                indices=list(range(len(train)));random.Random(seed+epoch*1000003).shuffle(indices)
                expected_stream.extend(train[i]['sample_id'] for i in indices)
            assert actual_stream==expected_stream and set(Counter(actual_stream).values())=={2}
            history=result['dev_history'];assert [h['step'] for h in history]==cfg['dev_steps']
            for observation in history:
                subset=[r for r in rows if r['label']==f'dev_step{observation["step"]}']
                assert len(subset)==128 and {r['sample_id'] for r in subset}=={r['sample_id'] for r in manifest['splits']['dev']}
                assert abs(statistics.mean(r['score'] for r in subset)-observation['score'])<1e-12
            chosen=max(history,key=lambda r:(r['score'],-r['step']))
            assert chosen['step']==result['retained_step']
            assert job['selected_checkpoint_health']['status']=='PASS_BYTES_AND_ADAPTER_SHA'
            checked.update(optimizer_calls=len(training),retained_selected_step=chosen['step'],training_source_counts='1024 sources exactly twice, seed-prespecified order',
                           dev_only_selection_recomputed=True,checkpoint_bytes_and_adapter_SHA='PASS_REMOTE_READBACK')
        checks[stage]=checked
    comparison=json.loads((run/'comparison.json').read_text(encoding='utf-8'))
    for split in ['test','transfer']:
        ids=sorted(predictions['frozen'][split]);means=[statistics.mean(predictions['seed'+str(s)][split].values()) for s in cfg['seeds']]
        deltas=[statistics.mean(predictions['seed'+str(s)][split][k] for s in cfg['seeds'])-predictions['frozen'][split][k] for k in ids]
        reported=comparison['splits'][split]
        assert all(abs(a-b)<1e-12 for a,b in zip(means,reported['SFT_seed_scores']))
        assert abs(statistics.mean(means)-reported['SFT_mean'])<1e-12
        assert abs(statistics.stdev(means)-reported['SFT_training_seed_SD'])<1e-12
        assert abs(statistics.mean(deltas)-reported['paired_delta_mean'])<1e-12
    assert by_stage['canary']['result']['resume_exact'] and by_stage['canary']['result']['public_adapter_load_exact']
    hashes={by_stage[s]['result']['frozen_base_SHA256'] for s in ['canary','seed42','seed123','seed3407']};assert len(hashes)==1
    return {'status':'PASS_READ_ONLY_FINAL_COHORT_AUDIT','checked_UTC':datetime.now(timezone.utc).isoformat(),'split_check':split_check,'checks':checks,
       'GPU_hours':snapshot['budget']['actual_GPU_hours'],'GPU_overlap_detected':False,'new_GPU_jobs_submitted_by_audit':0,
       'frozen_base_hash_equal_across_canary_and_seeds':True,'retained_stage_retries':len(jobs)-5,
       'input_SHA256':{'manifest':digest(manifest_path),'remote_protocol':digest(base/'protocol.json'),'local_protocol_copy':digest(run/'protocol.json'),'snapshot':digest(run/'latest_snapshot.json'),'comparison':digest(run/'comparison.json')},
       'protocol_note':'Remote byte SHA matches locked receipt; local Windows CRLF serialization differs in bytes, JSON values are identical.',
       'limits':['source photo bytes attested by cluster preparation; local audit checks manifest/prediction hashes, not absent local image files',
                 'source-bootstrap uncertainty conditional on three training seeds; seed SD separately reported',
                 'train2014 source-held-out VQA only; pretraining exposure unknown; not independent task-family RSI evidence',
                 'perceptual near-duplicate/global historical exposure audit NOT_RUN; generator fidelity gate still NOT_CLEARED']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);a=parser.parse_args()
    value=audit(a.run);path=Path(a.run)/'final_read_only_audit.json';path.write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value))
