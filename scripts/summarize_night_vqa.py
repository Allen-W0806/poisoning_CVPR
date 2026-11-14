"""Read-only paired analysis; never feeds a score back to training/selection."""
import argparse
import json
from pathlib import Path
import random
import statistics


def source_bootstrap(values,resamples=10000,seed=20261112):
    rng=random.Random(seed);n=len(values)
    draws=sorted(sum(rng.choices(values,k=n))/n for _ in range(resamples))
    def quantile(p):
        index=(len(draws)-1)*p;lo=int(index);hi=min(lo+1,len(draws)-1)
        return draws[lo]+(draws[hi]-draws[lo])*(index-lo)
    return [quantile(.025),quantile(.975)]


def summarize(base,snapshot):
    base=Path(base);success={}
    for job in snapshot['jobs']:
        if job.get('state')=='COMPLETED' and job.get('result',{}).get('status')=='PASS':success[job['stage']]=job
    expected=['frozen','seed42','seed123','seed3407']
    if not all(s in success for s in expected):return {'status':'NOT_COMPLETE','available_stages':[s for s in expected if s in success],'training_seed_SD':'NOT_ESTIMATED_UNTIL_3_SEEDS_COMPLETE'}
    def rows(stage,split):
        job=success[stage];path=base/'stages'/stage/f'attempt_{job["attempt"]}'/'predictions.jsonl'
        label=('frozen_' if stage=='frozen' else 'selected_')+split
        result={}
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line)
            if row['label']!=label:continue
            if row['sample_id'] in result:raise ValueError('Duplicate final prediction')
            result[row['sample_id']]=row
        return result
    result={'status':'COMPLETE_3_SEED_READ_ONLY_ANALYSIS','statistical_unit':'one question per source; seed-averaged paired deltas resampled over sources',
       'uncertainty_limit':'Source-bootstrap CI conditional on these three seeds. Training-seed SD separately reported; three seeds are not a large-seed population CI.',
       'analysis_seed':20261112,'bootstrap_resamples':10000,'transfer_scope':'source-held-out same VQA family, not independent task-family RSI evidence','splits':{}}
    for split,count in [('test',512),('transfer',128)]:
        frozen=rows('frozen',split);ids=sorted(frozen)
        if len(ids)!=count:raise ValueError('Unexpected source count')
        arms=[rows(s,split) for s in expected[1:]]
        if any(set(arm)!=set(ids) for arm in arms):raise ValueError('Unpaired evaluation sources')
        for arm in arms:
            if any(arm[k]['source_SHA256']!=frozen[k]['source_SHA256'] for k in ids):raise ValueError('Source content mismatch')
        frozen_mean=statistics.mean(frozen[k]['score'] for k in ids)
        means=[statistics.mean(arm[k]['score'] for k in ids) for arm in arms]
        delta=[statistics.mean(arm[k]['score'] for arm in arms)-frozen[k]['score'] for k in ids]
        by_type={kind:{'n':sum(frozen[k]['answer_type']==kind for k in ids),
                      'frozen':statistics.mean(frozen[k]['score'] for k in ids if frozen[k]['answer_type']==kind),
                      'SFT_seed_mean':statistics.mean(statistics.mean(arm[k]['score'] for arm in arms) for k in ids if frozen[k]['answer_type']==kind)} for kind in sorted({r['answer_type'] for r in frozen.values()})}
        result['splits'][split]={'n_sources':count,'seeds':[42,123,3407],'frozen':frozen_mean,'SFT_seed_scores':means,
              'SFT_mean':statistics.mean(means),'SFT_training_seed_SD':statistics.stdev(means),'paired_delta_mean':statistics.mean(delta),
              'paired_delta_source_bootstrap_95CI':source_bootstrap(delta),'answer_type':by_type}
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    value=summarize(a.evidence,json.loads(Path(a.snapshot).read_text(encoding='utf-8')));Path(a.output).write_text(json.dumps(value,indent=2),encoding='utf-8');print(json.dumps(value))
