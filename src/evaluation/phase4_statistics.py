"""Source-paired descriptive uncertainty, never seed-population inference."""
import numpy as np

def paired_summary(current,reference):
    if len(current)!=len(reference) or not current:raise ValueError('Paired nonempty observations required')
    if [x['sample_id'] for x in current]!=[x['sample_id'] for x in reference]:raise ValueError('Source order mismatch')
    if [x['seed'] for x in current]!=[x['seed'] for x in reference]:raise ValueError('Sampling seed mismatch')
    x=np.array([a['reward']['reward']-b['reward']['reward'] for a,b in zip(current,reference)])
    rng=np.random.default_rng(7);boot=x[rng.integers(0,len(x),size=(10000,len(x)))].mean(1)
    lo,hi=np.quantile(boot,[.025,.975])
    return {'mean_delta':float(x.mean()),'ci95':[float(lo),float(hi)],'ci_half_width':float((hi-lo)/2),
        'paired_deltas':x.tolist(),'independent_source_count':len(x),'bootstrap_seed':7,'bootstrap_resamples':10000,
        'scope':'source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory'}

def rsi_plateau(looks):
    return len(looks)>=2 and all(x['mean_delta']<=x['ci_half_width'] for x in looks[-2:])
