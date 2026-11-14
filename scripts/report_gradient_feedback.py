"""Recompute all benign validation tables/plots from immutable raw evidence.

python -B scripts/report_gradient_feedback.py
No inferred/missing model measurements are filled in. Repeated seeds are not
counted as 480 independent images. Bootstrap resamples unique source images.
"""
from collections import Counter,defaultdict
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
SCRATCH=ROOT/'tmp/gradient_feedback_reporting';SCRATCH.mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(SCRATCH/'mpl')
import numpy as np
from scipy.stats import ttest_ind
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def readcsv(path):
    with path.open(encoding='utf-8') as f:return list(csv.DictReader(f))


def writecsv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def jread(path):return json.loads(path.read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jwrite(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    source=ROOT/'results/benign_models_20261028_v1'; syn=ROOT/'results/gradient_feedback_20261028_remote_v2'
    out=ROOT/'results/gradient_feedback_summary_20261028';out.mkdir(parents=True,exist_ok=True)
    plots=out/'plots';plots.mkdir(exist_ok=True)
    manifest=jread(source/'evaluation_manifest.json'); manifest_hash=digest(source/'evaluation_manifest.json')
    raw=[json.loads(line) for line in (source/'predictions.jsonl').read_text().splitlines()]
    rows=readcsv(source/'benign_vlm_evaluation.csv')
    expected={(r['sample_id'],r['evaluation_seed'],r['condition']) for r in manifest['records']}
    actual=[(r['sample_id'],r['seed'],r['condition']) for r in raw]
    assert len(actual)==len(set(actual))==480 and set(actual)==expected
    assert len(rows)==480 and all(r['manifest_sha256']==manifest_hash for r in raw)
    from src.evaluation.vqa_official_compatible import score_answer
    for r in raw:
        assert abs(score_answer(r['raw_answer'],{'answers':r['references']})['score']-r['consensus_score'])<1e-12
    # Prove the CSV is identical to flushed JSONL scientific outputs.
    for csvrow,record in zip(rows,raw):
        assert csvrow['sample_id']==record['sample_id'] and int(csvrow['seed'])==record['seed']
        assert float(csvrow['consensus_score'])==record['consensus_score']
        assert csvrow['raw_answer']==record['raw_answer']
    base=[r for r in raw if r['seed']==42];assert len(base)==160
    seeds=sorted({r['seed'] for r in raw});rng=np.random.default_rng(20261028)
    cfg=jread(source/'protocol.json');boots=cfg['evaluation']['bootstrap_replicates']
    def ci(values):
        v=np.asarray(values,float);means=v[rng.integers(0,len(v),(boots,len(v)))].mean(1)
        return [float(x) for x in np.quantile(means,[.025,.975])]
    batchrows=[]
    for seed in seeds:
        for b in range(10):
            rr=[r for r in raw if r['seed']==seed and r['batch_id']==b];scores=np.array([r['consensus_score'] for r in rr]);low,high=ci(scores)
            batchrows.append({'seed':seed,'batch_id':b,'condition':'original','n':len(rr),'mean_reward':float(scores.mean()),'within_batch_variance':float(scores.var(ddof=1)),
                'ci95_low':low,'ci95_high':high,'unique_scores':len(set(scores)),'tied_score_pairs':sum(c*(c-1)//2 for c in Counter(scores).values()),
                'at_zero':int((scores==0).sum()),'at_one':int((scores==1).sum()),'invalid':sum(not r['valid'] for r in rr),
                'missing':16-len(rr),'category_counts':json.dumps(dict(Counter(r['category'] for r in rr))),
                'sample_ids':json.dumps([r['sample_id'] for r in rr]),'manifest_sha256':manifest_hash,'confidence_method':'percentile bootstrap, 5000 resamples of 16 images; descriptive'})
    writecsv(out/'batch_feedback.csv',batchrows);writecsv(ROOT/'results/batch_feedback.csv',batchrows)
    # Composition sensitivity: permute scores WITHIN answer category, preserving
    # the exact category counts and 16-example capacity of every batch.
    scores=np.array([r['consensus_score'] for r in base]);bid=np.array([r['batch_id'] for r in base]);cat=np.array([r['category'] for r in base])
    means=np.array([scores[bid==b].mean() for b in range(10)]);between=float(means.var(ddof=1))
    categories=sorted(set(cat));null=[];null_range=[]
    for _ in range(boots):
        perm=scores.copy()
        for c in categories:idx=np.flatnonzero(cat==c);perm[idx]=rng.permutation(perm[idx])
        m=np.array([perm[bid==b].mean() for b in range(10)]);null.append(float(m.var(ddof=1)));null_range.append(float(np.ptp(m)))
    perm_p=(1+sum(x>=between for x in null))/(boots+1)
    pairwise=[]
    for a in range(10):
        for b in range(a+1,10):
            test=ttest_ind(scores[bid==a],scores[bid==b],equal_var=False)
            pairwise.append({'batch_a':a,'batch_b':b,'mean_difference':float(means[a]-means[b]),'welch_p':float(test.pvalue)})
    # Holm step-down adjusted p-values; small n + bounded discrete values:
    # exploratory only, not equivalence or a candidate-learning signal.
    order=sorted(range(len(pairwise)),key=lambda i:pairwise[i]['welch_p']);prev=0.
    for rank,i in enumerate(order):
        prev=max(prev,min(1.,(len(pairwise)-rank)*pairwise[i]['welch_p']));pairwise[i]['holm_p']=prev
    writecsv(out/'batch_pairwise_exploratory.csv',pairwise)
    categories_out={c:{'n':int((cat==c).sum()),'consensus_accuracy':float(scores[cat==c].mean()),'ci95':ci(scores[cat==c])} for c in categories}
    repeat={sid:[r for r in raw if r['sample_id']==sid] for sid in {r['sample_id'] for r in raw}}
    checks={'manifest_records':480,'unique_images':160,'no_duplicate_records':True,'all_csv_predictions_recomputed':True,
        'manifest_hash_verified':manifest_hash==jread(source/'manifest_sha256.json')['sha256'],
        'raw_answer_seed_exact_fraction':sum(len({x['raw_answer'] for x in rr})==1 for rr in repeat.values())/160,
        'max_per_example_seed_score_variance':max(float(np.var([x['consensus_score'] for x in rr],ddof=1)) for rr in repeat.values()),
        'missing_records':len(expected-set(actual)),'invalid_records':sum(not r['valid'] for r in raw)}
    summary={'data_integrity':checks,'n_independent_images':160,'n_inference_calls':480,'seeds':seeds,
        'consensus_accuracy':float(scores.mean()),'consensus_ci95':ci(scores),
        'exact_any_reference_accuracy':float(np.mean([r['exact_any_reference'] for r in base])),
        'exact_any_reference_ci95':ci([r['exact_any_reference'] for r in base]),
        'per_category':categories_out,'between_batch_variance':between,'pooled_within_batch_variance':float(np.mean([r['within_batch_variance'] for r in batchrows if r['seed']==42])),
        'batch_min':float(means.min()),'batch_max':float(means.max()),'batch_score_ties':10-len(set(means)),
        'score_at_zero':int((scores==0).sum()),'score_at_one':int((scores==1).sum()),
        'category_stratified_composition_permutation_p':perm_p,'null_between_variance_ci95':[float(v) for v in np.quantile(null,[.025,.975])],
        'null_range_ci95':[float(v) for v in np.quantile(null_range,[.025,.975])],
        'holm_significant_batch_pairs':sum(r['holm_p']<.05 for r in pairwise),
        'mean_latency_seconds':float(np.mean([r['latency_seconds'] for r in raw])),
        'latency_ci95':ci([r['latency_seconds'] for r in base]),
        'latency_by_seed':{str(s):float(np.mean([r['latency_seconds'] for r in raw if r['seed']==s])) for s in seeds},
        'inference_seconds_sum':float(sum(r['latency_seconds'] for r in raw)),
        'interpretation':'Batch variation is a composition/difficulty diagnostic of one frozen system. Repeated seeds have no observed answer variability. No batch-specific policy/system effects are identified; batches cannot serve as matched GRPO candidates.',
        'leakage_scope':'Image IDs and SHA256 checked against internal train/test by model runner. Previously used development set; target pretraining contamination unknown. Not untouched held-out efficacy evidence.'}
    jwrite(out/'feedback_statistics.json',summary)
    # Canonical requested CSVs are regenerated views; raw run files never modified.
    shutil.copyfile(syn/'gradient_validation.csv',ROOT/'results/gradient_validation.csv')
    combined=list(rows)
    editpath=ROOT/'results/reviewed_edits_20261028_v1/benign_vlm_evaluation.csv'
    if editpath.exists():combined+=readcsv(editpath)
    writecsv(ROOT/'results/benign_vlm_evaluation.csv',combined)
    parameters=readcsv(syn/'parameter_update_audit.csv')+readcsv(source/'parameter_update_audit.csv')
    tiny=ROOT/'results/ddpo_wiring_20261028_v3/tiny_parameter_updates.json'
    if tiny.exists():parameters+=[{'model':'tiny_random_UNet_NOT_pretrained_IP2Pix',**r} for r in jread(tiny)]
    writecsv(ROOT/'results/parameter_update_audit.csv',parameters)
    gr=readcsv(syn/'gradient_validation.csv'); plt.rcParams.update({'font.size':11,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    gradient_stats=[]
    for method in sorted({r['estimator'] for r in gr}):
        rr=[r for r in gr if r['estimator']==method]
        for n in sorted({int(r['n']) for r in rr}):
            for component in sorted({int(r['component']) for r in rr}):
                ss=[r for r in rr if int(r['n'])==n and int(r['component'])==component]
                values=np.array([float(r['estimate']) for r in ss]);exact=float(ss[0]['exact'])
                se=float(values.std(ddof=1)/np.sqrt(len(values)))
                gradient_stats.append({'estimator':method,'n':n,'component':component,'independent_seeds':len(values),'exact':exact,
                    'mean_estimate':float(values.mean()),'mean_bias_estimate':float(values.mean()-exact),'empirical_estimator_variance':float(values.var(ddof=1)),
                    'mean_ci95_low_normal_approx':float(values.mean()-1.96*se),'mean_ci95_high_normal_approx':float(values.mean()+1.96*se)})
    writecsv(out/'gradient_statistics.csv',gradient_stats)
    def save(fig,name):
        fig.tight_layout();fig.savefig(plots/(name+'.png'),bbox_inches='tight');fig.savefig(plots/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for method in sorted({r['estimator'] for r in gr}):
        rr=[r for r in gr if r['estimator']==method];nn=max(int(r['n']) for r in rr)
        rr=[r for r in rr if int(r['n'])==nn]; comps=sorted({int(r['component']) for r in rr})
        exact=[float(next(r['exact'] for r in rr if int(r['component'])==c)) for c in comps]
        estimate=[np.mean([float(r['estimate']) for r in rr if int(r['component'])==c]) for c in comps]
        se=[np.std([float(r['estimate']) for r in rr if int(r['component'])==c],ddof=1)/np.sqrt(32) for c in comps]
        ax.errorbar(exact,estimate,yerr=1.96*np.array(se),fmt='o',markersize=4,label=method.replace('existing_',''),alpha=.75)
    lim=ax.get_xlim();ax.plot(lim,lim,'--',color='gray');ax.set(xlabel='Exact reward gradient',ylabel='Mean estimated gradient ± 1.96 SE');ax.legend(fontsize=7);save(fig,'exact_vs_estimated_gradients')
    fig,ax=plt.subplots(figsize=(7,4))
    for method in sorted({r['estimator'] for r in gr}):
        rr=[r for r in gr if r['estimator']==method];ns=sorted({int(r['n']) for r in rr});rm=[]
        for n in ns:
            subset=[r for r in rr if int(r['n'])==n];seeds0=sorted({int(r['seed']) for r in subset})
            errors=[]
            for s in seeds0:
                ss=[r for r in subset if int(r['seed'])==s];e=np.array([float(r['exact']) for r in ss]);g=np.array([float(r['estimate']) for r in ss]);errors.append(np.sum((e-g)**2)/np.sum(e**2))
            rm.append(float(np.sqrt(np.mean(errors))))
        ax.loglog(ns,rm,'o-',label=method.replace('existing_',''))
    ax.set(xlabel='Independent Monte Carlo samples per seed',ylabel='Relative gradient RMSE (32 seeds)');ax.legend(fontsize=7);save(fig,'gradient_error_vs_sample_count')
    fig,ax=plt.subplots(figsize=(7,4));br=[r for r in batchrows if r['seed']==42]
    ax.errorbar(range(10),means,yerr=[means-np.array([r['ci95_low'] for r in br]),np.array([r['ci95_high'] for r in br])-means],fmt='o',capsize=4)
    ax.axhline(scores.mean(),color='gray',ls='--');ax.set(xlabel='Disjoint diagnostic batch (16 images)',ylabel='Mean VQA reward + 95% bootstrap CI',ylim=(0,1.03),xticks=range(10));save(fig,'reward_variance_across_batches')
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(scores,bins=np.linspace(-.05,1.05,12),edgecolor='white');ax.set(xlabel='Official VQA consensus score',ylabel='Unique source images (N=160)');save(fig,'benign_vlm_score_distribution')
    fig,ax=plt.subplots(figsize=(7,4));ax.boxplot([[r['latency_seconds'] for r in raw if r['seed']==s] for s in seeds],tick_labels=[str(s) for s in seeds],showfliers=True);ax.set(xlabel='Evaluation seed (greedy decoding)',ylabel='Seconds per image/question');save(fig,'computation_time_per_evaluation')
    fig,ax=plt.subplots(figsize=(7,4));learn=readcsv(syn/'group_learning.csv')
    for method in sorted({r['method'] for r in learn}):
        steps=sorted({int(r['step']) for r in learn if r['method']==method});values=np.array([[float(r['expected_reward']) for r in learn if r['method']==method and int(r['step'])==s] for s in steps])
        ax.plot(steps,values.mean(1),label=method);ax.fill_between(steps,values.min(1),values.max(1),alpha=.2)
    ax.set(xlabel='Synthetic policy updates',ylabel='Exactly enumerated expected reward');ax.legend(fontsize=8);save(fig,'group_relative_learning')
    # English dashboard is a lightweight local deliverable with live-stage provenance.
    synchecks=jread(syn/'synthetic_checks.json');wiringpath=ROOT/'results/ddpo_wiring_20261028_v3/wiring_checks.json'
    wiring=jread(wiringpath) if wiringpath.exists() else {'passed':False,'checks':[],'status':'pending'}
    editorpath=ROOT/'results/benign_editor_20261028_v2/model_checks.json'
    editor=jread(editorpath) if editorpath.exists() else {'passed':False,'checks':[],'status':'pending'}
    state={'phase':'Complete: bounded validation; scientific training remains NO-GO','synthetic_gradient_passed':synchecks['passed'],
        'production_tiny_DDPO_wiring_passed':wiring.get('passed'),'frozen_VQA_reproducibility_passed':all([checks['raw_answer_seed_exact_fraction']==1,checks['missing_records']==0,checks['invalid_records']==0]),
        'editor_inference_engineering_passed':editor.get('passed'),'editor_semantic_preservation_passed':False,
        'feedback_record_integrity_passed':True,'batch_candidate_comparability_passed':False,
        'large_scale_finetuning_authorized_by_these_results':False,'statistics':summary,
        'failed_runs_retained':['gradient_feedback_20261028_v1: local diffusers absent','gradient_feedback_20261028_remote_v1: DDPO precision insufficient','benign_models_20261028_v1: NumPy bool JSON logging failure after inference','ddpo_wiring_20261028_v1: synthetic pipe omitted frozen module placeholders','ddpo_wiring_20261028_v2: sampling/replay SDPA mismatch; strict replay assertion retained'],
        'model_revisions':{k:cfg['evaluation'][k] for k in ['model_revision','editor_revision']},
        'resource_accounting':jread(out/'resource_accounting.json') if (out/'resource_accounting.json').exists() else {},
        'last_GPU_utilization':jread(ROOT/'results/benign_editor_20261028_v2/live.json').get('gpu') if (ROOT/'results/benign_editor_20261028_v2/live.json').exists() else 'unavailable',
        'gradient_diagnostics':[c for c in synchecks['checks'] if 'statistical_agreement' in c['test'] or 'convergence' in c['test']]}
    jwrite(out/'dashboard_data.json',state)
    (ROOT/'dashboard/gradient_feedback_validation.html').write_text('''<!doctype html><html lang="en"><meta charset="utf-8"><title>RSI Gradient and Feedback Validation</title><style>body{font:16px system-ui;margin:32px auto;max-width:1100px;color:#182537;background:#f4f6fa}section{background:white;padding:22px;margin:18px 0;border-radius:12px}img{max-width:100%;width:500px}pre{white-space:pre-wrap;font-size:13px}a{color:#205dcc}</style><h1>RSI: Gradient and Feedback Validation</h1><p>Benign diagnostics only. Large-scale fine-tuning remains disabled.</p><section><h2>Current phase and acceptance</h2><pre>'''+html.escape(json.dumps({k:v for k,v in state.items() if k!='statistics'},indent=2))+'''</pre></section><section><h2>Frozen target VQA</h2><p>160 unique images × 3 seeds. Greedy replay is exact; seeds do not triple the effective sample size.</p><pre>'''+html.escape(json.dumps(summary,indent=2))+'''</pre></section><section><h2>Plots</h2>'''+''.join(f'<a href="../results/gradient_feedback_summary_20261028/plots/{p.name}"><img src="../results/gradient_feedback_summary_20261028/plots/{p.name}"></a>' for p in sorted(plots.glob('*.png')))+'''</section><section><h2>Reports and evidence</h2><ul><li><a href="../reports/GRADIENT_AND_FEEDBACK_AUDIT.md">Code and gradient audit</a></li><li><a href="../reports/EXPERIMENT_RESULTS.md">Experiment results</a></li><li><a href="../results/benign_models_20261028_v1/evaluation_manifest.json">Immutable evaluation manifest</a></li><li><a href="../results/gradient_feedback_20261028_remote_v2/synthetic_checks.json">Gradient diagnostics</a></li><li><a href="../logs/gradient_feedback_20261028/">Raw execution logs</a></li><li><a href="../results/benign_editor_20261028_v2/live.json">GPU utilization and model-stage timing</a></li><li><a href="../results/gradient_feedback_summary_20261028/resource_accounting.json">Job resource ledger</a></li><li><a href="../results/editor_visual_review_20261028.json">Editor semantic failures and exclusions</a></li></ul></section></html>''',encoding='utf-8')
    jwrite(out/'report_generation.json',{'source_sha256':digest(Path(__file__)),'raw_predictions_sha256':digest(source/'predictions.jsonl'),
        'manifest_sha256':manifest_hash,'generated_unix':time.time(),'input_run':str(source.relative_to(ROOT)),
        'outputs':{str(p.relative_to(ROOT)):digest(p) for p in plots.iterdir() if p.is_file()}})
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    try:main()
    finally:
        resolved=SCRATCH.resolve()
        if resolved.parent!=(ROOT/'tmp').resolve() or resolved.name!='gradient_feedback_reporting':raise RuntimeError('Unsafe cleanup target')
        shutil.rmtree(resolved)
