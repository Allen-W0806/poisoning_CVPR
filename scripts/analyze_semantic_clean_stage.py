"""Saved-prediction reanalysis and genuine dual-human review ingestion. No inference."""
import argparse,csv,hashlib,json,sys
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from analyze_matched_feedback import holm,csvwrite,dump
ATTRS=['scene_preserved','objects_preserved','identity_count_preserved','color_spatial_preserved','reference_answer_valid','instruction_followed']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validated_reviews(paths,cases,adjudication=None):
    byid={c['source_image_id']:c for c in cases};reviews=defaultdict(dict);rejected=[];receipts=[]
    for path in paths:
        receipts.append({'path':str(path),'sha256':sha(path)})
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            for number,r in enumerate(csv.DictReader(f),2):
                if not r.get('reviewer_id','').strip() and not r.get('semantic_label','').strip():continue
                try:
                    c=byid[r['source_image_id']];rid=r['reviewer_id'].strip();assert rid and r['independence_attestation']=='yes'
                    assert r['original_sha256']==c['original_sha256'] and r['edited_sha256']==c['edited_sha256'];assert r['question_id']==c['question_id']
                    timestamp=datetime.fromisoformat(r['reviewed_utc'].replace('Z','+00:00'));assert timestamp.tzinfo is not None
                    assert r['semantic_label'] in {'valid','invalid','ambiguous'} and r['confidence'] in {'low','medium','high'}
                    assert all(r[k] in {'yes','no','ambiguous','not_applicable'} for k in ATTRS)
                    assert r['reference_answer_valid'] in {'yes','no','ambiguous'}
                    if r['semantic_label']=='valid':assert r['reference_answer_valid']=='yes' and all(r[k] in {'yes','not_applicable'} for k in ATTRS[:4])
                    else:assert r['failure_reason'].strip()
                    assert rid not in reviews[c['source_image_id']]
                    reviews[c['source_image_id']][rid]=r
                except (AssertionError,KeyError,ValueError):rejected.append({'path':str(path),'line':number,'reason':'incomplete, contradictory, duplicate, non-independent or unmatched provenance'})
    adjud={}
    if adjudication:
        receipts.append({'path':str(adjudication),'sha256':sha(adjudication)})
        with Path(adjudication).open(encoding='utf-8-sig',newline='') as f:
            for number,r in enumerate(csv.DictReader(f),2):
                if not r.get('adjudicator_id','').strip():continue
                try:
                    i=r['source_image_id'];c=byid[i];assert len(reviews[i])==2;assert r['adjudicator_id'].strip() not in reviews[i]
                    assert r['edited_sha256']==c['edited_sha256'];assert r['semantic_label'] in {'valid','invalid','ambiguous'};assert r['reference_answer_valid'] in {'yes','no','ambiguous'};assert r['reason'].strip()
                    assert datetime.fromisoformat(r['reviewed_utc'].replace('Z','+00:00')).tzinfo is not None
                    if r['semantic_label']=='valid':assert r['reference_answer_valid']=='yes'
                    assert i not in adjud;adjud[i]=r
                except (AssertionError,KeyError,ValueError):rejected.append({'path':str(adjudication),'line':number,'reason':'invalid adjudication provenance or incomplete independent pair'})
    labels={};pairs=[];disagreements=[]
    for i,rr in reviews.items():
        if len(rr)!=2:continue
        a,b=rr.values();pairs.append((a['semantic_label'],b['semantic_label']))
        if i in adjud:labels[i]=adjud[i]['semantic_label']
        elif a['semantic_label']==b['semantic_label'] and a['reference_answer_valid']==b['reference_answer_valid']:labels[i]=a['semantic_label']
        else:disagreements.append(i)
    n=len(pairs);agreement=sum(a==b for a,b in pairs)/n if n else None;kappa=None
    if n:
        left=Counter(a for a,b in pairs);right=Counter(b for a,b in pairs);pe=sum(left[k]*right[k] for k in {'valid','invalid','ambiguous'})/n**2
        if pe<1:kappa=(agreement-pe)/(1-pe)
    return {'accepted_records':sum(map(len,reviews.values())),'images_with_two_reviews':n,'final_labels':labels,'agreement':agreement,'cohen_kappa':kappa,'kappa_note':'undefined if no paired reviews or one constant marginal category','unresolved_disagreements':disagreements,'rejected_records':rejected,'input_receipts':receipts,'provenance_limit':'Human independence is attested, not biometric or externally authenticated.'},reviews
def summarize(matrix,categories,mask,seed):
    x=matrix[mask];cat=np.array(categories)[mask];n=len(x)
    if not n:return {'status':'BLOCKED','n':0,'pairs':[],'conditions':[],'categories':[]}
    rng=np.random.default_rng(seed);indices=rng.integers(0,n,(10000,n));boot=x[indices].mean(1);pairs=[]
    for j,k,name in [(4,0,'C4-C0'),(4,5,'C4-C5')]:
        d=x[:,j]-x[:,k];bd=boot[:,j]-boot[:,k];nz=d[np.abs(d)>1e-12]
        p=float((1+(np.abs((rng.choice([-1.,1.],(10000,len(nz)))*nz).sum(1)/n)>=abs(d.mean())-1e-12).sum())/10001) if len(nz) else 1.
        pairs.append({'comparison':name,'mean_difference':float(d.mean()),'ci95_low':float(np.quantile(bd,.025)),'ci95_high':float(np.quantile(bd,.975)),'wins':float((d>0).mean()),'ties':float((d==0).mean()),'losses':float((d<0).mean()),'permutation_p':p})
    holm(pairs)
    conditions=[{'condition':f'C{j}','mean':float(x[:,j].mean()),'ci95_low':float(np.quantile(boot[:,j],.025)),'ci95_high':float(np.quantile(boot[:,j],.975)),'at_zero':float((x[:,j]==0).mean()),'at_one':float((x[:,j]==1).mean())} for j in range(6)]
    groups=np.var(x[:,:5],axis=1,ddof=1)
    strata=[]
    for c in sorted(set(cat)):
        sub=x[cat==c];bs=sub[rng.integers(0,len(sub),(10000,len(sub)))].mean(1)
        for j,k,name in [(4,0,'C4-C0'),(4,5,'C4-C5')]:
            d=sub[:,j]-sub[:,k];bd=bs[:,j]-bs[:,k];nz=d[np.abs(d)>1e-12]
            p=float((1+(np.abs((rng.choice([-1.,1.],(10000,len(nz)))*nz).sum(1)/len(sub))>=abs(d.mean())-1e-12).sum())/10001) if len(nz) else 1.
            strata.append({'category':c,'n':len(sub),'comparison':name,'mean_difference':float(d.mean()),'ci95_low':float(np.quantile(bd,.025)),'ci95_high':float(np.quantile(bd,.975)),'permutation_p':p,'wins':float((d>0).mean()),'ties':float((d==0).mean()),'losses':float((d<0).mean())})
    holm(strata)
    return {'status':'PASS','n':n,'pairs':pairs,'conditions':conditions,'categories':strata,'all_candidate_tie_rate':float((groups<1e-24).mean()),'nondegenerate_fraction':float((groups>=1e-24).mean()),'within_image_reward_variance_mean':float(groups.mean()),'inference_note':'Inherited original-reference agreement. Group diagnostics cover C0-C4, C5 is auxiliary. C5 semantic validity itself has not been dual-reviewed.'}
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--review-csv',action='append',default=[]);p.add_argument('--adjudication-csv');p.add_argument('--analysis-id',default='analysis_initial');a=p.parse_args()
    run=ROOT/'results'/a.run_id;out=run/a.analysis_id;out.mkdir(exist_ok=False);old=ROOT/'results/matched_feedback_20261028_v2'
    cfg=json.loads((run/'protocol.json').read_text());cases=json.loads((run/'semantic_review/blinded_manifest.json').read_text())['cases'];review,reviews=validated_reviews(a.review_csv,cases,a.adjudication_csv)
    source=json.loads((old/'main_manifest.json').read_text())['records'];cand=json.loads((old/'main_repair_v1/candidate_manifest.json').read_text())['records'];raw=[json.loads(l) for l in (old/'main_repair_v1/predictions.jsonl').read_text().splitlines()];key={(r['source_image_id'],r['candidate_id']):r for r in raw if r['evaluation_seed']==42}
    assert len(key)==960;assert len({(r['source_image_id'],r['candidate_id'],r['evaluation_seed']) for r in raw})==len(raw)==1032
    from src.evaluation.vqa_official_compatible import score_answer
    for r in raw:assert r['execution_status']=='PASS' and abs(score_answer(r['raw_vlm_answer'],{'answers':r['reference_answers']})['score']-r['official_vqa_consensus_score'])<1e-12
    ids=[r['source_image_id'] for r in source];categories=[r['answer_type'] for r in source];matrix=np.array([[key[i,f'C{j}']['all_attempted_utility_score'] for j in range(6)] for i in ids]);labels=review['final_labels']
    risks={c['source_image_id']:any(v for k,v in c['diagnostics'].items() if k.startswith('risk_')) for c in cand if c['candidate_id']=='C4'}
    masks={'A_all_attempted':np.ones(160,dtype=bool),'B_human_valid':np.array([labels.get(i)=='valid' for i in ids]),'C_human_invalid_ambiguous':np.array([labels.get(i) in ['invalid','ambiguous'] for i in ids]),'auto_flagged_descriptive':np.array([risks[i] for i in ids]),'auto_unflagged_descriptive':np.array([not risks[i] for i in ids])}
    result={name:summarize(matrix,categories,mask,cfg['analysis_seed']+j) for j,(name,mask) in enumerate(masks.items())}
    for name,value in result.items():
        value.update(selection_fraction=value['n']/160,selection_bias='Post-edit subgroup selection; effects not randomized between subgroups' if name!='A_all_attempted' else 'Balanced high-consensus diagnostic pool, not population representative')
        if name.startswith('auto'):value['semantic_certification']=False
    rows=[{'source_image_id':i,'question_id':source[k]['question_id'],'category':categories[k],'semantic_label':labels.get(i,'pending'),'automated_risk_flag':risks[i],'within_C0_C4_variance':float(np.var(matrix[k,:5],ddof=1)),**{f'C{j}':float(matrix[k,j]) for j in range(6)},'C4_minus_C0':float(matrix[k,4]-matrix[k,0]),'C4_minus_C5':float(matrix[k,4]-matrix[k,5])} for k,i in enumerate(ids)]
    csvwrite(out/'per_image_feedback.csv',rows);csvwrite(out/'pairwise_statistics.csv',[{'analysis':name,'n':x['n'],**r} for name,x in result.items() for r in x['pairs']]);csvwrite(out/'category_statistics.csv',[{'analysis':name,**r} for name,x in result.items() for r in x['categories']])
    repeats=[]
    for s in source:
        if s['repeat_case']:
            for j in range(6):
                rr=[r for r in raw if r['source_image_id']==s['source_image_id'] and r['candidate_id']==f'C{j}'];assert len(rr)==3;repeats.append(len({(r['raw_vlm_answer'],r['official_vqa_consensus_score']) for r in rr})==1)
    failure=Counter(r['failure_reason'] for rr in reviews.values() for r in rr.values() if r['semantic_label']!='valid');dump(out/'failure_category_summary.json',{'status':'PASS' if reviews else 'BLOCKED','counts_per_accepted_review_record':dict(failure),'confirmed_case_labels':dict(Counter(labels.values())),'pending_cases':160-len(labels),'population_failure_prevalence':None if not labels else 'partial coverage; see labels and selection fraction'})
    dump(out/'semantic_review_audit.json',review)
    summary={'run_id':a.run_id,'analyses':result,'bootstrap_resamples':10000,'statistical_unit':'source image; 160 unique; repeated seeds excluded','multiplicity':'Holm on two overall comparisons in each descriptive/secondary analysis; separate six-category-comparison family. Previous planned four-comparison family remains in parent unchanged. Subgroup hypotheses are exploratory.','human_semantic_status':'BLOCKED' if not labels else ('PASS' if len(labels)==160 else 'NOT VALIDATED'),'human_final_case_count':len(labels),'human_semantic_valid_rate':sum(v=='valid' for v in labels.values())/160 if labels else None,'human_semantic_invalid_rate':sum(v=='invalid' for v in labels.values())/160 if labels else None,'human_semantic_ambiguous_rate':sum(v=='ambiguous' for v in labels.values())/160 if labels else None,'repeat_cases':len(repeats),'repeat_exact_fraction':float(np.mean(repeats)),'main_prediction_rows':len(raw),'original_failed_main_preserved':True,'prediction_sha256':sha(old/'main_repair_v1/predictions.jsonl'),'new_model_inference_calls':0,'new_generation_calls':0}
    dump(out/'analysis.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='analyses'},indent=2))
if __name__=='__main__':main()
