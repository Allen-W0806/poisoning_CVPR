"""Paired image-clustered analysis; no editing of raw predictions/manifests.
Primary uncertainty: 10,000 paired image bootstrap resamples. Semantic-valid
secondary analysis is blocked unless independent reviews are actually supplied.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import textwrap
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np


def csvwrite(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def holm(rows,raw='permutation_p',output='holm_p'):
    ordered=sorted(range(len(rows)),key=lambda i:rows[i][raw]);previous=0.
    for rank,i in enumerate(ordered):
        previous=max(previous,min(1.,rows[i][raw]*(len(rows)-rank)));rows[i][output]=previous


def analyze_phase(run,phase,cfg):
    path=run/phase
    revisions=json.loads((run/'accepted_phase_revisions.json').read_text()) if (run/'accepted_phase_revisions.json').exists() else {}
    input_path=run/revisions.get(phase,phase)
    raw=[json.loads(x) for x in (input_path/'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest=json.loads((input_path/'candidate_manifest.json').read_text());records=manifest['records']
    source=json.loads((run/f'{phase}_manifest.json').read_text())['records']
    primary=[r for r in raw if r['evaluation_seed']==42]
    conditions=cfg['primary_conditions']+cfg['auxiliary_conditions'];ids=[r['source_image_id'] for r in source]
    key={(r['source_image_id'],r['candidate_id']):r for r in primary}
    assert len(primary)==len(source)*len(conditions)==len(key)
    expected={(r['source_image_id'],c,seed) for r in source for c in conditions for seed in cfg['evaluation_seeds'] if seed==42 or r['repeat_case']}
    actual={(r['source_image_id'],r['candidate_id'],r['evaluation_seed']) for r in raw}
    assert len(raw)==len(actual) and actual==expected
    from src.evaluation.vqa_official_compatible import score_answer
    for r in raw:
        if r['execution_status']=='PASS':
            assert abs(score_answer(r['raw_vlm_answer'],{'answers':r['reference_answers']})['score']-r['official_vqa_consensus_score'])<1e-12
    matrix=np.array([[key[(i,c)]['all_attempted_utility_score'] for c in conditions] for i in ids]);n=len(ids)
    rng=np.random.default_rng(cfg['analysis_seed']+(0 if phase=='pilot' else 1));boot=rng.integers(0,n,(10000,n))
    resampled=matrix[boot].mean(1);means=matrix.mean(0);ci=np.quantile(resampled,[.025,.975],axis=0)
    condition_stats=[]
    for j,c in enumerate(conditions):
        diffs=matrix[:,j]-matrix[:,0];bd=resampled[:,j]-resampled[:,0]
        invalid=sum(key[(i,c)]['execution_status']!='PASS' for i in ids)
        condition_stats.append({'candidate_id':c,'n_unique_sources':n,'role':'primary' if c!='C5' else 'auxiliary geometry control',
            'mean_score_all_attempted_utility':float(means[j]),'mean_ci95_low':float(ci[0,j]),'mean_ci95_high':float(ci[1,j]),
            'paired_difference_C0':float(diffs.mean()),'paired_ci95_low':float(np.quantile(bd,.025)),'paired_ci95_high':float(np.quantile(bd,.975)),
            'score_variance':float(matrix[:,j].var(ddof=1)),'at_zero_fraction':float((matrix[:,j]==0).mean()),'at_one_fraction':float((matrix[:,j]==1).mean()),
            'invalid_records':invalid,'missing_records':0,'wins_C0':float((diffs>0).mean()),'ties_C0':float((diffs==0).mean()),'losses_C0':float((diffs<0).mean()),
            'score_interpretation':'inherited-reference agreement; semantic validity not certified' if c not in ['C0','C1'] else 'official VQA consensus'})
    def permutation(difference):
        nz=difference[np.abs(difference)>1e-12]
        if not len(nz):return 1.
        observed=abs(float(difference.mean()))
        signs=rng.choice([-1.,1.],(10000,len(nz)));values=np.abs((signs*nz).sum(1)/n)
        return float((1+np.sum(values>=observed-1e-12))/10001)
    pairwise=[]
    for j,c in enumerate(cfg['primary_conditions']):
        for k,d in enumerate(cfg['primary_conditions'][:j]):
            diff=matrix[:,j]-matrix[:,k];bd=resampled[:,j]-resampled[:,k]
            pairwise.append({'candidate_a':c,'candidate_b':d,'mean_difference':float(diff.mean()),
                'ci95_low':float(np.quantile(bd,.025)),'ci95_high':float(np.quantile(bd,.975)),
                'wins':float((diff>0).mean()),'ties':float((diff==0).mean()),'losses':float((diff<0).mean()),'permutation_p':permutation(diff)})
    holm(pairwise)
    planned=[dict(r) for r in pairwise if r['candidate_b']=='C0'];holm(planned,output='primary_family_holm_p')
    auxdiff=matrix[:,4]-matrix[:,5];auxboot=resampled[:,4]-resampled[:,5]
    auxiliary={'comparison':'C4-C5','mean_difference':float(auxdiff.mean()),'ci95':[float(v) for v in np.quantile(auxboot,[.025,.975])],
        'permutation_p':permutation(auxdiff),'interpretation':'generation beyond geometry preprocessing, original inherited references; not semantic certification'}
    group=[]
    for index,r in enumerate(source):
        values=matrix[index,:5];group.append({'phase':phase,'source_image_id':r['source_image_id'],'question_id':r['question_id'],'category':r['answer_type'],
            'candidate_reward_variance':float(values.var(ddof=1)),'all_candidates_tied':bool(np.ptp(values)==0),'unique_candidate_rewards':len(set(values)),
            'saturated_zero_candidates':int((values==0).sum()),'saturated_one_candidates':int((values==1).sum())})
    category=[]
    for cat in ['yes/no','number','other']:
        indices=[i for i,r in enumerate(source) if r['answer_type']==cat];v=matrix[indices]
        # Resample images jointly across candidates within each category.
        b=v[rng.integers(0,len(v),(10000,len(v)))].mean(1)
        for j,c in enumerate(conditions):
            diff=v[:,j]-v[:,0];interval=np.quantile(b[:,j]-b[:,0],[.025,.975])
            category.append({'category':cat,'candidate_id':c,'n':len(indices),'mean_score':float(v[:,j].mean()),'paired_difference_C0':float(diff.mean()),
                'paired_ci95_low':float(interval[0]),'paired_ci95_high':float(interval[1])})
    repeats=[]
    for i in ids:
        for c in conditions:
            rr=[r for r in raw if r['source_image_id']==i and r['candidate_id']==c]
            if len(rr)>1:repeats.append(len({r['raw_vlm_answer'] for r in rr})==1 and len({r['official_vqa_consensus_score'] for r in rr})==1)
    stats={'phase':phase,'status':'PASS','input_revision':input_path.name,'original_failed_attempt_retained':input_path!=path,'n_unique_sources':n,'n_primary_candidate_conditions':5,'auxiliary_condition':'C5',
        'n_primary_records':n*5,'n_all_records_including_auxiliary_and_seed_repeats':len(raw),'effective_sample_size':n,
        'conditions':condition_stats,'planned_comparisons':planned,'exploratory_pairwise':pairwise,'auxiliary_geometry_comparison':auxiliary,
        'group_statistics':{'mean_within_source_variance':float(np.mean([g['candidate_reward_variance'] for g in group])),
            'all_candidates_tied_fraction':sum(g['all_candidates_tied'] for g in group)/n,'nondegenerate_fraction':sum(not g['all_candidates_tied'] for g in group)/n},
        'repeated_candidate_cases':len(repeats),'repeat_exact_fraction':sum(repeats)/len(repeats),
        'semantic_validated_secondary':{'status':'BLOCKED','verified_independent_human_C4_cases':0,'matched_all_condition_valid_cases':0,'paired_effect':None,
            'reason':'No independent human review supplied. All edits retained. Identity control exact by construction; automated indicators do not certify C2/C3/C4/C5.'},
        'source_identities_reconciled':True,'statistical_unit':'source image','CI_method':'10,000 paired image bootstrap resamples; repeated seeds excluded',
        'zero_technical_failure_policy':'all-attempted utility zero; raw official consensus remains missing for failed inference/generation',
        'interpretation_limit':'C2/C3/C4 scores use inherited original references. Candidate agreement differences alone do not prove useful semantic-preserving feedback or generator learning.'}
    dump(path/'paired_statistics.json',stats);csvwrite(path/'paired_feedback.csv',raw);csvwrite(path/'candidate_group_statistics.csv',group)
    distribution=[{'phase':phase,'candidate_id':c,'score':float(score),'count':int(count),'fraction':count/n} for j,c in enumerate(conditions) for score,count in sorted(Counter(matrix[:,j]).items())]
    csvwrite(path/'feedback_distribution.csv',distribution);csvwrite(path/'category_statistics.csv',category)
    csvwrite(path/'condition_statistics.csv',condition_stats);csvwrite(path/'pairwise_statistics.csv',pairwise)
    # Queue contains every attempted non-original candidate, including failures.
    queue=[]
    for c in records:
        if c['candidate_id']=='C0':continue
        prediction=key[(c['source_image_id'],c['candidate_id'])];original=key[(c['source_image_id'],'C0')]
        queue.append({'phase':phase,'source_image_id':c['source_image_id'],'question_id':c['question_id'],'candidate_id':c['candidate_id'],
            'original_path':next(r['image_path'] for r in source if r['source_image_id']==c['source_image_id']),
            'edited_path':c['candidate_path'],'original_hash':c['original_image_hash'],'candidate_hash':c['candidate_image_hash'],
            'question':c['question'],'original_reference_answers':json.dumps(c['answers']),'instruction':cfg['editor']['instruction'] if c['candidate_id']=='C4' else c['transformation_type'],
            'original_model_answer':original['raw_vlm_answer'],'edited_model_answer':prediction['raw_vlm_answer'],
            'automated_diagnostics':json.dumps(c['diagnostics']),'technical_status':c['generation_status'],
            'review_status':'PASS' if c['candidate_id']=='C1' else 'NOT VALIDATED','review_provenance':'pixel-exact identity' if c['candidate_id']=='C1' else 'awaiting independent human review',
            'independent_reviewer_id':'','review_timestamp':'',
            **{name:'' for name in ['scene_identity','object_existence','object_count','object_identity','object_colors','spatial_relationships','text_symbols','VQA_answer_preserved','instruction_compliance']}})
    csvwrite(path/'semantic_review_queue.csv',queue)
    c4=[c for c in records if c['candidate_id']=='C4'];valid=sum(c['generation_status']=='PASS' for c in c4)
    flags=Counter(k for c in c4 for k,v in c['diagnostics'].items() if k.startswith('risk_') and v)
    anyflag=sum(any(v for k,v in c['diagnostics'].items() if k.startswith('risk_')) for c in c4)
    qachanges=sum(key[(i,'C4')]['normalized_answer']!=key[(i,'C0')]['normalized_answer'] for i in ids)
    audit={'phase':phase,'attempted_C4_edits':n,'technically_valid_C4_edits':valid,'technical_validity_rate':valid/n,'technical_invalid_count':n-valid,
        'independent_human_reviewed_count':0,'confirmed_semantic_preservation_rate':None,'confirmed_semantic_preservation_status':'BLOCKED',
        'unverified_C4_count':n,'human_confirmed_invalid_count':None,'human_confirmed_ambiguous_count':None,
        'automated_risk_flag_counts':dict(flags),'automated_any_risk_count':anyflag,'automated_any_risk_rate':anyflag/n,
        'original_edited_normalized_answer_change_count':qachanges,'answer_agreement_rate':1-qachanges/n,
        'scope':'Risk flags and target-answer agreement are not certified semantics or failure-rate ground truth. Human preservation rate unidentifiable without reviews.',
        'risk_flag_thresholds':cfg['risk_flags_only'],'all_outputs_retained':True}
    dump(path/'semantic_audit.json',audit)
    return stats,audit,raw,group,distribution,queue


def plots(run,phase,stats):
    import matplotlib
    matplotlib.use('Agg');import matplotlib.pyplot as plt
    target=run/phase/'plots';target.mkdir(exist_ok=True);plt.rcParams.update({'font.size':11,'figure.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    with (run/phase/'feedback_distribution.csv').open() as f:d=list(csv.DictReader(f))
    cstats=stats['conditions'][:5];labels=[r['candidate_id'] for r in cstats];x=np.arange(5)
    def save(fig,name):fig.tight_layout();fig.savefig(target/(name+'.png'));fig.savefig(target/(name+'.svg'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    vals=[[float(r['score']) for r in d if r['candidate_id']==c for _ in range(int(r['count']))] for c in labels]
    ax.boxplot(vals,tick_labels=labels,showmeans=True);ax.set(xlabel='Matched candidate condition',ylabel='Inherited-reference VQA consensus score',ylim=(-.05,1.05));save(fig,'condition_distributions')
    fig,ax=plt.subplots(figsize=(8,4));m=np.array([r['paired_difference_C0'] for r in cstats]);lo=np.array([r['paired_ci95_low'] for r in cstats]);hi=np.array([r['paired_ci95_high'] for r in cstats])
    ax.errorbar(x,m,yerr=[m-lo,hi-m],fmt='o',capsize=5);ax.axhline(0,color='gray',ls='--');ax.set(xticks=x,xticklabels=labels,ylabel='Paired score difference from C0',xlabel='10,000 image-cluster bootstrap resamples');save(fig,'paired_differences_CI')
    fig,ax=plt.subplots(figsize=(8,4));means=np.array([r['mean_score_all_attempted_utility'] for r in cstats]);low=np.array([r['mean_ci95_low'] for r in cstats]);high=np.array([r['mean_ci95_high'] for r in cstats])
    ax.errorbar(x,means,yerr=[means-low,high-means],fmt='o',capsize=5);ax.set(xticks=x,xticklabels=labels,ylabel='Condition mean with paired-bootstrap CI',ylim=(0,1.05));save(fig,'bootstrap_condition_CI')
    with (run/phase/'candidate_group_statistics.csv').open() as f:g=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4));ax.hist([float(r['candidate_reward_variance']) for r in g],bins=15,edgecolor='white');ax.set(xlabel='Within-source variance across C0–C4',ylabel='Unique source images');save(fig,'candidate_reward_variance')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(x-.18,[r['at_zero_fraction'] for r in cstats],.36,label='Score 0');ax.bar(x+.18,[r['at_one_fraction'] for r in cstats],.36,label='Score 1')
    ax.axhline(stats['group_statistics']['all_candidates_tied_fraction'],ls='--',color='black',label='All-candidate group tie fraction');ax.set(xticks=x,xticklabels=labels,ylabel='Fraction',ylim=(0,1.05));ax.legend();save(fig,'saturation_ties')
    with (run/phase/'category_statistics.csv').open() as f:cats=list(csv.DictReader(f))
    fig,ax=plt.subplots(figsize=(8,4))
    for category in ['yes/no','number','other']:ax.plot(x,[float(next(r['mean_score'] for r in cats if r['candidate_id']==c and r['category']==category)) for c in labels],'o-',label=category)
    ax.set(xticks=x,xticklabels=labels,ylabel='Category-specific reference agreement',ylim=(0,1.05));ax.legend();save(fig,'category_performance')


def contact_sheets(run,phase,cfg):
    from PIL import Image,ImageDraw,ImageFont,ImageOps
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    except OSError:font=ImageFont.load_default(size=17)
    revisions=json.loads((run/'accepted_phase_revisions.json').read_text()) if (run/'accepted_phase_revisions.json').exists() else {}
    input_path=run/revisions.get(phase,phase)
    source=json.loads((run/f'{phase}_manifest.json').read_text())['records'];candidates=json.loads((input_path/'candidate_manifest.json').read_text())['records']
    raw=[json.loads(x) for x in (input_path/'predictions.jsonl').read_text().splitlines()];answers={(r['source_image_id'],r['candidate_id']):r['raw_vlm_answer'] for r in raw if r['evaluation_seed']==42}
    target=run/phase/'contact_sheets';target.mkdir(exist_ok=True)
    for start in range(0,len(source),4):
        sheet=Image.new('RGB',(1200,4*300),'white');draw=ImageDraw.Draw(sheet)
        for offset,r in enumerate(source[start:start+4]):
            y=offset*300;cond={c['candidate_id']:c for c in candidates if c['source_image_id']==r['source_image_id']}
            text=f"ID {r['source_image_id']} | {r['answer_type']} | Q: {r['question']} | References: {', '.join(r['answers'])}"
            draw.text((8,y+2),textwrap.fill(text,140),font=font,fill='black')
            for j,c in enumerate(['C0','C1','C2','C3','C4','C5']):
                path=ROOT/cond[c]['candidate_path']
                if path.exists():
                    with Image.open(path) as image:thumb=ImageOps.contain(image.convert('RGB'),(196,150))
                    sheet.paste(thumb,(j*200,y+68))
                draw.text((j*200+4,y+218),c+': '+answers.get((r['source_image_id'],c),'<invalid>')[:22],font=font,fill='black')
            diag=cond['C4']['diagnostics'];detail=f"C4 lighting-only; preserve objects/counts/positions/colors. SSIM={diag.get('SSIM',float('nan')):.3f}; luminance delta={diag.get('mean_luminance_difference',float('nan')):.1f}; independent human review pending."
            draw.text((8,y+247),textwrap.fill(detail,140),font=font,fill='black')
        sheet.save(target/f'{phase}_{start//4:03d}.png')
    shutil.copyfile(target/f'{phase}_000.png',run/phase/'editor_contact_sheet.png')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='matched_feedback_20261028_v2');p.add_argument('--phase',choices=['pilot','main','all'],default='all');a=p.parse_args()
    run=ROOT/'results'/a.run_id;cfg=json.loads((run/'protocol.json').read_text());scratch=ROOT/'tmp'/f'{a.run_id}_analysis';scratch.mkdir(exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(scratch/'mpl');combined=[[],[],[],[]];stats={};audits={}
    try:
        for phase in (['pilot','main'] if a.phase=='all' else [a.phase]):
            result=analyze_phase(run,phase,cfg);s,au,raw,group,dist,queue=result;stats[phase]=s;audits[phase]=au
            for container,values in zip(combined,[raw,group,dist,queue]):container.extend(values)
            plots(run,phase,s);contact_sheets(run,phase,cfg)
        for filename,values in zip(['paired_feedback.csv','candidate_group_statistics.csv','feedback_distribution.csv','semantic_review_queue.csv'],combined):csvwrite(run/filename,values)
        all_attempts=[]
        for phase in (['pilot','main'] if a.phase=='all' else [a.phase]):
            for line in (run/phase/'predictions.jsonl').read_text().splitlines():all_attempts.append({'attempt_view':'original_'+phase,**json.loads(line)})
        if (run/'main_repair_v1/repair_predictions.jsonl').exists():
            for line in (run/'main_repair_v1/repair_predictions.jsonl').read_text().splitlines():all_attempts.append({'attempt_view':'engineering_repair_control',**json.loads(line)})
        csvwrite(run/'all_attempts_feedback.csv',all_attempts)
        dump(run/'paired_statistics.json',stats);dump(run/'semantic_audit.json',audits)
        chosen='main' if 'main' in stats else 'pilot';shutil.copyfile(run/chosen/'editor_contact_sheet.png',run/'editor_contact_sheet.png')
        print(json.dumps({phase:{'conditions':s['conditions'],'group_statistics':s['group_statistics'],'planned_comparisons':s['planned_comparisons']} for phase,s in stats.items()},indent=2))
    finally:
        resolved=scratch.resolve()
        if resolved.parent!=(ROOT/'tmp').resolve() or resolved.name!=f'{a.run_id}_analysis':raise RuntimeError('Unsafe cleanup target')
        shutil.rmtree(resolved)


if __name__=='__main__':main()
