"""Independent review metadata/coverage analysis; never invents or releases truth."""
import itertools
import json
import math
import statistics
from collections import Counter,defaultdict
from datetime import datetime
from pathlib import Path
from ..monitoring.common import read_json,now
from ..utils.logging import write_json

DIMENSIONS={
 'instruction_completion':['UNKNOWN','FULFILLED','PARTIAL','NOT_FULFILLED'],
 'image_realism':['UNKNOWN','PLAUSIBLE','IMPLAUSIBLE'],
 'content_preservation':['UNKNOWN','PRESERVED','CHANGED'],
 'question_answer_factual_validity':['UNKNOWN','ANSWERABLE','UNANSWERABLE'],
 'image_question_semantic_relationship':['UNKNOWN','SC','SI-V'],
 'visible_artifacts':['UNKNOWN','NONE','MINOR','MAJOR'],
 'evaluation_uncertainty':['UNKNOWN','LOW','MEDIUM','HIGH']}

def validate_submission(value,manifest):
    if value.get('schema_version')!='independent_edit_review_v1':raise ValueError('Unknown submission schema')
    provenance=value.get('provenance',{})
    required=['reviewer_id','review_session_id','reviewed_at','reviewer_type','independence_attestation','model_assisted']
    if any(k not in provenance for k in required):raise ValueError('Incomplete reviewer provenance')
    if not isinstance(provenance['reviewer_id'],str) or not provenance['reviewer_id'].strip():raise ValueError('Reviewer identity missing')
    if not isinstance(provenance['review_session_id'],str) or not provenance['review_session_id'].strip():raise ValueError('Review session missing')
    time=datetime.fromisoformat(provenance['reviewed_at'].replace('Z','+00:00'))
    if time.tzinfo is None:raise ValueError('Review timestamp must include timezone')
    if provenance['reviewer_type']!='human' or provenance['independence_attestation'] is not True or provenance['model_assisted'] is not False:
        raise ValueError('Independent human provenance not declared; retain as unverified/assisted, not human truth')
    pairs={p['pair_id']:p for p in manifest['pairs']};seen=set()
    if not isinstance(value.get('reviews'),list) or not value['reviews']:raise ValueError('No review rows supplied')
    for row in value['reviews']:
        pair=pairs.get(row.get('pair_id'))
        if not pair or row['pair_id'] in seen:raise ValueError('Unknown/duplicated pair identifier')
        seen.add(row['pair_id'])
        for field in ['original_image_sha256','edited_image_sha256']:
            if row.get(field)!=pair[field]:raise ValueError('Review/image identity mismatch: '+field)
        if set(row.get('dimensions',{}))!=set(DIMENSIONS):raise ValueError('Seven separate assessment dimensions required')
        for name,allowed in DIMENSIONS.items():
            if row['dimensions'][name] not in allowed:raise ValueError('Invalid categorical review: '+name)
        answer=row.get('edited_factual_answer')
        if not isinstance(answer,dict) or answer.get('status') not in {'UNKNOWN','REVIEWED','DISPUTED'}:raise ValueError('Edited answer status required')
        if answer['status'] in {'UNKNOWN','DISPUTED'}:
            if answer.get('value') is not None:raise ValueError('Unknown/disputed edited answer must stay null')
        elif not isinstance(answer.get('value'),list) or not answer['value'] or any(not isinstance(a,str) or not a.strip() for a in answer['value']):
            raise ValueError('Reviewed edited answer requires nonempty strings')
        if not isinstance(row.get('reason'),str) or not row['reason'].strip():raise ValueError('Review reason/uncertainty explanation required')
    return value

def agreement(rows):
    if not rows:return {'n':0,'agreement':None,'cohen_kappa':None,'reason':'no overlapping non-UNKNOWN ratings'}
    left=Counter(a for a,b in rows);right=Counter(b for a,b in rows);n=len(rows)
    observed=sum(a==b for a,b in rows)/n;expected=sum(left[k]*right[k] for k in left.keys()|right.keys())/(n*n)
    return {'n':n,'agreement':observed,'cohen_kappa':None if math.isclose(expected,1.) else (observed-expected)/(1-expected),
      'constant_marginal_kappa_undefined':math.isclose(expected,1.)}

def analyze_submissions(manifest,submissions,operator_attestation=None):
    trusted=operator_attestation or {};approved=set(trusted.get('verified_human_reviewer_ids',[])) if trusted.get('verified_by') and trusted.get('evidence_reference') else set()
    valid=[];errors=[];seen=set()
    for name,value in submissions:
        try:
            validate_submission(value,manifest)
            identity=(value['provenance']['reviewer_id'],value['provenance']['review_session_id'])
            if identity in seen:raise ValueError('Duplicated reviewer/session; no duplicated agreement evidence')
            seen.add(identity);valid.append((name,value))
        except (ValueError,TypeError,KeyError,AttributeError) as error:errors.append({'file':name,'error':str(error)})
    by_reviewer=defaultdict(dict);conflicts=[]
    for name,value in valid:
        rid=value['provenance']['reviewer_id']
        for row in value['reviews']:
            old=by_reviewer[rid].get(row['pair_id'])
            if old and old!=row:
                conflicts.append({'reviewer_id':rid,'pair_id':row['pair_id'],'reason':'conflicting submissions retained; excluded from agreement'})
                by_reviewer[rid][row['pair_id']]=None
            elif row['pair_id'] not in by_reviewer[rid]:by_reviewer[rid][row['pair_id']]=row
    dimensions={};disagreements=[];coverage={}
    for dim in DIMENSIONS:
        comparisons=[]
        for a,b in itertools.combinations(sorted(by_reviewer),2):
            pairs=[]
            for pid in by_reviewer[a].keys()&by_reviewer[b].keys():
                ra,rb=by_reviewer[a][pid],by_reviewer[b][pid]
                if not ra or not rb:continue
                va,vb=ra['dimensions'][dim],rb['dimensions'][dim]
                if va=='UNKNOWN' or vb=='UNKNOWN':continue
                pairs.append((va,vb))
                if va!=vb:disagreements.append({'dimension':dim,'pair_id':pid,'reviewer_ids':[a,b],'values':[va,vb]})
            comparisons.append({'reviewers':[a,b],**agreement(pairs),'operator_verified_reviewers':a in approved and b in approved})
        dimensions[dim]=comparisons
        coverage[dim]={'unique_rated_pairs':len({pid for rows in by_reviewer.values() for pid,r in rows.items() if r and r['dimensions'][dim]!='UNKNOWN'}),'available_pairs':len(manifest['pairs'])}
    for a,b in itertools.combinations(sorted(by_reviewer),2):
        for pid in by_reviewer[a].keys()&by_reviewer[b].keys():
            ra,rb=by_reviewer[a][pid],by_reviewer[b][pid]
            if not ra or not rb or any(r['edited_factual_answer']['status']!='REVIEWED' for r in [ra,rb]):continue
            if {x.strip().casefold() for x in ra['edited_factual_answer']['value']}!={x.strip().casefold() for x in rb['edited_factual_answer']['value']}:
                disagreements.append({'dimension':'edited_factual_answer','pair_id':pid,'reviewer_ids':[a,b],
                  'values':[ra['edited_factual_answer']['value'],rb['edited_factual_answer']['value']]})
    verified_pairs=[];proxy_groups=defaultdict(lambda:defaultdict(list));answer_agreement=[]
    from ..evaluation.vqa_official_compatible import normalize_answer
    for pair in manifest['pairs']:
        ratings=[(rid,rows[pair['pair_id']]) for rid,rows in by_reviewer.items() if rid in approved and rows.get(pair['pair_id'])]
        complete=[(rid,row) for rid,row in ratings if all(v!='UNKNOWN' for v in row['dimensions'].values()) and row['edited_factual_answer']['status']=='REVIEWED']
        signatures=[(r['dimensions'],{normalize_answer(a) for a in r['edited_factual_answer']['value']}) for rid,r in complete]
        if len(complete)>=2 and all(s==signatures[0] for s in signatures):verified_pairs.append(pair['pair_id'])
        for rid,row in ratings:
            for dim,label in row['dimensions'].items():
                if label=='UNKNOWN':continue
                for metric,value in pair['automatic_metrics'].items():
                    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):proxy_groups[dim+' / '+label][metric].append(value)
            if row['edited_factual_answer']['status']=='REVIEWED':
                automatic=pair.get('automatic_edited_answer')
                if isinstance(automatic,str):answer_agreement.append(normalize_answer(automatic) in {normalize_answer(a) for a in row['edited_factual_answer']['value']})
    return {'updated_utc':now(),'submission_files':len(submissions),'provenance_consistent_submissions':len(valid),'validation_errors':errors,
      'operator_attestation_present':bool(approved),'verified_reviewer_ids':sorted(approved),'reviewer_conflicts':conflicts,
      'independently_reviewed_consensus_pairs':len(verified_pairs),'consensus_pair_ids':verified_pairs,
      'available_pairs':len(manifest['pairs']),'dimension_coverage':coverage,'reviewer_pair_agreement':dimensions,
      'disagreements':disagreements,'proxy_group_summaries':{g:{m:{'n_ratings':len(v),'mean':statistics.mean(v),'min':min(v),'max':max(v)} for m,v in cols.items()} for g,cols in proxy_groups.items()},
      'proxy_answer_match':{'numerator':sum(answer_agreement),'denominator':len(answer_agreement),'rate':statistics.mean(answer_agreement) if answer_agreement else None,'scope':'normalized match to reviewed edited answers, not official10-answer VQA accuracy'},
      'analysis_unit':'prioritized unique source pair; repeated ratings retained separately, not additional independent images',
      'training_reward_released':False,'scientific_pilot_ready':False,
      'scope':'metadata and agreement do not independently prove human authorship or establish representative semantic efficacy'}

def refresh_review_analysis(root):
    root=Path(root);manifest=read_json(root/'reviews/workflow/manifest.json')
    if manifest is None:return {'status':'WORKFLOW_NOT_PREPARED','independently_reviewed_consensus_pairs':0}
    submissions=[];load_errors=[]
    for p in sorted((root/'reviews/incoming').glob('*.json')):
        try:submissions.append((p.name,read_json(p)))
        except (ValueError,OSError) as error:load_errors.append({'file':p.name,'error':str(error)})
    result=analyze_submissions(manifest,submissions,read_json(root/'reviews/reviewer_provenance.json',{}))
    result['validation_errors']+=load_errors
    write_json(root/'reports/phase6_followup/review_analysis.json',result)
    return result
