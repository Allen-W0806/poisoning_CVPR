"""Neutral factual metadata. Never creates edited answers or target labels."""
import copy
import re

SCHEMA='semantic_image_record_v2'
STATUSES={'UNKNOWN','VERIFIED','DISPUTED'}
CATEGORIES={'color','animal','vehicle','food','other','UNKNOWN'}
TERMS={'color':'red green blue yellow purple pink brown black white',
       'animal':'cat dog cow sheep horse bird',
       'vehicle':'car bus truck motorcycle bicycle train boat plane',
       'food':'pizza cake donut cookie burger sandwich salad'}

def candidate_categories(question):
    words=set(re.findall(r'\b\w+\b',question.lower()))
    return [k for k,v in TERMS.items() if words&set(v.split())]

def unknown():return {'status':'UNKNOWN','value':None,'source':None}

def from_original(row):
    value={'schema_version':SCHEMA,'sample_id':str(row['sample_id']),'split':row['split'],
        'original':{'image_id':str(row['image_id']),'sha256':row['image_sha256'],
                    'image_path':row['image_path'],'question':row['question'],'dataset_answers':list(row['answers'])},
        'edited':None,'edited_factual_answer':unknown(),
        'supplied_training_annotation':{'status':'NOT_PROVIDED','value':None,'authorization_id':None},
        'semantic_category':'UNKNOWN','candidate_semantic_categories':candidate_categories(row['question']),
        'image_question_consistency':unknown(),'visual_edit_validity':unknown(),
        'verification':unknown(),'model_edit_provenance':{}}
    validate(value)
    return value

def _evidence(value,name,allowed=None):
    if set(value)!={'status','value','source'} or value['status'] not in STATUSES:raise ValueError(name+' malformed evidence')
    if value['status']=='UNKNOWN':
        if value['value'] is not None or value['source'] is not None:raise ValueError(name+' UNKNOWN must not invent a value')
    elif not isinstance(value['source'],dict) or not value['source'].get('kind') or not value['source'].get('artifact'):
        raise ValueError(name+' requires verification provenance')
    if value['status']=='VERIFIED' and allowed is not None and value['value'] not in allowed:
        raise ValueError(name+' invalid verified category')

def validate(value):
    required={'schema_version','sample_id','split','original','edited','edited_factual_answer',
        'supplied_training_annotation','semantic_category','candidate_semantic_categories',
        'image_question_consistency','visual_edit_validity','verification','model_edit_provenance'}
    if set(value)!=required or value['schema_version']!=SCHEMA:raise ValueError('Semantic contract schema mismatch')
    if value['split'] not in {'train','development','gate','test'}:raise ValueError('Invalid v2 split')
    o=value['original']
    if set(o)!={'image_id','sha256','image_path','question','dataset_answers'}:raise ValueError('Incomplete original factual provenance')
    if not re.fullmatch('[0-9a-f]{64}',o['sha256']) or not o['image_id'] or not o['question']:raise ValueError('Invalid original identity')
    if not isinstance(o['dataset_answers'],list) or len(o['dataset_answers'])!=10 or any(not isinstance(a,str) or not a for a in o['dataset_answers']):
        raise ValueError('Original annotator answers required, not a model target')
    if value['edited'] is not None:
        e=value['edited']
        if set(e)!={'image_id','sha256','image_path'} or not re.fullmatch('[0-9a-f]{64}',e['sha256']):raise ValueError('Invalid edited provenance')
    _evidence(value['edited_factual_answer'],'edited_factual_answer')
    answer=value['edited_factual_answer']
    if answer['status']=='DISPUTED' and answer['value'] is not None:
        raise ValueError('Disputed factual answers must remain UNKNOWN; preserve alternatives in the source artifact')
    if answer['status']=='VERIFIED' and (value['edited'] is None or not isinstance(answer['value'],list) or not answer['value']):
        raise ValueError('Verified edited answers require an edited image and supplied evidence')
    if answer['status']=='VERIFIED' and any(not isinstance(a,str) or not a.strip() for a in answer['value']):
        raise ValueError('Verified edited factual answers must contain nonempty answer strings')
    _evidence(value['image_question_consistency'],'consistency',{'SC','SI-V'})
    _evidence(value['visual_edit_validity'],'validity',{'VALID','INVALID'})
    _evidence(value['verification'],'verification')
    if value['semantic_category'] not in CATEGORIES or any(x not in CATEGORIES-{'UNKNOWN'} for x in value['candidate_semantic_categories']):
        raise ValueError('Invalid semantic category')
    annotation=value['supplied_training_annotation']
    if set(annotation)!={'status','value','authorization_id'}:raise ValueError('Invalid supplied annotation')
    if annotation['status']=='NOT_PROVIDED':
        if annotation['value'] is not None or annotation['authorization_id'] is not None:raise ValueError('Do not invent training annotations')
    elif annotation['status']!='SUPPLIED' or not annotation['authorization_id'] or annotation['value'] is None:
        raise ValueError('An authorized source must supply the training annotation')
    return copy.deepcopy(value)

def eligible_for_semantic_study(value):
    validate(value)
    return bool(value['edited'] is not None and value['semantic_category']!='UNKNOWN'
        and value['image_question_consistency']['status']=='VERIFIED'
        and value['edited_factual_answer']['status']=='VERIFIED'
        and value['visual_edit_validity']['status']=='VERIFIED' and value['visual_edit_validity']['value']=='VALID'
        and value['verification']['status']=='VERIFIED')
