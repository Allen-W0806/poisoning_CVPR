"""Strict, explicit validity eligibility around the historical semantic proxy."""
import math
from .semantic_validation import semantic_decision

def validate_validity(value):
    required={'status','verified','score','provisionally_valid','source'}
    if set(value)!=required:raise ValueError('Missing/extra validity field')
    if value['status'] not in {'VALID','INVALID','UNKNOWN','DISPUTED'}:raise ValueError('Invalid validity status')
    if type(value['verified']) is not bool or type(value['provisionally_valid']) is not bool:raise ValueError('Ambiguous validity flags')
    if value['status']=='VALID' and not value['verified']:raise ValueError('Unverified edits must remain UNKNOWN')
    if value['verified'] and value['status'] not in {'VALID','INVALID'}:raise ValueError('Verified uncertain status')
    if value['status'] in {'INVALID','DISPUTED'} and value['provisionally_valid']:raise ValueError('Contradictory validity')
    if value['verified'] and value['provisionally_valid']:raise ValueError('Verified and provisional are separate')
    if not isinstance(value['source'],str) or not value['source']:raise ValueError('Validation source required')
    score=value['score']
    if isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1:raise ValueError('Malformed validity score')
    if value['status']=='VALID' and score!=1:raise ValueError('Verified valid score must be one')
    if value['status']=='INVALID' and score!=0:raise ValueError('Invalid score must be zero')
    if value['provisionally_valid'] and (value['status']!='UNKNOWN' or score!=.5):raise ValueError('Invalid provisional compatibility score')
    return value

def decision_v2(original,edited,record,consistency,judgment):
    identity=original.size==edited.size and original.tobytes()==edited.tobytes()
    if not identity:
        keys=['object_identity','color','count','spatial','attributes','answerable','answer_preserved']
        if judgment.get('parse_ok') is not True or any(type(judgment.get(k)) is not bool for k in keys):
            raise ValueError('Malformed/ambiguous automatic validation, no reward assigned')
    old=semantic_decision(original,edited,record,consistency,judgment)
    value={'status':('VALID' if old['valid'] else 'INVALID') if old['verified'] else 'UNKNOWN',
        'verified':old['verified'],'score':old['score'],'provisionally_valid':old.get('provisionally_valid',False),
        'source':old['verifier']}
    return validate_validity(value)

def consistency_eligible(value):
    validate_validity(value)
    return (value['status']=='VALID' and value['verified']) or value['provisionally_valid']
