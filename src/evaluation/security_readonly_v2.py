"""Aggregate supplied independent observations; never train or generate targets."""
import math
import re
import hashlib
import json
from pathlib import Path

QWEN_REV='89644892e4d85e24eaac8bacfd4f463576704203'

def validate_lineage(node):
    required={'artifact_id','stage','model_id','revision','parent_artifact_id','training_data_sha256','authorization_id','artifact_manifest_sha256'}
    if set(node)!=required or node['model_id']!='Qwen/Qwen3-VL-2B-Instruct':raise ValueError('Incomplete or wrong primary-VLM lineage')
    if not node['artifact_id'] or not re.fullmatch('[0-9a-f]{64}',node['artifact_manifest_sha256']):raise ValueError('Unidentified fixed artifact')
    if node['stage']=='pretrained':
        if node['revision']!=QWEN_REV or node['parent_artifact_id'] is not None or node['training_data_sha256'] is not None:raise ValueError('Pretrained lineage mismatch')
    elif node['stage'] in {'clean_finetuned','authorized_fixed_condition'}:
        if not node['parent_artifact_id'] or not node['authorization_id'] or not re.fullmatch('[0-9a-f]{64}',node['training_data_sha256'] or ''):
            raise ValueError('Post-training lineage requires documented parent/data/authorization')
    else:raise ValueError('Unknown model lineage stage')
    return node

def verify_fixed_artifact(node,manifest_path):
    path=Path(manifest_path).resolve()
    if hashlib.sha256(path.read_bytes()).hexdigest()!=node['artifact_manifest_sha256']:raise ValueError('Fixed-artifact manifest checksum mismatch')
    manifest=json.loads(path.read_text())
    for key in ['artifact_id','stage','model_id','revision','parent_artifact_id','training_data_sha256']:
        if manifest.get(key)!=node[key]:raise ValueError('Lineage manifest association mismatch')
    files=manifest.get('checkpoint_files')
    if not isinstance(files,list) or not files:raise ValueError('No supplied fixed checkpoint evidence')
    for row in files:
        p=(path.parent/row['path']).resolve()
        if not p.is_relative_to(path.parent):raise ValueError('Checkpoint outside supplied fixed-artifact bundle')
        h=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=row['sha256']:raise ValueError('Supplied checkpoint bytes mismatch')
    return True

def aggregate_fixed_observations(node,protocol,rows,artifact_manifest_path=None):
    validate_lineage(node)
    required={'definition_id','evaluation_artifact_id','source','independent_evaluator','fixture_kind'}
    if set(protocol)!=required or protocol['source']!='https://arxiv.org/html/2506.07214' or not protocol['independent_evaluator']:
        raise ValueError('Independent frozen security evaluation definition required')
    if protocol['fixture_kind'] not in {'synthetic_engineering','authorized_fixed_real'}:raise ValueError('Unknown fixture provenance')
    if len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Duplicate observation sample')
    for row in rows:
        if row['model_artifact_id']!=node['artifact_id'] or row['definition_id']!=protocol['definition_id']:raise ValueError('Artifact/definition association mismatch')
        if row['condition'] not in {'clean','SC','SI-V'} or type(row['predefined_target_observed']) is not bool:raise ValueError('Malformed supplied condition/observation')
        if row['phase'] not in {'inference_only','post_training'}:raise ValueError('Missing behavior phase')
        if row['phase']=='post_training' and node['stage']=='pretrained':raise ValueError('Pretrained inference is not a post-training lineage')
        if not row.get('source_image_id'):raise ValueError('Source image units required')
        score=row.get('factual_reference_score')
        if score is not None and (isinstance(score,bool) or not isinstance(score,(int,float)) or not math.isfinite(score) or not 0<=score<=1):raise ValueError('Invalid clean/semantic factual score')
        if score is not None and row.get('factual_reference_verified') is not True:raise ValueError('Unverified edited reference is not accuracy')
    def rate(group):return sum(r['predefined_target_observed'] for r in group)/len(group) if group else None
    def accuracy(group):
        scored=[r for r in group if r.get('factual_reference_score') is not None]
        scores=[r['factual_reference_score'] for r in scored]
        all_sources={r['source_image_id'] for r in group};scored_sources={r['source_image_id'] for r in scored}
        return {'value':sum(scores)/len(scores) if scores else None,'denominator':len(scores),
            'available_observations':len(group),'missing_verified_score_count':len(group)-len(scores),
            'available_unique_source_images':len(all_sources),'scored_unique_source_images':len(scored_sources),
            'unique_source_images_without_verified_score':len(all_sources-scored_sources)}
    populations={}
    for phase in ['inference_only','post_training']:
        group=[r for r in rows if r['phase']==phase]
        clean=[r for r in group if r['condition']=='clean'];sc=[r for r in group if r['condition']=='SC'];si=[r for r in group if r['condition']=='SI-V']
        populations[phase]={'clean_accuracy':accuracy(clean),'semantic_consistency_accuracy':accuracy(sc),
           'supplied_conditioned_behavior_rate':rate(si),'conditioned_behavior_denominator':len(si),
           'conditioned_unique_source_images':len({r['source_image_id'] for r in si}),
           'SC_false_activation_rate':rate(sc),'SC_false_activation_denominator':len(sc),
           'clean_false_activation_rate':rate(clean),'clean_false_activation_denominator':len(clean)}
    post=[r for r in rows if r['condition']=='SI-V' and r['phase']=='post_training']
    real=protocol['fixture_kind']=='authorized_fixed_real' and node['stage']=='authorized_fixed_condition' and bool(post)
    if real:
        if artifact_manifest_path is None:raise ValueError('Real post-training assessment requires the supplied immutable artifact bundle')
        verify_fixed_artifact(node,artifact_manifest_path)
    return {'schema_version':'readonly_security_v2','model_artifact_id':node['artifact_id'],'definition':protocol,
        'by_phase':populations,
        'post_training_ASR':rate(post) if real else None,'post_training_ASR_denominator':len(post) if real else 0,
        'real_security_result':bool(real and post),'input_observations_only':True,
        'scope':'Frequency of independently supplied predefined-target observations; no target or training data generated'}
