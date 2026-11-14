"""Content groups and neutral metadata, with immutable v1 held-out membership."""
import copy
import hashlib
from collections import Counter
from .semantic_contract_v2 import from_original,eligible_for_semantic_study
from .dataset_v2 import validate_content_disjoint

def prepare_v2(old,known_development_images=()):
    rows=[copy.deepcopy(r) for split in ['train','validation','test'] for r in old[split]]
    by_image={}
    for r in rows:
        if r['image_id'] in by_image and by_image[r['image_id']]!=r['image_sha256']:raise ValueError('One image ID has conflicting content hashes')
        by_image[r['image_id']]=r['image_sha256']
    groups={}
    for row in rows:groups.setdefault(row['image_sha256'],[]).append(row)
    allocation={};legacy_conflicts=[];devgroups=[]
    for digest,group in groups.items():
        splits={r['split'] for r in group}
        if 'test' in splits:
            allocation[digest]='test'
            if len(splits)>1:legacy_conflicts.append({'sha256':digest,'source_splits':sorted(splits),'policy':'old heldout takes precedence','sample_ids':[r['sample_id'] for r in group]})
        elif 'train' in splits:allocation[digest]='train'
        else:devgroups.append(digest)
    known=set(known_development_images)
    devgroups.sort(key=lambda x:hashlib.sha256(('phase5_v2_seed42:'+x).encode()).hexdigest())
    available=[]
    for digest in devgroups:
        if any(r['image_id'] in known for r in groups[digest]):allocation[digest]='development'
        else:available.append(digest)
    for i,digest in enumerate(available):allocation[digest]='gate' if i%2==0 else 'development'
    out={s:[] for s in ['train','development','gate','test']}
    for digest,group in groups.items():
        for row in group:
            row['source_v1_split']=row['split'];row['split']=allocation[digest]
            row['source_question_image_id']=row['image_id'];row['semantic_contract']=from_original(row)
            out[row['split']].append(row)
    for values in out.values():values.sort(key=lambda r:r['sample_id'])
    assert sum(map(len,out.values()))==len(rows)
    assert {r['sample_id'] for r in old['test']}<={r['sample_id'] for r in out['test']}
    validate_content_disjoint(out)
    categories={}
    for split,values in out.items():
        counts=Counter(c for r in values for c in r['semantic_contract']['candidate_semantic_categories'])
        categories[split]={'records':len(values),'unique_images':len({r['image_id'] for r in values}),
            'unique_content_groups':len({r['image_sha256'] for r in values}),
            'keyword_candidates':dict(counts),'independently_eligible':sum(eligible_for_semantic_study(r['semantic_contract']) for r in values),
            'verified_SC':0,'verified_SI_V':0,'semantic_unknown':len(values)}
    return out,{'schema_version':'vqav2_content_groups_v2','records_preserved':len(rows),
        'old_test_records_remain_test':True,'legacy_cross_split_groups':legacy_conflicts,
        'category_counts':categories,'near_duplicates':'separate candidate audit, not automatic deletion',
        'development_gate_exposure':'historically exposed v1 validation; not a fresh scientific selection set',
        'scientific_gate_ready':False,'independent_semantic_labels':0}
