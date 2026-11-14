"""CPU preparation of immutable content-deduplicated v2, without new inference."""
import json
import zipfile
import hashlib
from pathlib import Path
import _bootstrap
from src.data.prepare_v2 import prepare_v2
from src.data.dataset import sha256
from src.data.dataset_v2 import VerifiedVQADataset
from src.utils.logging import Experiment,write_json

def near_duplicate_candidates(root,old):
    from PIL import Image
    import numpy as np
    images={r['image_id']:r for rows in old.values() for r in rows};signatures=[]
    for image_id,row in images.items():
        with Image.open(root/row['image_path']) as image:a=np.asarray(image.convert('L').resize((9,8)))
        bits=(a[:,1:]>a[:,:-1]).reshape(-1);value=sum(int(b)<<i for i,b in enumerate(bits))
        signatures.append((image_id,row['image_sha256'],value))
    pairs=[]
    for i,(left,digest,a) in enumerate(signatures):
        for right,other,b in signatures[:i]:
            distance=(a^b).bit_count()
            if digest!=other and distance<=4:pairs.append({'image_ids':[right,left],'dhash_hamming':distance})
    return {'method':'64-bit horizontal dHash on9x8 grayscale, threshold<=4','pairs':pairs,
       'candidate_pairs':len(pairs),'status':'UNREVIEWED_CANDIDATES_NOT_CONFIRMED_DUPLICATES','automatic_exclusions':0}

def main():
    root=_bootstrap.ROOT;source=root/'data/processed/vqav2_seed42_v1'
    old={s:json.loads((source/f'{s}.json').read_text()) for s in ['train','validation','test']}
    protocol=json.loads((root/'configs/phase2_vqa_protocol.json').read_text())
    for split in old:assert sha256(source/f'{split}.json')==protocol['split_hashes'][split]
    from src.data.prepare_vqa import OFFICIAL
    archives={}
    for kind,url in OFFICIAL.items():
        with zipfile.ZipFile(root/'data/raw/vqav2'/Path(url).name) as z:
            member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
            archives[kind]=json.loads(z.read(member))
    questions={str(q['question_id']):q for q in archives['questions']['questions']}
    annotations={str(a['question_id']):a for a in archives['annotations']['annotations']}
    for rows in old.values():
        for row in rows:
            assert sha256(root/row['image_path'])==row['image_sha256']
            q=questions[row['sample_id']];a=annotations[row['sample_id']]
            assert str(q['image_id'])==str(a['image_id'])==row['image_id'] and q['question']==row['question']
            assert [v['answer'] for v in a['answers']]==row['answers']
    frozen=json.loads((root/'experiments/EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR/metrics.json').read_text())
    known={o['source_image_id'] for e in frozen['evaluations'] for o in e['observations']}
    rows,provenance=prepare_v2(old,known)
    target=root/'data/processed/vqav2_content_v2';target.mkdir(exist_ok=False)
    for split,records in rows.items():
        with (target/f'{split}.json').open('x') as f:json.dump(records,f,indent=2)
    provenance['split_sha256']={s:sha256(target/f'{s}.json') for s in rows}
    provenance['parent_split_sha256']={s:sha256(source/f'{s}.json') for s in old}
    provenance['official_joins_reverified']=1500
    near=near_duplicate_candidates(root,old)
    with (target/'near_duplicate_candidates.json').open('x') as f:json.dump(near,f,indent=2)
    provenance['near_duplicate_candidate_pairs']=near['candidate_pairs']
    with (target/'provenance.json').open('x') as f:json.dump(provenance,f,indent=2)
    for split in rows:VerifiedVQADataset(target/f'{split}.json',root,split,provenance['split_sha256'][split])
    with Experiment(root,'phase5_dataset_v2',{'phase':5,'source':'vqav2_seed42_v1','new_model_calls':0}) as run:
        run.metrics(provenance);run.finish(summary='Content-grouped v2 preserved all1500 original records and isolated old test; no edited factual labels manufactured.')
        write_json(root/'reports/phase5_dataset_v2.json',{'id':run.id,'path':str(target),'provenance':provenance})
    print(json.dumps(provenance['category_counts']))

if __name__=='__main__':main()
