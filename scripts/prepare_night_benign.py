"""CPU-only locked public clean-QA cohort plus paired benign restoration data."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import random
import re
import sys
import time
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256,validate_splits
from src.evaluation.vqa_official_compatible import normalize_answer


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2),encoding='utf-8');os.replace(tmp,path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--campaign',required=True);ap.add_argument('--assets-root',required=True);a=ap.parse_args()
    run,assets=Path(a.campaign).resolve(),Path(a.assets_root).resolve();cfg=json.loads((run/'protocol.json').read_text());started=time.time()
    if (run/'prepare_receipt.json').exists():raise ValueError('Cohort already finalized')
    progress=run/'prepare_progress.json';put(progress,{'stage':'reading_official_annotations','updated_unix':time.time()})
    used=set();census=assets/'results/matched_feedback_20261028_v2/prior_exposure_census.json'
    if census.is_file():used.update(str(i) for i in json.loads(census.read_text())['used_image_ids'])
    for base in [assets/'data/raw/vqav2/images',assets/'results',assets/'experiments',assets/'outputs']:
        if base.exists():
            for path in base.rglob('COCO_*.jpg'):
                if path.is_relative_to(run):continue
                match=re.search(r'COCO_(?:train|val)2014_(\d+)\.jpg$',path.name)
                if match:used.add(str(int(match.group(1))))
    def load(filename,key):
        path=assets/'data/raw/vqav2'/filename
        with zipfile.ZipFile(path) as archive:
            name=next(n for n in archive.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
            return json.loads(archive.read(name))[key]
    archives={name:sha256(assets/'data/raw/vqav2'/name) for name in ['v2_Annotations_Train_mscoco.zip','v2_Questions_Train_mscoco.zip']}
    questions={r['question_id']:r for r in load('v2_Questions_Train_mscoco.zip','questions')}
    annotations=load('v2_Annotations_Train_mscoco.zip','annotations');annotations.sort(key=lambda r:r['question_id'])
    random.Random(cfg['selection_seed']).shuffle(annotations)
    pending=run/'source_selection_locked.json'
    if pending.exists():splits=json.loads(pending.read_text())['splits']
    else:
        splits={};cursor=iter(annotations)
        for split,count in cfg['split_sizes'].items():
            selected=[]
            for ann in cursor:
                iid=str(ann['image_id'])
                if iid in used:continue
                answers=[x['answer'] for x in ann['answers']]
                if len(answers)!=10:continue
                target,n=Counter(normalize_answer(x) for x in answers).most_common(1)[0]
                if not target or (split=='train' and n<cfg['training_answer_min_consensus']):continue
                question=questions[ann['question_id']]
                if question['image_id']!=ann['image_id']:raise ValueError('Official QA association mismatch')
                filename=f'COCO_train2014_{int(iid):012d}.jpg'
                selected.append({'sample_id':str(ann['question_id']),'image_id':iid,'question':question['question'],
                  'answers':answers,'answer_type':ann['answer_type'],'training_answer':target,'human_consensus_count':n,
                  'image_path':(run/'images'/filename).relative_to(assets).as_posix(),
                  'URL':'https://s3.amazonaws.com/images.cocodataset.org/train2014/'+filename})
                used.add(iid)
                if len(selected)==count:break
            if len(selected)!=count:raise ValueError('Insufficient clean source pool')
            splits[split]=selected
        put(pending,{'splits':splits,'archives':archives,'selection_seed':cfg['selection_seed'],
                     'locked_before_model_calls':True,'selection':'question-uniform shuffle, one question per source; train consensus>=3; evaluation unfiltered by consensus'})
    images=run/'images';images.mkdir(exist_ok=True)
    allrows=[r for rows in splits.values() for r in rows]
    def fetch(row):
        from PIL import Image
        path=assets/row['image_path'];temporary=path.with_suffix('.download_partial')
        if not path.exists():
            for attempt in range(2):
                try:
                    with urllib.request.urlopen(row['URL'],timeout=35) as response:temporary.write_bytes(response.read())
                    with Image.open(temporary) as image:image.verify()
                    os.replace(temporary,path);break
                except Exception:
                    if attempt:raise
                    time.sleep(.5)
        with Image.open(path) as image:image.convert('RGB').load()
        return row['sample_id'],sha256(path),path.stat().st_size
    completed={};errors=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures={pool.submit(fetch,row):row for row in allrows}
        for future in as_completed(futures):
            row=futures[future]
            try:sample,h,size=future.result();completed[sample]=(h,size)
            except Exception as error:errors.append({'sample_id':row['sample_id'],'error':repr(error)})
            if (len(completed)+len(errors))%32==0:
                put(progress,{'stage':'download_original_public_images','completed':len(completed),'total':len(allrows),'errors':errors,'updated_unix':time.time()})
    if errors:put(run/'prepare_errors.json',errors);raise RuntimeError('Public image retrieval failed; no substitute sources or GPU launch')
    for row in allrows:row['image_sha256']=completed[row['sample_id']][0]
    audit=validate_splits(splits,cfg['split_sizes'],assets)
    manifest={'splits':splits,'audit':audit,'archive_SHA256':archives,'selection_locked_SHA256':sha256(pending),
       'protocol_SHA256':sha256(run/'protocol.json'),'prior_cached_source_exclusion':True,
       'scope':cfg['scope'],'answer_type_counts':{k:dict(Counter(r['answer_type'] for r in rows)) for k,rows in splits.items()}}
    put(run/'manifest.json',manifest)
    # Actual paired image task preparation, CPU only. Target is the original
    # public photo, not a model-generated pseudo-reference.
    from PIL import Image,ImageEnhance,ImageFilter
    p1=run/'p1_preparation';pairs={}
    for split,count in cfg['P1_data_sizes'].items():
        rows=[]
        for row in splits[split][:count]:
            with Image.open(assets/row['image_path']) as im:original=im.convert('RGB')
            scale=min(1.,512/max(original.size));reference=original.resize((round(original.width*scale),round(original.height*scale)),Image.Resampling.LANCZOS)
            if split=='transfer':degraded=reference.filter(ImageFilter.GaussianBlur(1.5));task='deblur';instruction='Restore the sharpness of this photograph while preserving its scene and objects.'
            else:degraded=ImageEnhance.Brightness(reference).enhance(.7);task='restore_exposure';instruction='Restore normal exposure in this photograph while preserving its colors and objects.'
            directory=p1/split;directory.mkdir(parents=True,exist_ok=True)
            source=directory/(row['sample_id']+'_input.png');target=directory/(row['sample_id']+'_reference.png')
            degraded.save(source);reference.save(target)
            rows.append({'sample_id':row['sample_id'],'image_id':row['image_id'],'original_SHA256':row['image_sha256'],
               'input':source.relative_to(run).as_posix(),'target':target.relative_to(run).as_posix(),
               'input_SHA256':sha256(source),'target_SHA256':sha256(target),'task_family':task,'instruction':instruction,
               'reference_type':'unaltered real-photo content, resized; input has known synthetic degradation'})
        pairs[split]=rows
    put(p1/'paired_manifest.json',{'splits':pairs,'clean_QA_manifest_SHA256':sha256(run/'manifest.json'),
       'arms':cfg['P1_arms'],'seeds':cfg['seeds'],'GPU_status':'BLOCKED_FROZEN_QUALITY_AND_512_TRAINING_NUMERICS',
       'all_comparators_share_same_pairs':True,'independent_empirical_RSI':'NOT_RUN'})
    receipt={'status':'PASS_DATA_PREPARED','manifest_SHA256':sha256(run/'manifest.json'),
       'protocol_SHA256':sha256(run/'protocol.json'),'source_images':len(allrows),'image_bytes':sum(v[1] for v in completed.values()),
       'P1_pairs':sum(len(v) for v in pairs.values()),'P1_GPU_status':'BLOCKED','elapsed_seconds':time.time()-started}
    put(run/'prepare_receipt.json',receipt);put(progress,{'stage':'COMPLETE','updated_unix':time.time(),**receipt});print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
