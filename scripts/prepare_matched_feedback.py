"""Census prior exposure, stage new official images, freeze pilot/main manifests.
No inference or training. Original project splits and raw evidence unchanged.
"""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import sys
import time
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from src.data.dataset import sha256
from src.evaluation.vqa_official_compatible import normalize_answer,reference_metadata


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def archive(path,field):
    with zipfile.ZipFile(path) as z:
        member=next(x for x in z.namelist() if x.endswith('.json') and not x.startswith('__MACOSX'))
        return json.loads(z.read(member))[field]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='configs/matched_feedback_20261028_v1.json');a=parser.parse_args()
    config_path=ROOT/a.config;cfg=json.loads(config_path.read_text());run=ROOT/'results'/cfg['run_id']
    run.mkdir(parents=True,exist_ok=False);dump(run/'protocol.json',cfg)
    (run/'preparation_source.py').write_bytes(Path(__file__).read_bytes())
    csv.field_size_limit(32*1024*1024)
    dump(run/'protocol_freeze.json',{'unix':time.time(),'config_sha256':sha256(config_path),'source_sha256':sha256(Path(__file__))})
    raw=ROOT/'data/raw/vqav2';annotations=archive(raw/'v2_Annotations_Train_mscoco.zip','annotations')
    questions=archive(raw/'v2_Questions_Train_mscoco.zip','questions');qmap={q['question_id']:q for q in questions}
    question_image={a['question_id']:str(a['image_id']) for a in annotations}
    used_ids=set();used_hashes=set();used_questions=set();files=[];errors=[]
    idkeys={'image_id','source_image_id','source_image_ids','image_ids','source_ids','original_image_id'}
    hashkeys={'image_sha256','source_image_sha256','original_image_hash','candidate_image_hash','original_sha256','expected_sha256'}
    def collect(value,key=''):
        if isinstance(value,dict):
            for k,v in value.items():collect(v,k)
        elif isinstance(value,list):
            for v in value:collect(v,key)
        elif isinstance(value,(str,int)):
            text=str(value)
            if key in idkeys and text.isdigit():used_ids.add(str(int(text)))
            if key in {'sample_id','question_id','sample_ids','question_ids'} and text.isdigit():used_questions.add(int(text))
            if key in hashkeys and re.fullmatch('[a-fA-F0-9]{64}',text):used_hashes.add(text.lower())
            # Encoded JSON lists occur in CSV manifests; retain their exact identifiers.
            if text.startswith(('[','{')):
                try:collect(json.loads(text),key)
                except ValueError:pass
    for top in ['data/processed','data/manifests','experiments','reports','results','dashboard_package']:
        for path in sorted((ROOT/top).rglob('*')):
            if not path.is_file() or run in path.parents or path.suffix not in {'.json','.jsonl','.csv'}:continue
            if 'source_snapshot' in path.parts or 'source_versions' in path.parts:continue
            try:
                if path.suffix=='.json':collect(json.loads(path.read_text(encoding='utf-8-sig')))
                elif path.suffix=='.jsonl':
                    for line in path.read_text(encoding='utf-8-sig').splitlines():
                        if line.strip():collect(json.loads(line))
                else:
                    with path.open(encoding='utf-8-sig',newline='') as f:
                        for row in csv.DictReader(f):collect(row)
                files.append({'path':str(path.relative_to(ROOT)),'sha256':sha256(path),'bytes':path.stat().st_size})
            except Exception as e:errors.append({'path':str(path.relative_to(ROOT)),'error':repr(e)})
    used_ids.update(question_image[q] for q in used_questions if q in question_image)
    cached=[]
    for path in sorted((raw/'images').glob('*')):
        if not path.is_file():continue
        match=re.search(r'_(\d+)\.',path.name)
        if match:cached.append({'image_id':str(int(match[1])),'sha256':sha256(path),'path':str(path.relative_to(ROOT))})
    fresh_cached=[r for r in cached if r['image_id'] not in used_ids and r['sha256'] not in used_hashes]
    # Malformed historical partial editor JSON has complete original manifest,
    # CSV and JSONL evidence; no new source IDs can be concealed in that file.
    known_partial='results/benign_models_20261028_v1/editor_checks.json'
    empty_files=[e['path'] for e in errors if (ROOT/e['path']).stat().st_size==0]
    unexpected=[e for e in errors if e['path']!=known_partial and e['path'] not in empty_files]
    if unexpected:dump(run/'exposure_scan_errors.json',errors);raise RuntimeError('Cannot certify exposure census with unexplained unreadable records')
    buckets=defaultdict(list)
    for record in annotations:
        if str(record['image_id']) in used_ids:continue
        answers=[r['answer'] for r in record['answers']]
        if len(answers)!=10 or max(Counter(normalize_answer(x) for x in answers).values())<cfg['minimum_human_answer_agreement']:continue
        buckets[record['answer_type']].append(record)
    universe={str(a['image_id']) for values in buckets.values() for a in values}
    census={'scan_unix':time.time(),'scanned_files':files,'unreadable_records':errors,'known_partial_exception':known_partial,
        'empty_historical_receipts_no_identifiers':empty_files,
        'used_image_ids':sorted(used_ids,key=int),'used_content_hashes':sorted(used_hashes),'used_question_ids':sorted(used_questions),
        'cached_source_images':len(cached),'cached_unused_images':len(fresh_cached),'cached_unused_inventory':fresh_cached,
        'official_annotation_image_population':len({a['image_id'] for a in annotations}),
        'eligible_unused_official_images':len(universe),'eligible_unused_by_category':{c:len({a['image_id'] for a in values}) for c,values in buckets.items()},
        'upstream_full_annotation_archive_is_not_a_project_evaluation_split':True,
        'data_action':'reuse cached annotations; download only fixed selected unused images from official COCO bucket' if len(fresh_cached)<184 else 'cached pool sufficient',
        'required_fresh_images':184,'pretraining_contamination':'unknown; project-level exposure census only'}
    dump(run/'prior_exposure_census.json',census)
    rng=random.Random(cfg['selection_seed']);[rng.shuffle(values) for values in buckets.values()]
    selected=[];selected_ids=set()
    for phase in ['pilot','main']:
        for category,count in cfg[f'{phase}_category_quotas'].items():
            accepted=0
            for ann in buckets[category]:
                image_id=str(ann['image_id'])
                if image_id in selected_ids:continue
                selected_ids.add(image_id);q=qmap[ann['question_id']]
                selected.append({'phase':phase,'source_image_id':image_id,'image_id':image_id,'question_id':str(ann['question_id']),
                    'sample_id':str(ann['question_id']),'question':q['question'],'answers':[r['answer'] for r in ann['answers']],
                    'answer_type':category,'question_type':ann.get('question_type'),'split':'new_matched_evaluation',
                    'image_path':f"results/{cfg['run_id']}/source_images/COCO_train2014_{int(image_id):012d}.jpg"})
                accepted+=1
                if accepted==count:break
            if accepted!=count:raise RuntimeError('Insufficient disjoint official source population')
    dump(run/'selection_before_download.json',selected)
    if shutil.disk_usage(ROOT).free<2*1024**3:raise RuntimeError('Insufficient staging storage')
    def fetch(row):
        target=ROOT/row['image_path'];target.parent.mkdir(parents=True,exist_ok=True)
        old=raw/'images'/target.name
        if old.exists():shutil.copyfile(old,target)
        else:
            request=urllib.request.Request(cfg['source_urls']['images']+target.name,headers={'User-Agent':'RSI-benign-matched-validation/1'})
            with urllib.request.urlopen(request,timeout=60) as response,target.open('xb') as output:shutil.copyfileobj(response,output)
        with Image.open(target) as im:im.verify()
        return sha256(target)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for row,h in zip(selected,pool.map(fetch,selected)):row['image_sha256']=h
    hashes={r['image_sha256'] for r in selected}
    assert len(hashes)==184 and not hashes & used_hashes and not selected_ids & used_ids
    for phase in ['pilot','main']:
        records=[r for r in selected if r['phase']==phase];rng.shuffle(records)
        # First two source IDs per answer category repeat generation/inference.
        repeated=set()
        for category in ['yes/no','number','other']:repeated.update(r['source_image_id'] for r in [x for x in records if x['answer_type']==category][:2])
        for i,row in enumerate(records):row['source_index']=i;row['repeat_case']=row['source_image_id'] in repeated
        manifest={'schema':'matched_sources_v1','phase':phase,'run_id':cfg['run_id'],'frozen_unix':time.time(),
            'protocol_sha256':sha256(run/'protocol.json'),'census_sha256':sha256(run/'prior_exposure_census.json'),
            'source_archive_sha256':{name:sha256(raw/name) for name in ['v2_Annotations_Train_mscoco.zip','v2_Questions_Train_mscoco.zip']},
            'scorer_reference':reference_metadata(),'conditions':cfg['conditions'],'model_revisions':{'vlm':cfg['vlm']['revision'],'editor':cfg['editor']['revision']},
            'license':{'annotations':'CC BY 4.0','terms':'https://visualqa.org/terms.html','images':'Original COCO/Flickr per-image terms; cached official public research images','image_terms':'https://cocodataset.org/#termsofuse'},
            'selection_scope':'balanced categories and human answer agreement >=6/10; not natural-population representative','records':records}
        dump(run/f'{phase}_manifest.json',manifest)
    dump(run/'data_checks.json',{'status':'PASS','source_images':184,'pilot':24,'main':160,'source_ids_unique':True,'content_hashes_unique':True,
        'overlap_existing_ids':0,'overlap_existing_content':0,'pilot_main_overlap':0,
        'manifest_sha256':{s:sha256(run/f'{s}_manifest.json') for s in ['pilot','main']},'staging_bytes':sum((ROOT/r['image_path']).stat().st_size for r in selected)})
    print(json.dumps({k:v for k,v in census.items() if k not in ['scanned_files','used_image_ids','used_content_hashes','used_question_ids','cached_unused_inventory']}),flush=True)


if __name__=='__main__':main()
