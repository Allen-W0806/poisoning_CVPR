"""Freeze new public clean QA sources before model use; no test-based selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256,validate_splits
from src.evaluation.vqa_official_compatible import normalize_answer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets-root',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    assets,run=Path(a.assets_root),Path(a.run);config=ROOT/'configs/p0_foundations_20261111.json'
    protocol=json.loads(config.read_text())
    if (run/'manifest.json').exists():raise ValueError('Refusing to revise locked manifest')
    used_ids,used_hashes=set(),set();exclusions=[]
    census=assets/'results/matched_feedback_20261028_v2/prior_exposure_census.json'
    if census.is_file():
        value=json.loads(census.read_text());used_ids.update(str(x) for x in value['used_image_ids']);used_hashes.update(value['used_content_hashes'])
        exclusions.append({'path':str(census),'sha256':sha256(census)})
    # Exclude all cached original COCO IDs without opening any historical test
    # annotations or the prospective reservation. This also excludes old pilot
    # and prior clean-replication images saved outside data/raw.
    for base in [assets/'data/raw/vqav2/images',assets/'results',assets/'experiments',assets/'outputs']:
        if not base.exists():continue
        for path in base.rglob('COCO_*.jpg'):
            if path.is_relative_to(run):continue
            match=re.search(r'COCO_(?:train|val)2014_(\d+)\.jpg$',path.name)
            if match:used_ids.add(str(int(match.group(1))))
    paths=[assets/'data/raw/vqav2'/name for name in ['v2_Annotations_Train_mscoco.zip','v2_Questions_Train_mscoco.zip']]
    def load(path,key):
        with zipfile.ZipFile(path) as archive:
            name=next(n for n in archive.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
            return json.loads(archive.read(name))[key]
    annotations=load(paths[0],'annotations');questions={x['question_id']:x for x in load(paths[1],'questions')}
    random.Random(protocol['source_selection_seed']).shuffle(annotations)
    image_dir=run/'source_images';image_dir.mkdir(exist_ok=False)
    selected=[];decisions=[]
    for ann in annotations:
        image_id=str(ann['image_id'])
        if image_id in used_ids:continue
        answers=[x['answer'] for x in ann['answers']]
        target,count=Counter(normalize_answer(x) for x in answers).most_common(1)[0]
        if count<protocol['qa_min_human_consensus'] or not target:continue
        path=image_dir/f'COCO_train2014_{int(image_id):012d}.jpg'
        url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+path.name
        with urllib.request.urlopen(url,timeout=30) as response:path.write_bytes(response.read())
        h=sha256(path)
        if h in used_hashes:
            decisions.append({'image_id':image_id,'status':'EXCLUDED_EXACT_CONTENT_DUPLICATE','url':url});continue
        from PIL import Image
        with Image.open(path) as image:image.convert('RGB').load()
        q=questions[ann['question_id']]
        if q['image_id']!=ann['image_id']:raise ValueError('Question annotation mismatch')
        row={'sample_id':str(ann['question_id']),'image_id':image_id,'question':q['question'],
             'answers':answers,'answer_type':ann['answer_type'],'training_answer':target,
             'human_consensus_count':count,'image_path':path.relative_to(assets).as_posix(),
             'image_sha256':h,'image_url':url,'annotation_source':'official VQAv2 train2014 archive'}
        selected.append(row);used_ids.add(image_id);used_hashes.add(h)
        decisions.append({'image_id':image_id,'status':'ADMITTED_BEFORE_INFERENCE','url':url})
        if len(selected)==sum(protocol['split_sizes'].values()):break
    if len(selected)!=sum(protocol['split_sizes'].values()):raise ValueError('Not enough clean sources')
    splits={};offset=0
    for split,size in protocol['split_sizes'].items():splits[split]=selected[offset:offset+size];offset+=size
    validation=validate_splits(splits,protocol['split_sizes'],assets)
    manifest={'splits':splits,'audit':validation,'selection_seed':protocol['source_selection_seed'],
              'source_selection':'new original COCO IDs excluding cached original images and prior exposure census; >=6/10 human answer consensus',
              'test_role':'source-disjoint run-specific clean evaluation; no tuning on test',
              'transfer_role':'separate source split; independent editing task-family evidence NOT_RUN',
              'prior_exclusion_sources':exclusions,'official_archives':{p.name:sha256(p) for p in paths},
              'historical_test_or_prospective_annotations_opened':False,'decisions':decisions}
    (run/'manifest.json').write_text(json.dumps(manifest,indent=2))
    lock={'manifest_SHA256':sha256(run/'manifest.json'),'protocol_SHA256':sha256(config),
          'selection_final_before_any_model_call':True,'downloaded_original_image_bytes':sum(p.stat().st_size for p in image_dir.iterdir())}
    (run/'manifest_lock.json').write_text(json.dumps(lock,indent=2))
    print(json.dumps({'audit':validation,'lock':lock},indent=2))


if __name__=='__main__':main()
