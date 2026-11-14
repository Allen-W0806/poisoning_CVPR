"""Freeze clean three-seed replication, unused original VQA evaluation and budget."""
import argparse,copy,hashlib,json,shutil,sys,time,urllib.request,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.vqa_official_compatible import normalize_answer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;run.mkdir(exist_ok=False);(run/'logs').mkdir();(run/'source_images').mkdir();(run/'audit').mkdir()
    parent=ROOT/'results/semantic_clean_20261029_v1';old=ROOT/'results/matched_feedback_20261028_v2';census=json.loads((old/'prior_exposure_census.json').read_text())
    used={str(x) for x in census['used_image_ids']};hashes=set(census['used_content_hashes'])
    for name in ['pilot','main']:
        for r in json.loads((old/f'{name}_manifest.json').read_text())['records']:used.add(r['image_id']);hashes.add(r['image_sha256'])
    for p in (ROOT/'data/raw/vqav2/images').glob('*'):
        if p.is_file():
            part=p.stem.rsplit('_',1)[-1]
            if part.isdigit():used.add(str(int(part)))
            hashes.add(sha(p))
    for r in json.loads((parent/'clean_data_manifest.json').read_text())['train']+json.loads((parent/'clean_data_manifest.json').read_text())['evaluation']:used.add(r['image_id']);hashes.add(r['image_sha256'])
    cfg={'run_id':a.run_id,'seeds':[42,123,3407],'logical_steps_per_seed':16,'evaluation_n':32,'training_n':32,'GPU_budget_seconds':3600,'charged_prior_seconds':1002,'prior_charges':{'gradient_validation':198,'matched_feedback':584,'clean_baseline':220},'P1_wall_seconds':300,'P2_wall_seconds':600,'max_retries_per_task':2,'continuation_acceptance':'bitwise equality; never relax thresholds','source_selection':'previous exposure census + matched sources + current original-image cache; one unique original source/content per question; >=6/10 normalized human consensus; fixed balanced diagnostic pool','model_revision':'89644892e4d85e24eaac8bacfd4f463576704203','scope':'clean VQA only and benign constant-coefficient numerical IP2Pix diagnostic; no attack rewards or target-error optimization'}
    dump(run/'protocol.json',cfg)
    records={}
    for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
        with zipfile.ZipFile(ROOT/'data/raw/vqav2'/filename) as z:member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));records[key]=json.loads(z.read(member))[key]
    questions={r['question_id']:r for r in records['questions']};quotas={'yes/no':10,'number':11,'other':11};chosen=[];candidates=[]
    for ann in sorted(records['annotations'],key=lambda r:r['question_id']):
        category=ann['answer_type'];i=str(ann['image_id']);answers=[x['answer'] for x in ann['answers']];answer,count=Counter(normalize_answer(x) for x in answers).most_common(1)[0]
        if i in used or count<6 or quotas[category]<=0:continue
        candidates.append({'image_id':i,'question_id':str(ann['question_id']),'category':category,'answer':answer,'count':count})
        path=run/'source_images'/f'COCO_train2014_{int(i):012d}.jpg';url='https://s3.amazonaws.com/images.cocodataset.org/train2014/'+path.name
        urllib.request.urlretrieve(url,path);digest=sha(path)
        if digest in hashes:
            # Keep fetched duplicate as provenance; it cannot enter evaluation.
            candidates[-1]['excluded']='duplicate content';continue
        from PIL import Image
        with Image.open(path) as image:image.convert('RGB').load()
        q=questions[ann['question_id']];assert q['image_id']==ann['image_id']
        chosen.append({'sample_id':str(ann['question_id']),'image_id':i,'image_path':str(path.relative_to(ROOT)),'image_sha256':digest,'question':q['question'],'answers':answers,'answer_type':category,'question_type':ann['question_type'],'source':'VQAv2 train2014-2017-04-26 official cached questions/annotations','training_answer':answer,'human_normalized_agreement':count,'source_url':url,'role':'fresh_evaluation'})
        used.add(i);hashes.add(digest);quotas[category]-=1
        if len(chosen)==32:break
    assert len(chosen)==32
    original=json.loads((parent/'clean_data_manifest.json').read_text());children=[]
    for seed in cfg['seeds']:
        child=ROOT/'results'/f'{a.run_id}_qwen_s{seed}_a0';child.mkdir(exist_ok=False)
        cc=json.loads((parent/'protocol.json').read_text());cc.update(run_id=child.name,seed=seed,main_steps=16,evaluation_scope='fresh image-ID/content-disjoint original VQA; no known prior project evaluation; pretraining exposure unknown',fresh_object_continuation=True)
        dump(child/'protocol.json',cc);dump(child/'clean_data_manifest.json',{'protocol_sha256':sha(child/'protocol.json'),'train':original['train'],'evaluation':chosen,'source_provenance':{'prior_census_sha256':sha(old/'prior_exposure_census.json'),'parent_train_manifest_sha256':sha(parent/'clean_data_manifest.json'),'selection_receipt':str((run/'fresh_source_receipt.json').relative_to(ROOT))}})
        dump(child/'accepted_clean_phases.json',{'pilot':'../semantic_clean_20261029_v1/clean_pilot_repair_v2','note':'Reuse previously passed bounded clean pilot; fresh-object next-step test is added and must independently pass.'});children.append({'task':f'P1_seed{seed}','kind':'qwen','seed':seed,'attempt':0,'run_id':child.name,'wall_seconds':300})
    p2=ROOT/'results'/f'{a.run_id}_ip2pix_a0';p2.mkdir(exist_ok=False);dump(p2/'protocol.json',{'parent_run':a.run_id,'source_record':original['train'][0],'criterion':'bitwise exact; descriptive errors do not relax acceptance','diagnostic_reward':1.0,'reward_meaning':'constant, independent of image content/VLM/security/attack outcomes','logical_batch_size':2,'num_inference_steps':2,'max_optimizer_calls':4,'seed':42,'gradient_checkpointing':True})
    children.append({'task':'P2_IP2Pix','kind':'ip2pix','attempt':0,'run_id':p2.name,'wall_seconds':600})
    dump(run/'tasks.json',children);dump(run/'fresh_source_receipt.json',{'status':'PASS','selected':chosen,'candidate_order':candidates,'known_prior_exposure_excluded':True,'exact_ID_and_hash_disjoint':True,'category_counts':dict(Counter(r['answer_type'] for r in chosen)),'images_downloaded_bytes':sum(p.stat().st_size for p in (run/'source_images').glob('*')),'dataset_license':'Annotations CC BY4.0; images retain individual COCO/Flickr terms','annotation_archive_sha256':sha(ROOT/'data/raw/vqav2/v2_Annotations_Train_mscoco.zip'),'question_archive_sha256':sha(ROOT/'data/raw/vqav2/v2_Questions_Train_mscoco.zip'),'pretraining_contamination':'unknown'})
    print(json.dumps({'run_id':a.run_id,'status':'PASS','children':children,'fresh_evaluation_n':32,'downloaded_bytes':sum(p.stat().st_size for p in (run/'source_images').glob('*'))}))
if __name__=='__main__':main()
