"""Reconcile clean labels/questions to the existing official annotation archives."""
import argparse,hashlib,json,sys,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.vqa_official_compatible import normalize_answer
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;manifest=json.loads((run/'clean_data_manifest.json').read_text());rows=manifest['train']+manifest['evaluation'];wanted={int(r['sample_id']) for r in rows};maps={};archives={}
    for filename,field in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
        path=ROOT/'data/raw/vqav2'/filename;archives[filename]=hashlib.sha256(path.read_bytes()).hexdigest()
        with zipfile.ZipFile(path) as z:member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));items=json.loads(z.read(member))[field]
        maps[field]={r['question_id']:r for r in items if r['question_id'] in wanted}
    checks=[]
    for r in rows:
        q=maps['questions'][int(r['sample_id'])];ann=maps['annotations'][int(r['sample_id'])];counts=Counter(normalize_answer(x) for x in r['answers'])
        checks.append({'sample_id':r['sample_id'],'official_image_id_equal':str(q['image_id'])==r['image_id']==str(ann['image_id']),'official_question_equal':q['question']==r['question'],'all_ten_official_answers_equal':[x['answer'] for x in ann['answers']]==r['answers'],'declared_majority_target_correct':counts[r['training_answer']]==r['human_normalized_agreement']>=6})
    result={'status':'PASS' if all(all(v for k,v in r.items() if k!='sample_id') for r in checks) else 'FAIL','dataset':'VQAv2 train2014, 2017-04-26 release','archive_sha256':archives,'checks':checks,'license':{'annotations':'CC BY 4.0','source':'https://visualqa.org/terms.html','images':'COCO/Flickr individual terms; no new redistribution','image_terms':'https://cocodataset.org/#termsofuse'},'source_download':'No download this stage; existing official cached archives and original images','caveat':'Original human consensus is not a guarantee against annotator error or model pretraining exposure.'}
    with (run/'audit/clean_original_label_provenance.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'verified_original_examples':len(checks),'archives':archives}));assert result['status']=='PASS'
if __name__=='__main__':main()
