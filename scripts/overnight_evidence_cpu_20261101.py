"""Changed-stage independent 32-pair correspondence verification; no model loads.

Unchanged historical feedback/Qwen analyses are checksum-bound and reused.
"""
import csv,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)

def main(run):
 import numpy as np
 from PIL import Image,ImageOps
 from skimage.metrics import structural_similarity
 from analyze_semantic_clean_stage import validated_reviews
 cfg=read(run/'protocol.json');assert all(sha(ROOT/n)==h for n,h in cfg['inputs'].items())
 kit=ROOT/cfg['kit'];cases=read(kit/'blinded_manifest.json')['cases'];full=ROOT/'results/semantic_clean_20261029_v1/semantic_review';population={x['source_image_id']:x for x in read(full/'blinded_manifest.json')['cases']};assert len(cases)==32 and len({x['source_image_id'] for x in cases})==32
 official={}
 for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
  with zipfile.ZipFile(ROOT/'data/raw/vqav2'/filename) as z:
   n=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));official[key]={str(x['question_id']):x for x in json.loads(z.read(n))[key] if str(x['question_id']) in {c['question_id'] for c in cases}}
  assert len(official[key])==32
 prior={x['source_image_id']:x for x in csv.DictReader((ROOT/'results/next_validation_20261101T0040Z_v1/aligned_image_quality.csv').open()) if x['condition']=='generated'};rows=[]
 for c in cases:
  p=population[c['source_image_id']];assert c==p;im={}
  for key in ['original','edited','geometry_control']:
   x=kit/c[key];y=full/p[key];assert x.resolve().is_relative_to(kit.resolve()) and x.is_file();assert sha(x)==sha(y)
   if key in ['original','edited']:assert sha(x)==c[key+'_sha256']
   with Image.open(x) as img:img.verify()
   with Image.open(x) as img:im[key]=img.convert('RGB')
  q=official['questions'][c['question_id']];a=official['annotations'][c['question_id']];assert c['question']==q['question'] and str(q['image_id'])==str(a['image_id'])==c['source_image_id'] and c['answers']==[x['answer'] for x in a['answers']]
  expected=np.asarray(ImageOps.pad(im['original'],(256,256),method=Image.Resampling.BICUBIC,color=(255,255,255)));ref=np.asarray(im['geometry_control']);gen=np.asarray(im['edited']);assert ref.shape==gen.shape==(256,256,3) and np.array_equal(expected,ref)
  s=float(structural_similarity(ref,gen,channel_axis=2,data_range=255));assert s==float(prior[c['source_image_id']]['SSIM'])
  scaled=float(structural_similarity(ref.astype(np.float32)/255,gen.astype(np.float32)/255,channel_axis=2,data_range=1));rows.append(dict(source_image_id=c['source_image_id'],original_sha256=sha(kit/c['original']),edited_sha256=sha(kit/c['edited']),correspondence=True,official_question_all_answers_match=True,geometry_reconstruction_exact=True,SSIM_uint8_range255=s,SSIM_float32_range1=scaled,precision_difference_descriptive=scaled-s,semantic_validity='UNREVIEWED',instruction_compliance='UNREVIEWED'))
 submitted=sorted((kit/'submitted').glob('*.csv'));adjud=[p for p in submitted if 'adjudicator_id' in p.open(encoding='utf-8-sig').readline()];reviews=[p for p in submitted if p not in adjud];assert len(adjud)<=1
 labels,_=validated_reviews(reviews,cases,adjud[0] if adjud else None)
 # Templates must remain unfilled. Genuine submissions belong in submitted/.
 for p in kit.glob('*template.csv'):
  for r in csv.DictReader(p.open(encoding='utf-8-sig')):assert not r.get('reviewer_id','').strip() and not r.get('adjudicator_id','').strip() and not r.get('semantic_label','').strip()
 put(run/'review32_verified.json',dict(status='PASS_INTEGRITY_ONLY',rows=rows,human_review_status='BLOCKED' if len(labels['final_labels'])<32 else 'REQUIRES_INDEPENDENCE_AUDIT',human_review=labels,official_archive_sha256={n:sha(ROOT/'data/raw/vqav2'/n) for n in ['v2_Annotations_Train_mscoco.zip','v2_Questions_Train_mscoco.zip']},no_fabricated_labels=True))
 reused={}
 for n in cfg['reuse_evidence']:
  reused[n]=dict(sha256=sha(ROOT/n),results=read(ROOT/n))
 put(run/'unchanged_evidence_reused.json',dict(status='REUSED_UNCHANGED_NOT_RERUN',records=reused))
 modules={n:importlib.util.find_spec(n) is not None for n in ['lpips','piq','torchmetrics']};put(run/'quality32_verified.json',dict(status='PASS_ALIGNMENT_ONLY',sources=32,SSIM_mean=float(np.mean([r['SSIM_uint8_range255'] for r in rows])),geometry='ImageOps.pad original RGB to256 white/BICUBIC; full-frame with padding; C5 vs itself is only a pipeline control',pixel_range=[0,255],LPIPS=None,calibrated_LPIPS='MISSING_NOT_SUBSTITUTED',available_modules=modules,semantics_independently_validated=False,generation_cost='No new generation; historical actual GPU accounting retained separately'))
 # Rehash saved clean adapters against their published inventories, no retraining.
 adapters=[]
 for seed in [42,123,3407]:
  base=ROOT/f'results/small_scale_validation_20261031T1928Z_v1_qwen_s{seed}/clean_main';inv=read(base/'artifact_inventory.json')
  for x in inv['files']:
   if x['path'].startswith('adapter/'):
    assert sha(base/x['path'])==x['sha256'];adapters.append(dict(seed=seed,path=x['path'],sha256=x['sha256']))
 assert adapters;put(run/'clean_adapters_verified.json',dict(status='PASS_INTEGRITY_ONLY',files=adapters,new_training=False,new_inference=False,scientific_gain='NOT ESTABLISHED; reuse existing paired CI and exposure analysis'))
 put(run/'artifact_inventory.json',dict(files=[dict(path=p.relative_to(run).as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(run.rglob('*.json')) if p.is_file()]))
 put(run/'checks.json',dict(status='PASS',diagnostic_complete=True,reviewed_pairs=len(labels['final_labels']),human_semantic_gate=False,reward_release=False,n_sources=32,new_model_calls=0,LPIPS=None,scientific_improvement='NOT ESTABLISHED'))
 print(json.dumps(read(run/'checks.json')))

if __name__=='__main__':main(Path(sys.argv[1]))
