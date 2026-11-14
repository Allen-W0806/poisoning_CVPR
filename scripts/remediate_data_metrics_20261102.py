"""New provenance acquisition, calibrated LPIPS and human-review v3 preparation.

No target-model inference, generation, reward training, model labels or ASR.
Old development cohorts remain development. Candidate bytes are acquired before
any evaluation and never treated as an already completed final test.
"""
import collections,concurrent.futures,csv,datetime,hashlib,io,json,os,re,shutil,subprocess,sys,time,traceback,urllib.request,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='execution_remediation_20261102_v1';KIT=ROOT/'results/next_validation_20261101T0040Z_v1/semantic_review32'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvout(p,rows):
 with Path(p).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def review(run):
 from src.evaluation.independent_review_v3 import ATTRS
 d=read(KIT/'blinded_manifest.json');assert len(d['cases'])==32
 html=(KIT/'index.html').read_text();assert 'text_preserved' not in d['fields'];idx=d['fields'].index('reference_answer_valid');d['fields'].insert(idx,'text_preserved')
 # Replace the embedded data and add one required rubric choice. All labels stay blank.
 html=re.sub(r'const DATA=\{.*?\};let index=',lambda m:'const DATA='+json.dumps(d)+';let index=',html,count=1,flags=re.S)
 if '"text_preserved"' not in html:
  # Original app keeps DATA and initialization on one line; preserve its remaining JS exactly.
  start=html.index('const DATA=');decoder=json.JSONDecoder();_,length=decoder.raw_decode(html[start+11:]);html=html[:start+11]+json.dumps(d)+html[start+11+length:]
 html=html.replace("reference_answer_valid:'5.","text_preserved:'Visible text preserved (choose not_applicable if no text)',reference_answer_valid:'5.")
 html=html.replace("'color_spatial_preserved'].some", "'color_spatial_preserved','text_preserved'].some")
 html=html.replace("['yes','no','ambiguous','not_applicable']", "['yes','no','uncertain','not_applicable']")
 html=html.replace('independent-semantic-32-next_validation_20261101T0040Z_v1:', 'independent-semantic-v3-ROLE:')
 out=run/'semantic_review_v3';out.mkdir();(out/'images').mkdir()
 for c in d['cases']:
  for k in ['original','edited','geometry_control']:
   p=KIT/c[k]
   if k!='geometry_control':assert sha(p)==c[k+'_sha256']
   shutil.copy2(p,out/c[k])
 (out/'index.html').write_text(html);put(out/'blinded_manifest.json',d)
 blank=[{k:c.get(k,'') for k in d['fields']} for c in d['cases']]
 for n in ['reviewer_template.csv','adjudication_template.csv']:csvout(out/n,blank)
 instructions='''# Independent 32-pair review v3
Two distinct real humans review independently, without model outputs or each other's labels.
Review instruction completion, scene/objects/identity/count/colors/spatial relations, visible text, original VQA answer validity and ambiguity. All required choices start blank. Original ten human answers are dataset references, not edited-image labels.
Use not_applicable only when a dimension is truly inapplicable. Valid semantics does not imply instruction success. Invalid/ambiguous judgments need a reason. Export both original CSVs unchanged. A third independent human adjudicates any disagreement. The assistant must never fill human judgments.
The two ZIPs have independent browser-storage namespaces. No human labels have been created. Keep raw exports; put them in semantic_review_v3/submitted/ for coordinator ingestion. Do not overwrite templates.
'''
 (out/'README.md').write_text(instructions);(out/'submitted').mkdir()
 for role in ['A','B']:
  with zipfile.ZipFile(run/f'independent_review32_v3_{role}.zip','x',zipfile.ZIP_DEFLATED) as z:
   for p in out.rglob('*'):
    if not p.is_file():continue
    b=p.read_bytes()
    if p.name=='index.html':b=b.replace(b'v3-ROLE:',('v3-reviewer-'+role+':').encode())
    z.writestr(str(p.relative_to(out)),b)
 assert all(not x[k] for x in blank for k in ['reviewer_id','semantic_label','reviewed_utc','text_preserved'])
 result=dict(status='READY_REAL_HUMANS_REQUIRED',cases=32,fields=d['fields'],genuine_review_records=0,genuine_final_labels=0,source_manifest_sha256=sha(KIT/'blinded_manifest.json'),source_assets_unchanged=True,text_preservation_required=True)
 put(run/'review_v3_receipt.json',result);return result

def quality(run):
 import numpy as np,torch
 from PIL import Image
 from skimage.metrics import structural_similarity
 receipt=read(run/'lpips_dependency_verification.json');assert receipt['status']=='VERIFIED_DEPENDENCY_READY_NOT_YET_MEASURED'
 sys.path.insert(0,receipt['module_root']);import lpips
 torch.set_num_threads(2);metric=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,verbose=False).eval().requires_grad_(False)
 calibration=torch.load(Path(receipt['module_root'])/'lpips/weights/v0.1/alex.pth',map_location='cpu',weights_only=True)
 assert all(torch.equal(metric.state_dict()[k],v) for k,v in calibration.items())
 records=[];identity=[];clock=time.perf_counter()
 for c in read(KIT/'blinded_manifest.json')['cases']:
  a=KIT/c['geometry_control'];b=KIT/c['edited'];assert sha(b)==c['edited_sha256']
  with Image.open(a) as im:x=np.asarray(im.convert('RGB')).copy()
  with Image.open(b) as im:y=np.asarray(im.convert('RGB')).copy()
  assert x.shape==y.shape==(256,256,3)
  t=lambda v:torch.from_numpy(v.copy()).permute(2,0,1).float().unsqueeze(0)/255.
  with torch.inference_mode():v=float(metric(t(x),t(y),normalize=True));zero=float(metric(t(x),t(x),normalize=True))
  assert __import__('math').isfinite(v) and abs(zero)<=1e-12
  identity.append(zero);records.append(dict(source_image_id=c['source_image_id'],original_sha256=c['original_sha256'],geometry_sha256=sha(a),edited_sha256=sha(b),SSIM=float(structural_similarity(x,y,channel_axis=2,data_range=255)),LPIPS_v01_alex=v,exact_noop=bool(np.array_equal(x,y)),independent_instruction_success=None,semantic_label='UNREVIEWED',geometry='stored256x256_RGB_geometry_control_vs_edited',metric_device='CPU'))
 csvout(run/'review32_calibrated_quality.csv',records)
 result=dict(status='PASS_DESCRIPTIVE_METRIC_ONLY',pairs=len(records),LPIPS_mean=float(np.mean([x['LPIPS_v01_alex'] for x in records])),SSIM_mean=float(np.mean([x['SSIM'] for x in records])),exact_noop_count=sum(x['exact_noop'] for x in records),no_op_denominator=len(records),instruction_success=None,human_labels=0,identity_distance_max=max(identity),calibrated_layer_weights_exact=True,calibration_sha256=receipt['calibration_sha256'],base_sha256=receipt['base_sha256'],seconds=time.perf_counter()-clock,new_GPU_seconds=0,no_training_or_model_semantic_judgments=True)
 put(run/'calibrated_quality_summary.json',result);return result

def prospective(run):
 from PIL import Image
 import numpy as np
 from scipy.fft import dctn
 from src.evaluation.vqa_official_compatible import normalize_answer
 old=ROOT/'results/next_scientific_20261101T1905Z_v1/prospective_evaluation';reservations=read(old/'candidate_reservations_METADATA_ONLY.json')['records'];known=read(old/'known_exposure_exclusions.json');assert len(reservations)==160
 out=run/'prospective';out.mkdir();assets=ROOT/'data/raw/vqav2/prospective_20261102_v1';assets.mkdir(exist_ok=False)
 protocol=dict(created_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),reservations_sha256=sha(old/'candidate_reservations_METADATA_ONLY.json'),planned_sources=160,acquisition_only=True,maximum_download_bytes=160*5*1024*1024,free_bytes_before=shutil.disk_usage(ROOT).free,near_duplicate_method='64-bit grayscale DCT pHash on32x32; Hamming<=4 flags review, not semantic truth',no_model_inference=True,no_feedback_or_selection_using_predictions=True,paper_test_release=False,statistical_power='not yet specified; no final-paper inference authorized',source_URL='https://s3.amazonaws.com/images.cocodataset.org/train2014/',license_scope='Original COCO image rights retained; per-image license metadata not newly certified')
 assert protocol['free_bytes_before']>protocol['maximum_download_bytes']*2;put(out/'protocol.json',protocol)
 unresolved=[]
 for item in read(old/'unresolved_historical_records.json'):
  p=ROOT/item['path'];assert sha(p)==item['sha256']
  if p.stat().st_size==0:status='EMPTY_NONDATA_REPORT_RETAINED';witness=[]
  elif 'threeway/' in item['path']:
   q=ROOT/'scripts/benign_threeway_20261101.py';s=q.read_text();assert 'UNet2DConditionModel(sample_size=8' in s and 'cos(' in s;status='SYNTHETIC_COMPANION_FOUND_EXACT_JOB_LINEAGE_PENDING';witness=[str(q.relative_to(ROOT))]
  else:
   q=ROOT/'results/benign_models_20261028_v1/evaluation_manifest.json';d=read(q);rr=d['records'];assert any(str(x['sample_id'])=='61054004' for x in rr);assert {str(x['image_id']) for x in rr}<=set(known['source_image_ids']);status='INPUT_MANIFEST_COMPANION_FOUND_EXACT_JOB_LINEAGE_PENDING';witness=[str(q.relative_to(ROOT))]
  unresolved.append(dict(path=item['path'],original_corruption_preserved=True,status=status,witnesses={n:sha(ROOT/n) for n in witness}))
 put(out/'historical_corruption_resolution.json',unresolved)
 def acquire(c):
  iid=c['source_image_id'];assert iid.isdigit();p=assets/f'COCO_train2014_{int(iid):012d}.jpg';url=protocol['source_URL']+p.name
  try:
   with urllib.request.urlopen(url,timeout=30) as f:b=f.read(5*1024*1024+1)
   assert len(b)<=5*1024*1024
   with Image.open(io.BytesIO(b)) as im:im.verify()
   with p.open('xb') as f:f.write(b)
   return {**c,'image_path':str(p.relative_to(ROOT)),'image_sha256':sha(p),'source_url':url,'bytes':len(b)}
  except Exception as e:return {**c,'acquisition_error':repr(e)}
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:acquired=list(pool.map(acquire,reservations))
 wanted={int(x['question_id']) for x in reservations};official={};archive_hashes={}
 for fn,k in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
  p=ROOT/'data/raw/vqav2'/fn;archive_hashes[fn]=sha(p)
  with zipfile.ZipFile(p) as z:member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));allrows=json.loads(z.read(member))[k]
  official[k]={int(x['question_id']):x for x in allrows if int(x['question_id']) in wanted};del allrows
 def phash(p):
  with Image.open(p) as im:a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
  v=dctn(a,type=2,norm='ortho')[:8,:8].flatten();bits=v>np.median(v[1:]);return sum(int(b)<<i for i,b in enumerate(bits))
 refs=[];seen=set()
 paths=list((ROOT/'data/raw/vqav2/images').glob('*'))+list((ROOT/'results/replication_monitor_20261029_v1/source_images').glob('*'))
 for p in paths:
  if not p.is_file():continue
  h=sha(p)
  if h in seen:continue
  seen.add(h)
  try:refs.append((str(p.relative_to(ROOT)),h,phash(p)))
  except Exception:continue
 candidates=[]
 for x in acquired:
  if 'acquisition_error' in x:candidates.append(x);continue
  qid=int(x['question_id']);q=official['questions'][qid];a=official['annotations'][qid];assert str(q['image_id'])==str(a['image_id'])==x['source_image_id'];answers=[v['answer'] for v in a['answers']];target,n=collections.Counter(normalize_answer(v) for v in answers).most_common(1)[0];assert len(answers)==10 and n>=6
  h=phash(ROOT/x['image_path']);distances=[((h^v).bit_count(),name) for name,_,v in refs];near=[dict(path=name,hamming=d) for d,name in distances if d<=4]
  candidates.append({**x,'question':q['question'],'answers':answers,'training_answer':target,'known_id_overlap':x['source_image_id'] in set(known['source_image_ids']),'known_exact_hash_overlap':x['image_sha256'] in set(known['content_sha256']) or x['image_sha256'] in seen,'pHash64':f'{h:016x}','nearest_historical_hamming':min(d for d,_ in distances) if distances else None,'historical_near_duplicate_flags':near,'role':'PROSPECTIVE_UNOPENED_SOURCE_RESERVATION','model_inference_count':0})
 pair_flags=[]
 valid=[x for x in candidates if 'pHash64' in x]
 for i,x in enumerate(valid):
  for y in valid[:i]:
   d=(int(x['pHash64'],16)^int(y['pHash64'],16)).bit_count()
   if d<=4 or x['image_sha256']==y['image_sha256']:pair_flags.append(dict(a=x['source_image_id'],b=y['source_image_id'],Hamming=d,exact_bytes=x['image_sha256']==y['image_sha256']))
 put(out/'candidate_content_manifest.json',dict(records=candidates,official_archive_sha256=archive_hashes));put(out/'within_candidate_duplicate_flags.json',pair_flags)
 result=dict(status='CONTENT_AUDITED_FINAL_TEST_NOT_RELEASED',candidate_count=160,acquired=len(valid),acquisition_failures=sum('acquisition_error' in x for x in candidates),new_image_bytes=sum(x.get('bytes',0) for x in candidates),known_ID_overlaps=sum(x.get('known_id_overlap',False) for x in candidates),known_exact_hash_overlaps=sum(x.get('known_exact_hash_overlap',False) for x in candidates),historical_pHash_flagged_sources=sum(bool(x.get('historical_near_duplicate_flags')) for x in candidates),within_candidate_flags=len(pair_flags),historical_unique_images_screened=len(refs),corrupt_records_covered_by_complete_witnesses=False,corrupt_record_lineage='UNKNOWN; exact run-to-input binding not established by a shared source ID',paper_test_release=False,remaining=['Review any near-duplicate flags','Freeze power/minimum effect and complete project job-to-source exposure registry','Per-image license/provenance release check','No inference/selection or human-label claims have been made'],new_model_calls=0,new_GPU_seconds=0)
 put(out/'admission_status.json',result);return result

def main():
 run=ROOT/'results'/RUN;result={};errors={};clock=time.perf_counter()
 for name,fn in [('review_v3',review),('calibrated_quality',quality),('prospective_sources',prospective)]:
  try:result[name]=fn(run)
  except Exception:errors[name]=traceback.format_exc();(run/(name+'_failure.txt')).write_text(errors[name])
 cmd=[sys.executable,'-B','-m','unittest','tests.test_semantic_review_v3','-v'];p=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT);(run/'review_v3_regressions.log').write_text(p.stdout+p.stderr)
 summary=dict(status='PASS_SCOPED_ENGINEERING' if not errors and p.returncode==0 else 'PARTIAL_FAILURE_PRESERVED',job_id=os.getenv('SLURM_JOB_ID'),results=result,errors=errors,review_v3_test_returncode=p.returncode,new_GPU_seconds=0,seconds=time.perf_counter()-clock);put(run/'data_metric_remediation_summary.json',summary);print(json.dumps(summary),flush=True);return 0 if not errors and p.returncode==0 else 1

if __name__=='__main__':raise SystemExit(main())
