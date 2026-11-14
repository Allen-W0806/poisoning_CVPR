"""Benign 32/16/32 engineering pilot. No attack labels, rewards or guard changes."""
import argparse,copy,csv,hashlib,json,os,random,re,subprocess,sys,time,zipfile
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def dump(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def table(p,rows):
 if not rows:return
 with Path(p).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def execute(run,name,args):
 t=time.perf_counter()
 with (run/'logs'/f'{name}.log').open('x') as log:r=subprocess.run([sys.executable,'-B',*map(str,args)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 result=dict(task=name,returncode=r.returncode,seconds=time.perf_counter()-t,command=list(map(str,args)))
 dump(run/'executions'/f'{name}.json',result);return result
def prepare(run):
 from PIL import Image
 from src.evaluation.vqa_official_compatible import normalize_answer
 raw=ROOT/'data/raw/vqav2';archives={};values={}
 for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
  path=raw/filename;archives[filename]=sha(path)
  with zipfile.ZipFile(path) as z:
   member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));values[key]=json.loads(z.read(member))[key]
 qmap={r['question_id']:r for r in values['questions']};used=set();errors=[];scan=[]
 def collect(v,key=''):
  if isinstance(v,dict):
   for k,x in v.items():collect(x,k)
  elif isinstance(v,list):
   for x in v:collect(x,key)
  elif isinstance(v,(str,int)):
   s=str(v)
   if key in {'image_id','source_image_id','original_image_id','used_image_ids','image_ids'} and s.isdigit():used.add(str(int(s)))
   if s.startswith(('[','{')):
    try:collect(json.loads(s),key)
    except ValueError:pass
 csv.field_size_limit(32*1024*1024)
 for top in ('results','experiments','data/processed','data/manifests','reports'):
  for p in sorted((ROOT/top).rglob('*')):
   if not p.is_file() or run in p.parents or p.suffix not in {'.json','.jsonl','.csv'} or any(x in p.parts for x in ('source_snapshot','source_versions','audit')):continue
   try:
    if p.suffix=='.json':collect(read(p))
    elif p.suffix=='.jsonl':
     for line in p.read_text(encoding='utf-8-sig').splitlines():
      if line.strip():collect(json.loads(line))
    else:
     with p.open(encoding='utf-8-sig',newline='') as f:
      for row in csv.DictReader(f):collect(row)
    scan.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
   except Exception as e:errors.append(dict(path=str(p.relative_to(ROOT)),error=repr(e)))
 cached={}
 for p in sorted((raw/'images').glob('*')):
  m=re.search(r'_(\d+)\.',p.name)
  if m:cached[str(int(m[1]))]=p
 candidates=[]
 for ann in sorted(values['annotations'],key=lambda a:a['question_id']):
  ident=str(ann['image_id'])
  if ident not in cached:continue
  answers=[x['answer'] for x in ann['answers']];answer,n=Counter(normalize_answer(x) for x in answers).most_common(1)[0]
  if len(answers)!=10 or n<6:continue
  q=qmap[ann['question_id']];assert q['image_id']==ann['image_id'];p=cached[ident]
  candidates.append(dict(sample_id=str(ann['question_id']),image_id=ident,image_path=str(p.relative_to(ROOT)),image_sha256=sha(p),question=q['question'],answers=answers,answer_type=ann['answer_type'],training_answer=answer,human_normalized_agreement=n,source='official cached VQAv2 train2014 2017-04-26',known_prior_project_exposure=ident in used))
 random.Random(20261031).shuffle(candidates);candidates.sort(key=lambda r:r['known_prior_project_exposure'])
 def dhash(p):
  import numpy as np
  with Image.open(p) as im:a=np.array(im.convert('L').resize((9,8)))
  return int(''.join('1' if x else '0' for x in (a[:,1:]>a[:,:-1]).ravel()),2)
 selected=[];ids=set();hashes=set();codes=[];rejected=[]
 for r in candidates:
  if r['image_id'] in ids or r['image_sha256'] in hashes:continue
  code=dhash(ROOT/r['image_path']);near=next((s['image_id'] for s,c in zip(selected,codes) if (c^code).bit_count()<=4),None)
  if near:rejected.append(dict(image_id=r['image_id'],near_image_id=near,criterion='dHash64 Hamming <=4'));continue
  selected.append(r);ids.add(r['image_id']);hashes.add(r['image_sha256']);codes.append(code)
  if len(selected)==80:break
 assert len(selected)==80,'Insufficient source/content/near-duplicate separated cached examples'
 # Assignment is frozen before inference, independent of all evaluation scores.
 random.Random(2026103101).shuffle(selected)
 splits=dict(train=selected[:32],development=selected[32:48],independent_evaluation=selected[48:])
 for name,records in splits.items():
  for r in records:r['role']=name
 fresh=not errors and not any(r['known_prior_project_exposure'] for r in splits['independent_evaluation'])
 scope='project-unexposed source-disjoint assessment; pretraining contamination unknown' if fresh else 'historical development assessment; not an untouched scientific test'
 parent=read(ROOT/'results/semantic_clean_20261029_v1/protocol.json')
 cfg=dict(run_id=run.name,seeds=[42,123,3407],split_sizes={k:len(v) for k,v in splits.items()},logical_steps=16,gpu_cap_seconds=900,global_cap_seconds=3600,global_prior_seconds=1344,evaluation_scope=scope,model_revision=parent['model_revision'],controls={'identity':'pixel exact RGB PNG','brightness':1.03,'contrast':1.03,'geometry':'ImageOps.pad 256 white, method=3'},no_editor_optimization=True,held_scientific_claims=['attack efficacy','autonomous RSI benefit','general BF16 continuation'],near_duplicate_rule='dHash64 Hamming <=4 excluded across all selected records; imperfect screen; no learned embedding guarantee')
 dump(run/'protocol.json',cfg);dump(run/'split_manifest.json',dict(splits=splits,archive_sha256=archives,evaluation_scope=scope,selection_seeds=[20261031,2026103101],official_question_and_all_ten_answers_preserved=True))
 dump(run/'audit/data_integrity.json',dict(status='PASS',unique_ids=len(ids),unique_content_hashes=len(hashes),cached_images=len(cached),exposure_census_files=scan,exposure_scan_errors=errors,known_prior_image_count=len(used),fresh_scientific_test=fresh,near_duplicate_rejections=rejected,scope=scope,category_counts={k:dict(Counter(r['answer_type'] for r in v)) for k,v in splits.items()},licenses={'annotations':'CC BY 4.0 https://visualqa.org/terms.html','images':'individual COCO/Flickr terms https://cocodataset.org/#termsofuse'},download='none; cached originals only'))
 for seed in cfg['seeds']:
  child=ROOT/'results'/f'{run.name}_qwen_s{seed}';child.mkdir(exist_ok=False);cc=copy.deepcopy(parent);cc.update(run_id=child.name,seed=seed,main_steps=16,evaluation_scope=scope,fresh_object_continuation=True)
  dump(child/'protocol.json',cc);dump(child/'clean_data_manifest.json',dict(protocol_sha256=sha(child/'protocol.json'),train=splits['train'],evaluation=splits['development']+splits['independent_evaluation']))
  dump(child/'accepted_clean_phases.json',dict(pilot='../semantic_clean_20261029_v1/clean_pilot_repair_v2',note='Reuse passed unchanged engineering pilot; new source split and full next-step checks required'))
 return cfg,splits
def synthetic_rsi(run):
 from src.rsi.harness.engineering_fixture import create_cases
 from src.rsi.harness.harness_executor import HarnessExecutor
 from src.rsi.harness.harness_spec import HarnessSpec
 from src.rsi.harness.trace_collector import TraceCollector
 from src.rsi.harness.failure_investigator import FailureInvestigator
 from src.rsi.harness.revision_proposer import propose
 from src.rsi.harness.cost_aware_gate import IndependentGate
 from src.rsi.harness.version_manager import VersionManager
 results=[]
 for repeat in range(2):
  root=run/'synthetic_rsi'/f'repeat{repeat}';root.mkdir(parents=True)
  train,_=create_cases(root,'train',4);cases,oracle=create_cases(root,'gate',32);executor=HarnessExecutor(root);initial=HarnessSpec('fixed_geometry')
  inv=FailureInvestigator(executor,TraceCollector(root/'trace.jsonl'),train);diagnosis=inv.investigate_integrity_failure(train[0]);proposal=propose(initial,diagnosis,'H0000')
  manager=VersionManager(root/'versions',initial);gate=IndependentGate(executor,cases,oracle)
  fixed=gate.compare(initial,initial);bad=copy.deepcopy(proposal);bad['candidate']=HarnessSpec('decode_only',('decode',)).to_dict();rejected=manager.evaluate(bad,gate);accepted=manager.evaluate(proposal,gate)
  assert not rejected['accepted'] and accepted['accepted'];accepted_id=manager.incumbent
  reg=copy.deepcopy(proposal);reg['parent_version']=accepted_id;reg['candidate']=initial.to_dict();regression=manager.evaluate(reg,gate);assert not regression['accepted'] and manager.incumbent==accepted_id
  manager.rollback('H0000','preregistered rollback integrity test');assert manager.incumbent=='H0000'
  operations=inv.resources.diagnostic_operations+sum(r['measured_validation'][k]['diagnostic_operations'] for r in (rejected,accepted,regression) for k in ('parent_resources','candidate_resources'))+sum(fixed[k]['diagnostic_operations'] for k in ('parent_resources','candidate_resources'))
  results.append(dict(fixed_score=fixed['candidate_accuracy'],no_revision_score=fixed['parent_accuracy'],revised_score=accepted['measured_validation']['candidate_accuracy'],rejected=rejected['accepted'],accepted=accepted['accepted'],regression_rejected=not regression['accepted'],rollback=manager.incumbent,diagnostic_operations_including_rejected=operations,diagnosis=diagnosis))
 assert [{k:v for k,v in r.items() if k!='diagnosis'} for r in results][0]==[{k:v for k,v in r.items() if k!='diagnosis'} for r in results][1]
 dump(run/'synthetic_rsi/results.json',dict(status='PASS',reproducible=True,runs=results,scope='deterministic integrity oracle; no natural-image or autonomous scientific efficacy claim'))
def images(run,splits):
 from PIL import Image,ImageEnhance,ImageOps
 import numpy as np
 from skimage.metrics import structural_similarity
 import importlib.util
 records=[];pairs=[];folder=run/'image_controls';folder.mkdir()
 for r in sum(splits.values(),[]):
  p=ROOT/r['image_path']
  with Image.open(p) as im:original=im.convert('RGB')
  for cond in ('original','identity','brightness','contrast','geometry'):
   t=time.perf_counter()
   image=original if cond=='original' else original.copy() if cond=='identity' else ImageEnhance.Brightness(original).enhance(1.03) if cond=='brightness' else ImageEnhance.Contrast(original).enhance(1.03) if cond=='contrast' else ImageOps.pad(original,(256,256),color='white',method=3)
   out=p if cond=='original' else folder/f"{r['sample_id']}_{cond}.png"
   if cond!='original':image.save(out)
   with Image.open(out) as im:im.load();assert im.mode in ('RGB','L') and im.size==image.size
   ref=ImageOps.pad(original,(256,256),color='white',method=3) if cond=='geometry' else original
   a=np.array(ref).astype(float);b=np.array(image).astype(float)
   pixel=bool(np.array_equal(a,b))
   if cond=='identity':assert pixel
   row=dict(sample_id=r['sample_id'],image_id=r['image_id'],role=r['role'],condition=cond,input_sha256=r['image_sha256'],output_sha256=sha(out),output_path=str(out.relative_to(ROOT)),original_size=list(original.size),output_size=list(image.size),SSIM=float(structural_similarity(a,b,channel_axis=2,data_range=255)),RGB_MAE_255=float(np.abs(a-b).mean()),pixel_equal=pixel,processing_seconds=time.perf_counter()-t,LPIPS=None,LPIPS_status='BLOCKED: lpips package absent in unchanged cached environment',reference_answer_consistency='original official references / identity pixels' if cond in ('original','identity') else 'UNREVIEWED',semantic_label='label-preserving by identity' if cond in ('original','identity') else 'ambiguous pending independent verification',runtime_device='cpu',gpu_seconds=0,metric_reference='geometry-matched original' if cond=='geometry' else 'original RGB')
   records.append(row)
 old=read(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json')['records']
 for r in old:
  if r['candidate_id']!='C4':continue
  p=ROOT/r['candidate_path'];assert sha(p)==r['candidate_image_hash']
  pairs.append(dict(source_image_id=r['source_image_id'],output_path=r['candidate_path'],output_sha256=sha(p),original_sha256=r['original_image_hash'],engineering=r['engineering'],diagnostics=r.get('diagnostics',{}),semantic_status='UNREVIEWED',generation_latency='see immutable parent generation.jsonl',LPIPS=None,LPIPS_status='BLOCKED: package absent; not replaced by SSIM'))
 table(run/'image_controls.csv',records);dump(run/'frozen_ip2pix_descriptive.json',dict(records=pairs,n=len(pairs),source='existing frozen outputs only; no generation rerun',human_review='BLOCKED',LPIPS_status='BLOCKED'))
def cpu(run):
 cfg,splits=prepare(run);synthetic_rsi(run);images(run,splits)
 pg=execute(run,'finite_gradient',['scripts/validate_policy_gradients.py','--run-id',run.name+'_gradient','--protocol','configs/gradient_feedback_validation_v2.json']);assert pg['returncode']==0,'Finite-gradient reference failed; preserve evidence and block dependent GPU'
 guard=ROOT/'src/evaluation/reward_release.py'
 if not guard.exists():guard=ROOT/'src/rewards/reward_release.py'
 candidates=list(ROOT.rglob('reward_release.py'));guard_hashes={str(p.relative_to(ROOT)):sha(p) for p in candidates if 'src' in p.parts}
 dump(run/'cpu_complete.json',dict(status='PASS',at_unix=time.time(),data=True,synthetic_rsi=True,image_controls=400,finite_gradient=True,lpips='BLOCKED',generator_reward_guards=guard_hashes,pretrained_BF16='NOT VALIDATED; negative results preserved'))
def gpu(run):
 assert read(run/'cpu_complete.json')['status']=='PASS'
 for name,digest in read(run/'source_manifest.json').items():assert sha(ROOT/name)==digest,('Source drift',name)
 cfg=read(run/'protocol.json');results=[]
 for seed in cfg['seeds']:
  r=execute(run,f'clean_seed{seed}',['scripts/run_qwen_replication.py','--run-id',f'{run.name}_qwen_s{seed}','--phase','main']);results.append(r)
  if r['returncode']:break
 # Independent benign production-optimizer wiring test. Only random tiny UNet updates.
 results.append(execute(run,'production_ddpo',['scripts/validate_ddpo_wiring.py','--run-id',run.name+'_ddpo']))
 dump(run/'gpu_complete.json',dict(status='PASS' if all(r['returncode']==0 for r in results) and len(results)==4 else 'FAIL',results=results,job_id=os.environ.get('SLURM_JOB_ID'),no_pretrained_generator_steps=True))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['cpu','gpu']);p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id
 if a.mode=='cpu':
  run.mkdir(exist_ok=False);(run/'logs').mkdir();(run/'audit').mkdir();(run/'executions').mkdir()
 try:globals()[a.mode](run)
 except Exception:
  import traceback
  dump(run/(a.mode+'_failure.json'),dict(traceback=traceback.format_exc(),preserved=True));raise
