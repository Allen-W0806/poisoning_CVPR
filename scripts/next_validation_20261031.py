"""Bounded benign follow-up; immutable protocol, real reviewers never synthesized."""
import argparse,copy,csv,hashlib,importlib.util,json,os,random,re,shutil,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
PARENT='small_scale_validation_20261031T1928Z_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def table(p,rows):
 with Path(p).open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def live(run,stage):
 (run/'live.json').write_text(json.dumps(dict(stage=stage,UTC=datetime.now(timezone.utc).isoformat(),job=os.getenv('SLURM_JOB_ID'))))
def prepare(run):
 from PIL import Image
 import numpy as np
 from skimage.metrics import structural_similarity
 old=ROOT/'results/semantic_clean_20261029_v1/semantic_review';manifest=read(old/'blinded_manifest.json');cases=sorted(manifest['cases'],key=lambda r:r['source_image_id']);assert len(cases)==160
 protocol=dict(run_id=run.name,parent=PARENT,created_utc=datetime.now(timezone.utc).isoformat(),selection_seed=20261031032,review_n=32,review_population=160,review_manifest_sha256=sha(old/'blinded_manifest.json'),gpu_cap_seconds=600,global_validation_cap_seconds=3600,paired_BF16=['ordinary','instrumented'],BF16_repetitions_per_path=1,BF16_acceptance='original bitwise assertions unchanged; no rerun-until-pass',Qwen_evaluation='existing 32 assessment originals, development-only: historical exposure unresolved',Qwen_decoding='unchanged pinned greedy math SDPA; two fresh loads each adapter',integrated=dict(objective='terminal cos(first latent coordinate)',batch=10,steps_per_arm=4,replay_steps_per_arm=1,diffusion_steps=2,seeds={'initial':42,'train':[100,200,300,400],'gate':9000},candidate_lr=2e-5,fixed_lr=1e-5,gate_n=16,acceptance='strict positive paired validation mean vs equal-step fixed arm; finite and budget gates',no_test_selection=True),reward_guard_sha256=sha(ROOT/'src/rl/phase5_backend.py'),no_attack_rewards=True)
 put(run/'protocol.json',protocol) # Written BEFORE random selection, metrics or inference.
 selected=random.Random(protocol['selection_seed']).sample(cases,32);kit=run/'semantic_review32';kit.mkdir();(kit/'images').mkdir();(kit/'submitted').mkdir()
 for c in selected:
  for key in ['original','edited','geometry_control']:
   p=old/c[key];target=kit/c[key];target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,target);assert sha(p)==sha(target)
 data={**manifest,'cases':selected};put(kit/'blinded_manifest.json',data)
 page=(old/'index.html').read_text(encoding='utf-8')
 start=page.index('const DATA=');end=page.index(';const $=',start);page=page[:start]+'const DATA='+json.dumps(data)+page[end:]
 page=page.replace('160 matched edits','32 randomly selected matched edits').replace('/ 160','/ 32').replace('Math.min(159,index+1)','Math.min(DATA.cases.length-1,index+1)').replace('independent-semantic-v1:','independent-semantic-32-'+run.name+':')
 (kit/'index.html').write_text(page,encoding='utf-8')
 for name in ['reviewer_1_template.csv','reviewer_2_template.csv','adjudication_template.csv']:
  with (old/name).open(encoding='utf-8-sig',newline='') as f:reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
  with (kit/name).open('x',newline='',encoding='utf-8') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(r for r in rows if r.get('source_image_id') in {c['source_image_id'] for c in selected})
 (kit/'INSTRUCTIONS.md').write_text('Two distinct genuine human reviewers must work independently without predictions or each other\'s labels. Review whether the original official answer remains visually correct (yes), changes (no), or is ambiguous. Export separately into submitted/. Do not treat blank templates as reviews. A third independent adjudicator resolves disagreements separately; retain both original labels. This predeclared subset uses seed 20261031032, 32 of 160, without model-score selection. The interface includes original, edited, geometry-only, original question and all official reference answers. Full 160-image kit remains unchanged.\n',encoding='utf-8')
 put(kit/'selection_receipt.json',dict(protocol_sha256=sha(run/'protocol.json'),selected_ids=[c['source_image_id'] for c in selected],blinded=True,reviewer_exports_received=0,independent_review_status='BLOCKED',disagreement_status='NOT YET MEASURABLE'))
 metrics=[]
 for c in cases:
  images={}
  for key in ['original','edited','geometry_control']:
   p=old/c[key]
   with Image.open(p) as im:im.load();images[key]=np.array(im.convert('RGB'))
   expected=c.get(key+'_sha256');assert not expected or sha(p)==expected
  assert images['edited'].shape==images['geometry_control'].shape
  ref=images['geometry_control']
  for condition,img in [('geometry_only',ref),('generated',images['edited'])]:
   metrics.append(dict(source_image_id=c['source_image_id'],condition=condition,reference='stored C5 geometry-matched original',SSIM=float(structural_similarity(ref,img,channel_axis=2,data_range=255)),RGB_MAE_255=float(np.abs(ref.astype(float)-img.astype(float)).mean()),LPIPS=None,LPIPS_status='BLOCKED: calibrated LPIPS implementation unavailable' if importlib.util.find_spec('lpips') is None else 'NOT TESTED',original_size=c['original_size'],evaluated_size=list(img.shape[:2]),semantic_status='UNREVIEWED'))
 table(run/'aligned_image_quality.csv',metrics);put(run/'image_quality_summary.json',dict(n_sources=160,all_pairs_decode_hash_PASS=True,generated_SSIM_mean=float(np.mean([r['SSIM'] for r in metrics if r['condition']=='generated'])),generated_SSIM_range=[float(min(r['SSIM'] for r in metrics if r['condition']=='generated')),float(max(r['SSIM'] for r in metrics if r['condition']=='generated'))],generated_MAE_mean=float(np.mean([r['RGB_MAE_255'] for r in metrics if r['condition']=='generated'])),LPIPS=None,cached_metric_modules={n:importlib.util.find_spec(n) is not None for n in ['lpips','torchmetrics','piq']},geometry_interpretation='SSIM 1 for C5 vs itself is a pipeline control, not native-resolution fidelity or semantic proof'))
 # Freshness is a provenance property. Do not rename known historical assessment as a test.
 census=read(ROOT/'results'/PARENT/'audit/data_integrity.json');put(run/'evaluation_provenance.json',dict(status='BLOCKED',scope='development-only',reason='No genuinely unused cached subset certified; unresolved historical census records and previous feedback exposure',historical_census_errors=census['exposure_scan_errors'],previous_known_image_count=census['known_prior_image_count'],cached_images=census['cached_images'],new_model_selection=False))
 source=read(ROOT/'results/replication_monitor_20261029_v1_ip2pix_a0/protocol.json')
 for kind in ['ordinary','instrumented']:
  child=ROOT/'results'/f'{run.name}_BF16_{kind}';child.mkdir();put(child/'protocol.json',{**source,'parent_run':run.name})
 put(run/'cpu_complete.json',dict(status='PASS',review_pairs=32,audited_pairs=160,LPIPS='BLOCKED',human_reviews=0))
def execute(run,name,args):
 start=time.perf_counter()
 with (run/'logs'/f'{name}.log').open('x') as f:r=subprocess.run([sys.executable,'-B',*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 row=dict(task=name,returncode=r.returncode,seconds=time.perf_counter()-start,args=args);put(run/'executions'/f'{name}.json',row);return row
def qwen(run):
 import torch,numpy as np,gc
 from PIL import Image
 from peft import PeftModel
 from src.models.qwen_loader import QwenVQA
 from src.evaluation.vqa_official_compatible import score_answer
 from src.utils.seed import seed_everything
 cfg=read(ROOT/'results'/f'{PARENT}_qwen_s42/protocol.json');rows=read(ROOT/'results'/PARENT/'split_manifest.json')['splits']['independent_evaluation'];records=[];integrity=[]
 for seed in [None,42,123,3407]:
  for repeat in range(2):
   seed_everything(42 if seed is None else seed);q=QwenVQA(model_id=cfg['model_id'],revision=cfg['model_revision'],device='cuda',max_new_tokens=cfg['max_new_tokens'],min_pixels=cfg['min_pixels'],max_pixels=cfg['max_pixels'],seed=42 if seed is None else seed)
   if seed is not None:
    out=ROOT/'results'/f'{PARENT}_qwen_s{seed}/clean_main';inventory=read(out/'artifact_inventory.json')
    for r in inventory['files']:
     if r['path'].startswith('adapter/') or r['path'] in ['continuation_checkpoint.pt','final_training_state.pt']:
      integrity.append(dict(seed=seed,repeat=repeat,path=r['path'],matches=sha(out/r['path'])==r['sha256']))
    assert all(x['matches'] for x in integrity)
    q.model=PeftModel.from_pretrained(q.model,out/'adapter',is_trainable=False);q.model.eval().requires_grad_(False)
   assert all(not p.requires_grad for p in q.model.parameters())
   with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
    for r in rows:
     assert sha(ROOT/r['image_path'])==r['image_sha256']
     with Image.open(ROOT/r['image_path']) as im:im=im.convert('RGB')
     torch.cuda.synchronize();t=time.perf_counter();answer=q.answer([im],[r['question']])[0];torch.cuda.synchronize();sc=score_answer(answer,r)
     row=dict(seed=seed,repeat=repeat,sample_id=r['sample_id'],image_id=r['image_id'],raw_answer=answer,normalized_answer=sc['normalized_answer'],score=sc['score'],latency_seconds=time.perf_counter()-t);records.append(row)
     with (run/'qwen_predictions.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   del q;gc.collect();torch.cuda.empty_cache()
 base={r['sample_id']:r for r in records if r['seed'] is None and r['repeat']==0};stats=[]
 for seed in [None,42,123,3407]:
  r0={r['sample_id']:r for r in records if r['seed']==seed and r['repeat']==0};r1={r['sample_id']:r for r in records if r['seed']==seed and r['repeat']==1};assert set(r0)==set(r1)==set(base)
  delta=np.array([r0[i]['score']-base[i]['score'] for i in base]);g=np.random.default_rng(20261031);boot=g.integers(32,size=(5000,32));equal=all(r0[i]['normalized_answer']==r1[i]['normalized_answer'] for i in base)
  stats.append(dict(seed=seed,n=32,accuracy=float(np.mean([r0[i]['score'] for i in base])),paired_difference=float(delta.mean()),paired_ci95=np.quantile(delta[boot].mean(1),[.025,.975]).tolist(),changed_answers=sum(r0[i]['normalized_answer']!=base[i]['normalized_answer'] for i in base),fresh_object_repeat_exact=equal,mean_latency=float(np.mean([r['latency_seconds'] for r in records if r['seed']==seed])),scope='historical development only'))
 put(run/'qwen_recheck.json',dict(status='PASS' if all(r['fresh_object_repeat_exact'] for r in stats) else 'FAIL',stats=stats,checkpoint_integrity=integrity,optimizer_steps=0,n_calls=len(records),model_revision=cfg['model_revision'],config_sha256=sha(ROOT/'results'/f'{PARENT}_qwen_s42/protocol.json')))
def gpu(run):
 import torch
 torch.set_num_threads(4);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 assert os.getenv('SLURM_JOB_ID') and torch.cuda.is_available()
 p=read(run/'protocol.json');assert sha(ROOT/'src/rl/phase5_backend.py')==p['reward_guard_sha256'];executions=[]
 for kind,script in [('ordinary','run_ip2pix_checkpoint_diagnostic.py'),('instrumented','run_ip2pix_checkpoint_instrumented.py')]:
  live(run,'BF16 '+kind);executions.append(execute(run,'BF16_'+kind,['scripts/'+script,'--run-id',run.name+'_BF16_'+kind]))
 live(run,'integrated benign DDPO RSI');executions.append(execute(run,'integrated',['scripts/integrated_benign_rsi_20261031.py','--run-id',run.name]))
 live(run,'existing Qwen checkpoints, no training')
 try:qwen(run)
 except Exception:(run/'qwen_failure.txt').write_text(traceback.format_exc());executions.append(dict(task='Qwen',returncode=1))
 assert sha(ROOT/'src/rl/phase5_backend.py')==p['reward_guard_sha256']
 put(run/'gpu_complete.json',dict(status='COMPLETED_WITH_LIMITATIONS',executions=executions,ordinary_failures_preserved=True,reward_guard_unchanged=True,GPU=torch.cuda.get_device_name(),allocated_job=os.environ['SLURM_JOB_ID']))
 live(run,'completed; inspect all assertions')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['prepare','gpu']);ap.add_argument('--run-id',required=True);a=ap.parse_args();run=ROOT/'results'/a.run_id
 if a.mode=='prepare':run.mkdir(exist_ok=False);(run/'logs').mkdir();(run/'executions').mkdir();prepare(run)
 else:gpu(run)
