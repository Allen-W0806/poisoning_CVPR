"""CPU-only comparison of selected pretrained-pilot outputs; never trains.

The archived four-source observations are reproduced first. New image metrics
are then computed on those same saved PNGs. No final-pool image is opened.
"""
import csv, hashlib, itertools, json, math, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PILOT=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot'
ARM_NAMES={'frozen':'Frozen','ddpo_only':'DDPO-only','ordinary_scheduler':'Ordinary Scheduler','rsi':'DDPO + RSI'}
ARMS=list(ARM_NAMES)
BASE='31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
 Path(p).parent.mkdir(parents=True,exist_ok=True)
 Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def csvout(p,rows):
 with Path(p).open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def query(args):
 p=subprocess.run(args,cwd=ROOT,env={**os.environ,'TZ':'UTC'},capture_output=True,text=True,timeout=60)
 p.check_returncode();return {'command':args,'stdout':p.stdout,'stderr':p.stderr}
def interval(x):return '['+f'{x[0]:.8f}, {x[1]:.8f}'+']'
def mdtable(headers,rows):return '| '+' | '.join(headers)+' |\n|'+ '|'.join(['---']*len(headers))+'|\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)

def main():
 import numpy as np,torch
 from PIL import Image,ImageOps
 from skimage.metrics import structural_similarity
 wall,cpu=time.perf_counter(),time.process_time();torch.set_num_threads(2)
 stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 out=ROOT/'results'/('quick_comparison_'+stamp);out.mkdir(exist_ok=False)
 cfg,src,observations,result=map(read,[PILOT/'protocol.json',PILOT/'source_manifest.json',PILOT/'prediction_observations.json',PILOT/'results.json'])
 assert sha(PILOT/'source_manifest.json')==cfg['source_manifest_sha256']
 original_by={x['arm']:x for x in result['branches']}
 assessment=src['assessment'];ids=[str(x['image_id']) for x in assessment]
 train_ids={str(x['image_id']) for x in src['train']};train_hashes={x['image_sha256'] for x in src['train']}
 assert len(ids)==len(set(ids))==4 and not set(ids)&train_ids
 assert not {x['image_sha256'] for x in assessment}&train_hashes
 # First reproduce saved observations, with no image metric recomputation.
 rows={a:[x for x in observations if x['phase']==a+'/selected_assessment'] for a in ARMS}
 saved_reproduction=[]
 for a in ARMS:
  rr=rows[a];assert len(rr)==4 and [str(x['source_image_id']) for x in rr]==ids
  assert [x['seed'] for x in rr]==[4220000,4220001,4220002,4220003]
  values=[x['reward'] for x in rr];err=abs(float(np.mean(values))-original_by[a]['assessment_mean'])
  assert err<=1e-12 and np.array_equal(values,original_by[a]['assessment_scores'])
  saved_reproduction.append(dict(arm=a,source_ids=ids,mean=float(np.mean(values)),max_mean_reproduction_error=err,exact_saved_score_vector=True))
 put(out/'saved_observation_reproduction.json',saved_reproduction)
 # Read current allocation and admission evidence, never call controllers.
 live={}
 live['squeue']=query(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%M|%S|%b'])
 live['sacct']=query(['sacct','-X','-nP','-u',os.environ['USER'],'--starttime=2026-11-02T00:00:00','--endtime=2026-11-04T00:00:00','--format=JobIDRaw,JobName%60,State,Submit,Start,End,ElapsedRaw,AllocTRES%160,WorkDir%200,SubmitLine%2200'])
 live['pilot_allocation']=query(['sacct','-X','-nP','-j','88392','--format=JobIDRaw,State,Submit,Start,End,ElapsedRaw,AllocTRES%160,ExitCode'])
 live['scheduler']=read(ROOT/'reports/monitoring/scheduler.json')
 live['overnight_state']=read(ROOT/'reports/overnight_benign_20261102_v2/state.json')
 live['urgent_budget']=read(ROOT/'reports/numerical_fix_20261102_v1/budget.json')
 live['old_validation_state']=read(ROOT/'reports/overnight_supervisor/state.json')
 live['authorization_expired']=time.time()>=live['overnight_state']['deadline_epoch']
 assert live['authorization_expired'] and live['overnight_state']['admissions_closed']
 alloc=live['pilot_allocation']['stdout'].strip().split('|');assert alloc[0]=='88392' and alloc[1]=='COMPLETED'
 pilot_gpu_seconds=int(alloc[5]);assert 'gres/gpu=1' in alloc[6]
 put(out/'live_resource_and_authorization.json',live)
 # Verify selected checkpoints, sidecars and content-bound pretrained base.
 meta=read(ROOT/'reports/phase2_model_metadata.json')['timbrooks/instruct-pix2pix'];assert meta['revision']==BASE
 base_files=[]
 for name in meta['files']:
  p=Path(meta['path'])/name;digest=sha(p);resolved=p.resolve()
  blob_match=digest==resolved.name if len(resolved.name)==64 else None
  assert blob_match is not False
  base_files.append(dict(file=name,sha256=digest,bytes=p.stat().st_size,cache_blob_sha256_matches=blob_match))
 lineage=[]
 for a in ARMS:
  p=PILOT/a/'selected/training_state.pt';digest=sha(p);side=read(p.parent/'active_runtime_config.json')
  state=torch.load(p,map_location='cpu',weights_only=False)
  assert digest==original_by[a]['selected_checkpoint_sha256']==side['checkpoint_sha256']
  assert state['step']==original_by[a]['retained_checkpoint_step']
  assert state['contract']['model']['revision']==BASE
  kept=0 if a=='frozen' else (10 if a=='rsi' else 20)
  retained=src['train'][:kept]
  lineage.append(dict(arm=a,method=ARM_NAMES[a],checkpoint=str(p.relative_to(ROOT)),sha256=digest,bytes=p.stat().st_size,selected_step=state['step'],attempted_steps=original_by[a]['attempted_updates'],attempted_training_sources=0 if a=='frozen' else 20,retained_training_sources=kept,retained_training_source_ids=[str(x['image_id']) for x in retained],attempted_training_source_ids=[] if a=='frozen' else sorted(train_ids),validation_sources=4,base_model=meta['id'],base_revision=BASE,base_variant=meta['variant'],contract_model=state['contract']['model'],active_LR=side['active_optimizer_lrs'],adapter_tensors=len(state['adapter']),adapter_elements=sum(v.numel() for v in state['adapter'].values()),adapter_dtypes=sorted({str(v.dtype) for v in state['adapter'].values()}),training_job='88392',selected_before_assessment=True,selection='fixed step0' if a=='frozen' else ('own validation rejected candidate step2; restored own step1 prefix' if a=='rsi' else 'predeclared step2'),assessment_train_ID_overlap=0,assessment_train_exact_hash_overlap=0))
  del state
 put(out/'checkpoint_and_base_lineage.json',dict(checkpoints=lineage,base_files=base_files,source_manifest_sha256=sha(PILOT/'source_manifest.json')))
 # Calibrated LPIPS: verify both trunk bytes and learned calibration tensors.
 dependency=read(ROOT/'results/execution_remediation_20261102_v1/lpips_dependency_verification.json')
 assert sha(dependency['base_file'])==dependency['base_sha256']
 calpath=Path(dependency['module_root'])/'lpips/weights/v0.1/alex.pth';assert sha(calpath)==dependency['calibration_sha256']
 sys.path.insert(0,dependency['module_root']);import lpips
 metric=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,verbose=False).cpu().eval().requires_grad_(False)
 calibration=torch.load(calpath,map_location='cpu',weights_only=True)
 assert all(torch.equal(metric.state_dict()[k],v) for k,v in calibration.items())
 t=lambda x:torch.from_numpy(x.copy()).permute(2,0,1).unsqueeze(0).float()/255.
 per_source=[];metric_seconds={a:0. for a in ARMS};identity=[]
 for i,r in enumerate(assessment):
  original=ROOT/r['image_path'];assert sha(original)==r['image_sha256']
  with Image.open(original) as im:reference=ImageOps.pad(im.convert('RGB'),(512,512),color='white',method=3)
  x=np.asarray(reference).copy();luminance=lambda a:float(((a.astype(np.float64)/255.)*np.array([.2126,.7152,.0722])).sum(axis=2).mean())
  before=luminance(x);target=min(1.,1.03*before)
  with torch.inference_mode():identity.append(float(metric(t(x),t(x),normalize=True)))
  for a in ARMS:
   start=time.perf_counter();p=PILOT/a/'selected_assessment'/f'{i}.png';obs=rows[a][i]
   assert sha(p)==obs['image_sha256']
   with Image.open(p) as im:y=np.asarray(im.convert('RGB')).copy()
   assert x.shape==y.shape==(512,512,3)
   after=luminance(y);reward=abs(before-target)-abs(after-target)
   assert abs(reward-obs['reward'])<1e-12 and abs(before-obs['source_mean_luminance'])<1e-12
   with torch.inference_mode():distance=float(metric(t(x),t(y),normalize=True))
   elapsed=time.perf_counter()-start;metric_seconds[a]+=elapsed
   per_source.append(dict(arm=a,source_image_id=str(r['image_id']),sample_id=r['sample_id'],generation_seed=obs['seed'],source_sha256=r['image_sha256'],saved_edit_sha256=obs['image_sha256'],brightness_proxy=reward,SSIM=float(structural_similarity(x,y,channel_axis=2,data_range=255)),calibrated_LPIPS=distance,exact_noop=int(np.array_equal(x,y)),original_reference_VQA_agreement=None,metric_CPU_wall_seconds=elapsed,source_scope='DEVELOPMENT ONLY',generation_job='88392'))
 assert max(map(abs,identity))<=1e-12
 csvout(out/'per_source_metrics.csv',per_source)
 # Exact enumeration of all 4^4 paired source bootstrap draws, shared for arms.
 indices=np.asarray(list(itertools.product(range(4),repeat=4)))
 ci=lambda v:np.quantile(np.asarray(v)[indices].mean(axis=1),[.025,.975]).tolist()
 get=lambda a,k:np.asarray([x[k] for x in per_source if x['arm']==a],dtype=float)
 summaries=[]
 for k in ['brightness_proxy','SSIM','calibrated_LPIPS','exact_noop','original_reference_VQA_agreement']:
  for line in lineage:
   a=line['arm'];v=get(a,k);b=get('frozen',k);available=k!='original_reference_VQA_agreement'
   if available:
    mean=float(v.mean());delta=float((v-b).mean());bounds=ci(v);paired=ci(v-b)
    if k=='exact_noop':
     z=1.959963984540054;n=len(v);phat=mean;center=(phat+z*z/(2*n))/(1+z*z/n);half=z*math.sqrt(phat*(1-phat)/n+z*z/(4*n*n))/(1+z*z/n);bounds=[max(0.,center-half),min(1.,center+half)]
   else:mean=delta=None;bounds=paired=[None,None]
   summaries.append(dict(method=ARM_NAMES[a],arm=a,checkpoint=line['checkpoint'],checkpoint_sha256=line['sha256'],selected_optimizer_step=line['selected_step'],attempted_training_source_count=line['attempted_training_sources'],retained_training_source_count=line['retained_training_sources'],evaluation_source_count=4,metric=k,value=mean,CI95_low=bounds[0],CI95_high=bounds[1],paired_difference_vs_Frozen=delta,paired_CI95_low=paired[0],paired_CI95_high=paired[1],uncertainty_method='NOT TESTED' if not available else ('Wilson source-rate interval; paired exact bootstrap' if k=='exact_noop' else 'exact enumeration of256 paired source-bootstrap draws'),metric_status='MEASURED' if available else 'NOT TESTED',generation_job_ID='88392',generation_allocation_GPU_seconds_shared=pilot_gpu_seconds,per_arm_allocation_GPU_seconds=None,assessment_UNet_forward_calls=24,new_GPU_seconds=0,new_metric_CPU_wall_seconds=metric_seconds[a],scope='DEVELOPMENT ONLY; 1 training seed; no independent edited labels',numerical_limitation='BF16 ordinary/raw-gradient inconsistency unresolved; fixed-checkpoint read-only description remains valid',ASR='NOT TESTED'))
 csvout(ROOT/'reports/QUICK_BASELINE_COMPARISON.csv',summaries)
 # Fix the prospective expanded development cohort before any new inference.
 # Only ID/hash exclusion metadata is consulted for the sealed final pool.
 sealed_path=ROOT/'results/execution_remediation_20261102_v1/prospective/candidate_content_manifest.json'
 sealed=read(sealed_path);sealed_ids=set();sealed_hashes=set()
 def scan(v):
  if isinstance(v,dict):
   for k,x in v.items():
    if k in ['image_id','source_image_id'] and isinstance(x,(str,int)):sealed_ids.add(str(x))
    if k in ['image_sha256','sha256','content_sha256','original_sha256'] and isinstance(x,str) and len(x)==64:sealed_hashes.add(x)
    if isinstance(x,(dict,list)):scan(x)
  elif isinstance(v,list):
   for x in v:scan(x)
 scan(sealed)
 assert len(sealed_ids)==160
 candidate_manifest=ROOT/'data/processed/vqav2_seed42_v1/validation.json';candidate_rows=read(candidate_manifest)
 used_ids=train_ids|{str(x['image_id']) for x in src['validation']}|set(ids)|sealed_ids
 used_hashes=train_hashes|{x['image_sha256'] for x in src['validation']}|{x['image_sha256'] for x in assessment}|sealed_hashes
 eligible=[]
 # Selection uses IDs only; no reward, metric, answer, or checkpoint score.
 for r in sorted(candidate_rows,key=lambda r:hashlib.sha256(('quick-dev-v1:'+str(r['image_id'])).encode()).hexdigest()):
  iid=str(r['image_id']);h=r['image_sha256']
  if iid in used_ids or h in used_hashes:continue
  p=ROOT/r['image_path']
  if not p.is_file() or sha(p)!=h:continue
  eligible.append(r);used_ids.add(iid);used_hashes.add(h)
  if len(eligible)==60:break
 assert len(eligible)==60
 expanded=[]
 for i,r in enumerate(assessment+eligible):expanded.append(dict(source_image_id=str(r['image_id']),sample_id=r['sample_id'],image_path=r['image_path'],image_sha256=r['image_sha256'],seed=4220000+i,source_manifest=str(candidate_manifest.relative_to(ROOT)) if i>=4 else str((PILOT/'source_manifest.json').relative_to(ROOT)),scope='DEVELOPMENT ONLY',previous_selection_feedback='previous four-source assessment' if i<4 else 'historical development manifest; not used for these four-arm checkpoint selection'))
 assert len(expanded)==len({x['source_image_id'] for x in expanded})==64
 assert not {x['source_image_id'] for x in expanded}&(train_ids|sealed_ids|{str(x['image_id']) for x in src['validation']})
 put(out/'expanded_development_cohort.json',expanded)
 proposal=dict(status='PREPARED_NOT_EXECUTED_GPU_AUTHORIZATION_REQUIRED',checkpoints=lineage,cohort_sha256=sha(out/'expanded_development_cohort.json'),source_count=64,added_sources=60,common_for_every_arm=True,excluded_all_attempted_training_sources=20,excluded_selection_validation_sources=4,sealed_source_IDs_excluded=160,no_sealed_images_opened=True,no_checkpoints_selected_from_evaluation=True,conditions=dict(instruction='Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.',preprocessing='PIL RGB, ImageOps.pad512 square white BICUBIC, no crop',num_inference_steps=2,eta=1.0,guidance_scale=7.5,image_guidance_scale=1.5,CFG_execution='sequential',SDPA='MATH',precision='original BF16 path; FP32 LoRA and guidance',base_revision=BASE,parameter_updates=0,checkpoint_load_then_all_requires_grad_false=True),required_generated_images=256,generation_UNet_NFE=1536,pricing=dict(measured_reference='Frozen branch8 images plus save/reload bookkeeping',reference_seconds=original_by['frozen']['incremental_GPU_held_seconds'],projected_generation_component_seconds=original_by['frozen']['incremental_GPU_held_seconds']/8*256,requested_cap_A100_GPU_seconds=600,requested_GPU_count=1,requested_wall_minutes=10,CPU_metric_GPU_seconds=0,currency_price='UNAVAILABLE; no cluster dollar rate supplied',optional_VQA='NOT INCLUDED; no four-arm saved VQA answers. Separate pinned-scorer phase needs separate priced authorization and original-reference-only label.'),uncertainty='source paired bootstrap fixed seed20261102;10000 resamples across64 IDs;one-seed descriptive intervals only',acceptance=['Every arm has same64 IDs,seeds,instruction,preprocessing','No train/selection/final-pool ID or exact-content overlap','Verify fixed checkpoint/base/input/output hashes and parameter invariance before/after','No backward,optimizer step,reward feedback,or checkpoint selection','Publish all finite results including negative values; no gain gate','Report missing/error sources; paired comparisons only complete common IDs, never impute','No independent edited-label correctness or security-effectiveness claim'],currently_authorized_GPU_seconds=0)
 put(out/'expanded_readonly_protocol.json',proposal)
 wall_seconds=time.perf_counter()-wall;cpu_seconds=time.process_time()-cpu
 put(out/'result_summary.json',dict(created_UTC=datetime.now(timezone.utc).isoformat(),saved_observations_reproduced=True,source_ids=ids,independent_source_count=4,training_seeds=1,metric_rows=summaries,identity_LPIPS_max=max(map(abs,identity)),calibrated_layers_exact=True,LPIPS_calibration_sha256=dependency['calibration_sha256'],LPIPS_trunk_sha256=dependency['base_sha256'],CPU_wall_seconds=wall_seconds,CPU_process_seconds=cpu_seconds,new_GPU_seconds=0,new_model_generation_calls=0,new_VQA_calls=0,evidence=str(out.relative_to(ROOT)),expanded_protocol=proposal))
 lookup={(x['arm'],x['metric']):x for x in summaries}
 table=[]
 for line in lineage:
  a=line['arm'];b,s,l,n=(lookup[a,k] for k in ['brightness_proxy','SSIM','calibrated_LPIPS','exact_noop'])
  table.append([ARM_NAMES[a],f"{line['attempted_training_sources']} / {line['retained_training_sources']}",line['selected_step'],4,f"{b['value']:.9f}",f"{b['paired_difference_vs_Frozen']:+.9f} {interval([b['paired_CI95_low'],b['paired_CI95_high']])}",f"{s['value']:.9f}",f"{l['value']:.9f}",'0/4'])
 report=f'''# Quick pretrained IP2Pix comparison

PROJECT_ID: RSI-POISONING. Generated UTC {datetime.now(timezone.utc).isoformat()}. **DEVELOPMENT ONLY: four independent source IDs, one training seed42.** Actual saved selected checkpoints and PNGs from job88392 are compared. Read-only evaluation does not depend on backward bitwise reproducibility. No parameters were updated, no checkpoint was reselected, and no sealed final-pool image was opened.

{mdtable(['Method','Training IDs attempted/retained','Selected step','Eval IDs','Brightness proxy (higher better)','Paired delta vs Frozen [95% source CI]','SSIM (higher similarity)','Calibrated LPIPS (lower distance)','Exact no-op'],table)}
The trained-arm brightness means are all below Frozen. All paired brightness intervals include zero. This is a measured negative development result, not evidence of useful learning or causal RSI superiority. Negative brightness proxy means the edit moved global luminance farther from the target than doing nothing. Similarity to the source is not instruction-following or semantic correctness.

## Reproduction and metric definitions

Saved four-source score vectors and arm means reproduce within1e-12 (actual maximum error {max(x['max_mean_reproduction_error'] for x in saved_reproduction):.3g}). Newly computed PNG brightness reproduces every saved score within1e-12. Same assessment source IDs `{', '.join(ids)}` and generation seeds4220000–4220003 appear in every arm. No source-ID or exact-content overlap with any of the20 attempted pilot training sources. The retained RSI checkpoint contains10 source exposures; its rejected branch attempted another10. All20 are conservatively excluded from held-out and expanded comparisons. Four separate validation sources were used for scheduling/RSI decisions and are excluded here.

Brightness proxy: |L_source−min(1,1.03 L_source)|−|L_edit−min(1,1.03 L_source)|. L is mean RGB[0,1] weighted by (0.2126,0.7152,0.0722), without sRGB linearization. Units are normalized pixel luminance, not percent accuracy. SSIM uses skimage structural_similarity on geometry-aligned512×512 RGB uint8 arrays, channel_axis2,data_range255,default7-pixel uniform window. LPIPS is calibrated AlexNet v0.1 from lpips0.1.4; normalize=True maps RGB[0,1] to[-1,1]. Learned calibration SHA256 `{dependency['calibration_sha256']}` and trunk SHA256 `{dependency['base_sha256']}` were checked; all loaded calibration tensors match and identity LPIPS is0. Exact no-op is pixel equality after identical source preprocessing; its operational definition detects exact identity only.

All means and paired contrasts, including SSIM, LPIPS and no-op, with intervals are in the CSV and table below. Intervals enumerate all256 equally weighted ordered bootstrap resamples of4 source IDs. The paired resample indices are shared across arms. These exploratory conditional intervals do not capture training-seed variability, dataset selection bias or independent semantic uncertainty. No-op marginal intervals use Wilson95%;0/4 has an upper bound around0.49, despite an empirical paired-bootstrap zero interval when every observed pair is zero.

'''
 alltable=[]
 for x in summaries:
  alltable.append([x['method'],x['metric'],'NOT TESTED' if x['value'] is None else f"{x['value']:.9f}",'NOT TESTED' if x['value'] is None else interval([x['CI95_low'],x['CI95_high']]),'NOT TESTED' if x['value'] is None else f"{x['paired_difference_vs_Frozen']:+.9f}",'NOT TESTED' if x['value'] is None else interval([x['paired_CI95_low'],x['paired_CI95_high']])])
 report+=mdtable(['Method','Metric','Mean','95% source interval','Delta Frozen','Paired95% interval'],alltable)
 report+='\n## Checkpoint and training lineage\n\n'
 report+=mdtable(['Method','Selected checkpoint relative to canonical project','SHA256','Training/selection history'],[[x['method'],'`'+x['checkpoint']+'`','`'+x['sha256']+'`',x['selection']] for x in lineage])
 report+=f'''\nPinned base: `timbrooks/instruct-pix2pix@{BASE}`, fp16 artifact variant loaded in BF16; exact selected-state contracts and runtime sidecars agree. All four selected files were hashed now, and their hashes equal the original result receipts and sidecar bindings. Base model files were hashed now; safetensors hashes match their content-addressed cache blobs. Full file hashes and selected contracts are in checkpoint_and_base_lineage.json. These .pt files contain LoRA and optimizer state, not duplicated full base models; each has256 FP32 LoRA tensors /1,594,368 elements. LoRA was already FP32, which did not prevent the historical numerical mismatch.

Job88392 performed four physical optimizer calls total: one shared prefix plus three branch candidates. Each trained arm attempted two logical updates; RSI rejected its LR5e-6 step2 candidate using its own validation and restored its own LR1e-5 step1 prefix. DDPO retained step2 LR1e-5; ordinary ReduceLROnPlateau retained step2 LR5e-6. Fixed step rules and validation rollback preceded assessment. Common shared-prefix cost is counted once.

## Compute and live resource control

Live sacct rechecked job88392: submit {alloc[2]}, start {alloc[3]}, end {alloc[4]} UTC, COMPLETED, **{pilot_gpu_seconds} A100 GPU-seconds shared by the entire pilot**. The allocation includes loading, training, selection, validation, assessment and writes; evaluation-only GPU time was not separately measured and cannot be attributed to individual rows. Four assessment images per arm imply24 UNet forward calls each,96 total. The frozen8-image validation+assessment branch component took {original_by['frozen']['incremental_GPU_held_seconds']:.6f}s; that includes checkpoint bookkeeping and is not a pure inference benchmark. Per-arm branch timings are retained in original receipts, not multiplied into four independent allocation costs.

This result-processing run used **0 GPU-seconds**, {wall_seconds:.3f}s CPU wall time and {cpu_seconds:.3f}s CPU process time on the existing login CPU process (no new Slurm job). It made0 editor-generation and0 VQA calls. Current live queue and full allocation records are saved in live_resource_and_authorization.json. Overnight and urgent-debug admissions remain closed; the10-hour authorization expired at {live['overnight_state']['deadline_UTC']}. No new valid GPU allocation authorization was found in the current request or live control ledgers. The remaining expired balances are not usable. Monitor {live['scheduler']['next_job_id']} remains due {live['scheduler']['next_due_utc']}; no duplicate monitor was created.

## Expanded read-only evaluation, prepared and separately priced

Prepared a fixed **64-source DEVELOPMENT ONLY** cohort: the existing4 assessment IDs plus60 deterministically selected available unique historical validation sources. Selection hashes source IDs only and does not inspect scores. All20 attempted pilot training IDs/content hashes,4 selection-validation IDs/content hashes,160 sealed final IDs/content hashes and duplicate IDs/content are excluded. Sealed metadata was consulted only for exclusion; no final image or answer was used. The same64 sources, fixed seeds, editing instruction,512 white-pad geometry,2-step DDIM eta1,7.5/1.5 guidance, sequential CFG and math SDPA apply to all four fixed selected checkpoints. Cohort and protocol hashes are frozen in `{out.relative_to(ROOT)}/expanded_development_cohort.json` and expanded_readonly_protocol.json. No further checkpoint selection is permitted.

Price:256 generated images /1536 UNet forward calls. Scaling the measured frozen branch component gives about {proposal['pricing']['projected_generation_component_seconds']:.1f} A100-seconds, with substantial setup/throughput uncertainty. Proposed hard allocation cap **600 A100 GPU-seconds**,1 GPU,10-minute wall limit, followed by CPU-only SSIM/calibrated LPIPS/brightness/no-op processing. This is a proposed cap, not granted authorization or a guaranteed estimate. Dollar pricing is unavailable. The expanded evaluation has **NOT RUN** because no renewed GPU budget/window exists. Its eligibility does not depend on a BF16 backward fix. Optional original-reference VQA is a separate unpriced/unexecuted extension requiring its own pinned scorer and resource authorization; it is not silently included in the600-second quote.

## Scientific limits

All exposed sources remain DEVELOPMENT ONLY. Original-reference VQA agreement is **NOT TESTED for these four selected-checkpoint outputs**: the pilot made no Qwen calls. Historical frozen-pipeline answers cannot substitute. Independent edited-label correctness, Clean Accuracy and ASR are **NOT TESTED**. No human labels were created. The original scientific reward remains unreleased; this isolated global-brightness engineering proxy does not change it.

BF16 ordinary-path gradient/optimizer inconsistencies remain preserved. They limit reproducible training and attribution of arm differences, but do not invalidate read-only descriptions of these explicitly hashed fixed outputs. No numerical fix or security-effectiveness claim is made. Full per-source metrics, exact source bootstrap, live allocation evidence, checkpoint/base lineage and proposal are retained under `{out.relative_to(ROOT)}`. No old C0–C5 or Tiny-UNet table was regenerated.
'''
 (ROOT/'reports/QUICK_BASELINE_COMPARISON.md').write_text(report,encoding='utf8')
 tex=[r'\begin{table*}[t]',r'\centering',r'\small',r'\caption{DEVELOPMENT ONLY: four independent source IDs, one training seed, fixed pretrained IP2Pix checkpoints from job88392. Brightness is normalized pixel-luminance error improvement. Intervals use all256 paired source-bootstrap draws and do not represent independent semantic validity or training-seed uncertainty. Shared pilot allocation:242 A100 GPU-seconds; new CPU processing:0 GPU-seconds.}',r'\label{tab:quick_ip2pix_development}',r'\begin{tabular}{lrrrrrr}',r'\hline',r'Method & Retained step & Brightness & $\Delta$Frozen [95\% CI] & SSIM & LPIPS & No-op \\',r'\hline']
 for line in lineage:
  a=line['arm'];b,s,l=(lookup[a,k] for k in ['brightness_proxy','SSIM','calibrated_LPIPS'])
  tex.append(f"{ARM_NAMES[a]} & {line['selected_step']} & {b['value']:.9f} & {b['paired_difference_vs_Frozen']:+.9f} [{b['paired_CI95_low']:.9f}, {b['paired_CI95_high']:.9f}] & {s['value']:.9f} & {l['value']:.9f} & 0/4 "+r'\\')
 tex += [r'\hline',r'\end{tabular}',r'\parbox{\textwidth}{\footnotesize Training source counts attempted/retained: Frozen0/0, DDPO20/20, Scheduler20/20, RSI20/10 after own-prefix rollback. No-op Wilson95\% interval is[0,0.4899] for each arm. Complete per-metric intervals, paired contrasts, full checkpoint hashes, compute scope and source counts are in the companion CSV/Markdown. Original-reference VQA agreement, independent edited-label accuracy and ASR are NOT TESTED. All trained-arm brightness means are below Frozen; all paired brightness intervals include zero. Historical BF16 training inconsistency remains unresolved.}',r'\end{table*}']
 (ROOT/'reports/QUICK_BASELINE_COMPARISON.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
 put(out/'delivery_manifest.json',dict(evidence=str(out.relative_to(ROOT)),reports={n:sha(ROOT/'reports'/n) for n in ['QUICK_BASELINE_COMPARISON.md','QUICK_BASELINE_COMPARISON.csv','QUICK_BASELINE_COMPARISON.tex']},source_script_sha256=sha(__file__),source_files={str(p.relative_to(ROOT)):sha(p) for p in [PILOT/'protocol.json',PILOT/'source_manifest.json',PILOT/'prediction_observations.json',PILOT/'results.json']},new_GPU_seconds=0))
 print(json.dumps(dict(evidence=str(out.relative_to(ROOT)),CPU_wall_seconds=wall_seconds,CPU_process_seconds=cpu_seconds,new_GPU_seconds=0,comparison=table,expanded_sources=64,requested_future_A100_seconds=600)))

if __name__=='__main__':main()
