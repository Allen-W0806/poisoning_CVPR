"""Recompute saved matched controls and tiny ablation; descriptive CPU work only."""
import argparse,collections,csv,hashlib,importlib.util,json,math,os,sys,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
from skimage.metrics import structural_similarity
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
REAL=ROOT/'results/matched_feedback_20261028_v2'
ABL=ROOT/'results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,d):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,indent=2,allow_nan=False)
def csvwrite(p,rows):
 fields=list(dict.fromkeys(k for row in rows for k in row))
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def mdtable(head,rows):return '| '+' | '.join(head)+' |\n|'+'|'.join(['---']*len(head))+'|\n'+''.join('| '+' | '.join(str(v) for v in row)+' |\n' for row in rows)
def ci(x):return np.quantile(x,[.025,.975]).tolist()
def interval(x,scale=1,digits=3):return '['+', '.join(f'{v*scale:.{digits}f}' for v in x)+']'
def emit(run,name,text):
 out=run/'publication';out.mkdir(exist_ok=True);p=out/name;p.write_text(text,encoding='utf-8')
 target=ROOT/'reports'/name
 if target.exists() and target.read_bytes()!=p.read_bytes():
  archive=run/'prior_reports';archive.mkdir(exist_ok=True);(archive/(name+'.'+sha(target)[:12])).write_bytes(target.read_bytes())
 tmp=target.with_name(target.name+'.baseline.partial');tmp.write_bytes(p.read_bytes());tmp.replace(target)
def pct(v):return f'{v*100:.4f}'

def real(run,protocol):
 out=run/'real_image';out.mkdir();sources=read(REAL/'main_manifest.json')['records'];assert len(sources)==160
 ids=[str(x['source_image_id']) for x in sources];assert len(set(ids))==160;source={str(x['source_image_id']):x for x in sources}
 allraw=[json.loads(l) for l in (REAL/'main_repair_v1/predictions.jsonl').read_text().splitlines() if l.strip()]
 primary=[x for x in allraw if x['evaluation_seed']==42];by={(str(x['source_image_id']),x['candidate_id']):x for x in primary};assert len(primary)==len(by)==960
 candidate={(str(x['source_image_id']),x['candidate_id']):x for x in read(REAL/'main_repair_v1/candidate_manifest.json')['records']};assert len(candidate)==960
 quality=list(csv.DictReader((ROOT/'results/next_validation_20261101T0040Z_v1/aligned_image_quality.csv').open(encoding='utf-8')));priorq={(x['source_image_id'],x['condition']):float(x['SSIM']) for x in quality}
 rng=np.random.default_rng(protocol['source_bootstrap_seed']);indices=rng.integers(0,160,size=(protocol['source_bootstrap_resamples'],160));matrix=np.zeros((160,6));ssim=np.zeros_like(matrix);detail=[];counts=collections.Counter();timing=collections.defaultdict(list)
 for i,iid in enumerate(ids):
  refrow=candidate[iid,'C5'];refpath=ROOT/refrow['candidate_path'];assert sha(refpath)==refrow['candidate_image_hash']
  with Image.open(refpath) as im:ref=np.asarray(im.convert('RGB'))
  assert ref.shape==(256,256,3)
  for j in range(6):
   c='C'+str(j);x=by[iid,c];record=source[iid];cand=candidate[iid,c]
   assert x['question']==record['question'] and x['reference_answers']==record['answers'] and str(x['question_id'])==str(record['question_id'])
   assert x['candidate_image_hash']==cand['candidate_image_hash'] and x['original_image_hash']==record['image_sha256']
   path=ROOT/x['candidate_path'];exists=path.is_file();counts[c+'_missing']+=not exists
   if exists:assert sha(path)==x['candidate_image_hash']
   valid=x['execution_status']=='PASS' and x['generation_status']=='PASS' and exists;counts[c+'_invalid']+=not valid
   score=score_answer(x['raw_vlm_answer'],dict(answers=x['reference_answers']))['score'] if valid else 0.
   assert x['official_vqa_consensus_score'] is None or abs(score-x['official_vqa_consensus_score'])<1e-12
   normalized=normalize_answer(x['raw_vlm_answer']);assert normalized==x['normalized_answer'];matrix[i,j]=score
   changed=normalized!=normalize_answer(by[iid,'C0']['raw_vlm_answer']);counts[c+'_changed']+=changed
   if j in [4,5]:
    value=priorq[iid,'generated' if j==4 else 'geometry_only'];measurement='reused existing aligned SSIM, hash-bound'
   else:
    with Image.open(path) as im:aligned=np.asarray(ImageOps.pad(im.convert('RGB'),(256,256),method=Image.Resampling.BICUBIC,color=(255,255,255)))
    if j in [0,1]:assert np.array_equal(aligned,ref),'Common geometry control mismatch'
    value=float(structural_similarity(ref,aligned,channel_axis=2,data_range=255));measurement='new CPU-only common-geometry SSIM'
   ssim[i,j]=value;latency=x.get('inference_latency_seconds')
   if latency is not None:assert math.isfinite(latency) and latency>=0;timing[c].append(latency)
   detail.append(dict(source_image_id=iid,question_id=x['question_id'],condition=c,raw_answer=x['raw_vlm_answer'],normalized_answer=normalized,recomputed_original_reference_score=score,changed_vs_C0=changed,SSIM_aligned256_to_C5=value,SSIM_measurement=measurement,valid=valid,missing=not exists,inference_seconds=latency,candidate_sha256=x['candidate_image_hash']))
 assert np.array_equal(matrix[:,0],matrix[:,1]);historical={'C0':.84875,'C4':.649375,'C5':.764375}
 for c,v in historical.items():assert abs(matrix[:,int(c[1])].mean()-v)<1e-12
 boots=matrix[indices].mean(1);qboots=ssim[indices].mean(1);gen=read(REAL/'main/generation_records.json');gen_seconds=[x['metadata']['generation_seconds'] for x in gen if x.get('status','PASS')=='PASS' and 'metadata' in x]
 summary=[];names=['Original','Identity','Brightness x1.03','Contrast x1.03','Frozen IP2Pix','Geometry only']
 for j in range(6):
  c='C'+str(j);d0=matrix[:,j]-matrix[:,0];d5=matrix[:,j]-matrix[:,5];sc=ci(boots[:,j]);c0=ci(boots[:,j]-boots[:,0]);c5=ci(boots[:,j]-boots[:,5]);qc=ci(qboots[:,j]);lats=timing[c]
  summary.append(dict(condition=c,name=names[j],unique_sources=160,original_reference_agreement=float(matrix[:,j].mean()),score_ci95_low=sc[0],score_ci95_high=sc[1],paired_delta_C0=float(d0.mean()),delta_C0_ci95_low=c0[0],delta_C0_ci95_high=c0[1],paired_delta_C5=float(d5.mean()),delta_C5_ci95_low=c5[0],delta_C5_ci95_high=c5[1],changed_predictions_vs_C0=counts[c+'_changed'],changed_predictions_vs_C5=sum(normalize_answer(by[iid,c]['raw_vlm_answer'])!=normalize_answer(by[iid,'C5']['raw_vlm_answer']) for iid in ids),SSIM_aligned256_mean=float(ssim[:,j].mean()),SSIM_ci95_low=qc[0],SSIM_ci95_high=qc[1],SSIM_sample_sd=float(ssim[:,j].std(ddof=1)),LPIPS=None,LPIPS_status='UNAVAILABLE_CALIBRATED_IMPLEMENTATION',missing=counts[c+'_missing'],invalid=counts[c+'_invalid'],VLM_primary_record_inference_seconds=sum(lats),VLM_mean_record_inference_seconds=float(np.mean(lats)),VLM_timed_records=len(lats),generation_primary_seconds=sum(gen_seconds) if j==4 else None,generation_timed_sources=len(gen_seconds) if j==4 else None,allocated_GPU_seconds=None,allocated_GPU_scope='shared historical run; not attributable per row',evaluation_scope='frozen original-reference agreement, development-only',new_GPU_seconds=0))
 csvwrite(out/'per_source_recomputed.csv',detail);put(out/'summary.json',dict(conditions=summary,raw_rows=len(allraw),primary_rows=960,independent_sources=160,repeated_rows_excluded=len(allraw)-960,historical_point_estimates=historical,historical_points_reconciled=True,bootstrap_seed=protocol['source_bootstrap_seed'],resamples=len(indices),resource_accounting=read(REAL/'resource_accounting.json'),new_generation_calls=0,new_VLM_calls=0,LPIPS_module_available=importlib.util.find_spec('lpips') is not None,new_descriptive_SSIM_rows=640))
 csvwrite(out/'BASELINE_REAL_IMAGE_RESULTS.csv',summary);emit(run,'BASELINE_REAL_IMAGE_RESULTS.csv',(out/'BASELINE_REAL_IMAGE_RESULTS.csv').read_text())
 publish_real(run,summary,protocol);return summary

def publish_real(run,s,protocol):
 rows=[[x['condition']+' '+x['name'],160,pct(x['original_reference_agreement']),interval([x['score_ci95_low'],x['score_ci95_high']],100),f"{x['paired_delta_C0']*100:+.4f}",interval([x['delta_C0_ci95_low'],x['delta_C0_ci95_high']],100),f"{x['paired_delta_C5']*100:+.4f}",interval([x['delta_C5_ci95_low'],x['delta_C5_ci95_high']],100),x['changed_predictions_vs_C0'],f"{x['SSIM_aligned256_mean']:.6f}",str(x['invalid'])+'/'+str(x['missing'])] for x in s]
 report='# Natural-image control comparison — internally measured\n\n'+f"Recomputed UTC {datetime.now(timezone.utc).isoformat()}. Immutable run: {run.name}. **160 unique matched sources,960 primary records;72 dependent repeats excluded from n.** No new inference, generation or training.\n\n"
 report+='Metric: **VQA original-reference agreement**, not verified edited-image accuracy or post-training ASR. Same original questions, ten references and pinned scorer across all conditions. Historically exposed development data.\n\n'
 report+=mdtable(['Condition','n','Agreement %','Source95%CI %','Delta C0 pp','Paired95%CI pp','Delta C5 pp','Paired95%CI pp','Changed/C0','SSIM aligned256','Invalid/missing'],rows)
 report+='\nHistorical C0=.84875, C4=.649375 and C5=.764375 point estimates reconcile using the unchanged1e-12 arithmetic check. Intervals use10,000 paired source resamples, fixed seed'+str(protocol['source_bootstrap_seed'])+'. These are descriptive, unadjusted intervals. Historical interval endpoint differences reflect bootstrap streams, not changed data. All C5 contrasts are pipeline diagnostics; C4-C5 isolates generation beyond geometry, not semantic correctness.\n\n'
 report+='SSIM uses a common256 RGB reference: stored C5, white padding, BICUBIC. C0-C3 pass through this same geometry (640 new CPU measurements); C4/C5 reuse hash-bound aligned measurements. No human prerequisite is imposed on descriptive image metrics. Identity/geometry SSIM=1 is not instruction success. LPIPS is null because calibrated implementation/weights are unavailable, not replaced by another metric.\n\n'
 report+=mdtable(['Condition','SSIM95%CI','VLM primary-record seconds','VLM mean seconds','C4 generation seconds / timed sources'],[[x['condition'],interval([x['SSIM_ci95_low'],x['SSIM_ci95_high']],digits=6),f"{x['VLM_primary_record_inference_seconds']:.3f}",f"{x['VLM_mean_record_inference_seconds']:.6f}",(f"{x['generation_primary_seconds']:.3f} / {x['generation_timed_sources']}" if x['generation_primary_seconds'] is not None else 'not separately measured')] for x in s])
 report+='\nTiming sums cover identified saved primary records. Historical job costs also include pilot, repairs, repeats, loads/writes. Shared allocation is never charged to every condition; full receipt is retained in this run real_image/summary.json. Component timing is not a synchronized latency benchmark. All missing metrics remain null.\n\nThe C4 decline is a frozen-pipeline observation, not backdoor implantation. Independent edited labels remain0. C2/C3 tie for largest observed agreement, without established superiority.\n\nEvidence: '+str(REAL/'main_repair_v1/predictions.jsonl')+'; '+str(REAL/'main_repair_v1/candidate_manifest.json')+'; '+str(run/'real_image/per_source_recomputed.csv')+'.\n'
 emit(run,'BASELINE_REAL_IMAGE_RESULTS.md',report)
 ranks=sorted({round(x['original_reference_agreement'],12) for x in s},reverse=True)
 tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Internally measured natural-image controls on160 matched VQAv2 sources. Agreement uses original references; edited labels are unverified. Source-level bootstrap95\% intervals. Bold/underline indicate largest/second distinct observed agreement, not significant superiority.}',r'\label{tab:rsi_controls_measured}',r'\small',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrrrrr}',r'\toprule',r'Condition & $n$ & Agreement [95\% CI] (\%) & $\Delta$C0 [95\% CI] (pp) & $\Delta$C5 [95\% CI] (pp) & Changed/C0 & Invalid & Missing \\',r'\midrule']
 for x in s:
  val=pct(x['original_reference_agreement']);rank=ranks.index(round(x['original_reference_agreement'],12));val=(r'\textbf{'+val+'}') if rank==0 else (r'\underline{'+val+'}') if rank==1 else val
  tex.append(f"{x['condition']} {x['name']} &160& {val} {interval([x['score_ci95_low'],x['score_ci95_high']],100)} & {x['paired_delta_C0']*100:+.4f} {interval([x['delta_C0_ci95_low'],x['delta_C0_ci95_high']],100)} & {x['paired_delta_C5']*100:+.4f} {interval([x['delta_C5_ci95_low'],x['delta_C5_ci95_high']],100)} & {x['changed_predictions_vs_C0']} &{x['invalid']}&{x['missing']} "+r'\\')
 tex +=[r'\bottomrule',r'\end{tabular}}',r'\par\vspace{3pt}',r'\resizebox{.95\textwidth}{!}{%',r'\begin{tabular}{lrrrr}',r'\toprule',r'Condition & Aligned256 SSIM [95\% CI] & LPIPS & VLM recorded seconds & Generation recorded seconds \\',r'\midrule']
 for x in s:tex.append(f"{x['condition']} & {x['SSIM_aligned256_mean']:.6f} {interval([x['SSIM_ci95_low'],x['SSIM_ci95_high']],digits=6)} & -- & {x['VLM_primary_record_inference_seconds']:.3f} & "+(f"{x['generation_primary_seconds']:.3f}" if x['generation_primary_seconds'] is not None else '--')+r' \\')
 tex +=[r'\bottomrule',r'\end{tabular}}',r'\par\footnotesize Timing is saved component wall time, not separately attributable GPU allocation. SSIM is descriptive in common256 RGB geometry, not semantic certification. LPIPS unavailable.',r'\end{table*}','']
 emit(run,'BASELINE_REAL_IMAGE_RESULTS.tex','\n'.join(tex))

def ablation(run,protocol):
 out=run/'ablation';out.mkdir();d=read(ABL/'threeway/results.json');branches=d['branches'];seeds=[42,123,3407];arms=['fixed','adaptive_scheduler','rsi'];assert len(branches)==9 and all(x['passed'] for x in d['checks'])
 by={(x['seed'],x['arm']):x for x in branches};a=np.array([[by[s,arm]['selected_test'] for s in seeds] for arm in arms]);assert a.shape==(3,3,32)
 for b in branches:
  assert abs(np.mean(b['selected_test'])-b['test_mean'])<1e-12
  assert (b['optimizer_calls'],b['forward_NFE'],b['validation_queries'],b['test_queries'])==(4,320,48,32)
 disk=read(ROOT/'results/audit_remediation_20261101T0120Z_v1/audit/preadaptation_disk_verification.json')
 g=np.random.default_rng(20261101);contrasts=[]
 for oi,other in enumerate(['fixed','adaptive_scheduler']):
  delta=a[2]-a[oi];si=g.integers(3,size=(5000,3));ciidx=g.integers(32,size=(5000,32));draws=np.array([delta[s][:,c].mean() for s,c in zip(si,ciidx)]);bounds=ci(draws);stored=next(x for x in d['comparisons'] if x['comparison']=='rsi_minus_'+other)
  assert abs(delta.mean()-stored['mean'])<1e-12 and np.array_equal(np.asarray(bounds),stored['crossed_seed_condition_bootstrap95'])
  contrasts.append(dict(row_type='contrast',comparison='rsi_minus_'+other,mean=float(delta.mean()),ci95_low=bounds[0],ci95_high=bounds[1],unit='cosine terminal reward, not accuracy',CI_method='original5000 crossed seed/shared-condition resamples, seed20261101'))
 rows=[];aggregate=[]
 for k,arm in enumerate(arms):
  scores=a[k].mean(1);held=np.array([by[s,arm]['GPU_held_seconds_measured'] for s in seeds]);aggregate.append(dict(row_type='arm_summary',arm=arm,seeds=3,mean=float(scores.mean()),sample_sd=float(scores.std(ddof=1)),GPU_held_seconds_mean=float(held.mean()),GPU_held_seconds_sample_sd=float(held.std(ddof=1)),optimizer_calls_per_seed=4,forward_NFE_per_seed=320,validation_queries_per_seed=48,test_queries_per_seed=32,allocated_GPU_seconds=None,allocated_scope='144 shared across9 successful branches; do not multiply by row count'))
  for i,s in enumerate(seeds):
   b=by[s,arm];delta=a[k,i]-a[0,i];idx=np.random.default_rng(20261101+s).integers(32,size=(5000,32));bounds=ci(delta[idx].mean(1))
   rows.append(dict(row_type='seed_arm',seed=s,arm=arm,mean=float(scores[i]),paired_delta_fixed=float(delta.mean()),ci95_low=bounds[0],ci95_high=bounds[1],optimizer_calls=b['optimizer_calls'],forward_NFE=b['forward_NFE'],validation_queries=b['validation_queries'],test_queries=b['test_queries'],GPU_held_seconds=b['GPU_held_seconds_measured'],candidate_lr=b['candidate_lr'],retained_lr=b['retained_lr'],updated_adapter_tensors=b['updated_adapter_tensors'],allocated_GPU_seconds=None,allocated_scope='shared job88032,144 seconds',preintervention_equality='existing full-state/disk assertions PASS, not rerun',selected_checkpoint_sha256=b['selected_checkpoint_sha256']))
 allocation=[dict(row_type='shared_allocation',job_id='88032',allocated_GPU_seconds=144,scope='all9 successful branches plus overhead'),dict(row_type='shared_allocation',job_id='88029',allocated_GPU_seconds=147,scope='prior mixed diagnostic/partial-ablation attempt, internal serialization failure preserved')]
 put(out/'recomputed.json',dict(per_seed=rows,aggregates=aggregate,contrasts=contrasts,existing_preadaptation_disk_evidence=disk,checks=d['checks'],component_GPU_seconds=d['total_component_GPU_held_seconds'],allocations=allocation,scientific_RSI_superiority='NOT ESTABLISHED',new_GPU_seconds=0))
 csvwrite(out/'RL_RSI_ABLATION_RESULTS.csv',rows+aggregate+contrasts+allocation);emit(run,'RL_RSI_ABLATION_RESULTS.csv',(out/'RL_RSI_ABLATION_RESULTS.csv').read_text())
 publish_ablation(run,rows,aggregate,contrasts,d);return aggregate,contrasts

def publish_ablation(run,rows,aggregate,contrasts,d):
 report='# Benign DDPO / scheduler / RSI ablation\n\n**Controlled Tiny FP32 engineering/benign-learning ablation; not natural-image or backdoor efficacy.** Three independent training seeds share32 synthetic test conditions. Objective: predeclared harmless cosine of the first terminal latent coordinate. No ASR/security reward or new optimizer steps.\n\n'
 report+=mdtable(['Arm','Seeds','Test mean','Sample SD across seeds','GPU-held seconds mean±SD','Updates/NFE per seed','Validation/test queries'],[[x['arm'],3,f"{x['mean']:.12f}",f"{x['sample_sd']:.12f}",f"{x['GPU_held_seconds_mean']:.6f} ± {x['GPU_held_seconds_sample_sd']:.6f}",'4 /320','48 /32'] for x in aggregate])
 report+='\n'+mdtable(['Seed','Arm','Outcome','Delta fixed','Paired source95%CI','GPU-held seconds','LR candidate→retained'],[[x['seed'],x['arm'],f"{x['mean']:.12f}",f"{x['paired_delta_fixed']:+.12f}",interval([x['ci95_low'],x['ci95_high']],digits=12),f"{x['GPU_held_seconds']:.6f}",f"{x['candidate_lr']:g}→{x['retained_lr']:g}"] for x in rows])
 report+='\n'+mdtable(['Contrast','Mean difference','Crossed seed/source95%CI','Conclusion'],[[x['comparison'],f"{x['mean']:+.12f}",interval([x['ci95_low'],x['ci95_high']],digits=12),'RSI superiority NOT ESTABLISHED'] for x in contrasts])
 report+='\nOriginal crossed-bootstrap intervals reproduce exactly from saved paired observations. SD uses three seed means;96 seed-condition outcomes are not96 independent conditions. Intervals are exploratory with only3 training seeds.\n\nExisting full-state preintervention evidence is reused and SHA-bound, including12 initial/prefix cross-branch disk comparisons: parameters, optimizer/scheduler, saved RNG, gradients, predictions, contracts/version/trajectory IDs. Four updates include rejected computation. RSI rollback restores its OWN prefix. Every branch reports320 measured forward NFE,48 validation and32 test queries.\n\n'
 report+='Branch GPU-held seconds total'+f"{d['total_component_GPU_held_seconds']:.6f}"+'. Successful job88032 allocated144 GPU-seconds SHARED across9 branches; do not charge144 to each row. Prior job88029 allocated147 seconds to mixed diagnostics and a partial ablation with strict-JSON sentinel failure. Its failure/costs remain and cannot be precisely assigned to an arm. Sequential same-node timings do not establish speed superiority.\n\n## Historical pretrained IP2Pix — separate, noncausal\n\n'
 report+='\nM0 is the frozen reference (zero optimizer updates; one checkpoint evaluated at seeds42/123). M1 is DDPO-only and M2 is DDPO+RSI (two training seeds, eight updates each). These labels are distinct from the later Tiny fixed/adaptive_scheduler/rsi arms.\n\n'
 old=read(ROOT/'reports/phase4_results.json');report+=mdtable(['Arm','seed','Final legacy proxy reward'],[[x['arm'],x['seed'],f"{x['reward']:.12f}"] for x in old['raw_final_rows']])
 report+='\nEach trained historical arm uses8 updates for seeds42/123. Before LR intervention at step4, adapter max differences are4.34328e-5/6.64559e-5; optimizer moments already differ. Thus final pretrained RL/RSI differences cannot support causal improvement. New Tiny FP32 equality does not retroactively repair the historical BF16 comparison. No ranking is applied to confounded historical arms.\n\nEvidence: '+str(ABL/'threeway/results.json')+'; '+str(run/'ablation/recomputed.json')+'; original protocol and preadaptation disk verification.\n'
 emit(run,'RL_RSI_ABLATION_RESULTS.md',report)
 tex=[r'\begin{table*}[t]',r'\centering',r'\caption{Internally measured controlled Tiny FP32 ablation: three training seeds,32 shared synthetic test conditions. Mean$\pm$sample SD over seed means. The harmless cosine task is not natural-image VQA or backdoor efficacy. Intervals resample training seeds and shared conditions.}',r'\label{tab:rsi_tiny_ablation}',r'\small',r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Arm & Test reward mean$\pm$SD & $\Delta$Fixed [95\% CI] & Updates & NFE & Val/Test queries & GPU-held seconds mean$\pm$SD \\',r'\midrule']
 fixed=aggregate[0]['mean'];rank=sorted({x['mean'] for x in aggregate},reverse=True)
 for x in aggregate:
  c=contrasts[0] if x['arm']=='rsi' else None;effect=(f"{c['mean']:+.9f} {interval([c['ci95_low'],c['ci95_high']],digits=9)}" if c else 'reference' if x['arm']=='fixed' else f"{x['mean']-fixed:+.9f} (descriptive)")
  value=f"{x['mean']:.9f}";value=(r'\textbf{'+value+'}') if x['mean']==rank[0] else (r'\underline{'+value+'}') if x['mean']==rank[1] else value
  tex.append(x['arm'].replace('_',r'\_')+f" & {value}"+r'$\pm$'+f"{x['sample_sd']:.9f} & {effect} &4&320&48/32&{x['GPU_held_seconds_mean']:.4f}"+r'$\pm$'+f"{x['GPU_held_seconds_sample_sd']:.4f}"+r' \\')
 tex +=[r'\bottomrule',r'\end{tabular}}',r'\par\footnotesize RSI--scheduler difference: '+f"{contrasts[1]['mean']:+.9f} {interval([contrasts[1]['ci95_low'],contrasts[1]['ci95_high']],digits=9)}"+'. Both intervals include zero. Bold/underline are descriptive only. Total successful shared allocation144 GPU-seconds; branch-held sum'+f"{d['total_component_GPU_held_seconds']:.3f}"+' seconds. Prior failed/mixed allocation147 seconds is separate.',r'\end{table*}','']
 emit(run,'RL_RSI_ABLATION_RESULTS.tex','\n'.join(tex))

def main(run):
 assert os.getenv('CUDA_VISIBLE_DEVICES')=='';beg=time.perf_counter();protocol=read(run/'protocol.json');assert protocol['new_GPU_calls']==0
 inputs=[REAL/'main_manifest.json',REAL/'main_repair_v1/predictions.jsonl',REAL/'main_repair_v1/candidate_manifest.json',REAL/'main/generation_records.json',REAL/'resource_accounting.json',ROOT/'results/next_validation_20261101T0040Z_v1/aligned_image_quality.csv',ABL/'threeway/results.json',ABL/'protocol.json',ROOT/'results/audit_remediation_20261101T0120Z_v1/audit/preadaptation_disk_verification.json',ROOT/'src/evaluation/vqa_official_compatible.py',ROOT/'src/rl/phase5_backend.py']
 before={str(p.relative_to(ROOT)):sha(p) for p in inputs};put(run/'input_hashes.json',before);r=real(run,protocol);a,c=ablation(run,protocol);assert all(sha(ROOT/n)==h for n,h in before.items())
 put(run/'checks.json',dict(status='PASS_COMPUTED_TABLES_ONLY',CPU_elapsed_seconds=time.perf_counter()-beg,real_unique_sources=160,primary_records=960,new_common_geometry_SSIM_rows=640,historical_points_reconciled=True,original_ablation_intervals_exact=True,input_files_unchanged=True,new_GPU_seconds=0,new_generation_calls=0,new_VLM_calls=0,new_training_steps=0,scientific_gate='NO-GO',LPIPS='UNAVAILABLE'))
 print(json.dumps(dict(status='PASS_COMPUTED_TABLES_ONLY',real=[{k:x[k] for k in ['condition','original_reference_agreement','changed_predictions_vs_C0','SSIM_aligned256_mean']} for x in r],ablation=a,contrasts=c,seconds=time.perf_counter()-beg)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();main(Path(a.run))
