"""CPU evidence analysis, existing fixed-model metrics and honest reports."""
import copy,csv,hashlib,html,json,math,re,time
from pathlib import Path
from scripts.numerical_boundaries_20261102 import compare,load,sha,put

def analyze_numerical(out):
 import torch
 torch.set_num_threads(2);entries={k:load(out/p) for k,p in [('fresh0','fresh0/update0.pt'),('fresh1','fresh1/update0.pt'),('resident0','resident/update0.pt'),('resident1','resident/update1.pt')]};pairs={};first=None
 for a,b in [('fresh0','fresh1'),('resident0','resident1'),('fresh0','resident0'),('fresh0','resident1')]:
  x,y=entries[a],entries[b];rows={}
  rows['initial']=compare(x['before'],y['before'])
  ca=load(out/('resident' if a.startswith('resident') else a)/'conditioning.pt');cb=load(out/('resident' if b.startswith('resident') else b)/'conditioning.pt');rows['conditioning']=compare(ca,cb)
  for key in ['raw','runtime_total_norm','runtime_coefficient_formula','postclip']:rows[key]=compare(x['capture'][key],y['capture'][key])
  rows['Adam_and_parameters']=compare(x['after'],y['after']);pairs[a+'__'+b]=rows
  for key in ['initial','conditioning','raw','runtime_total_norm','runtime_coefficient_formula','postclip','Adam_and_parameters']:
   if not rows[key]['exact']:
    if first is None:first=dict(pair=a+'__'+b,boundary=key)
    break
 exact=all(v['exact'] for row in pairs.values() for v in row.values());tol=all(v['tolerance_pass'] for row in pairs.values() for v in row.values())
 classification='NOT_REPRODUCED_IN_MINIMAL_CAPTURE' if exact else {'raw':'RAW_BACKWARD_GRADIENT_MISMATCH','runtime_total_norm':'NORM_REDUCTION_MISMATCH_WITH_MATCHED_RAW','runtime_coefficient_formula':'COEFFICIENT_MISMATCH','postclip':'CLIPPING_MISMATCH','Adam_and_parameters':'OPTIMIZER_OR_FINAL_STATE_MISMATCH'}.get(first['boundary'],'PRE_UPDATE_STATE_MISMATCH')
 result=dict(exact_pass=exact,tolerance_pass=tol,classification=classification,first_boundary=first,pairs=pairs,root_cause_established=False,fix_confirmed=False,ordinary_uninstrumented_test=False,scientific_release=False,scope='One fixed trajectory fixture, matched initial states, separate fresh and resident processes; minimal clipping wrapper can perturb execution, never a complete fix')
 put(out/'numerical_results.json',result);put(out/'checks.json',dict(status='PASS' if exact else 'FAIL',diagnostic_complete=True,numerical_retry_allowed=False,classification=classification));return {k:v for k,v in result.items() if k!='pairs'}

def analyze_resume(out):
 rows={}
 for a,b,name in [('ordinary0','ordinary1','ordinary_repeat'),('ordinary0','resume','resumed_next_update')]:
  phases=['after2/training_state.pt','next_trajectory.pt','postclip.pt']+(['before/training_state.pt','after1/training_state.pt'] if b=='ordinary1' else [])
  rows[name]={phase:compare(load(out/a/phase),load(out/b/phase)) for phase in phases}
  ra=json.loads((out/a/'receipt.json').read_text());rb=json.loads((out/b/'receipt.json').read_text())
  if len(ra.get('updates',[]))!=2 or len(rb.get('updates',[]))!=(2 if b=='ordinary1' else 1):raise ValueError('Missing executed update receipts: ordinary requires2,resume requires1')
  for receipt in [ra,rb]:
   for update in receipt['updates']:
    for key in ['gradient_norm','loss']:
     if type(update.get(key)) not in (int,float) or not math.isfinite(update[key]):raise ValueError('Missing/nonfinite actual '+key)
    if update['gradient_norm']<=0:raise ValueError('Missing positive preclip gradient norm')
  import torch
  # Recorded preclip norm/loss are mandatory, even if final tensors agree.
  ua=ra['updates'] if b=='ordinary1' else ra['updates'][-1:];ub=rb['updates']
  rows[name]['recorded_preclip_norm_and_loss']=compare([torch.tensor([v['gradient_norm'],v['loss']],dtype=torch.float64) for v in ua],[torch.tensor([v['gradient_norm'],v['loss']],dtype=torch.float64) for v in ub])
  for receipt in [ra,rb]:assert all(receipt[k] for k in ['ownership_exact','frozen_base_unchanged','unmodified_canonical_update','original_guard_unchanged'])
 exact=all(x['exact'] for r in rows.values() for x in r.values());tol=all(x['tolerance_pass'] for r in rows.values() for x in r.values());put(out/'resume_comparisons.json',rows)
 result=dict(exact_pass=exact,tolerance_pass=tol,classification='ORDINARY_AND_RESUME_PASS_SCOPED' if exact else 'ORDINARY_OR_RESUMED_UPDATE_FAIL',first_boundary=next((dict(pair=k,boundary=p) for k,r in rows.items() for p,v in r.items() if not v['exact']),None),ordinary_uninstrumented=True,resumed_next_optimizer_update=True,fix_confirmed=False,historical_failures_preserved=True)
 put(out/'resume_results.json',result);put(out/'checks.json',dict(status='PASS' if exact else 'FAIL',diagnostic_complete=True,numerical_retry_allowed=False));return result

def bootstrap_ci(values,seed=20261102):
 import numpy as np
 a=np.asarray(values,dtype=np.float64);rng=np.random.default_rng(seed);means=a[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1);return [float(x) for x in np.quantile(means,[.025,.975])]

def fixed_model_metrics(root,out):
 import numpy as np,torch,sys
 from PIL import Image,ImageOps
 from skimage.metrics import structural_similarity
 torch.set_num_threads(2);pilot=root/'results/execution_remediation_20261102_v1_pretrained_pilot';base=root/'results/execution_remediation_20261102_v1';data=json.loads((pilot/'source_manifest.json').read_text());saved=json.loads((pilot/'results.json').read_text());obs=json.loads((pilot/'prediction_observations.json').read_text());dep=json.loads((base/'lpips_dependency_verification.json').read_text());sys.path.insert(0,dep['module_root']);import lpips
 assert sha(Path(dep['module_root'])/'lpips/weights/v0.1/alex.pth')==dep['calibration_sha256']
 metric=lpips.LPIPS(net='alex',version='0.1',lpips=True,pretrained=True,pnet_rand=False,verbose=False).eval().requires_grad_(False);calibration=torch.load(Path(dep['module_root'])/'lpips/weights/v0.1/alex.pth',map_location='cpu',weights_only=True);assert all(torch.equal(metric.state_dict()[k],v) for k,v in calibration.items())
 results=[];metrics=[];frozen=next(b['assessment_scores'] for b in saved['branches'] if b['arm']=='frozen')
 for b in saved['branches']:
  arm=b['arm'];quality=[];rewards=[]
  for i,r in enumerate(data['assessment']):
   source=root/r['image_path'];edit=pilot/arm/'selected_assessment'/f'{i}.png';assert sha(source)==r['image_sha256'];match=next(o for o in obs if o['phase']==arm+'/selected_assessment' and str(o['source_image_id'])==str(r['image_id']));assert sha(edit)==match['image_sha256']
   with Image.open(edit) as im:y=np.asarray(im.convert('RGB')).copy()
   with Image.open(source) as im:aligned=ImageOps.pad(im.convert('RGB'),(y.shape[1],y.shape[0]),color='white',method=3);x=np.asarray(aligned).copy()
   tensor=lambda z:torch.from_numpy(z.copy()).permute(2,0,1).float().unsqueeze(0)/255.
   with torch.inference_mode():distance=float(metric(tensor(x),tensor(y),normalize=True))
   lum=lambda z:float(((z.astype(np.float64)/255.)*np.array([.2126,.7152,.0722])).sum(axis=2).mean());before,after=lum(x),lum(y);target=min(1.,1.03*before);reward=abs(before-target)-abs(after-target);assert abs(reward-match['reward'])<1e-12
   q=dict(method=arm,image_id=r['image_id'],image_sha256=r['image_sha256'],edited_sha256=sha(edit),SSIM=float(structural_similarity(x,y,channel_axis=2,data_range=255)),LPIPS=distance,no_op=bool(np.array_equal(x,y)),proxy=reward,independent_edit_validity=None);quality.append(q);rewards.append(reward)
  ci=bootstrap_ci(rewards);paired=[v-w for v,w in zip(rewards,frozen)];paired_ci=bootstrap_ci(paired);selected=pilot/arm/'selected/training_state.pt';start=pilot/'shared_initial/training_state.pt'
  retained=10*b['retained_checkpoint_step'];attempted=0 if arm=='frozen' else 20
  results.append(dict(Method=arm,Training_seed=42,Initial_checkpoint_hash=sha(start),Selected_checkpoint_hash=sha(selected),Training_sources=retained,Attempted_training_sources=attempted,Training_source_IDs=';'.join(str(r['image_id']) for r in data['train'][:retained]),Evaluation_sources=4,Evaluation_source_IDs=';'.join(str(r['image_id']) for r in data['assessment']),Optimizer_steps=b['retained_checkpoint_step'],Attempted_optimizer_steps=b['attempted_updates'],Evaluation_queries=b['actual_arm_assessment_queries'],Validation_queries=b['actual_arm_validation_queries'],UNet_NFE=b['incremental_forward_UNet_NFE'],UNet_NFE_scope='branch incremental;shared prefix168 once;job total720',Benign_proxy_mean=float(np.mean(rewards)),Metric_CI_low=ci[0],Metric_CI_high=ci[1],Paired_minus_Frozen_mean=float(np.mean(paired)),Paired_CI_low=paired_ci[0],Paired_CI_high=paired_ci[1],SSIM=float(np.mean([q['SSIM'] for q in quality])),LPIPS=float(np.mean([q['LPIPS'] for q in quality])),No_op_rate=float(np.mean([q['no_op'] for q in quality])),Actual_allocated_GPU_seconds=None,Historical_shared_job_GPU_seconds=242,Branch_GPU_held_seconds=b['incremental_GPU_held_seconds'],Training_job_id='88392',Execution_status='HISTORICAL_TRAINING_NEW_CPU_FIXED_MODEL_METRICS',Limitations='Historical exposed development only;n=4;1seed;numerical confound;unreviewed semantics;shared allocation not attributable per arm;not overnight training',Clean_VQA_accuracy=None,ASR='NOT TESTED',Independent_edit_validity=None,Dataset_manifest_sha256=sha(pilot/'source_manifest.json'),Checkpoint_path=str(selected.relative_to(root))))
  metrics.extend(quality)
 put(out/'fixed_model_metrics.json',dict(arms=results,per_source=metrics,scope='New CPU quality metrics and bootstrap on EXISTING fixed models only; no inference/regeneration/training; historical training budget kept separate',CI='source-bootstrap percentile95%;10000 draws;seed20261102;n=4 exploratory, no significance/RSI superiority claim',UNet_NFE_accounting='shared prefix168 once;incremental48+168+168+168;historical total720',overnight_GPU_seconds=0,final_sources_opened=0))

def render(s,root,base,out):
 from scripts.overnight_benign_controller_20261102 import ledger,now
 b=ledger(s);m=json.loads((out/'fixed_model_metrics.json').read_text()) if (out/'fixed_model_metrics.json').exists() else {'arms':[]};arms=m['arms']
 for arm in arms:arm['PROJECT_ID']='RSI-POISONING'
 fields=list(arms[0]) if arms else ['PROJECT_ID','Method','Training_seed','Initial_checkpoint_hash','Selected_checkpoint_hash','Training_sources','Evaluation_sources','Optimizer_steps','Evaluation_queries','UNet_NFE','Benign_proxy_mean','Metric_CI_low','Metric_CI_high','SSIM','LPIPS','No_op_rate','Actual_allocated_GPU_seconds','Execution_status','Limitations']
 with (root/'reports/OVERNIGHT_EXPERIMENT_RESULTS.csv').open('w',newline='',encoding='utf8') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(arms)
 tex=['% PROJECT_ID = RSI-POISONING','% Historical fixed-model CPU evaluation only; no new overnight learning arms.','\\begin{tabular}{lrrrrrr}','\\toprule','Method & Proxy & 95\\% CI & SSIM & LPIPS & No-op & GPU s \\\\','\\midrule']
 for a in arms:tex.append(a['Method'].replace('_','\\_')+f" & {a['Benign_proxy_mean']:.6f} & [{a['Metric_CI_low']:.6f},{a['Metric_CI_high']:.6f}] & {a['SSIM']:.4f} & {a['LPIPS']:.4f} & {a['No_op_rate']:.2f} & -- "+'\\\\')
 if not arms:tex.append('No completed eligible arms & -- & -- & -- & -- & -- & -- \\\\')
 tex+=['\\bottomrule','\\end{tabular}','\\par\\smallskip','\\begin{tabular}{lrrrrrrr}','\\toprule','Method & Seed & Sources$^{r/a}$ & Eval. & Steps$^{r/a}$ & Queries$^{e/v}$ & NFE$^{inc}$ & GPU s \\\\','\\midrule']
 for a in arms:tex.append(a['Method'].replace('_','\\_')+f" & {a['Training_seed']} & {a['Training_sources']}/{a.get('Attempted_training_sources',a['Training_sources'])} & {a['Evaluation_sources']} & {a['Optimizer_steps']}/{a['Attempted_optimizer_steps']} & {a['Evaluation_queries']}/{a['Validation_queries']} & {a['UNet_NFE']} & -- "+'\\\\')
 tex+=['\\bottomrule','\\end{tabular}']
 if arms:
  tex+=['\\par\\smallskip\\begin{minipage}{\\textwidth}\\scriptsize','Common initial SHA256: \\texttt{'+arms[0]['Initial_checkpoint_hash']+'}.\\\\']
  for a in arms:tex.append(a['Method'].replace('_','\\_')+' selected SHA256: \\texttt{'+a['Selected_checkpoint_hash']+'}.\\\\')
  tex+=['Historical VQAv2 development only; one training seed and four exposed assessment sources. $r/a$: retained/attempted. $e/v$: assessment/validation. Incremental NFE excludes the shared prefix168, counted once; historical total720. GPU s unavailable per arm; shared allocation88392 was242 s once. No new overnight model training. Numerical confound and unreviewed semantic validity prevent scientific comparison claims. ASR NOT TESTED.','\\end{minipage}']
 tex+=['% Requires booktabs. Null cost is --, never zero. Descriptive source-bootstrap95% CI;10000 draws;seed20261102.']
 (root/'reports/OVERNIGHT_EXPERIMENT_RESULTS.tex').write_text('\n'.join(tex)+'\n',encoding='utf8')
 rows=[]
 for t in s['GPU_jobs']:rows.append(f"|{t.get('job_id','UNCERTAIN')}|{t['stage']}|{t.get('scheduler_state')}|{t.get('elapsed_seconds',0)}|{t.get('classification','PENDING')}|{t.get('exit_code','NOT TERMINAL')}|")
 text=f'''# Overnight execution status

PROJECT_ID: RSI-POISONING. Run `{s['run_id']}`. Updated UTC {now()}.

Status: **{s['status']}**. No confirmed BF16 fix. Learning eligible: **{s.get('learning_eligible',False)}**.160 prospective final-test sources stay sealed. Scientific reward guards remain intact; ASR NOT TESTED.

NEW ledger only: cap36000 A100 seconds; consumed{b['consumed']};reserved{b['reserved']};remaining{b['remaining']}. Initialization{s['initialized_UTC']};deadline{s['deadline_UTC']}. One GPU at a time; numerical sub-budget1200 seconds. Previous3600-second validation ledger is separate.

| GPU job | Stage | State | Actual GPU seconds | Classification | Exit |
|---|---|---|---:|---|---|
{chr(10).join(rows)}

Registered CPU controller jobs: {json.dumps([{k:t.get(k) for k in ['job_id','purpose','parent_GPU_job']} for t in s['CPU_jobs']])}. Completion handlers use afterany, persistent state, lock, durable held submission, no polling. Existing two-hour monitor is preserved. Source directory: `reports/{s['run_id']}`;evidence: `results/{s['run_id']}`.

Passed scoped stages: {s['passed_stages']}. Failures: {json.dumps(s['failure_history'])}. Numerical comparisons and runtime exits are retained per-job. No automatic numerical retry or hyperparameter search. NODE_FAIL permits at most one recorded excluded-node infrastructure repair within budget.

{len(arms)} historical model arms have new CPU-only quality metrics when present. Training job88392 is historical and is not charged to this overnight budget. They are NOT newly trained overnight arms or final-test evidence. Arm-specific allocated GPU cost is null because the historical242-second allocation was shared; branch held time and NFE are separate fields. Frozen/DDPO/scheduler/RSI assessment uses the same4 historically exposed sources; uncertainty is exploratory and does not establish superiority.

New efficacy comparisons are blocked by numerical failure when present, unreleased original scientific reward and independent development evaluation not admitted. No source from the sealed160 is used. If no eligible stage remains, stop GPU submissions rather than fill budget. Real human labels remain0.

Cloud publication is separately receipt-verified by the connected Drive workflow; report generation alone is not a cloud synchronization claim.
'''
 (root/'reports/OVERNIGHT_EXECUTION_STATUS.md').write_text(text,encoding='utf8');(root/'reports/OVERNIGHT_FINAL_REPORT.md').write_text(text+'\n## Next falsifiable work\n\nIf raw gradients differ, localize backward accumulation/upstream adjoints; if raw match but norm differs, inspect norm reduction dtype/foreach/order; if norm matches but postclip differs, inspect clipping; if postclip matches but Adam differs, inspect optimizer step ordering. No repair is claimed without ordinary uninstrumented and resumed next-update checks.\n',encoding='utf8')
 latest=[t for t in s['GPU_jobs'] if t.get('result')]
 if latest:
  t=latest[-1];r=t['result'];first=r.get('first_boundary');extra=f"# Numerical root cause — latest\n\nPROJECT_ID: RSI-POISONING. Updated {now()}. Overnight run{s['run_id']};job{t['job_id']}.\n\nStatus **{r.get('classification',t['classification'])}**; exact {r['exact_pass']};tolerance {r.get('tolerance_pass')}. First observed boundary: `{first}`. Root cause NOT ESTABLISHED; no production repair or science release.\n\nThe new experiment keeps trajectories/model/data/optimizer/SDPA/precision fixed and varies process residency. Every compared update has the same minimal end-of-backward witness. Constant coefficient1 is a numerical fixture, not the scientific reward. Ordered raw/postclip gradients, runtime norm/coefficient, independent CPU FP64 reference, per-tensor norms and Adam transitions are retained in `{t['run_directory']}`. No activation dumps. Complete per-tensor affected counts,max-absolute and relative-L2 errors are in `numerical_results.json`.\n\nOrdinary uninstrumented/resumed next-update checks are required separately. Passed stages:{s['passed_stages']}. Earlier88422 ordinary preclip-norm failure remains preserved at `results/numerical_root_cause_20261102_v1`; original report is archived in `results/{s['run_id']}/before/NUMERICAL_ROOT_CAUSE_LATEST.md`. A bounded PASS cannot erase it.\n"
  result_path=root/t['run_directory']/'numerical_results.json'
  if result_path.exists():
   comparisons=json.loads(result_path.read_text())['pairs'];extra+='\n| Contrast | Raw affected tensors/elements | Raw max absolute | Raw relative L2 | Raw exact/tolerance | Runtime norm exact | Postclip exact |\n|---|---|---:|---:|---|---|---|\n'
   for pair,v in comparisons.items():
    g=v['raw'];extra+=f"|{pair}|{g['affected_tensors']}/{g['affected_elements']}|{g['max_abs']:.12g}|{g['relative_L2']:.12g}|{g['exact']}/{g['tolerance_pass']}|{v['runtime_total_norm']['exact']}|{v['postclip']['exact']}|\n"
   extra+='\nAll starts and conditioning must be checked before raw gradients; no forward/adjoint tensors were retained in this minimal experiment. Gradient-order paths are not a temporal first-CUDA-operation trace. Runtime total norm was observed; the clipping coefficient field is **reconstructed afterward** from that norm using the pinned Torch2.6 formula, not an intercepted internal intermediate. Root cause is bracketed at raw backward gradients before clipping; a residency-specific or clipping-specific cause is not established.\n'
  (root/'reports/NUMERICAL_ROOT_CAUSE_LATEST.md').write_text(extra,encoding='utf8')
 (root/'reports/NEXT_ELIGIBLE_EXPERIMENT.md').write_text(f"# Next eligible experiment\n\nPROJECT_ID: RSI-POISONING.\n\nRun{s['run_id']}.Status{s['status']}.New ledger remaining{b['remaining']},deadline{s['deadline_UTC']}.\n\n{('Wait for registered afterany CPU completion; no duplicate allocation.' if any(t.get('scheduler_state') in {'PENDING','RUNNING','UNKNOWN'} for t in s['GPU_jobs']) else 'No learned-model efficacy stage is admitted. Preserve failures and continue independent CPU evaluation only.')}\n\nNumerical success must be followed by ordinary uninstrumented repeat and checkpoint save/reload plus the next optimizer update. Historical failures remain. Original scientific reward release is denied; independent development evaluation is not admitted;160 final sources stay sealed. Existing historical assessment is development-only. No score-driven extra steps or sweeps. The exact next falsifiable numerical branch is described in OVERNIGHT_FINAL_REPORT.md. Previous detailed benign task/cohort/acceptance protocol is preserved in results/{s['run_id']}/before/NEXT_ELIGIBLE_EXPERIMENT.md.\n",encoding='utf8')
 dashboard_only(s,root)

def dashboard_only(s,root):
 # Persist the new component even when the legacy two-hour renderer refreshes.
 from scripts.overnight_benign_controller_20261102 import ledger
 b=ledger(s)
 p=root/'dashboard/dashboard_data.json';data=json.loads(p.read_text(encoding='utf8'));data['overnight_execution']=dict(run_id=s['run_id'],status=s['status'],budget=b,GPU_jobs=[{k:t.get(k) for k in ['job_id','stage','scheduler_state','classification']} for t in s['GPU_jobs']],learning_eligible=False,ASR='NOT TESTED');encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 block='<section id="overnight-benign"><h2>RSI-Poisoning overnight / 夜间执行</h2><p>'+html.escape(s['status'])+'; GPU seconds '+str(b['consumed'])+' /36000; remaining '+str(b['remaining'])+'</p><p><a href="../reports/OVERNIGHT_EXECUTION_STATUS.md">Execution status</a> · <a href="../reports/OVERNIGHT_FINAL_REPORT.md">Results and limitations</a></p></section>'
 for name in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']:
  p=root/'dashboard'/name
  if not p.exists():continue
  body=p.read_text(encoding='utf8');body=re.sub(r'<!--OVERNIGHT_BENIGN_START-->.*?<!--OVERNIGHT_BENIGN_END-->','',body,flags=re.S);body=re.sub(r'(<body[^>]*>)',lambda z:z[1]+'<!--OVERNIGHT_BENIGN_START-->'+block+'<!--OVERNIGHT_BENIGN_END-->',body,count=1);body=re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)',lambda z:z[1]+encoded+z[2],body,flags=re.S);p.write_text(body,encoding='utf8')
 for name in ['dashboard_data.json','state.json']:(root/'dashboard'/name).write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf8')
