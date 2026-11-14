"""CPU-only final evidence, strict numeric verdict and two requested reports."""
import json,sys,subprocess,zipfile,hashlib,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put,sha
from scripts.overnight_supervisor_20261101 import accounting,read,write,now,inspect_tasks,BASE
from src.monitoring.common import FileLock
OUT=ROOT/'results/numerical_root_cause_20261102_v1'

def main():
 import torch
 torch.set_num_threads(2)
 numeric=read(OUT/'new_tensor_comparisons.json');execution=read(OUT/'execution_summary.json');ordinary=numeric['pairs']['ordinary_repeat'];base=ROOT/'results/execution_remediation_20261102_v1';pilot=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot'
 a=load(pilot/'ordinary_scheduler/candidate/training_state.pt');b=load(pilot/'rsi/candidate/training_state.pt');old={'adapter':compare(a['adapter'],b['adapter'])}
 for key in ['exp_avg','exp_avg_sq']:old[key]=compare({k:v[key] for k,v in a['optimizer']['state'].items()},{k:v[key] for k,v in b['optimizer']['state'].items()})
 put(OUT/'live_pilot_detailed_deltas.json',old)
 kit=base/'semantic_review_v3_r2';original=ROOT/'results/next_validation_20261101T0040Z_v1/semantic_review32';manifest=read(kit/'blinded_manifest.json');assets=[]
 for case in manifest['cases']:
  for key in ['original','edited','geometry_control']:
   p=kit/case[key];ref=original/case[key];h=sha(p);assets.append(dict(path=case[key],sha256=h,original_package_exact=h==sha(ref),manifest_hash_exact=h==case[key+'_sha256'] if key+'_sha256' in case else None))
 packages=[]
 for suffix in ['A','B']:
  p=base/f'independent_review32_v3_r2_{suffix}.zip'
  with zipfile.ZipFile(p) as z:
   entries=z.namelist();matched=0
   for asset in assets:
    choices=[n for n in entries if n==asset['path'] or n.endswith('/'+asset['path'])];assert len(choices)==1
    assert hashlib.sha256(z.read(choices[0])).hexdigest()==asset['sha256'];matched+=1
  packages.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),verified_assets=matched))
 submitted=list((kit/'submitted').rglob('*')) if (kit/'submitted').exists() else [];real_exports=[str(p.relative_to(ROOT)) for p in submitted if p.suffix.lower() in {'.csv','.json'} and p.is_file()]
 put(OUT/'human_review_ready.json',dict(cases=len(manifest['cases']),assets=assets,all_original_assets_exact=all(x['original_package_exact'] for x in assets),packages=packages,submitted_files=real_exports,accepted_real_annotations=0,adjudications=0,status='PREPARED_WAITING_FOR_REAL_HUMANS',new_annotations_generated=0))
 protected=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json');unchanged={n:sha(ROOT/n)==h for n,h in protected.items()};assert all(unchanged.values())
 # Preserve the original scoped execution PASS. Strict scientific/numeric verdict
 # is CPU-derived and supersedes only the current task's aggregate acceptance.
 verdict=dict(status='PASS' if numeric['all_saved_boundaries_exact'] else 'FAIL',diagnostic_complete=True,numerical_retry_allowed=False,numerical_status='NUMERICAL_MISMATCH_PRESERVED',execution_invariants_pass=all(execution['invariants'].values()),exact_pass=numeric['all_saved_boundaries_exact'],tolerance_pass=numeric['tolerance_pass'],root_cause_established=False,fix_confirmed=False,ordinary_path_released=False,checkpoint_roundtrip_pass=True,resumed_next_optimizer_update='NOT TESTED',scope='Original execution PASS was invariants only; strict CPU comparison FAIL is authoritative')
 put(OUT/'checks.json',verdict)
 with FileLock(BASE/'monitor.lock') as acquired:
  assert acquired
  s=read(BASE/'state.json');task=next(t for t in s['tasks'] if t.get('job_id')=='88422');prior=task.get('classification');task.update(result=verdict,classification='NUMERICAL_MISMATCH_PRESERVED',numerical_retry_allowed=False,CPU_acceptance_superseded_UTC=now());write(BASE/'state.json',s);budget=accounting(s)
 put(OUT/'acceptance_supersession.json',dict(UTC=now(),prior_classification=prior,current_classification='NUMERICAL_MISMATCH_PRESERVED',original_execution_receipt='execution_checks_original.json',reason='CPU exact and tolerance checks fail; no GPU rerun'))
 jobs=subprocess.run(['sacct','-X','-nP','-j','88386,88392,88395,88398,88422','--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,Start,End'],capture_output=True,text=True,check=True).stdout
 queue=subprocess.run(['squeue','-h','-u','ydk297','-o','%i|%j|%T|%S|%R'],capture_output=True,text=True,check=True).stdout
 heartbeat=read(BASE/'heartbeat.json');sizes={str(p.relative_to(OUT)):p.stat().st_size for p in OUT.rglob('*.pt')}
 put(OUT/'terminal_resource_receipt.json',dict(UTC=now(),jobs_sacct=jobs,queue=queue,budget=budget,existing_monitor=heartbeat['scheduler'],retained_tensor_bytes=sum(sizes.values()),tensor_files=sizes,large_C_writes=0,protected_sources_exact=unchanged,script_files={n:sha(ROOT/'scripts'/n) for n in ['run_numerical_boundary_fixture_20261102.py','numerical_boundaries_20261102.py','prepare_benign_learning_20261102.py','test_numerical_boundaries_20261102.py','manage_numerical_boundary_20261102.py']},preregistered_size_estimate=650000000,storage_estimate_exceeded=True,scratch_created=False))
 table=[]
 for label,key in [('Preclip norm scalar','preclip_norm_scalar'),('Postclip gradients','postclip'),('Adapter parameters','after_adapter'),('Adam exp_avg','after_exp_avg'),('Adam exp_avg_sq','after_exp_avg_sq')]:
  r=ordinary[key];table.append(f"| {label} | {r['affected_tensors']} | {r['affected_elements']:,} | {r['max_abs']:.12g} | {r['relative_L2']:.12g} | {'PASS' if r['tolerance_pass'] else 'FAIL'} |")
 report=f'''# Numerical root cause — latest

PROJECT_ID: RSI-POISONING. Evidence UTC: {now()}. Canonical root: `/storage/home/ydk297/projects/poisoning`.

**Status: NUMERICAL_MISMATCH_PRESERVED. Root cause NOT ESTABLISHED. No BF16 correction or training release.** One new diagnostic completed and stopped, job88422,137 A100 GPU-seconds (cap200; Slurm180; process alarm165). No smoke tests or four-arm campaign rerun.

Job88422 used constant coefficient1 instead of the original scientific rewards. The production DDPO loss implementation remained unchanged; no original scientific-reward efficacy was evaluated.

## Earliest observed boundary, in execution order

| Boundary | Existing saved replay88386 versus live pilot88392/CPU88398 | New job88422 ordinary A1/A2 |
|---|---|---|
| Initial model/optimizer | Saved replay initial states exact; live ordinary/RSI complete start states exact | Adapter, optimizer, scheduler, step/version/consumed state exact; same frozen base and ownership |
| LR/scheduler | Live candidate active LR5e-6 both; original outer object missing,88395 explicitly reconstructed it | Optimizer LR5e-6 and LambdaLR base5e-6 exact; original design contract LR1e-5 retained; scheduler _last_lr is historical metadata, equal between repeats |
| CPU/CUDA/NumPy/Python RNG and sampler | Live final RNG exact; original training internal sampler states not saved | Initial and before-update RNG exact. Explicit local CUDA generator seed4210000+i; source list fixed, no random source sampler; generator terminal state not saved |
| Source order/timesteps | Same10 source IDs/seeds and output PNGs; not proof of internal equality | Fixed existing train[10:20],B10/T2. All trajectory metadata/order/timesteps exact |
| Internal latent transitions | Saved replay inputs fixed; original live latent tensors NOT SAVED | C1/C2 initial latent, every state/action/noise/logp/std exact. PNGs are not the criterion |
| Forward activations/loss inputs | Saved capture hashes exact; original live activation/loss-input tensors NOT SAVED | A forward tensors NOT OBSERVED, scalar loss exact at -2.1148666068911552. B1/B2 guided epsilon and loss action/logp/mean/std exact, plus32 selected LoRA module inputs/outputs/upstream adjoints at transition0 only |
| Raw gradients / preclip norm |88386 captured raw exact;88392 raw NOT SAVED. Historical88099 raw-gradient mismatch retained separately | A raw tensor gradients NOT OBSERVED. **First new confirmed unequal measurement is preclip global norm**,1.0337369441986084 versus1.033760666847229. Cannot distinguish backward-gradient difference from norm-reduction difference. B1/B2 raw gradients and32-parameter contributions across20 transitions exact |
| Postclip, Adam, parameters |88392 has31 adapter and32 tensors in each moment family unequal | All256 postclip tensors unequal; all256 adapter and both Adam families unequal. Final RNG and scheduler exact |

This brackets the new mismatch after equal live trajectory collection and by the gradient-norm reduction before clipping. It does not identify the first CUDA primitive or prove ordinary forward tensors identical. Historical88099 remains an earlier run with a directly observed raw-backward-before-clipping tensor mismatch; it is not substituted for a missing88422 raw witness.

## Separate paths and exact versus tolerance results

| Contrast | Exact result | Tolerance result | Meaning |
|---|---|---|---|
| A1/A2 ordinary canonical updates and C1/C2 live collection | Initial/RNG/complete trajectories PASS; preclip norm/postclip/final model FAIL | Aggregate FAIL | Ordinary training reproducibility fails despite identical continuous trajectories |
| B1/B2 fixed C1 replay with capture | Every saved boundary PASS, including raw/contributions/selected adjoints/moments | PASS | Bounded replay result only |
| A1/B1 | Common saved post-training boundaries FAIL | FAIL | Instrumentation and execution order are confounded |
| A2/B1 | All common saved boundaries PASS | PASS | First-versus-later execution/process/allocator effects remain hypotheses; not an identified cause |

All comparisons use `torch.equal` plus separate elementwise abs-error <=1e-7+1e-5*abs(reference). These descriptive tolerance thresholds do not relax strict acceptance. Missing A forward/raw/adjoint records are explicitly NOT COMPARABLE in `new_tensor_comparisons.json`, never PASS.

| A1/A2 mismatch | Tensors affected | Elements affected | Max absolute error | Aggregate relative L2 | Tolerance |
|---|---:|---:|---:|---:|---|
{chr(10).join(table)}

Relative L2 uses A1 as reference and a concatenated vector within each tensor family. Individual tensor metrics identified by tensor path are retained in `new_tensor_comparisons.json`; no mixed-unit whole-state relative-L2 claim is made. Original88392 adapter mismatch:36,057 affected elements,max4.186404112260789e-8; separate moment metrics in `live_pilot_detailed_deltas.json`. All original files and negative receipts remain intact.

## Source corrections, regression and resume

No production source-code correction is justified by this experiment. The eight protected source/config hashes, scientific loss, reward guards and acceptance requirements remain unchanged. Newly introduced diagnostic/reporting utilities were added only. Their initial versions/receipts are retained under `executed_source/` and `*_initial.json`. The cohort-lock reporter was corrected to replace inherited `independent_evaluation` row labels with `development_assessment`, retaining `historical_role`. CPU reporting now exposes missing captures, aggregate relative-L2, preclip scalar and separate Adam families. A reviewer-found tolerance-reference inconsistency and zero-reference summary edge case were corrected with explicit elementwise first-tensor scaling and an undefined-count field; current numerical FAIL is unchanged.

Executed5 initial comparator regressions PASS and5 affected re-executions PASS. Final6 comparator regressions PASS (`comparator_regression_final.log`), including a new asymmetric-tolerance reference check and undefined-relative-L2 assertion. These are6 unique tests,16 executions. Eight targeted deterministic brightness checks PASS across28 existing development sources, repeated after the cohort-label correction. Existing88395 regressions were read, not rerun. All4 checkpoint save/restore state round-trips PASS, trainable optimizer ownership and frozen-base hash PASS. **A resumed next optimizer update was NOT TESTED; ordinary path FAIL.** No diagnostic PASS is called a complete BF16 fix.

Pinned runtime actually executed Torch{execution['torch']}, CUDA{execution['CUDA']}. Existing MATH SDPA context, sequential CFG, no activation checkpointing, fixed accumulation order, BF16 frozen forward/FP32 adapters, deterministic algorithms and CUBLAS_WORKSPACE_CONFIG=:4096:8 were retained. No multiple-setting sweep, backend/precision change or science-loss edit occurred.

## Budget, preservation and next eligibility

Job88386:139 GPU-seconds;88392:242 GPU-seconds;88395 and88398:CPU only; new88422:137 GPU-seconds,COMPLETED0:0 with strict numerical FAIL. `execution_checks_original.json` scoped invariant PASS is retained; final `checks.json` and existing task classification record numerical FAIL. CPU commands ran directly on the login host and consumed no allocated GPU-seconds. Current validation ledger: consumed{budget['consumed']},reserved{budget['reserved']},remaining{budget['remaining']} of3600. No second diagnostic was submitted.

Existing two-hour monitor remains job{heartbeat['scheduler']['next_job_id']},due{heartbeat['scheduler']['next_due_utc']}; no new timer or monitor job. Its previous heartbeat is historical, not relabeled as a current check. Terminal live queue/budget are in `terminal_resource_receipt.json`.

160 candidate files reverified byte-exact,total26,879,223 bytes; no new model calls. They remain an unopened reservation and are not certified independent final evaluation.96 review assets match the original32-pair kit; both ZIPs verified,0 accepted real annotations. Existing human ingestion remains pending. Human-review materials are ready, not annotated.

Retained new tensor evidence: {sum(sizes.values()):,} bytes on cluster, exceeding the650MB estimate because selected-module activation/adjoint copies are larger than estimated. No large C-drive files were written. Evidence is retained intentionally. No run scratch directory was created.

**Benign model learning is not eligible now.** Required next work and exact criteria are in `NEXT_ELIGIBLE_EXPERIMENT.md`; numerical localization and independent development-evaluation admission remain unresolved. No ASR, stealth or poisoning optimization was implemented. ASR remains NOT TESTED.
'''
 (ROOT/'reports/NUMERICAL_ROOT_CAUSE_LATEST.md').write_text(report,encoding='utf-8')
 next_report=f'''# Next eligible experiment

PROJECT_ID: RSI-POISONING. Updated UTC: {now()}.

**No new learned-model training release.** Job88422 has stopped after137 A100 seconds; ordinary exact and tolerance checks failed. Root cause and source repair remain unestablished.701 A100 seconds remain in the validation subtotal at this receipt. No further GPU submission is authorized by this document alone; the user's one-diagnostic limit for this turn has been reached.

## Next numerical experiment, specified but NOT LAUNCHED

Use the same byte-bound shared-prefix checkpoint, active LR5e-6,B10/T2,10 sources/explicit seeds, constant coefficient1, pinned Torch2.6.0. Keep MATH SDPA, CFG, BF16/FP32 choices, deterministic configuration and scientific loss unchanged. The next uncertainty is A1 raw gradients versus preclip norm reduction, not PNG quality. Prepare a versioned minimal capture at the existing clipping call after all backward calls: retain raw gradient tensors and returned norm, with no forward/leaf hooks. First CPU-review this wrapper and its no-mutation contract. Only after separately admitting a compute cap, compare a fresh-process ordinary repeat and a resident-process repeat with matched initial model/optimizer/RNG/source order. Process residency is the sole treatment; the same minimal capture must exist in every compared worker. Copying raw gradients changes execution and cannot by itself certify an uninstrumented fix.

Acceptance: report exact and elementwise tolerance separately on initial states, continuous trajectories, raw gradients, norm, postclip, Adam and final parameters. Any earlier raw mismatch localizes before clipping; exact raw with unequal norm localizes the reduction boundary. If those agree, retain unresolved uncaptured earlier boundaries. Never use order dependence as an established CUDA cause. A proposed repair would then require the original implementation preserved, targeted regressions, an ordinary uninstrumented execution check, **a restored checkpoint followed by the next optimizer update**, and frozen/trainable ownership verification. No seed search or retry-until-PASS.

## Benign learning test prepared, admission pending

Task: non-adversarial lighting brightening only. Use the isolated brightness proxy from88392, not the unreleased original scientific reward. For RGB uint8 converted to[0,1],Y=mean(.2126R+.7152G+.0722B),target=min(1,1.03Ysource),reward=abs(Ysource-target)-abs(Yedited-target). Reward units are normalized absolute luminance error reduction,bounded[-1,1],not accuracy,ASR or measured GPU cost. This is a weighted sRGB brightness proxy without gamma linearization, not calibrated physical luminance. The scientific DDPO loss and guards remain intact.

CPU deterministic controls already EXECUTED: identity produces0; uniform additive brightness with uint8 quantization gives positive proxy improvement for all28 existing sources; equal darkening gives negative improvement. Four literal RGB unit examples and four cohort/control checks PASS. This validates units/sign and establishes a simple control, not semantic validity or learned behavior.

`next_benign_cohort_lock.json` byte-binds20 training,4 selection-validation and4 development-assessment sources to the existing pilot manifest and initial checkpoint. Roles were explicitly corrected to development-only; historical labels retained. All28 are historically exposed. The4 assessment samples are NOT a new independent test. The160 candidate sources remain excluded/unopened; their original manifest is hash-bound, not silently consumed. Before learning, admit and hash-lock a truly source-disjoint independent **development evaluation** cohort with documented prior exposure/feedback history, power and no selection use. No such cohort is asserted currently. Preserve separate final test reservations.

When admission is met, run **one preregistered benign DDPO learning update** from the saved shared-initial checkpoint on one fixed B10 training batch; compare it with the exact frozen checkpoint using the same evaluation IDs,seeds,resolution,T2,CFG and query count. One learning arm only; no RSI/ordinary-scheduler/four-arm sweep. Set LR in advance; no adaptive choice after assessment. Compute cap must include generation/evaluation/load/shared/failed cost and be checked against the then-live residual budget; this report does not spend the remaining701 seconds.

Successful learning requires trained-minus-frozen mean proxy reward >0 with a paired source-bootstrap95% CI lower bound >0, plus reduced absolute target-luminance error, separately on development validation and the newly admitted independent development evaluation. Predetermine bootstrap seed,10,000 resamples and source as the unit; insufficient power means inconclusive. Compare against the simple deterministic control; do not call beating a weak frozen model scientific instruction-following success. Require real independent semantic preservation/instruction review and adjudication for the relevant edited outputs; no fabricated labels. Failure stops the experiment; do not add steps to chase a proxy.

## Work actually complete and pending

| Work | Actual evidence | Status |
|---|---|---|
| CPU existing-artifact comparisons | Saved replay88386 exact; live88392 candidate mismatch quantitatively retained | COMPLETE; numerical failure preserved |
| Single bounded diagnostic |88422,137 A100 seconds,ordinary mismatch/fixed replay exact | COMPLETE; STOPPED |
| New deterministic control |8 checks,28 content-bound historical development sources | PASS for proxy mechanics only |
|160 candidate sources |160/160 SHA256 and byte counts match,no model inference | PRESERVED; final admission pending |
|32-pair human review |96 assets and both A/B ZIPs verified | READY;0 accepted human annotations |
| Genuine independent dev evaluation | Historical assessment exposed;160 reservation excluded | NOT ADMITTED |
| Actual BF16 production repair | No cause established,no production edit,resumed next-update test absent | NOT ESTABLISHED |
| Benign learned-model experiment | Exact cohort/checkpoint and criteria prepared | NOT ELIGIBLE; NOT LAUNCHED |
| Two-hour Slurm monitoring | Existing job{heartbeat['scheduler']['next_job_id']} due{heartbeat['scheduler']['next_due_utc']} | PRESERVED; no duplicate scheduler |

Human review packages remain `results/execution_remediation_20261102_v1/independent_review32_v3_r2_A.zip` and `_B.zip`. Real CSV/JSON exports must pass the existing ingestion validator; none are fabricated. Reports may be published through the RSI-POISONING Drive text-only allowlist; datasets,images,weights,tensor witnesses and raw logs remain local/cluster evidence.
'''
 (ROOT/'reports/NEXT_ELIGIBLE_EXPERIMENT.md').write_text(next_report,encoding='utf-8')
 put(OUT/'report_manifest.json',dict(UTC=now(),reports={n:sha(ROOT/'reports'/n) for n in ['NUMERICAL_ROOT_CAUSE_LATEST.md','NEXT_ELIGIBLE_EXPERIMENT.md']}))
 put(OUT/'artifact_inventory.json',dict(files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']))
 print(json.dumps(dict(verdict=verdict,budget_remaining=budget['remaining'],retained_tensor_bytes=sum(sizes.values()),reports_written=True)))
if __name__=='__main__':main()

