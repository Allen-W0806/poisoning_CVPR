"""Write only targeted numerical deliverables from actual saved evidence."""
import json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import sha,put
from scripts.overnight_benign_controller_20261102 import read,write,now

def main():
 out=ROOT/'results/numerical_fix_20261102_v1';base=ROOT/'reports/numerical_fix_20261102_v1';state=read(base/'state.json');budget=read(base/'budget.json');v=read(out/'verification/mandatory_verification.json');assert v['status']=='ROOT_CAUSE_PARTIALLY_LOCALIZED' and not v['fix_verified'];assert state['admissions_closed']
 original=read(out/'CPU_existing_evidence.json');local=read(out/'localization_bounded/boundary_summary.json');bridge=read(out/'attention_bridge/attention_bridge_summary.json');h1=read(out/'h1_order/H1_summary.json');h3=read(out/'h3_launch/H3_summary.json');git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();jobs=[]
 results={'localization':'Capture size guard stopped worker;no paired numerical result','localization_bounded':'Identical selected forward,unequal incoming adjoint before local VJP','attention_bridge':'Captured first-source SDPA bridge exact;later transition12 upstream mismatch remains','h1_order':'Single-thread autograd fails exact and tolerance;not a fix','h3_launch':'Launch-blocked raw pair exact;candidate only,not ordinary verification','verification':'Raw pair exact;ordinary pair FAIL;resumed optimizer update TIMEOUT'}
 for j in state['jobs']:
  p=out/j['stage']/'protocol.json';jobs.append(dict(job_id=j['job_id'],stage=j['stage'],scheduler_state=j['state'],GPU_seconds=j['elapsed'],exit_code=j['exit_code'],result=results[j['stage']],protocol_sha256=sha(p),declared_source_hashes=read(p)['source_hashes'],configuration_path=str(p.relative_to(ROOT))))
 put(out/'job_revisions_and_results.json',dict(git_base=git,working_tree_not_clean_by_design=True,jobs=jobs))
 table='| Job | Stage | Slurm / exit | A100 seconds | Actual result |\n|---|---|---|---:|---|\n'+''.join(f"|{j['job_id']}|{j['stage']}|{j['scheduler_state']} / {j['exit_code']}|{j['GPU_seconds']}|{j['result']}|\n" for j in jobs)
 params=(out/'affected_gradients32.md').read_text(encoding='utf8');stamp=now()
 root=f'''# Numerical root cause — latest

PROJECT_ID: RSI-POISONING. Updated UTC: {stamp}. Canonical root `/storage/home/ydk297/projects/poisoning`.

**ROOT_CAUSE_PARTIALLY_LOCALIZED. No production numerical repair is verified. The launch-blocking candidate FAILED ordinary verification.**

## Earliest observed boundary

Existing88433 CPU analysis confirms the same32 affected LoRA parameters in all four fresh/resident contrasts, within `down_blocks.0.attentions.0/1`, self/cross attention projections. All exact names,module paths,shapes,dtypes,element counts,max absolute and relative-L2 errors are in `results/numerical_fix_20261102_v1/affected_gradients32.csv` (128 pair/parameter rows);the32-name table follows below. All256 gradients remain in strict acceptance; no failing tensor was excluded.

Job88497 used identical node/parameter hooks in two fresh processes,matched checkpoint/optimizer/RNG/conditioning,B10/T2,constant coefficient1. For `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight`, first source/transition0/CFG call0 has exact forward input/output but unequal incoming `grad_output`:32,768 differing elements,max abs2.2879531602626457e-11,relativeL2 .005070520188901492. Local `MmBackward0` weight VJP differs afterward (max3.972786544181872e-10),then parameter contribution and accumulated gradients. The responsible upstream CUDA operation is **NOT ESTABLISHED**. This is the earliest fully observed boundary in the selected path, not a full-network temporal trace.

Three-branch local VJP sums match a valid explicit FP32 order in every1280 audited case. The selected parameter's matching order is the same across paired workers. This does not identify leaf accumulation ordering as the first origin. It does not exclude upstream graph scheduling. Local LoRA-A already receives an unequal adjoint,so its local VJP or clipping cannot be blamed as the primary origin from this evidence.

Job88499 additionally captured Q/K/V,SDPA output/incoming gradient/dQ/dK/dV and the LoRA-B bridge for the first source's two transitions. Those6 covered SDPA calls and the covered LoRA-B bridge were exact. Under this larger intervention,the first selected upstream mismatch moved to transition12/call38 (208 differing values,max3.8009179137432625e-13). Full forward tensors at that later transition were not retained. Instrumentation changes reproduction; covered SDPA equality is not an ordinary-path release or proof about all SDPA calls.

## Controlled hypotheses and candidate

| Hypothesis | Actual evidence | Conclusion |
|---|---|---|
| H1 backward ordering | CPU1280 sum-order cases valid;88502 ABBA default/single-thread with no layer hooks | Turning off autograd multithreading does NOT fix exact/tolerance; upstream ordering remains unresolved |
| H2 BF16 attention/backward | Existing MATH/deterministic/checkpoint-off settings retained;88499 six observed SDPA bridges exact | No demonstrated offending SDPA primitive; no backend setting relabeled as a new fix |
| H3 CUDA/process execution |88503 only changed CUDA_LAUNCH_BLOCKING0/1 before process CUDA init | Blocked captured raw pair exact,async pair FAIL;candidate support only |
| H4 local VJP versus upstream |88497 selected input/output exact but incoming adjoint unequal;local VJP follows | First observed inequality precedes selected local LoRA VJP |
| H5 mixed precision/casting | Raw LoRA FP32;lossless packing requires exact roundtrip;independent FP64 local references retained | Reference rounding quantified,not a proof of expected nondeterminism or its cause |

No whole-model replacement,scientific-loss/reward coefficient change,SDPA switch,activation-checkpointing change,hyperparameter search or threshold relaxation occurred. The task is a non-adversarial constant-coefficient numerical fixture,not scientific reward evaluation.

An opt-in launch configuration candidate was implemented in `src/rl/cuda_execution_guard.py:9-14` and `src/models/pretrained_editor_loader.py:13-18` (current working-tree line numbers;strict_backward defaultsFalse). It checks CUDA initialization and sets CUDA_LAUNCH_BLOCKING1;late initialization without it fails closed. Original loader and exact executed failed-candidate versions are preserved in `candidate_original/` and `candidate_failed_source/`;the current guard has additional failure-warning documentation. The candidate passed captured raw checks but FAILED uninstrumented ordinary checks. It is not enabled by any production training caller. Guard docs explicitly mark it experimental. No justified mathematical/backward-operator source correction is established.

The88503 subprocess setting was established before interpreter CUDA use;88504 exercised a bootstrap that imports Torch then invokes the guard before Torch-reported CUDA initialization,then loader opt-in. The placement difference and observation styles are unresolved confounds,not an established explanation of failure. Do not generalize the captured pair's PASS to a complete BF16 fix.

## Actual execution and resource accounting

{table}

Total new GPU allocation: **{budget['consumed']}/960 A100 seconds**,including47-second capture-size failure and340-second bounded verification failure,reserved0,remaining{budget['remaining']}. This separate urgent-debug ledger uses the still-valid authorization window ending{budget['deadline_UTC']};the finalized overnight controller remains closed. Previous overnight spend172 plus this852 is1024 under the shared36000 envelope;the older3600 validation ledger is separate. No jobs from RSI-Video were charged,changed or canceled. No further GPU task is admitted.

Each job's exact protocol SHA256,declared source revisions,paths and results are in `job_revisions_and_results.json`;Git base `{git}` plus explicit working-tree hashes identify code,not a fictitious new commit. Failed source versions/logs remain.88504 returned124 because its own predeclared process timeout fired,not because Slurm reported a successful recovery.

## Required verification outcome

Raw captured pair and postclip/Adam/adapter/RNG exact and tolerance PASS. Ordinary first-update adapter:744,464 differing elements,max9.978311936720274e-7,exact/tolerance FAIL. Adam first moment:1,593,772 differing elements,max5.123263690620661e-6,exact/tolerance FAIL. Adam second moment:1,589,926 unequal,max1.657298298596288e-9,exact FAIL,tolerance PASS. Do not call its tolerance result bitwise equality.

Ordinary second-update adapter:1,387,812 differing elements,max6.400543043127982e-6. Ordinary second postclip:1,594,366 differing elements,max.0009832559153437614. These fail exact/tolerance. All ordinary initial checkpoints match. Raw0 after-state equals ordinary0 first-update common state;the ordinary pair still fails,so this one cross-mode match does not release the candidate.

The actual resume process restored the checkpoint and produced an exact92-tensor next trajectory. It did not complete/save the next optimizer update before timeout:after2,postclip,update receipt and final guard receipt are missing. Recovery-next-update is **NOT VERIFIED**,not PASS and not an invented numerical failure. Completed ordinary workers verified frozen weights,256-parameter optimizer ownership and unchanged reward guard. Final CPU verifier explicitly lists missing evidence.

Six new guard-unit tests and seven bounded-capture/API tests passed. Independent same-family provisional review cross-checked20 saved-tensor verification comparisons with zero discrepancies;it rejected a fix claim. Unit tests validate configuration/capture mechanics,not BF16 reliability. See `reports/BF16_FIX_REVIEW_20261102.md`.

{params}
'''
 (ROOT/'reports/NUMERICAL_ROOT_CAUSE_LATEST.md').write_text(root,encoding='utf8')
 verification=f'''# Numerical fix verification

PROJECT_ID: RSI-POISONING. UTC {stamp}.

**ROOT_CAUSE_PARTIALLY_LOCALIZED — candidate FAILED ordinary verification. No FIX_VERIFIED claim.**

| Required test | Actual result |
|---|---|
| Matched checkpoint/optimizer/data/RNG | PASS for compared starts |
| Captured exact raw gradients | PASS,raw0/raw1,all256 tensors/1,594,368 FP32 elements |
| Separate tolerance raw checks | PASS for captured pair only |
| Raw pair postclip/Adam/adapter | PASS exact/tolerance |
| Ordinary uninstrumented paired execution | FAIL from first optimizer update |
| Ordinary postclip and Adam comparison | FAIL;some second-moment tolerance checks pass but exact fails |
| Final ordinary adapter | FAIL exact/tolerance |
| Checkpoint save/load and next trajectory | Restore actually executed;next trajectory exact |
| Next optimizer update after checkpoint resume | NOT VERIFIED/TIMEOUT;no saved completed update |
| Frozen weights/trainable ownership | PASS in completed raw/ordinary workers;resume completion receipt missing |

Source change attempted: optional launch guard in `src/rl/cuda_execution_guard.py`,loader flag in `src/models/pretrained_editor_loader.py`. No production caller opt-in was enabled. Original versions and failed candidate source remain. Model/loss/coefficients/strict criteria are unchanged. The helper controls process configuration only and is explicitly not a reproducibility certificate. New CPU regressions6 guard +7 capture/API PASS;full ordinary acceptance FAIL. Per-tensor exact/tolerance,max absolute,relativeL2 and affected counts are stored in `verification/mandatory_verification.json`. Missing fields cannot become a vacuous PASS.

{table}

Actual budget852/960,reserved0,remaining108;old controller not reopened;deadline{budget['deadline_UTC']}. No retry of the failed candidate and no benign training job. Lowest CUDA primitive and software-defect versus numerical-property distinction remain unresolved. No broad audit or completed model smoke test was recreated.
'''
 (ROOT/'reports/NUMERICAL_FIX_VERIFICATION.md').write_text(verification,encoding='utf8')
 next_=f'''# Next eligible experiment

PROJECT_ID: RSI-POISONING. UTC {stamp}. Status **ROOT_CAUSE_PARTIALLY_LOCALIZED**.

This debug run is finalized:852/960 A100 seconds used,108 unreserved seconds remain,and the inherited authorization expires{budget['deadline_UTC']}. Do not silently reopen this run or the older finalized controller. No further GPU diagnostic or learning run is currently admitted. Existing two-hour monitoring stays unchanged.

The next falsifiable numerical question is whether full guided-UNet predictions,Gaussian loss inputs and dLoss/dUNet-output match when the selected LoRA upstream adjoint differs. The completed probes stored only selected intermediate forwards,not every relevant full-network/loss boundary. Prepare a minimal recorder for the small4-channel UNet outputs,loss mean/action/logp and output gradients across all20 transitions;avoid a network dump. Compare pre-interpreter CUDA_LAUNCH_BLOCKING1 against the late-bootstrap variant as a separately preregistered single factor if resources are newly authorized. This is an unresolved hypothesis,not a proven cause or approved retry-untilPASS. If upstream inputs match,trace only the first unequal node's causal path;do not start with clipping.

A repair is admissible only after justified localization,original-source retention,targeted regressions,ordinary uninstrumented paired updates,exact raw/postclip/Adam/final parameter checks,separate tolerance checks,checkpoint reload followed by a completed next optimizer update,and frozen/LoRA ownership verification. Current candidate fails these requirements and remains opt-in/defaultFalse.

Benign learning remains blocked. After numerical requirements AND reward/cohort/evaluation admission pass,first use a matched Frozen versus one DDPO-only harmless lighting-editing arm;expand to ordinary scheduler/RSI only after their prerequisites pass. Keep initial checkpoints,seeds,data,query/NFE/compute budgets matched;preserve negative scores and rejected RSI cost. Do not add steps or search parameters to chase a proxy. Original reward guard is not released.160 prospective final-test sources stay sealed;historical development remains development;no fabricated human labels. No poisoning,stealth,backdoor-ASR optimization or attack improvement. ASR NOT TESTED.
'''
 (ROOT/'reports/NEXT_ELIGIBLE_EXPERIMENT.md').write_text(next_,encoding='utf8')
 # Preserve negative domain gate; do not alter reward/training guard code.
 gate=read(ROOT/'reports/SCIENTIFIC_GATE_STATUS.json');gate['updated_UTC']=stamp;gate['statuses']['NUMERICAL_REPRODUCIBILITY_STATUS']=dict(status='ROOT_CAUSE_PARTIALLY_LOCALIZED',scope='Captured launch-blocked pair PASS;88504 uninstrumented ordinary FAIL and actual resumed next update TIMEOUT. No verified production fix.');write(ROOT/'reports/SCIENTIFIC_GATE_STATUS.json',gate)
 from scripts.render_execution_remediation_dashboard_20261102 import dashboard
 dashboard(ROOT)
 monitor=read(ROOT/'reports/monitoring/scheduler.json');protected=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json');assert all(sha(ROOT/p)==h for p,h in protected.items())
 sizes={str(p.relative_to(out)):p.stat().st_size for p in out.rglob('*.pt')};queue=subprocess.check_output(['squeue','-h','-u','ydk297','-o','%i|%j|%T|%S'],text=True)
 reportnames=['NUMERICAL_ROOT_CAUSE_LATEST.md','NUMERICAL_FIX_VERIFICATION.md','NEXT_ELIGIBLE_EXPERIMENT.md'];put(out/'final_delivery.json',dict(PROJECT_ID='RSI-POISONING',status='ROOT_CAUSE_PARTIALLY_LOCALIZED',UTC=stamp,budget=budget,jobs=jobs,reports={n:sha(ROOT/'reports'/n) for n in reportnames},protected_source_hashes_unchanged=True,next_monitor=monitor,tensor_bytes=sum(sizes.values()),tensor_files=sizes,queue=queue,large_C_files_created=False,production_training_caller_optin_enabled=False))
 put(out/'artifact_inventory.json',dict(files=[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=sha(p)) for p in out.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']))
 print(json.dumps(dict(status='ROOT_CAUSE_PARTIALLY_LOCALIZED',GPU_seconds=budget['consumed'],remaining=budget['remaining'],retained_tensor_bytes=sum(sizes.values()),next_monitor=monitor['next_job_id'],reports=reportnames)))
if __name__=='__main__':main()
