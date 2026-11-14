"""Finish CPU-only lineage checks and publish an honestly scoped FP32 report."""
import hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import compare,load,sha,put

def main():
 import torch
 torch.set_num_threads(2)
 comparison=ROOT/'results/quick_comparison_20261102T192245Z'
 pilot=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot'
 fp=ROOT/'results/fp32_reference_20261102_v1_math_cpu'
 branch=ROOT/'experiments/fp32_reference_20261102_v1/branch'
 result=json.loads((fp/'comparison.json').read_text());cross={}
 x,y=load(fp/'bf16a/witness.pt'),load(fp/'fp32a/witness.pt')
 for step in ['after1','after2']:
  for key in ['output','loss','raw','original_state','optimizer']:cross[step+'/'+key]=compare(x[step][key],y[step][key])
 put(fp/'cross_precision_boundary_comparisons.json',cross)
 history=[];before_hashes={}
 expected=json.loads((ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json').read_text())
 protected={n:dict(sha256=sha(ROOT/n),matches_preserved_reference=sha(ROOT/n)==h) for n,h in expected.items()}
 assert all(v['matches_preserved_reference'] for v in protected.values())
 for arm in ['frozen','ddpo_only','ordinary_scheduler','rsi']:
  path=pilot/arm/'selected/training_state.pt';before_hashes[arm]=sha(path);state=load(path)
  steps=sorted({int(v['step']) for v in state['optimizer']['state'].values()})
  expected_sources=0 if arm=='frozen' else (10 if arm=='rsi' else 20)
  assert len(state['consumed'])==expected_sources
  assert steps==([] if arm=='frozen' else [state['step']])
  r=dict(arm=arm,selected_step=state['step'],Adam_step_values=steps,actual_consumed_trajectory_count=len(state['consumed']),consumed_trajectory_IDs=sorted(state['consumed']),sha256=before_hashes[arm])
  if arm=='frozen':reference=load(pilot/'shared_initial/training_state.pt')
  elif arm=='rsi':reference=load(pilot/'shared_prefix/training_state.pt')
  else:reference=None
  if reference is not None:
   r['retained_reference_adapter_exact']=compare(reference['adapter'],state['adapter'])['exact'];r['retained_reference_optimizer_exact']=compare(reference['optimizer'],state['optimizer'])['exact']
   assert r['retained_reference_adapter_exact'] and r['retained_reference_optimizer_exact']
  history.append(r)
 put(comparison/'selected_training_history_reverification.json',history)
 put(fp/'production_protection.json',dict(checked_UTC=datetime.now(timezone.utc).isoformat(),protected=protected,main_branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip(),reference_branch=subprocess.check_output(['git','branch','--show-current'],cwd=branch,text=True).strip(),reference_HEAD=subprocess.check_output(['git','rev-parse','HEAD'],cwd=branch,text=True).strip(),reference_sources={n:sha(branch/n) for n in ['fp32_attention_reference_v1.py','run_cpu_reference_v1.py']},selected_checkpoints_unchanged=all(sha(pilot/a/'selected/training_state.pt')==h for a,h in before_hashes.items()),production_code_edits_in_this_task=0))
 table='| Comparison | Initial/input | First update | Second update | Exact | Numerical tolerance |\n|---|---|---|---|---|---|\n'
 for name in ['BF16_repeat','BF16_resume','FP32_repeat','FP32_resume']:
  c=result['comparisons'][name]
  table+=f"|{name}|identical|real forward/backward/Adam|real next update|{all(v['exact'] for v in c.values())}|{all(v['tolerance_pass'] for v in c.values())}|\n"
 detail='| Cross-precision boundary | Unequal elements | Max absolute | Relative L2 | Exact | Tolerance |\n|---|---:|---:|---:|---|---|\n'
 for key,c in cross.items():detail+=f"|{key}|{c['affected_elements']}|{c['max_abs']:.12g}|{c['relative_L2']:.12g}|{c['exact']}|{c['tolerance_pass']}|\n"
 workers=result['workers'];peak=max(x['process_peak_RSS_KiB'] for x in workers.values());pbytes=workers['fp32a']['parameter_storage_bytes'];temp=workers['fp32a']['temporary_FP32_frozen_weight_bytes']
 retained=sum(p.stat().st_size for d in [fp,ROOT/'results/fp32_reference_20261102_v1'] for p in d.rglob('*') if p.is_file())
 text=f'''# FP32 numerical alternative — isolated reference v1

PROJECT_ID: RSI-POISONING. Updated UTC {datetime.now(timezone.utc).isoformat()}.

**A versioned upstream-attention reference was implemented and tested on CPU. CUDA memory feasibility and full pretrained-UNet ordinary/resumed execution remain NOT TESTED. This is not a production BF16 fix.** Fixed-checkpoint read-only evaluation is independently valid and already delivered in QUICK_BASELINE_COMPARISON.md/csv/tex.

## Actual isolated implementation

Branch: `codex/fp32-reference-20261102-v1`, separate worktree `{branch}`; HEAD remains `06bd889d9392c4d53621c3e4af15ad718e84f6c5`, with the two experimental source files staged separately, not committed. Production remains on master; its existing working-tree edits were not changed. Exact experimental source hashes are recorded in production_protection.json. Sources are `fp32_attention_reference_v1.py` and `run_cpu_reference_v1.py` in that worktree; Windows copies are under experiments/fp32_reference_20261102_v1/source/.

The wrapper uses Torch2.6.0 torch.func.functional_call with temporary FP32 floating parameters/buffers and FP32 hidden/context/mask tensors under disabled autocast. The original Diffusers AttnProcessor2_0 and PEFT modules perform projection, scaled-dot-product attention and output projection. It upcasts frozen base attention weights and intermediates, not only LoRA. All original frozen base storage stays BF16 and optimizer-owned LoRA storage stays FP32. Temporary functional tensors do not replace optimizer-owned trainable objects. No scientific loss, original training routine, strict acceptance test or reward-release guard is modified.

The fixture comprises actual self- and cross-attention modules with query width320, context width768,8 heads×40,128 query tokens,77 context tokens, rank8/alpha16 LoRA,16 FP32 trainable tensors. The original BF16 path retains the production-style FP32 LoRA linear helper while base attention runs under BF16 autocast. Both paths explicitly select the MATH SDPA backend and use byte-identical initialization, inputs and a fixed constant output adjoint1/40960. This is a harmless arithmetic fixture, not a learning objective or production model. There is no checkpoint-activation recomputation or dropout in this test. An earlier automatic CPU-backend check and its executed sources are preserved separately under results/fp32_reference_20261102_v1; the table below uses only the explicit-MATH revision.

## Executed checks

Six fresh CPU worker processes ran under the pinned `{workers['fp32a']['torch']}` runtime: BF16 repeat/repeat/resume and FP32 repeat/repeat/resume. Each executes ordinary module forward/backward, verifies finite gradients, performs a real AdamW update, saves model/Adam/RNG, and performs a second real update. Resume workers construct fresh module/optimizer objects, load the disk checkpoint, restore RNG and execute the next update. Full saved outputs, gradients, model states and Adam states were compared with unchanged exact and numerical criteria.

{table}

All four repeat/resume comparisons pass exact and tolerance checks with max absolute error0, relative L2 error0 and0 differing elements. Every worker confirms finite outputs/loss/gradients, changed trainables,16 FP32 trainable tensors, exact optimizer ownership and invariant frozen-weight hashes. **BF16 CPU also passes**, so this is API/reference validation and supplies no evidence that FP32 cures the observed CUDA failure. The comparison utility is unchanged: per-element numerical tolerance abs_error≤1e-7+1e-5×abs(reference), reported separately from exact equality.

Across precision paths, the initial states and inputs match exactly. The first saved differing boundary is the first-update **forward output**, as expected when arithmetic precision changes. No internal CUDA primitive is identified:

{detail}

Cross-precision inequality is preserved and is not misreported as regression equivalence. Detailed comparisons include every tensor and structural difference in cross_precision_boundary_comparisons.json.

## Memory and resource measurements

Total elapsed CPU wall time including all six interpreter/module loads: {result['wall_seconds_including6_process_loads']:.3f}s. Worker peak process RSS: at most {peak:,}KiB ({peak/1024:.2f}MiB). Fixture parameter storage is {pbytes:,}bytes; temporary FP32 frozen-attention weights require {temp:,}bytes if simultaneously resident. Worker two-update sections range {min(w['CPU_wall_seconds'] for w in workers.values()):.3f}–{max(w['CPU_wall_seconds'] for w in workers.values()):.3f}s; these tiny CPU timings are not a GPU speed comparison. Both CPU backend revisions retain about {retained/1024**2:.2f}MiB of checkpoints, tensor witnesses, exact-source snapshots and receipts on cluster storage for reproducibility; these are evidence, not disposable scratch. No model, installer or multi-gigabyte intermediate was written to C:; the small Windows source/report copies are on H:. No run-specific scratch directory was needed.

**No Slurm job was submitted; new GPU consumption0 seconds.** These CPU memory figures do not establish full512×512 diffusion CUDA feasibility. For one eight-head FP32 self-attention score tensor at4096 tokens, the elementary B×H×N×N×4 storage is536,870,912bytes (512MiB), versus524,288bytes for128 tokens; backward and other activations add memory, and the actual SDPA implementation need not materialize that exact tensor. Repeated temporary weight casts also incur allocation/compute overhead. Full-UNet memory, performance, graph compatibility and optimizer-resume behavior remain unresolved.

The old10-hour window expired at2026-11-02T16:17:57.636501 UTC. No new GPU budget/window is authorized. Any eventual local CUDA feasibility test must be separately authorized, use one A100, stop on allocation/memory failure, and stay within a proposed200-GPU-second cap. First measure reset_peak_memory_stats/max_memory_allocated for the same harmless local fixture and then one representative full-size attention shape; do not allocate a larger UNet run until that result is known. Do not run read-only four-arm evaluation and this numerical experiment from a shared untracked allowance. The separate read-only64-source evaluation quote is600 A100 seconds, not a numerical-test authorization.

## What is still required for an actual repair

The production ordinary-path result from job88504 remains failed, and its resume-next-update remains incomplete. The CPU reference does not supersede either. A future experimental integration must preserve checkpoint key mapping and original trainable/frozen ownership; this wrapper currently adds an inner namespace and is **not** a drop-in checkpoint-loader patch. It has only been exercised through the local attention fixture's own state interface. It must also demonstrate actual CPU/CUDA functional-call+PEFT behavior under the full UNet, ordinary uninstrumented repeated updates, finite raw gradients, optimizer updates, frozen invariance, and a completed save/reload next optimizer update with original strict tests intact. The responsible CUDA primitive remains unresolved. Neither the scientific reward nor benign production training is released by these results.

Evidence: results/fp32_reference_20261102_v1_math_cpu/comparison.json, cross_precision_boundary_comparisons.json, production_protection.json, six worker receipts/checkpoints/witnesses. Production policy/trainer/trajectory/guard/config reference hashes all match their preserved reference. Four selected pilot checkpoints were rehashed unchanged; their actual consumed trajectory counts, Adam steps and Frozen/RSI initial-prefix references were reverified in selected_training_history_reverification.json. No poisoning, attack metric or security optimization was implemented. ASR NOT TESTED.
'''
 (ROOT/'reports/FP32_NUMERICAL_ALTERNATIVE.md').write_text(text,encoding='utf8')
 # Readable spacing only; numeric table values and code/hash strings are intact.
 p=ROOT/'reports/QUICK_BASELINE_COMPARISON.md';q=p.read_text()
 addition='\nAdditional CPU checkpoint history verification: actual consumed trajectory counts are0/20/20/10; saved Adam step values are empty/2/2/1. Frozen adapter/Adam match shared_initial exactly; selected RSI adapter/Adam match its own shared_prefix exactly. Selected file hashes remain unchanged. See selected_training_history_reverification.json.\n'
 if addition not in q:q+=addition
 p.write_text(q,encoding='utf8')
 manifest=json.loads((comparison/'delivery_manifest.json').read_text());manifest['reports']={n:sha(ROOT/'reports'/n) for n in ['QUICK_BASELINE_COMPARISON.md','QUICK_BASELINE_COMPARISON.csv','QUICK_BASELINE_COMPARISON.tex','FP32_NUMERICAL_ALTERNATIVE.md']};manifest['finalized_UTC']=datetime.now(timezone.utc).isoformat();put(comparison/'delivery_manifest.json',manifest)
 print(json.dumps(dict(reports=manifest['reports'],production_protected=True,actual_training_counts=[h['actual_consumed_trajectory_count'] for h in history],reference_status=result['status'],new_GPU_seconds=0)))

if __name__=='__main__':main()
