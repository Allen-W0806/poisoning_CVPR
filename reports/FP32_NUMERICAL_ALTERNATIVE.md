# FP32 numerical alternative — isolated reference v1

PROJECT_ID: RSI-POISONING. Updated UTC 2026-11-02T19:32:07.143056+00:00.

**A versioned upstream-attention reference was implemented and tested on CPU. CUDA memory feasibility and full pretrained-UNet ordinary/resumed execution remain NOT TESTED. This is not a production BF16 fix.** Fixed-checkpoint read-only evaluation is independently valid and already delivered in QUICK_BASELINE_COMPARISON.md/csv/tex.

## Actual isolated implementation

Branch: `codex/fp32-reference-20261102-v1`, separate worktree `/storage/home/ydk297/projects/poisoning/experiments/fp32_reference_20261102_v1/branch`; HEAD remains `06bd889d9392c4d53621c3e4af15ad718e84f6c5`, with the two experimental source files staged separately, not committed. Production remains on master; its existing working-tree edits were not changed. Exact experimental source hashes are recorded in production_protection.json. Sources are `fp32_attention_reference_v1.py` and `run_cpu_reference_v1.py` in that worktree; Windows copies are under experiments/fp32_reference_20261102_v1/source/.

The wrapper uses Torch2.6.0 torch.func.functional_call with temporary FP32 floating parameters/buffers and FP32 hidden/context/mask tensors under disabled autocast. The original Diffusers AttnProcessor2_0 and PEFT modules perform projection, scaled-dot-product attention and output projection. It upcasts frozen base attention weights and intermediates, not only LoRA. All original frozen base storage stays BF16 and optimizer-owned LoRA storage stays FP32. Temporary functional tensors do not replace optimizer-owned trainable objects. No scientific loss, original training routine, strict acceptance test or reward-release guard is modified.

The fixture comprises actual self- and cross-attention modules with query width320, context width768,8 heads×40,128 query tokens,77 context tokens, rank8/alpha16 LoRA,16 FP32 trainable tensors. The original BF16 path retains the production-style FP32 LoRA linear helper while base attention runs under BF16 autocast. Both paths explicitly select the MATH SDPA backend and use byte-identical initialization, inputs and a fixed constant output adjoint1/40960. This is a harmless arithmetic fixture, not a learning objective or production model. There is no checkpoint-activation recomputation or dropout in this test. An earlier automatic CPU-backend check and its executed sources are preserved separately under results/fp32_reference_20261102_v1; the table below uses only the explicit-MATH revision.

## Executed checks

Six fresh CPU worker processes ran under the pinned `2.6.0+cu124` runtime: BF16 repeat/repeat/resume and FP32 repeat/repeat/resume. Each executes ordinary module forward/backward, verifies finite gradients, performs a real AdamW update, saves model/Adam/RNG, and performs a second real update. Resume workers construct fresh module/optimizer objects, load the disk checkpoint, restore RNG and execute the next update. Full saved outputs, gradients, model states and Adam states were compared with unchanged exact and numerical criteria.

| Comparison | Initial/input | First update | Second update | Exact | Numerical tolerance |
|---|---|---|---|---|---|
|BF16_repeat|identical|real forward/backward/Adam|real next update|True|True|
|BF16_resume|identical|real forward/backward/Adam|real next update|True|True|
|FP32_repeat|identical|real forward/backward/Adam|real next update|True|True|
|FP32_resume|identical|real forward/backward/Adam|real next update|True|True|


All four repeat/resume comparisons pass exact and tolerance checks with max absolute error0, relative L2 error0 and0 differing elements. Every worker confirms finite outputs/loss/gradients, changed trainables,16 FP32 trainable tensors, exact optimizer ownership and invariant frozen-weight hashes. **BF16 CPU also passes**, so this is API/reference validation and supplies no evidence that FP32 cures the observed CUDA failure. The comparison utility is unchanged: per-element numerical tolerance abs_error≤1e-7+1e-5×abs(reference), reported separately from exact equality.

Across precision paths, the initial states and inputs match exactly. The first saved differing boundary is the first-update **forward output**, as expected when arithmetic precision changes. No internal CUDA primitive is identified:

| Cross-precision boundary | Unequal elements | Max absolute | Relative L2 | Exact | Tolerance |
|---|---:|---:|---:|---|---|
|after1/output|40959|0.00161439180374|0.00280610860613|False|False|
|after1/loss|1|2.40281224251e-06|0.00444424156348|False|False|
|after1/raw|48128|4.56431880593e-05|0.00264575747574|False|False|
|after1/original_state|6406|1.99349597096e-05|3.21581884155e-06|False|False|
|after1/optimizer|96256|4.5643246267e-06|1.60047818654e-05|False|False|
|after2/output|40960|0.00190338492393|0.00274274572768|False|False|
|after2/loss|1|6.50063157082e-07|0.00176143306964|False|False|
|after2/raw|48128|4.76241111755e-05|0.00280962685036|False|False|
|after2/original_state|22859|3.98447737098e-05|5.51280059946e-06|False|False|
|after2/optimizer|96256|8.8702654466e-06|1.48754994812e-05|False|False|


Cross-precision inequality is preserved and is not misreported as regression equivalence. Detailed comparisons include every tensor and structural difference in cross_precision_boundary_comparisons.json.

## Memory and resource measurements

Total elapsed CPU wall time including all six interpreter/module loads: 37.949s. Worker peak process RSS: at most 555,944KiB (542.91MiB). Fixture parameter storage is 2,405,632bytes; temporary FP32 frozen-attention weights require 4,426,240bytes if simultaneously resident. Worker two-update sections range 0.076–0.174s; these tiny CPU timings are not a GPU speed comparison. Both CPU backend revisions retain about 140.00MiB of checkpoints, tensor witnesses, exact-source snapshots and receipts on cluster storage for reproducibility; these are evidence, not disposable scratch. No model, installer or multi-gigabyte intermediate was written to C:; the small Windows source/report copies are on H:. No run-specific scratch directory was needed.

**No Slurm job was submitted; new GPU consumption0 seconds.** These CPU memory figures do not establish full512×512 diffusion CUDA feasibility. For one eight-head FP32 self-attention score tensor at4096 tokens, the elementary B×H×N×N×4 storage is536,870,912bytes (512MiB), versus524,288bytes for128 tokens; backward and other activations add memory, and the actual SDPA implementation need not materialize that exact tensor. Repeated temporary weight casts also incur allocation/compute overhead. Full-UNet memory, performance, graph compatibility and optimizer-resume behavior remain unresolved.

The old10-hour window expired at2026-11-02T16:17:57.636501 UTC. No new GPU budget/window is authorized. Any eventual local CUDA feasibility test must be separately authorized, use one A100, stop on allocation/memory failure, and stay within a proposed200-GPU-second cap. First measure reset_peak_memory_stats/max_memory_allocated for the same harmless local fixture and then one representative full-size attention shape; do not allocate a larger UNet run until that result is known. Do not run read-only four-arm evaluation and this numerical experiment from a shared untracked allowance. The separate read-only64-source evaluation quote is600 A100 seconds, not a numerical-test authorization.

## What is still required for an actual repair

The production ordinary-path result from job88504 remains failed, and its resume-next-update remains incomplete. The CPU reference does not supersede either. A future experimental integration must preserve checkpoint key mapping and original trainable/frozen ownership; this wrapper currently adds an inner namespace and is **not** a drop-in checkpoint-loader patch. It has only been exercised through the local attention fixture's own state interface. It must also demonstrate actual CPU/CUDA functional-call+PEFT behavior under the full UNet, ordinary uninstrumented repeated updates, finite raw gradients, optimizer updates, frozen invariance, and a completed save/reload next optimizer update with original strict tests intact. The responsible CUDA primitive remains unresolved. Neither the scientific reward nor benign production training is released by these results.

Evidence: results/fp32_reference_20261102_v1_math_cpu/comparison.json, cross_precision_boundary_comparisons.json, production_protection.json, six worker receipts/checkpoints/witnesses. Production policy/trainer/trajectory/guard/config reference hashes all match their preserved reference. Four selected pilot checkpoints were rehashed unchanged; their actual consumed trajectory counts, Adam steps and Frozen/RSI initial-prefix references were reverified in selected_training_history_reverification.json. No poisoning, attack metric or security optimization was implemented. ASR NOT TESTED.
