# Targeted numerical evidence review — 2026-11-02

Review independence: **same-family**. Acceptance status: **provisional**. Scope: the four requested scripts and seven requested result JSON files, plus the directly referenced saved tensor and content artifacts. No model inference, training, GPU submission, production edits, or new annotation was performed by this reviewer.

**Verdict: WARN on evidence description; numerical acceptance FAIL; root cause and fix eligibility remain unsupported.** The negative numerical evidence is real and must remain preserved. No fake annotations, phantom independent cohort, or demonstrated learning benefit was found in the scoped outputs.

## Verified evidence

- The four local review scripts match canonical cluster bytes. All five protocol-bound source hashes, the shared-prefix checkpoint hash, cohort manifest hash, and future baseline checkpoint hash match current canonical files.
- A1/A2 before-state, before-update RNG, and trajectories are exact. Independently recomputing all saved A1/A2 tensor metrics confirms 256 different postclip tensors, 1,594,333 affected elements, maximum absolute error `2.678763121366501e-05`, and maximum per-tensor relative L2 `0.0011192377571042473`. Exact and prescribed tolerance checks both fail. The after-state has 768 different tensors, maximum absolute error `2.6787747628986835e-06`.
- **A2/B1 are exact on every common saved boundary.** B1/B2 are reported exact, including their captured forward, raw-gradient, contribution and local adjoint boundaries. Thus A1/B1 differences do not isolate an instrumentation effect; fixed execution order and unrecorded process state remain confounders.
- Independently rehashed all 160 candidate images, 96 review assets, and both ZIP packages against the scoped receipt. Candidate files also match the candidate manifest. Actual accepted human pairs remain 0. All 28 deterministic brightness-control rows recompute within `1e-14`; no model learning occurred.

## Concrete findings and claim boundaries

1. **Missing ordinary raw/forward evidence prevents root-cause localization.** `scripts/numerical_boundaries_20261102.py:63-70` compares only the intersection of witness keys; A witnesses have no `raw`, `forward`, `contributions`, or `first_transition_downblock`. `postclip` is the earliest *saved mismatching* boundary, not the earliest divergent operation. Record missing boundaries explicitly. B-only exactness cannot release ordinary BF16 training or establish a fix.
2. **Capture scope is overstated.** `scripts/run_numerical_boundary_fixture_20261102.py:66-77` hooks only the 32 trainable attention LoRA weights under `down_blocks.0.attentions`; the saved witness contains 32 × 20 parameter contributions and 96 local module entries for transition 0 (32 × 3 CFG branches), with all 96 upstream adjoints present. `protocol.json:43` says “full first downblock inputs/output adjoints,” which is broader than captured evidence. Use “32 attention LoRA modules in downblock 0, transition 0.”
3. **`resume_exact` means checkpoint round-trip, not continuation.** `scripts/run_numerical_boundary_fixture_20261102.py:82` saves, immediately restores, then compares state. No resumed next update is executed. `execution_summary.json:43,86,129,172,179` supports round-trip equality only. The future lock correctly requires continuation, but this run does not satisfy it.
4. **Assessment rows retain a misleading historical role.** `next_benign_cohort_lock.json:631,656,681,706` says `role: independent_evaluation`, while lines 709 and 725 explicitly say historically exposed development data and `eligible_now: false`. The negative top-level eligibility is correct. Preserve the inherited value as `historical_role` and set the effective role to `development_assessment` in any supplemental corrected lock.
5. **Constant-coefficient probe is not the original scientific objective.** The update formula and production code hashes are unchanged, but every coefficient is 1.0 (`run_numerical_boundary_fixture_20261102.py:53`). `execution_summary.json:191` should be read as unchanged DDPO loss implementation, not unchanged scientific-reward evaluation or an efficacy test.
6. **Comparator edge cases need explicit definitions for future use.** `numerical_boundaries_20261102.py:16-17` uses the first tensor for relative-L2 denominator but `torch.isclose(first, second)` uses the second for tolerance scaling. Document that orientation or make it consistent. Line 27 maps undefined relative L2 for zero-reference/nonzero-error to 0 in the aggregate; report a separate aggregate undefined count instead. There are **0** such zero-denominator cases in both reviewed comparison files, so this does not change the current failed verdict.

## Integrity checklist

- A, provenance: **PASS with scope qualifier**. Numerical replay uses saved execution tensors, not semantic ground truth. Brightness target is an explicit source-derived proxy; inherited VQAv2 answers are not used to establish improvement.
- B, score normalization: **PASS for current results**, with the comparator edge cases above. Brightness reports absolute weighted-RGB error reduction; it is not percent accuracy or calibrated physical luminance.
- C, result existence: **PASS**. Referenced scoped outputs and underlying inspected artifacts exist and measured values match.
- D, live metric code: **PASS**. Saved metric outputs correspond to executed calculations; independent CPU recalculation reproduced the ordinary-pair metrics and brightness rows.
- E, scope: **WARN**. One process/device, one batch of 10 historical development sources, two transitions, fixed A1/A2/B1/B2 order, two repeats per mode, partial internal capture. No fresh-process reproducibility, root primitive, repair, learning, semantic validity, or scientific release is demonstrated.
- F, evaluation type: **engineering reproducibility probe** plus **self_supervised_proxy** brightness controls. Human evaluation remains pending, not completed.

The four witness files retain approximately **1.411 GB** on the cluster, exceeding the protocol's 650 MB estimate; these are experiment evidence and should be retained. This review created no large C: artifact and no scratch directory.

## Reviewed result hashes

The comparison JSON was regenerated during this review: its SHA256 changed from `2d3efa3d07e66fc1a884e55618f77c476e683dab065c4b6fda4e84622068b6e6` to `020c6a2bbc54e64eb318835839431a86ae8194ea4aae7fa1092f7a51403ca8ac`. The independent A1/A2 verification used the first snapshot. Preserve supplemental evidence separately; do not interpret byte identity as verified across regeneration.

| File | SHA256 at final provenance read |
|---|---|
| protocol.json | a77287fb2c51955451d35b4b5206dfdafaee10d9c918fd5f80407857adbf31e5 |
| execution_summary.json | 82dc00b9802d3e55e937910a7640846f7a0775cbe1fd3a50cd95682ddca1eb8c |
| existing_tensor_comparisons.json | 87ebe827fda652d0068b78ae1deefe2d45d6efb11cd9e886753cee0cdfa76fae |
| deterministic_benign_control.json | 243b75f16a5c88f360fb95d82320f4f4b04bd288fdf5ae2297fcd80b5d895179 |
| next_benign_cohort_lock.json | e1e8ff3f20abd6b97217cfd75e2a106ab5797f9dca88e8756ee5a178b748581c |
| candidate_and_review_preservation.json | b4cae0c6a2059db9c9be4b58fb6b6fec655f6aa497507872517e81a120d1901e |

Current-run checks above establish this report's findings. Prior memory was used only to preserve the known no-release/no-rerun interpretation; it was not substituted for current tensor evidence.
