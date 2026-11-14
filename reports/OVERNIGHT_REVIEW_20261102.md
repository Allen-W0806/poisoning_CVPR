# Overnight execution review, 2 November 2026

**Verdict — WARN.** The numerical failure and the resulting stop are supported. Job 88433 consumed **172 A100-seconds**. Its earliest differing measured boundary was the accumulated raw gradient before clipping. No ordinary/resume or learned-efficacy job followed. The executed controller and historical table have the limitations below.

This fresh same-family review is provisional. It covers `overnight_benign_20261102_v2` on UTC. The accompanying JSON preserves the exact reviewed input hashes, raw-witness hashes and machine-readable findings. This review used CPU reads of existing receipts and tensors. No Slurm job, model evaluation or sealed-cohort image read was performed by the reviewer.

## Resource and admission evidence

The new ledger opened at 06:17:57.636501 UTC with a deadline of 16:17:57.636501 UTC. It has a separate 36,000 GPU-second cap and a 1,200-second numerical sub-budget. Slurm accounting records one GPU for job 88433 on epyc00 from 06:18:07 to 06:20:59 UTC. The ledger released its 360-second reservation at completion and records 172 consumed, 0 reserved and 35,828 remaining. The existing 3,600-second ledger remained separately readable with 2,762 consumed and 838 remaining at inspection.

Intent and scheduler-reply files exist for all four submissions. Bootstrap 88432 and the `afterany:88433` handler 88434 completed with exit 0 on CPU. At initial inspection the one-shot deadline job 88431 was pending. The handler processed 88433 once. Its payload and all three workers exited 0. The scientific check recorded FAIL. The terminal state closed admissions, retained the failure, left the original reward release denied, and had zero passed stages.

Controller lines 32–55 reserve remaining time for pending, running and uncertain GPU jobs and charge elapsed time for completed and failed jobs. Lines 64–88 request one A100 with a Slurm deadline, write intentions before submission, and register the held completion callback before releasing the GPU job. Lines 155–168 include a late-start check and a bounded process-group timeout. The 18 saved CPU tests passed. These observed properties support this run's resource accounting. Recovery across event-delivery interruptions has the limits described below.

## Numerical evidence

All four saved updates contained 256 uniquely named raw gradients in the same order as their conditioning records. The reviewer independently recomputed affected tensor and element counts, maximum errors and the elementwise tolerance from the saved raw tensors. The acceptance test was exact equality. The additional per-element threshold used `abs(a-b) <= 1e-7 + 1e-5*abs(a)`.

| First / second update | Differing tensors | Differing elements | Maximum absolute error | Relative L2 | Tolerance |
|---|---:|---:|---:|---:|---|
| Fresh 0 / fresh 1 | 32/256 | 76,295 | 2.19675712e-7 | 2.22581203e-6 | FAIL |
| Resident 0 / resident 1 | 32/256 | 96,017 | 2.83461995e-5 | 3.40685713e-4 | FAIL |
| Fresh 0 / resident 0 | 32/256 | 95,710 | 2.80453824e-5 | 3.39563105e-4 | FAIL |
| Fresh 0 / resident 1 | 32/256 | 92,748 | 3.10502946e-6 | 3.60681320e-5 | FAIL |

Each pair had exact initial states and conditioning. The two fresh updates returned the same runtime norm of 1.033760666847229 and formula-derived clip factor of 0.9673410058021545 despite differing raw gradients. Their CPU FP64 reference norms were 1.0337607544359944 and 1.0337607815803325. In resident order, the FP64 values were 1.0337610898345555 and 1.0337614871943024. An independent dot-product reduction agreed with all four recorded FP64 totals to within 5e-14 and verified every per-tensor sum of squares. The snapshot clones detached CPU tensors and invokes the original clipping primitive exactly once. Its CPU fixture verified isolation from later mutations. Copying gradients can influence GPU synchronization.

The observed difference exists before global norm computation, including between two fresh processes. Its origin upstream of the captured boundary and any causal role of residency remain unresolved. The scientific guard hash matched the protocol, the frozen-base and ownership checks passed for every worker, and the numerical fixture used constant coefficient 1. Sources for this evidence are `gradient_boundary_capture_20261102.py:10–31`, `run_overnight_norm_fixture_20261102.py:17–47`, `diffusion_rl_trainer.py:39–58` and `overnight_benign_reporting_20261102.py:6–22`.

## Fixed historical model table

The four table rows matched the saved historical checkpoint hashes, 16 per-source observations, recorded branch NFE and branch-held times. Source-ID pairing independently reproduced the means and percentile intervals. SSIM, LPIPS and pixel no-op means matched their saved per-source values. LPIPS inference was not repeated for this review. The new CPU analysis used the same four previously exposed assessment sources across all arms with one historical training seed. Its brightness proxy, image similarity and pixel equality supply no independent judgment of edit validity. Clean VQA, human edit validity, ASR, RSI superiority and final-test performance remain unestablished.

The 28 historical training, validation and assessment source IDs and exact hashes had zero overlap with the 160 sealed manifest records. Only manifest identifiers and hashes were compared. Slurm independently confirms the historical 242 seconds for job 88392. Per-arm allocated GPU seconds are correctly empty. The single shared prefix used 168 forward evaluations, then the four branch increments used 48, 168, 168 and 168, for **720 total**. The historical allocation contributes zero seconds to the overnight ledger.

## Findings on the reviewed versions

1. **A one-shot callback can be lost.** Controller lines 135–150 return success immediately on lock contention. Lines 47–55 and 111 can return after a single stale accounting read without marking the parent processed. A separate health invocation can recover the work, but the successful callback exit alone does not ensure delivery. Registration also has a fail-closed interval between durable intent and returned job ID. It protects against blind duplication while leaving an orphaned held job possible until token reconciliation. No such loss occurred for 88433.
2. **Deadline finalization can replace the stop reason.** Lines 147–148 overwrite the top-level status with `DEADLINE_CLOSED`, including after a completed numerical stop, and skip result analysis for a terminal but unprocessed GPU job. Preserve a terminal scientific status and handle terminal receipts before closing admissions. This was a latent condition at inspection.
3. **Conditional ordinary verification omits its recorded preclip norm.** The ordinary worker saves canonical update receipts, but `analyze_resume` lines 24–31 compares only checkpoints, trajectory and postclip data. It omits the saved gradient norm and scalar loss for each corresponding update. Equal final tensors can therefore conceal a differing ordinary preclip norm, the earlier failure boundary. The conditional stage was never eligible in this run.
4. **Table labels obscure selected-model exposure and shared cost.** Reporting line 56 writes 20 training sources for each learned arm. Historical pilot lines 65, 82 and 87 show 20 attempted for RSI but only 10 contributing to its selected prefix after rollback. Label attempted and retained training sources separately. Label per-arm NFE as incremental and state that the 242-second historical allocation is counted once. Stored means and paired intervals are supported.
5. **The clipping coefficient is reconstructed.** Capture lines 26–30 observe the actual returned norm and reconstruct the coefficient from the pinned formula. The protocol and rendered narrative should distinguish this from intercepting the runtime multiplier. Fresh-pair equality remains valid for the formula-derived coefficient.
6. **An infrastructure retry can exceed one diagnostic submission.** Controller lines 104–108 allow an extra submission on Slurm NODE_FAIL without checking whether a partial diagnostic executed. Test lines 43–45 check the node-name regex only. The actual run had one attempt. A strict one-submission policy needs to remove that path or make the distinction between a submission and an unstarted diagnostic explicit and verifiable.

**Claim boundary.** The preserved result supports a reproducibility failure in this instrumented, fixed-input engineering fixture. There is no confirmed BF16 repair or learned efficacy result. Further work on controller resilience, table labels or CPU evidence quality cannot reclassify the observed numerical failure.
