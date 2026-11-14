# Execution Remediation — Latest

Project: **RSI-POISONING**. Canonical root: `/storage/home/ydk297/projects/poisoning`. Updated UTC: 2026-11-02T04:42:05.666593+00:00. Git base: `06bd889d9392c4d53621c3e4af15ad718e84f6c5`; working-tree fixes are archived and hash-bound.

Actual implementation fixes, regressions, one numerical experiment and one pretrained benign engineering pilot completed. **Engineering PASS does not imply numerical or scientific PASS.**

## 1. Bugs actually fixed

| Defect | Actual code correction | Executed evidence | Remaining limit / next distinct action |
|---|---|---|---|
| YAML reward metadata was not executed | `src/rl/reward.py:RewardWeights.from_config`; strict exact aliases/finite coefficients, no silent defaults. `scripts/smoke_generator_rl.py` now calls `src/rl/benign_reward_v3.py:caller_feedback` | 88387 nondefault-coefficient tests;500 historical rows reproduce exactly | Coefficients unchanged; construct validation absent |
| Verified identity lost consistency credit | v3 caller credits verified valid OR explicitly provisional observations | Identity regression gives1.4666666667 at unchanged cost proxy1/3; invalid/missing/ambiguous cases checked | No-op remains preferred; this is not an instruction-following reward fix |
| Missing/nonfinite components and unit/logging ambiguity | v3 raw/effective components, strict numeric schema, executed/recorded weights, clipping flags, explicit denoising-step cost proxy; actual dimensions/steps/resource fields logged | 88387 and legacy compatibility checks PASS | Allocation GPU-seconds remain a separate job measurement; no proxy-to-measured-cost substitution |
| Corrected diagnostic could enter old training callers | Scientific reward-release check before updates in smoke and phase4 callers; Phase5 guard bytes unchanged | Guard rejects both false and forged true training_eligible receipts | Original scientific reward still not released; isolated engineering pixel task is a separately declared experiment |
| Human UI/backend uncertainty enum mismatch | `src/evaluation/independent_review_v3.py` accepts/normalizes legacy ambiguous without modifying original exports; r2 UI emits uncertain; `scripts/ingest_semantic_review_v3.py` binds raw exports to exact image hashes | 88395 and88403; real empty input yields0 labels | Two real independent humans and third-human adjudication still required |
| Diagnostic PASS omitted initial/raw equality | `scripts/run_bf16_adjoints_20261102.py:strict_pair_acceptance` plus cross-mode starts; original strict criterion preserved | 88395 rechecks saved artifacts; prerequisite-negative tests PASS | Current bounded run passed, historical/live failures remain |
| Active LR versus initial contract / missing ordinary scheduler state | `src/rl/runtime_checkpoint_metadata.py` byte-binds active LR, LambdaLR bases and encoded outer state; future pilot saves/reloads it | 88395 verifies12 sidecars and outer state | The original GPU outer-scheduler object was not persisted; its historical sidecar is explicitly reconstructed from recorded scores/source |
| Query counts and matched-control acceptance were asserted from constants/subsets | Future pilot measures output records; saved observations independently counted. `matched_control_exact` requires starts/images/parameters/moments/RNG | 88395 measured4+4 per arm;88400 negative prerequisite tests | Same-LR live candidate control actually FAILS; no claim of superiority |
| Historical-corruption coverage was overstated | Removed unsupported COMPLETE coverage assertion; corrected receipt supersedes only that claim, original file preserved | 88395 records companions without pretending exact job-to-input lineage | Corrupt JSON remains negative evidence; exact old run-to-manifest binding unresolved |
| One global NO-GO was interpreted as all work blocked | Seven independent statuses, persistent existing-monitor/dashboard reporting | 88398 state/data embedding and idempotence tests PASS | Paper NO-GO and training guards remain; no second scheduler |

Legacy reward source is byte-identical `src/rl/reward_legacy_v1.py`, SHA256 `35b71b25358e8b2cdcc23723875cc47fe7bf1cd519d51158817684eca10425ad`. Original sources/reports are retained in this execution run’s `before/`. Original manuscript/objective and scientific coefficients were not edited.

## 2. Tests actually executed

- 27 tests PASS: `results/execution_remediation_20261102_v1/reward_regressions/unittest.stderr.log`.
- 7 tests PASS: `results/execution_remediation_20261102_v1/review_v3_regressions.log`.
- 17 tests PASS: `results/execution_remediation_20261102_v1/acceptance_recheck/unittest.log`.
- 2 tests PASS: `results/execution_remediation_20261102_v1/matched_lr_control/status_dashboard_regressions.log`.
- 3 tests PASS: `results/execution_remediation_20261102_v1/acceptance-final-88400.err`.
- 2 tests PASS: `results/execution_remediation_20261102_v1/human-ingest-88403.err`.

58 canonical test executions are not58 independent experiments. A Windows native OpenMP duplicate-runtime failure and one CPU batch submission rejected for a missing shebang are retained; neither started GPU work. The shebang was fixed before successful88403. No unsafe duplicate-OpenMP flag was used.

## 3. Numerical reliability

88386 used139 A100 seconds: fixed saved trajectories, coefficient1, ordinary/capture/capture/ordinary.384 paired adjoint boundaries, captured raw gradients, post-clip states, adapter/Adam/RNG all matched in that bounded replay. CPU88395 independently checked every acceptance prerequisite. This did not reproduce the old event and is not a numerical fix.

The new pretrained pilot supplies a separate negative control: scheduler and RSI candidate LR are both5e-6; starts and all10 generated training PNGs are exact; scalar losses and final RNG match. Yet31 candidate adapter tensors differ (max absolute4.1864041123e-8), and32 exp_avg plus32 exp_avg_sq tensors differ. Preclip norms0.1725901365/0.1725898683 are below the clip threshold1. This remains **NUMERICAL_MISMATCH_PRESERVED**. PNG equality does not establish equality of unrecorded internal trajectory/forward tensors. The first temporal CUDA primitive remains unknown. No bitwise threshold was relaxed.

## 4. Reward validity

Mechanical configuration/identity/field/logging fixes PASS;500 old rewards reproduce with error0. Original clipping/coefficient behavior remains: historical S=.5A on500/500,230/500 floor-saturated, no-op preference and same-model semantic dependence. No-op credit correction does not cure the construct. All v3 scientific receipts are diagnostic-only; Phase5 guard remains unchanged. The pilot below uses a separately named bounded pixel-brightness error-improvement proxy and is not a release/redefinition of the original scientific reward.

## 5. Data, image quality and semantics

160 preregistered candidate sources were acquired as26,879,223 bytes with official question/image joins, ten original human answers and normalized targets. Known-ID/hash overlaps0; pHash Hamming<=4 screening against1515 historical source images flags0, and within-candidate flags0. This is a screening result, not proof of global nonexposure or a released final test. Exact historical job lineage, statistical protocol/power and image-license admission remain incomplete. Original development cohorts stay development.

Verified LPIPS0.1 AlexNet uses the official calibrated linear weights (SHA256 df73285e35b22355a2df87cdb6b70b343713b667eddbda73e1977e0c860835c0), not classifier features alone; package and cached trunk were checked independently. On the existing32 geometry-aligned pairs: LPIPS mean0.154228289, SSIM mean0.765955211, exact no-op0/32. Identity LPIPS0 and exact calibration-layer loading passed. These are descriptive quality metrics, not semantic validity. Instruction success remains null; genuine human pairs0. The r2 package adds explicit visible-text preservation and requires all human fields before export/ingestion.

## 6. RL/RSI integration

A single pinned pretrained IP2Pix UNet is both generator and diffusion policy;256 shared LoRA tensors/1,594,368 elements are owned by the same optimizer. The backend remains DDPO_SF_v1, not GRPO. VQA question and editing instruction are distinct; the pixel pilot uses no Qwen/ASR. Bounded RSI memory, diagnosis, LR proposal, own-validation rejection and exact own-prefix rollback were actually executed. The broader constructed harness has existing provenance; this run does not relabel it a second generation network or prove scientific RSI benefit. Existing three-seed clean Qwen checkpoints/reload/scoring provenance were retained, not retrained.

## 7. Jobs and resource accounting

| Job | Actual UTC start–end | Terminal | GPU-seconds |
|---|---|---|---:|
| 88386 (poisoning-bf16-adjoints) | 2026-11-02T03:52:24–2026-11-02T03:54:43 | COMPLETED / 0:0 | 139 |
| 88387 (poisoning-reward-regressions) | 2026-11-02T03:54:35–2026-11-02T03:54:40 | COMPLETED / 0:0 | 0 |
| 88390 (poisoning-data-metric-fixes) | 2026-11-02T04:02:54–2026-11-02T04:03:30 | COMPLETED / 0:0 | 0 |
| 88392 (poisoning-pretrained-benign-pilot) | 2026-11-02T04:07:10–2026-11-02T04:11:12 | COMPLETED / 0:0 | 242 |
| 88395 (poisoning-remediation-recheck) | 2026-11-02T04:18:10–2026-11-02T04:18:18 | COMPLETED / 0:0 | 0 |
| 88398 (poisoning-matched-lr-control) | 2026-11-02T04:24:49–2026-11-02T04:24:52 | COMPLETED / 0:0 | 0 |
| 88400 (poisoning-acceptance-final) | 2026-11-02T04:30:15–2026-11-02T04:30:15 | COMPLETED / 0:0 | 0 |
| 88403 (poisoning-human-ingest-v3) | 2026-11-02T04:35:45–2026-11-02T04:35:46 | COMPLETED / 0:0 | 0 |

New GPU use381 seconds; validation subtotal2762/3600, reserved0, remaining838. The numerical diagnostic stayed under300 seconds; the separately preregistered pilot used242 of its360-second cap. Existing monitor 88402 is the unique successor, due 2026-11-02T06:32:35+00:00. No duplicate monitor was installed. Terminal source: `terminal_resource_receipt_v2.json`.

## 8. Updated pretrained benign comparison

Single training seed42;20 training sources,4 validation sources,4 historically exposed assessment sources. Two DDPO steps for trained arms include one physically shared prefix plus one branch update. Same exact prefix checkpoint and stochastic controls; per arm4 actual validation +4 assessment queries. Frozen is the0-update reference. Reward is reduction in absolute global-luminance target error, target1.03×source luminance; no semantic labels/ASR/distortion/cost objective.

| Arm | Attempted updates / retained step | Assessment proxy mean | Incremental UNet NFE | Incremental GPU-held seconds | Selection |
|---|---:|---:|---:|---:|---|
| frozen | 0 / 0 | -0.117785774292 | 48 | 10.966034 | frozen |
| ddpo_only | 2 / 2 | -0.118267059777 | 168 | 53.111324 | candidate retained |
| ordinary_scheduler | 2 / 2 | -0.118165582044 | 168 | 52.906212 | candidate retained |
| rsi | 2 / 1 | -0.118036655253 | 168 | 53.318524 | rejected; exact own-prefix rollback |

Shared prefix consumed168 UNet forward NFE once; total720. Physical optimizer calls4, not6; the prefix must not be charged once per arm. Branch GPU-held times are component timings, not independent allocated job costs or a speed benchmark. All trained branch proxy means are below Frozen. One seed/four assessment sources and the equal-LR numerical discrepancy prevent statistical, causal, semantic, generalization or RSI-superiority claims. Clean Accuracy and ASR remain NOT TESTED for this pilot. Intermediate/selected states and runtime sidecars are preserved.

## 9. Independent status and remaining blocks

| Dimension | Status | Scope |
|---|---|---|
| ENGINEERING_STATUS | PASS_SCOPED_FIXES_AND_EXECUTION | Reward wiring/strict fields/legacy compatibility; calibrated metrics; real pretrained updates; domain-specific guards remain intact |
| NUMERICAL_REPRODUCIBILITY_STATUS | FAIL_PRESERVED_LIVE_PILOT | Saved-trajectory O/C/C/O passed; equal-LR live candidates differ in31 adapter tensors,32 exp_avg and32 exp_avg_sq tensors. No confirmed BF16 fix |
| BENIGN_REWARD_VALIDITY_STATUS | IMPLEMENTATION_FIXED_CONSTRUCT_UNVALIDATED | YAML and identity accounting fixed; unchanged coefficients retain redundancy/saturation/no-op preference; no scientific reward release |
| DATASET_VALIDITY_STATUS | CONTENT_PASS_FINAL_ADMISSION_PENDING | 160 prospective sources acquired,0 known ID/hash/pHash flags; original labels/source bytes checked. Complete history lineage/power/final protocol remain pending; human labels0 |
| BENIGN_RL_STATUS | EXECUTED_DEVELOPMENT_ONLY_NO_OBSERVED_GAIN | One-seed pretrained4-arm pixel-brightness proxy; all trained branches below frozen on4 assessment sources; no generalization claim |
| RSI_CONTRIBUTION_STATUS | NOT_ESTABLISHED_NUMERICAL_CONFOUND | Own validation rejection and rollback executed; equal-LR candidate discrepancy prevents causal superiority attribution |
| MAIN_PAPER_EVIDENCE_STATUS | INCOMPLETE_NO_GO | No matched six-arm final evidence, no independent semantic labels, all real post-training ASR NOT TESTED |

Original88099/88105 failures and the new88398 numerical negative evidence remain. Independent human semantics cannot be manufactured. The original six-arm CVPR main experiment, modified-victim lineage and real post-training ASR remain absent. These do not block unrelated CPU/bookkeeping work.

## 10. Exact next validation

The next concrete validation is the existing32-pair independent human review. Use `independent_review32_v3_r2_A.zip` and `_B.zip` from this execution run, not the initial enum-incompatible packages. No model/assistant reviewer substitutes are permitted. Two genuine exports must exist before the following ingestion command can execute on real labels;88403 already verifies that empty inputs remain PENDING with0 accepted final pairs. Disagreement invokes a distinct human adjudicator via `--adjudication`, never automatic semantic resolution.

```powershell
python -B scripts/ingest_semantic_review_v3.py --kit results/execution_remediation_20261102_v1/semantic_review_v3_r2 --reviews results/execution_remediation_20261102_v1/semantic_review_v3_r2/submitted/reviewer_A.csv results/execution_remediation_20261102_v1/semantic_review_v3_r2/submitted/reviewer_B.csv --out results/semantic_review32_v3_real_v1
```

In parallel, a future numerical experiment must distinguish live B10 collection/trajectory context from saved-trajectory replay and capture first-down-block incoming adjoints versus leaf contributions. The current evidence cannot choose a CUDA fix. No additional GPU task is registered: the completed paired-adjoint diagnostic used139 of its300-second diagnostic ceiling, and the new equal-LR negative control makes immediate efficacy expansion uninformative. Do not repeat the same run until a random PASS. Remaining project budget838 A100 seconds is not authorization to relax a failed criterion.

Source-independence work can continue on CPU: bind old job86736 and failed job88029 to immutable executed-source/input-manifest witnesses. Matching one source ID or finding a current companion script is insufficient. Candidate160 content has been acquired/screened, but final test release still requires this registry closure, a frozen final statistical protocol/power decision and image-license/provenance review. The candidate pool has received0 target-model calls and0 training/feedback uses.
