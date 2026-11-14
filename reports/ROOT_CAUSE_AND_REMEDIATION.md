# Root cause and remediation

Snapshot 2026-11-01T01:28:15.514837+00:00. **NO-GO.** All23 historical findings reconciled. PASS is restricted to the named new engineering/documentation scope; it never rewrites a failed historical result.

Historical reference: reports/AUDIT_DECISION_MATRIX.md, SHA256 64f9814fdc08c78518c5020d1fb1d9cc276c2a7939d7b09f5da206cfa6d67c88. New audit data: results/audit_remediation_20261101T0120Z_v1/cpu_audit/. October31 reports and previous failures remain intact.

## F01 — Claim/estimand mismatch — PASS

**Evidence:** Historical matrix section F01; `docs/CURRENT_METHOD_20261101.md` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** The historical work measures benign preservation, not BadSem.
**Fix attempted:** Explicit current-method and preregistered benign synthetic scope; no objective change.
**Verification:** Protocol names cosine engineering task and no VLM/security reward.
**Remaining risk:** No BadSem reproduction or security efficacy has been established.

## F02 — Same-model semantic verification — BLOCKED

**Evidence:** Historical matrix section F02; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/review32_status.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** The reward provider is also the semantic proxy.
**Fix attempted:** Existing random32 kit reused and hashes checked; real dual-review ingestion inspected.
**Verification:** 0 genuine reviewed pairs; no human/model error rate available.
**Remaining risk:** Independent reviewer agreement and semantic validity remain unknown.

## F03 — Reward redundancy and floor censoring — FAIL

**Evidence:** Historical matrix section F03; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/feedback_distribution.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** S and A encode one gate; U/C are constant and clipping suppresses variation.
**Fix attempted:** Read-only reanalysis of500 observations; no coefficient edits.
**Verification:** 500/500 S=.5A;230/500 floor; centered component rank2.
**Remaining risk:** Feedback construct is not validated for scientific learning.

## F04 — Instruction success absent — BLOCKED

**Evidence:** Historical matrix section F04; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/identity_and_config_audit.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Preservation can be maximized by doing no edit.
**Fix attempted:** No-op controls, identity caller audit and independent instruction rubric retained.
**Verification:** C0/C1 scores identical for160; identity exceeds all historical changed rewards.
**Remaining risk:** No human-confirmed lighting success rate; no new objective introduced.

## F05 — RSI versus ordinary scheduling — BLOCKED

**Evidence:** Historical matrix section F05; `results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/threeway/results.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Historical RSI is a learning-rate schedule extension.
**Fix attempted:** Added fixed, ordinary ReduceLROnPlateau and RSIController branches with equal queries.
**Verification:** New comparison PASS; modest LR/rollback mechanism explicitly described.
**Remaining risk:** Providing a scheduler baseline does not prove novel RSI or autonomous superiority.

## F06 — Preadaptation causal confounding — PASS

**Evidence:** Historical matrix section F06; `results/audit_remediation_20261101T0120Z_v1/audit/preadaptation_disk_verification.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Historical arms differ before LR intervention.
**Fix attempted:** New identical independently replayed prefixes; strict full-state, gradient and prediction checks before adaptation.
**Verification:** Disk initial/prefix comparisons: 12; all equal=True.
**Remaining risk:** Only the new FP32 tiny-model comparison is repaired; historical BF16 causal claims remain invalid.

## F07 — Reused data and weak replication — BLOCKED

**Evidence:** Historical matrix section F07; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Old development images were repeatedly used for selection.
**Fix attempted:** Audited latest80 and three clean seeds; no new natural-image test opened.
**Verification:** Newest32 assessment sources:4 prior inference,10 in old opened-test manifest.
**Remaining risk:** No real-image generalization claim; three seeds cannot substitute independent data.

## F08 — Exact-content split leakage — PASS

**Evidence:** Historical matrix section F08; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Historical v1 train/test has one exact duplicate under different IDs.
**Fix attempted:** Preserved v1; independently checked new80 cross-split IDs and SHA.
**Verification:** Zero ID/hash overlaps within latest32/16/32; all80 file hashes valid.
**Remaining risk:** PASS for new internal split only, not historical v1 or exposure independence.

## F09 — Pretrained stochastic continuation — FAIL

**Evidence:** Historical matrix section F09; `results/audit_remediation_20261101T0120Z_v1_BF16_ordinary/diagnostic/checks.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Earlier exact backward/optimizer continuation failed.
**Fix attempted:** One preregistered ordinary+instrumented BF16 pair this run, with unchanged bitwise checks.
**Verification:** Both new paths PASS; earlier ordinary and instrumented failures preserved.
**Remaining risk:** Intermittent path-dependent history is unresolved; batch10 continuation is unaccepted.

## F10 — Question versus global preservation — BLOCKED

**Evidence:** Historical matrix section F10; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/review32_status.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** A question-specific answer proxy does not certify all objects/colors/counts.
**Fix attempted:** Blinded review covers global attributes, answer validity and instruction compliance.
**Verification:** Forms prepared; zero human labels.
**Remaining risk:** Rubric availability is not semantic remediation.

## F11 — Step proxy versus measured compute — PASS

**Evidence:** Historical matrix section F11; `results/audit_remediation_20261101T0120Z_v1/final_status.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Legacy C=10/30 excludes target/model/diagnostic overhead.
**Fix attempted:** Retain C unchanged and label it a step proxy; measure Slurm totals and synchronized one-GPU component time.
**Verification:** Both GPU attempts charged; per-branch actual GPU-held seconds retained.
**Remaining risk:** No claim that old C is measured cost or has become a validated cost objective.

## F12 — Identity loses consistency credit — FAIL

**Evidence:** Historical matrix section F12; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/identity_and_config_audit.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Caller gates A solely on provisionally_valid; verified identity lacks that key.
**Fix attempted:** Executed a clearly synthetic identity fixture; objective left unchanged.
**Verification:** Caller gives A0 and reward.9666667 versus1.4666667 with declared credit.
**Remaining risk:** Bug still present in legacy caller; no historical reward overwritten.

## F13 — Exposure versus realized compute — PASS

**Evidence:** Historical matrix section F13; `results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/threeway/results.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Matched exposure does not imply equal runtime.
**Fix attempted:** Equal update/NFE/query budgets; report actual time rather than claim runtime equivalence.
**Verification:** Each branch4 updates,320 forward NFEs,48 validation+32 test queries.
**Remaining risk:** Measured times differ; sequential same-node timings are not a latency benchmark.

## F14 — Suspected editing/padding failures — BLOCKED

**Evidence:** Historical matrix section F14; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/preprocessing_aligned_quality.csv` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Padding and content changes confound legacy pixel quality.
**Fix attempted:** Reconstructed all160 geometry controls exactly and retained full-frame metrics plus review rubric.
**Verification:** Geometry reconstruction160/160; aligned SSIM mean.7340322949; LPIPS absent.
**Remaining risk:** No human-confirmed edit/instruction failure rates.

## F15 — Active-contract reconstruction — BLOCKED

**Evidence:** Historical matrix section F15; `results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/threeway/checks.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Production revisions require matching current contract/config to resume.
**Fix attempted:** New tiny branches save initial/prefix/candidate/selected and check retained LR against optimizer.
**Verification:** New config/optimizer and rollback checks PASS.
**Remaining risk:** General production pretrained resume across a historical revision remains unvalidated.

## F16 — Obsolete method entry — PASS

**Evidence:** Historical matrix section F16; `docs/CURRENT_METHOD_20261101.md` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** METHOD.md described only the early PIL controller.
**Fix attempted:** Added dated current-method entry and explicit historical banner; archived original body.
**Verification:** Original body byte text preserved after removing banner; code paths checked.
**Remaining risk:** Documentation does not certify scientific benefit.

## F17 — Backward primitive not localized — BLOCKED

**Evidence:** Historical matrix section F17; `results/next_validation_20261101T0040Z_v1_BF16_instrumented/diagnostic/raw_backward_localization.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** First observed mismatch is backward, not restored forward/RNG.
**Fix attempted:** Preserved all historical first divergences and collected a new fixed paired diagnostic.
**Verification:** Previous instrumented raw error2.834317e-5; new pair exact; no stable path effect.
**Remaining risk:** No specific CUDA primitive or unique causal mechanism identified.

## F18 — Independent validity unknown — BLOCKED

**Evidence:** Historical matrix section F18; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/review32_status.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** No genuine dual-review evidence exists.
**Fix attempted:** Reused predeclared32 random subset, separate reviewer/adjudication files and validator.
**Verification:** 0 reviewed pairs; disagreement and validity rates remain unknown.
**Remaining risk:** Human-dependent task cannot be replaced by model flags.

## F19 — Reward config/default mismatch — FAIL

**Evidence:** Historical matrix section F19; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/identity_and_config_audit.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Dataclass defaults, not YAML, supply executed coefficients.
**Fix attempted:** Audited actual call and recorded defaults without rewiring reward.
**Verification:** Values match historical campaign; wiring gap persists.
**Remaining risk:** Editing YAML alone would not change coefficients; future version needs explicit contract.

## F20 — Runtime data guards weaker — FAIL

**Evidence:** Historical matrix section F20; `src/data/dataset.py` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Generic loader lacks per-read image SHA and official source-join enforcement.
**Fix attempted:** Run-specific source/content audits performed; no generic loader repair claimed.
**Verification:** All80 latest image hashes PASS; historical official join receipt retained.
**Remaining risk:** Audit success does not make runtime loader a corruption detector.

## F21 — Near-duplicates/pretraining exposure — BLOCKED

**Evidence:** Historical matrix section F21; `results/audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Content hashes cannot certify crop/re-encode or foundation-model nonexposure.
**Fix attempted:** Retained dHash screen and separated known inference/manifest exposure witnesses.
**Verification:** All80 prior-exposure flags true; latest split only source-disjoint.
**Remaining risk:** Foundation-model contamination and unrecorded prior exposure unknown.

## F22 — Real post-training scientific/security evidence absent — BLOCKED

**Evidence:** Historical matrix section F22; `reports/AUDIT_DECISION_MATRIX.md` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Synthetic aggregation and clean LoRA are not a controlled backdoor assessment.
**Fix attempted:** No security/ASR optimization or new post-training security campaign.
**Verification:** Absent denominators remain missing; scientific gate NO-GO.
**Remaining risk:** Natural-image efficacy and training-induced security behavior untested.

## F23 — Within-batch duplicate freshness gap — PASS

**Evidence:** Historical matrix section F23; `results/audit_remediation_20261101T0120Z_v1/audit/F23_after.json` (input hashes in cpu_audit/input_hashes.json).
**Root cause:** Previous IDs were added to consumed only after a complete update.
**Fix attempted:** Added non-mutating per-batch ID validation before model/gradient work; archived old source.
**Verification:** Prepatch duplicate accepted;5 regression tests PASS including real trainer early rejection.
**Remaining risk:** Distinct trajectories for the same condition remain explicitly permitted; no historical duplicate was inferred.

## Preserved engineering failure and repair

88029 completed its wrapper but the first three-way child failed while serializing the ordinary scheduler state: ReduceLROnPlateau deliberately stores mode_worse=-inf. No finite-gradient assertion failed. The failed partial JSON, checkpoints, logs and v1 source remain. Version2 encodes only this documented comparison sentinel as a string; strict allow_nan=False and all gradient/reward/state equality checks remain unchanged. Only the three-way test was rerun in a unique repair run; BF16 tests were not repeated. The earlier submission dict/list TypeError occurred before GPU allocation and is retained.

## No scientific objective change

Legacy reward coefficients/caller remain unchanged. Synthetic objective is the same terminal cosine as the prior tiny-UNet test. Qwen was not retrained or newly inferred in this stage. The generator reward-release guard hash is unchanged. Historical-source archives retain the pre-F23 code; new code and test hashes are versioned. Human labels were neither created nor contacted.
