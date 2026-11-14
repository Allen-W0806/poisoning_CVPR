# Main Experiment Gap Matrix — Current

Project: **RSI-POISONING**. Canonical root: `/storage/home/ydk297/projects/poisoning`. Updated UTC: 2026-11-02T04:42:05.666593+00:00. Git base: `06bd889d9392c4d53621c3e4af15ad718e84f6c5`; working-tree fixes are archived and hash-bound.

**Complete matched CVPR main experiment NOT COMPLETED. All real post-training ASR = NOT TESTED.** The new pretrained four-arm run is development-only engineering evidence, not a completed security main experiment.

| Proposed main arm | Actual data/training/model | Valid current evidence | Clean Accuracy | ASR | Remaining main evidence |
|---|---|---|---|---|---|
| Clean-only | Existing original VQAv2 clean Qwen LoRA3 seeds;16 steps/17 calls incl replay; original adapters retained | Response-only supervised/reload/scoring engineering | Latest exposed32-source scores78.4375/80.625/80.625%; not matched main test | NOT TESTED | Untouched matched final protocol and cohorts |
| Fixed Editing | Existing frozen IP2Pix/Qwen diagnostics; new0-update brightness reference uses same pretrained initial state | Frozen inference/quality and new4-source pixel proxy | NOT TESTED as main trained arm | NOT TESTED | Independent edited-label validity and trained-victim bundle |
| RL-only | Historical2-seed8-step IP2Pix DDPO; new20-source2-step attempted engineering fork, selected step2 | Real LoRA updates and negative development proxy result | NOT TESTED | NOT TESTED | Numerical reliability, validated original reward, independent assessment, main victim lineage |
| RL+RSI | Historical IP2Pix RSI; new shared-prefix LR candidate rejected, selected step1 | Memory/proposal/validation/rollback executed | NOT TESTED | NOT TESTED | Equal-LR candidates numerically differ; contribution not identified; main validity gaps persist |
| BadSem | No locally reproduced training/model/result | Published protocol context only | NOT TESTED | NOT TESTED | No admitted fixed-model reproduction bundle/observations/denominator |
| CBV | No locally reproduced training/model/result | Published protocol context only | NOT TESTED | NOT TESTED | Same admission gap; no external numbers counted as local measurements |

## New pretrained benign engineering result

| Arm | Attempted updates / retained step | Assessment proxy mean | Incremental UNet NFE | Incremental GPU-held seconds | Selection |
|---|---:|---:|---:|---:|---|
| frozen | 0 / 0 | -0.117785774292 | 48 | 10.966034 | frozen |
| ddpo_only | 2 / 2 | -0.118267059777 | 168 | 53.111324 | candidate retained |
| ordinary_scheduler | 2 / 2 | -0.118165582044 | 168 | 52.906212 | candidate retained |
| rsi | 2 / 1 | -0.118036655253 | 168 | 53.318524 | rejected; exact own-prefix rollback |

Same-LR numerical control FAIL:31 adapter,32 exp_avg and32 exp_avg_sq differences. No efficacy ranking. New model paths: `results/execution_remediation_20261102_v1_pretrained_pilot/{frozen,ddpo_only,ordinary_scheduler,rsi}/selected/training_state.pt`, each paired with byte-bound `active_runtime_config.json` and original pinned IP2Pix base. Detailed new hashes: `current_model_lineage.json`; original full checkpoint inventory remains `results/main_gap_audit_20261102_v1/checkpoint_inventory_final.{json,csv}`.

## Evaluation admission

Historical cohorts remain development-only. New160 originals: bytes/official labels checked,0 known exact/ID/pHash flags;0 target-model calls or feedback uses. No final-test release until job/source lineage, frozen statistical protocol and image provenance are resolved. Human semantics0; r2 review packages ready. Automated image similarity is not edited-answer validity.

## Scope

Original C0–C5/Tiny statistics were not regenerated. No full clean Qwen retraining, no poisoning/victim training or ASR optimization occurred. Scientific coefficients, manuscripts and strict guard/bitwise criteria are preserved.

| Dimension | Status | Scope |
|---|---|---|
| ENGINEERING_STATUS | PASS_SCOPED_FIXES_AND_EXECUTION | Reward wiring/strict fields/legacy compatibility; calibrated metrics; real pretrained updates; domain-specific guards remain intact |
| NUMERICAL_REPRODUCIBILITY_STATUS | FAIL_PRESERVED_LIVE_PILOT | Saved-trajectory O/C/C/O passed; equal-LR live candidates differ in31 adapter tensors,32 exp_avg and32 exp_avg_sq tensors. No confirmed BF16 fix |
| BENIGN_REWARD_VALIDITY_STATUS | IMPLEMENTATION_FIXED_CONSTRUCT_UNVALIDATED | YAML and identity accounting fixed; unchanged coefficients retain redundancy/saturation/no-op preference; no scientific reward release |
| DATASET_VALIDITY_STATUS | CONTENT_PASS_FINAL_ADMISSION_PENDING | 160 prospective sources acquired,0 known ID/hash/pHash flags; original labels/source bytes checked. Complete history lineage/power/final protocol remain pending; human labels0 |
| BENIGN_RL_STATUS | EXECUTED_DEVELOPMENT_ONLY_NO_OBSERVED_GAIN | One-seed pretrained4-arm pixel-brightness proxy; all trained branches below frozen on4 assessment sources; no generalization claim |
| RSI_CONTRIBUTION_STATUS | NOT_ESTABLISHED_NUMERICAL_CONFOUND | Own validation rejection and rollback executed; equal-LR candidate discrepancy prevents causal superiority attribution |
| MAIN_PAPER_EVIDENCE_STATUS | INCOMPLETE_NO_GO | No matched six-arm final evidence, no independent semantic labels, all real post-training ASR NOT TESTED |
