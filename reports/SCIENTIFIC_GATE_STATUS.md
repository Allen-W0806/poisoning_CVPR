# Scientific Gate Status

Project: **RSI-POISONING**. Canonical root: `/storage/home/ydk297/projects/poisoning`. Updated UTC: 2026-11-02T04:42:05.666593+00:00. Git base: `06bd889d9392c4d53621c3e4af15ad718e84f6c5`; working-tree fixes are archived and hash-bound.

| Dimension | Status | Scope |
|---|---|---|
| ENGINEERING_STATUS | PASS_SCOPED_FIXES_AND_EXECUTION | Reward wiring/strict fields/legacy compatibility; calibrated metrics; real pretrained updates; domain-specific guards remain intact |
| NUMERICAL_REPRODUCIBILITY_STATUS | FAIL_PRESERVED_LIVE_PILOT | Saved-trajectory O/C/C/O passed; equal-LR live candidates differ in31 adapter tensors,32 exp_avg and32 exp_avg_sq tensors. No confirmed BF16 fix |
| BENIGN_REWARD_VALIDITY_STATUS | IMPLEMENTATION_FIXED_CONSTRUCT_UNVALIDATED | YAML and identity accounting fixed; unchanged coefficients retain redundancy/saturation/no-op preference; no scientific reward release |
| DATASET_VALIDITY_STATUS | CONTENT_PASS_FINAL_ADMISSION_PENDING | 160 prospective sources acquired,0 known ID/hash/pHash flags; original labels/source bytes checked. Complete history lineage/power/final protocol remain pending; human labels0 |
| BENIGN_RL_STATUS | EXECUTED_DEVELOPMENT_ONLY_NO_OBSERVED_GAIN | One-seed pretrained4-arm pixel-brightness proxy; all trained branches below frozen on4 assessment sources; no generalization claim |
| RSI_CONTRIBUTION_STATUS | NOT_ESTABLISHED_NUMERICAL_CONFOUND | Own validation rejection and rollback executed; equal-LR candidate discrepancy prevents causal superiority attribution |
| MAIN_PAPER_EVIDENCE_STATUS | INCOMPLETE_NO_GO | No matched six-arm final evidence, no independent semantic labels, all real post-training ASR NOT TESTED |

Paper NO-GO is scoped to unsupported claims, not all engineering work. Original strict failures and the reward-release guard remain. See EXECUTION_REMEDIATION_LATEST.md for actual tests/jobs.

| Arm | Attempted updates / retained step | Assessment proxy mean | Incremental UNet NFE | Incremental GPU-held seconds | Selection |
|---|---:|---:|---:|---:|---|
| frozen | 0 / 0 | -0.117785774292 | 48 | 10.966034 | frozen |
| ddpo_only | 2 / 2 | -0.118267059777 | 168 | 53.111324 | candidate retained |
| ordinary_scheduler | 2 / 2 | -0.118165582044 | 168 | 52.906212 | candidate retained |
| rsi | 2 / 1 | -0.118036655253 | 168 | 53.318524 | rejected; exact own-prefix rollback |
