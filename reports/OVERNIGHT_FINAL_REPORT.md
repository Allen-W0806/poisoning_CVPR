# Overnight execution status

PROJECT_ID: RSI-POISONING. Run `overnight_benign_20261102_v2`. Updated UTC 2026-11-02T06:53:20.435487+00:00.

Status: **STOPPED_NUMERICAL_OR_EXECUTION_FAILURE**. No confirmed BF16 fix. Learning eligible: **False**.160 prospective final-test sources stay sealed. Scientific reward guards remain intact; ASR NOT TESTED.

NEW ledger only: cap36000 A100 seconds; consumed172;reserved0;remaining35828. Initialization2026-11-02T06:17:57.636501+00:00;deadline2026-11-02T16:17:57.636501+00:00. One GPU at a time; numerical sub-budget1200 seconds. Previous3600-second validation ledger is separate.

| GPU job | Stage | State | Actual GPU seconds | Classification | Exit |
|---|---|---|---:|---|---|
|88433|numerical|COMPLETED|172|NUMERICAL_MISMATCH_PRESERVED|0:0|

Registered CPU controller jobs: [{"job_id": "88431", "purpose": "one-shot wall deadline finalization", "parent_GPU_job": null}, {"job_id": "88432", "purpose": "CPU tests, dry run, stage admission", "parent_GPU_job": null}, {"job_id": "88434", "purpose": "afterany completion", "parent_GPU_job": "88433"}]. Completion handlers use afterany, persistent state, lock, durable held submission, no polling. Existing two-hour monitor is preserved. Source directory: `reports/overnight_benign_20261102_v2`;evidence: `results/overnight_benign_20261102_v2`.

Passed scoped stages: []. Failures: [{"job_id": "88433", "reason": "NUMERICAL_MISMATCH_PRESERVED", "first_boundary": {"pair": "fresh0__fresh1", "boundary": "raw"}, "UTC": "2026-11-02T06:21:12.704430+00:00"}]. Numerical comparisons and runtime exits are retained per-job. No automatic numerical retry or hyperparameter search. NODE_FAIL permits at most one recorded excluded-node infrastructure repair within budget.

4 historical model arms have new CPU-only quality metrics when present. Training job88392 is historical and is not charged to this overnight budget. They are NOT newly trained overnight arms or final-test evidence. Arm-specific allocated GPU cost is null because the historical242-second allocation was shared; branch held time and NFE are separate fields. Frozen/DDPO/scheduler/RSI assessment uses the same4 historically exposed sources; uncertainty is exploratory and does not establish superiority.

New efficacy comparisons are blocked by numerical failure when present, unreleased original scientific reward and independent development evaluation not admitted. No source from the sealed160 is used. If no eligible stage remains, stop GPU submissions rather than fill budget. Real human labels remain0.

Cloud publication is separately receipt-verified by the connected Drive workflow; report generation alone is not a cloud synchronization claim.

## Next falsifiable work

If raw gradients differ, localize backward accumulation/upstream adjoints; if raw match but norm differs, inspect norm reduction dtype/foreach/order; if norm matches but postclip differs, inspect clipping; if postclip matches but Adam differs, inspect optimizer step ordering. No repair is claimed without ordinary uninstrumented and resumed next-update checks.

## Final executed checks

Latest26 targeted CPU tests PASS (capture nonmutation/clip equivalence, real persisted callback dry runs, budget/deadline guards, one-shot accounting recovery, bounded NODE_FAIL policy, conditional preclip-norm acceptance and explicit rejection of empty update receipts, and failed-artifact classification). A real repeated callback for88433 returned IDEMPOTENT_ALREADY_PROCESSED and created no jobs. Install re-entry returned ALREADY_INSTALLED. The original monitor recognizes88433 as a separately authorized allocation and currently accounts2899/3600 in its own ledger; it has not merged the new172 seconds.

No production BF16 fix. Raw gradients differ before clipping, so ordinary/resume and learned-efficacy stages did not run. Selected RSI historical checkpoint retained10 training sources after its20-source candidate was rejected; attempted cost is preserved. New CPU model quality measurements do not count as new overnight training.

New tensor evidence retained: 218960454 bytes on cluster;no activation dumps or large C-drive writes. Original outputs and failed receipts are kept. CPU staging path error was corrected using the absolute canonical interpreter before deployment;GPU cost0. Only own unused deadline finalizer was canceled;the existing two-hour chain remains. Kernel-serialized completion callbacks have one delayed accounting recovery plus the existing two-hour fallback; unresolved scheduler submissions remain fail-closed and are not blindly retried. NODE_FAIL alone permits one identified/excluded-node infrastructure retry; none occurred in this run. Cloud verification is recorded separately after text-only upload.

## Text publication and table verification

An observed periodic desktop publisher was still copying the old42f approved index and could replace newer connector-published links. The scoped publisher correction accepts only a separately project-bound, remotely byte-verified index with exactly six report names, all new SHA256 values and all legacy history hashes. It creates a distinct immutable publication version while leaving the old approved report package unchanged.14 affected publisher safety tests PASS, including index persistence on repeated copy, tamper rejection and allowlist rejection. Connector delivery/readback receipts separately establish actual cloud publication.

CSV and booktabs LaTeX table source are provided. No TeX engine is available in the checked cluster/local PATH, so rendered typesetting is NOT TESTED; no compiled-PDF/layout PASS is claimed. No original scientific objective, reward guard or production optimizer was changed.
