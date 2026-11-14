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
