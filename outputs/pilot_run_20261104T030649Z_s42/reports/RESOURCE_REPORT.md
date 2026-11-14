# Resource settlement

Actual allocated GPU time: **3859/5400 seconds (1.071944 GPU hours)**.

Failed/time-capped allocations and numerical-witness updates are included. CPU-only audits consume zero GPU seconds. The original historical budgets and unrelated projects are not charged or modified.

| Job | State | Wall seconds | GPUs | GPU seconds | Exit |
|---|---|---:|---:|---:|---|
| 88916 | FAILED | 3312 | 1 | 3312 | 1:0 |
| 88919 | COMPLETED | 38 | 0 | 0 | 0:0 |
| 88930 | FAILED | 547 | 1 | 547 | 1:0 |
| 88932 | COMPLETED | 114 | 0 | 0 | 0:0 |

Actual arm training updates: `{"frozen": 0, "fixed": 0, "ddpo_only": 100, "rule_rsi": 100, "rl_rsi": 100}`. Physical generator optimizer calls including separate numerical witnesses: 307.

Large checkpoint/gradient/image files are retained as experiment evidence on cluster storage. The artifact inventory records exact bytes and SHA256. Local delivery contains reports, figures, logs and source, while model weights and full checkpoint evidence remain in the canonical cluster directory.
