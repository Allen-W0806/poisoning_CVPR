# Automated monitoring setup and actual verification

Enabled: **true**. Method: persistent CPU-only Slurm successor chain, every **7200 seconds**. Each short allocation queues one successor before its inspection. No login-node daemon or idle GPU. Actual start may be delayed by the queue.

Next eligibility: **2026-10-27T08:32:35+00:00 / 2026-10-27T04:32:35-04:00 America/New_York**. Sole future job **86320**, PENDING. First manual check 86309 started UTC 06:09:21, COMPLETED 0:0 in four CPU seconds. Corrected manual 86315 and periodic bootstraps 86317/86319 each completed 0:0 in two seconds. Each requested one CPU, 2G RAM, five-minute ceiling and no GPU. Pending 86318 was intentionally cancelled during native pause/resume validation with zero allocation time. A separate SSH connection confirms the resumed successor remains in Slurm.

Tracked **71 canonical experiments**, 980 files, and 85 source/edited associations. No active scientific experiment; no research submission or repair in actual idle checks. Fifteen observations consist of twelve historical failed parent/child records with later successful witnesses, one declared generator-RL blocker and two scientific-status observations. No new integrity failure or finalized artifact mutation detected.

## Scheduler and operations

Live probes: scrontab disabled, crontab absent, user systemd bus/lingering unavailable. Ordinary CPU Slurm submissions and successor creation from compute nodes were accepted. Project-specific name/comment ownership and separate nonblocking schedule/check locks prevent duplicates. Repeated native install retained the same next job; pause cancelled only that CPU monitor; resume restored one successor. Three consecutive failed/interrupted checks trip a persisted circuit breaker; skipped checks are not successful inspections. A controller outage/import failure before successor submission can break this chain; no high-availability watchdog is claimed.

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning && .venv/bin/python scripts/manage_monitor.py status'
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning && .venv/bin/python scripts/manage_monitor.py pause'
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning && .venv/bin/python scripts/manage_monitor.py resume'
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning && .venv/bin/python scripts/manage_monitor.py remove'
```

Remove disables recurrence and cancels only owned monitor jobs; evidence and source remain.

## Files and recovery boundaries

Entry points: scripts/monitor_experiments.py, scripts/diagnose_experiment.py EXP_ID, scripts/recover_experiment.py, scripts/manage_monitor.py. Configuration: configs/monitoring.yaml. Implementation: src/monitoring/. Rolling records: reports/MONITORING_LOG.md and MONITORING_STATUS.json. Timestamped records: reports/monitoring/CHECK_*.json/.md. Raw activation, lifecycle probes, scheduler_events.jsonl, Slurm logs and verification are alongside them.

Safe automatic repairs create active output directories, append missing registry records, and reconcile active terminal statuses. Without an original completion witness, COMPLETED_UNVERIFIED is used, never invented scientific PASS. New transient I/O/node/preemption failures may restart recognized same-source editor/supervised-training jobs: two total attempts per lineage, clean source at creation/currently, matching HEAD, same task/settings/reservation, and the existing 7200-GPU-second/one-GPU guard. Restarts are from initialization, not unsupported checkpoint resume. Temporary accounting outages preserve unattempted eligibility.

OOM/resource incompatibility, invalid checkpoints, duplicate IDs, nonfinite metrics, unproven code repairs and scientific issues are diagnosed/blocked. Generator microbatch is already one; there is no current validated resume CLI or registered deterministic source-repair rule. Permission flags do not authorize arbitrary code/resource changes. Old failures and pending jobs never trigger resubmission. Idle readiness does not load models on a GPU or unpickle/replay optimizer tensors; actual telemetry availability is explicit.

## Tests, provenance and preservation

**35 tests PASS** (21 monitoring scenarios plus 14 original tests), including all seven requested failure classes, concurrency, lineage, uncertain submission, between-check failure, source pixels/dimensions, provenance and accounting outage. See final_units.log. Two fresh same-family review rounds remain provisional; the last two fixes are executor-verified after the review budget. Review evidence remains under implement-stage/MONITORING_SWEEP_ROUND1.json, ROUND2.json and MONITORING_REVIEW_FOLLOWUP.md.

No scientific objective, reward, question/reference, data/split, frozen protocol or target revision changed. Held-out 300 was not reopened. Generator RL, independent semantic review and scientific advantage remain prior Phase 2 blockers. Scientific GPU allocation remains **1187 seconds**. Monitoring added **zero GPU seconds**; four completed checks used ten CPU allocation seconds in total. No new experiment was started to keep the scheduler busy.

Canonical source `7fb6a3b04b4501e22d1922c9d74355b0d49a634e`; 23 changed source files match the H: mirror after CRLF-to-LF normalization. Full scheduler/accounting receipt: activation_verification.json.

```text
JobIDRaw|State|ExitCode|ElapsedRaw|AllocTRES
86309|COMPLETED|0:0|4|billing=1,cpu=1,mem=2G,node=1
86315|COMPLETED|0:0|2|billing=1,cpu=1,mem=2G,node=1
86317|COMPLETED|0:0|2|billing=1,cpu=1,mem=2G,node=1
86318|CANCELLED by 100127|0:0|0|
86319|COMPLETED|0:0|2|billing=1,cpu=1,mem=2G,node=1
86320|PENDING|0:0|0|
```

The initial manual sbatch submission was rejected for CRLF before allocation. The rejected original script, activation_failures.jsonl and corrected LF script are retained. No failed experiment evidence was erased.

Sources: [UTC submission](https://utc.teamdynamix.com/TDClient/2717/Portal/KB/Article/163831/How-to-run-slurm-jobs-over-SSH), [CPU partition](https://utc.teamdynamix.com/TDClient/2717/Portal/KB/PrintArticle?ID=163830), [Slurm begin time](https://slurm.schedmd.com/sbatch.html), [scrontab availability](https://slurm.schedmd.com/scrontab.html). Live probes and actual allocations take precedence over older documentation snapshots.

Final housekeeping: run-specific staging was removed on both machines after checksum-verified report mirroring. Monitoring state, ownership locks, all logs, weights, environments and historical experiments remain. Retained local monitoring files total 1618018 bytes; no new multi-gigabyte artifact. Post-cleanup C: free 70934343680 bytes; H: free 235767754752 bytes. Both source trees clean. Native future CPU job 86320 remains enabled. See housekeeping_cluster.json and housekeeping_local.json.
