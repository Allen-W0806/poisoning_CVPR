# Live monitoring and execution evidence

Run `replication_monitor_20261029_v1`. Latest successful stage inspection **2026-10-29T01:41:07.260100-04:00** (America/New_York). Monitor PASS means inspection completed; it does not mean every experiment passed.

Initial live inspection found a genuine existing two-hour Slurm chain, pending86836, latest legacy heartbeat2026-10-29T04:33:01Z, but its123-entry legacy registry did not cover the new result directories or Latest dashboard. Earlier clean baseline jobs86804–86807 actually executed; the next replication/diagnostic recommendations had not yet been submitted. The new coverage and experiments were executed this stage.

`crontab -l`: command not found (127). `systemctl --user list-timers --all`: Failed to connect to bus: No medium found (1). `scrontab -l`: disabled on this cluster. These errors are preserved in [activation probes](../results/replication_monitor_20261029_v1/audit/activation_probe.json). The permitted native mechanism is a short CPU-only Slurm successor chain; no permanent login-node process, sleeping GPU or dependence on an open chat.

Installation reuses `scripts/manage_monitor.py` / `src.monitoring.scheduler.Scheduler`, extends its callback with `scripts/rsi_stage_monitor.py`, and registers the approved stage in `reports/rsi_stage_monitor/active_stage.json`. Native install/status confirms enabled=true, interval7200seconds, methodslurm_cpu_successor. Registration entry: `reports/monitoring/monitor_job.sbatch`; owned comment prefix `poisoning-monitor-v1:f916f24e0de6`; future job **86836**; next eligible **2026-10-29T02:32:35-04:00**, subject to queue delay. Do not interpret eligibility as a guaranteed start.

Actual registered entry-point verification job **86850 COMPLETED**,4 CPU-seconds, invoked the native tick inside an owned Slurm allocation, executed both old and new inspections, and retained the future successor. Completion CPU jobs86847/86849/86852 advanced all three seeds into P2 without chat/terminal supervision. Separate nonblocking scheduler and extension locks prevent overlap. Each experiment also has an afterany CPU check; this is an event hook, not a second recurring chain.

| Task | Attempt | Slurm job | Scheduler state | Numerical checks | GPU seconds |
|---|---:|---|---|---|---:|
| P1_seed42 | 0 | 86846 | COMPLETED | PASS | 79 |
| P1_seed123 | 0 | 86848 | COMPLETED | PASS | 79 |
| P1_seed3407 | 0 | 86851 | COMPLETED | PASS | 78 |
| P2_IP2Pix | 0 | 86853 | FAILED | FAIL | 53 |
| P2_IP2Pix | 1 | 86856 | COMPLETED | PASS | 53 |


The monitor inspects sacct states, pending/stalled progress, training JSONL finite values, numerical assertions, checkpoint/output hash inventories and expected step/evaluation counts. Known recoverable node/preemption/transient-I/O failures may retry from initialization, with unique run IDs, unchanged science and at most two additional attempts per task; budget checks include active reservations. A persisted uncertain submission blocks duplicate submission. Unexplained numerical failures are retained and blocked, not repeatedly seeded until PASS. Failure tests cover failed/node/pending/stalled/completed states, overlapping locks, checkpoint tampering and numerical assertions:9/9 PASS, zero real jobs disrupted. The instrumented P2 follow-up adds read-only measurements and preserves attempt0.

Heartbeat: `reports/rsi_stage_monitor/heartbeat.json`. Latest log: `reports/rsi_stage_monitor/check_20261029T054107_260286Z.json`. Periodic scheduler state: `reports/monitoring/scheduler.json`; periodic verification log: `reports/rsi_stage_monitor/periodic-entry-86850.out`. Dashboard: `dashboard/RSI_Experimental_Progress_Latest.html`, updated after every extension pass. All findings and failed states remain visible. No claim of high availability: the existing scheduler trips a circuit breaker after three failed/interrupted checks, and an import/controller failure before successor submission can interrupt recurrence.

This stage allocated **342 A100-seconds**. Including earlier gradient198 + matched584 + clean220 gives **1344/3600 seconds**, leaving **2256 seconds**. No active reservation remains when all listed jobs are terminal. Other account activity such as WANOP86841 belongs to `RSI_Video_Branching_20261027_v1`, not this approved validation scope; it was not modified. Dollar billing unavailable. [Raw accounting](../results/replication_monitor_20261029_v1/sacct_final.txt), [budget ledger](../results/replication_monitor_20261029_v1/resource_accounting.json).

Review handoff engineering PASS, human semantic review BLOCKED:160 cases/320 hashes checked, no genuine independent records. See [handoff](INDEPENDENT_REVIEW_HANDOFF.md). Generator reward-release guards remain active. No backdoor training, poisoning-policy optimization or ASR-driven optimization.
