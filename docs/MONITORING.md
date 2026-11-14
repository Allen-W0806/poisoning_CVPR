# Persistent cluster monitoring

Phase3 extension: the same owned Slurm successor chain now reads DDPO-SF likelihood replay, conditioning, actual adapter changes, gradients/reward variance, complete restart witnesses and per-update peak CUDA allocations. Separate `phase3_jobs.jsonl`/`submit_phase3.py` enforce an additional7200 GPU-second ceiling. A new append-only `phase3_protected_hashes_v1` baseline covers new scientific files; existing Phase2 expected hashes are unchanged. Initialization occurs only after authorized Phase3 implementation. No second scheduler is installed. An idle monitor never loads models or tensor state.

Phase3 transient restart routing uses its own budget guard and original clean-source/configuration gates with the same durable two-attempt lineage cap. This is a fresh identical-config restart, not resume. Numerical/probability/semantic failures are diagnosed for review and are never automatically repaired by changing reward, scheduler, precision, splits or scientific criteria. Only runtime-validated checkpoint resume would be permitted; the monitor currently does not invoke a resume CLI.

Canonical root: `/storage/home/ydk297/projects/poisoning`. The monitor uses short CPU-only Slurm jobs on `epyc-cpu`, account `allusers`, one CPU, 2G RAM, five-minute hard limit. Every job queues one successor at its scheduled UTC slot plus 7200 seconds **before** inspecting experiments. A late job skips missed slots rather than catching up in a burst. Eligibility is every two hours; Slurm queue delay may postpone execution. It survives SSH/Codex exit and occupies no GPU or waiting process.

Cluster probes showed `scrontab: fatal: scrontab is disabled on this cluster`, no crontab executable, and no user systemd bus/lingering. No system service or login-node daemon was installed. Ordinary CPU Slurm submission is the supported fallback; actual installation must be verified in MONITORING_SETUP_REPORT.md. See [UTC submission guidance](https://utc.teamdynamix.com/TDClient/2717/Portal/KB/Article/163831/How-to-run-slurm-jobs-over-SSH), [CPU partition guidance](https://utc.teamdynamix.com/TDClient/2717/Portal/KB/PrintArticle?ID=163830), and [Slurm begin-time semantics](https://slurm.schedmd.com/sbatch.html).

## Commands

Run from the canonical root using `.venv/bin/python`:

```bash
python scripts/manage_monitor.py install
python scripts/manage_monitor.py status
python scripts/monitor_experiments.py
python scripts/diagnose_experiment.py EXP_ID
python scripts/recover_experiment.py
python scripts/manage_monitor.py pause
python scripts/manage_monitor.py resume
python scripts/manage_monitor.py remove
```

Install/resume is idempotent. Pause disables recurrence and cancels only jobs with this exact project-derived monitor name/comment. Remove also clears the installation flag; evidence and scripts remain. Reports are CHECK timestamped JSON/Markdown plus rolling MONITORING_LOG.md and MONITORING_STATUS.json. scheduler.json and scheduler_events.jsonl retain ownership, next eligibility/job, failures and submission receipts. Separate nonblocking locks protect scheduling and checks. Uncertain submission blocks duplicate installation; three consecutive failed checks stop recurrence.

## Diagnosis and recovery

The monitor reads real queue/accounting/resource fields, current progress and gradients from events, metrics, checkpoint age, structural/checksum integrity, image dimensions, source/output associations, model revisions, dataset split hashes, prediction counts/questions/references, scientific source hashes, RSI decisions and disk/source status. Existing failed attempts remain visible with later same-kind success witnesses. Exact identity/reload PNG duplicates are expected and never deleted.

Safe automatic repairs create missing output directories **only for active experiments**, append missing registry records supported by existing manifests, and reconcile stale active statuses with terminal Slurm evidence. A completed allocation without original successful-finalization evidence becomes COMPLETED_UNVERIFIED, never an invented scientific PASS. Finalized measurements remain immutable; effective state changes are appended to experiments/state_observations.jsonl.

Only newly observed transient I/O, node-failure or preemption cases can restart a recognized current-source editor/supervised-training job. New failures/completions between checks are detected from creation time. Maximum two total attempts follow the original lineage across new IDs and failure categories. Clean source at original creation, matching HEAD and a currently clean worktree are required; legacy provenance cannot trigger restarts. The original reservation/configuration is retained and the existing 7200-GPU-second/one-GPU guard applies. This is restart from initialization, **not checkpoint resume**. Old failed runs and pending jobs never trigger retries. An idle check starts no research.

Resource and code-repair flags authorize only supported, verified compatible fixes. There is currently no registered deterministic source-repair rule, no current generator resume CLI, and generator microbatch is already one. OOM, incompatible state, CUDA/dependency issues or unsupported resource changes are diagnosed and blocked for a tested repair, rather than silently changing resolution/batch/objective. Automatic arbitrary source generation, pip installation into shared environments and method substitution are absent.

## Scientific and observation limits

No reward/dataset/held-out/prompt/scoring changes, no attacks or security optimization, no extra experiments to keep the scheduler busy. The held-out 300 remain once-only. Generator LoRA supervision and frozen-generator controller RL are distinct. No gain or noisy reward prompts automatic scientific tuning.

Idle readiness checks cache files, configurations and prior actual execution witnesses. It does not freshly load Qwen/editor on a GPU or replay optimizer tensors every two hours. RAM/CPU telemetry is requested via sstat only for active project jobs; live GPU utilization is N/A without experiment telemetry. Checkpoints are hashed and structurally verified without unpickling model data. Raw tensor finiteness uses existing runtime witnesses, not a new numerical audit. First activation hashes cannot detect pre-activation corruption without earlier trusted hashes.

A successor protects against an ordinary check crash after arming; a cluster outage before successor submission or broken scheduler-entry imports can break a user-job chain. Inspect `manage_monitor.py status` after such incidents. There is no claim of a separate high-availability watchdog or privileged system timer.
