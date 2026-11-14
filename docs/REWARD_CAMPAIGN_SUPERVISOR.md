# Bounded supervisor for this fixed campaign

Canonical root `/storage/home/ydk297/projects/poisoning`. Existing `.venv-editor`
and `.venv-shadowcast-20261104` ledgers apply, no environment rebuild. This is a
foreground helper for one run, maximum6hours, not a recurring automation.

Documented read-only snapshot (no CUDA or job submission):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python scripts/supervise_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --read-only
```

CPU gate tests, with disposable run-specific scratch:

```bash
PYTHONDONTWRITEBYTECODE=1 PILOT_TEST_TMP=/storage/home/ydk297/projects/poisoning/tmp/reward_debug_20261104T054605Z_s42_supervisor_tests .venv-editor/bin/python -m unittest tests.test_reward_campaign_supervisor -v
```

Authorized fixed sequence: existingPGD finishes; CPU audit verifies all14 full
4000-iteration traces, all277 outputs, selected50, and serialized8/255 bound;
only then train poison50. Clean0 already completed. Lock both final checkpoint
hashes before submitting ASR/clean-utility evaluations. Completed internalv3
can start the preregistered frozen-Qwen editing-quality diagnostic. Every GPU
reservation uses submit_reward_campaign.py and the72000GPU-second cumulative
guard, including startup failures and rejectedv2. CPU audit reserves0GPU.

The helper never changes hyperparameters, samples, prompts, scoring, reward
definitions or checkpoint selection; it never retries a failed stage. A fixed
stage directory reserves submission before sbatch and blocks duplication even
when the reply is ambiguous. Failures require an explicit reviewed repair.
Possible internal time-cap continuation is a separate candidate, never invoked
automatically by this helper. All reports/figures remain subject to final review.

Full loop invocation, after source/CPU checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python -u scripts/supervise_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --max-seconds 21600
```

It exits when the specified experiment spines and GPU settlements complete or
its own6hour wall cap fires. A dropped SSH session does not cancel Slurm jobs;
inspect stage receipts and queue before any restart. Stop this foreground helper
and verify its PID is gone when the run finishes. Keep failure and submission
evidence; clean only its recorded run-specific tmp directories after jobs end.
