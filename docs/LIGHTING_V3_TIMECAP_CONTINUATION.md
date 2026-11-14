# Conditional seed42 time-cap continuation

This is a prepared fallback, not a submitted or GPU-verified continuation.
The active primary job88957 has a300-minute Slurm limit and a17700-second
cooperative process limit. The user authorized remaining steps from a saved
checkpoint and a separate cumulative20GPU-hour campaign. Failed allocations
remain charged. Do not resume any BF16, objective, replay or numerical failure.

Canonical root `/storage/home/ydk297/projects/poisoning`, existing `.venv-editor`.
CPU-only controller/RNG and schedule-boundary verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python -m unittest tests.test_lighting_v3_continuation -v
```

Admit only after the primary job is terminal, run_state status is FAILED with a
cooperative TimeoutError, its original numerical gate is PASS, and remaining
budget plus all active reservations allows one additional3600GPU-seconds.
Check the interrupted adapter/Adam/RNG checkpoint, committed update log,
pending controller state and original source hashes. Never submit a duplicate
stage or resume while the primary job is alive. No continuation is needed if
the primary pilot completes.

```bash
.venv-editor/bin/python scripts/submit_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --stage internal_v3_continuation --minutes 60 -- .venv-editor/bin/python scripts/continue_lighting_v3.py --output_dir outputs/reward_debug_20261104T054605Z_s42/internal_pilot --wall-seconds 3480
```

The candidate requires the unique stage receipt and matching Slurm job ID,
the exact cooperative time-cap message, the original primary's FAILED sacct
state, and its absence from squeue. Arbitrary TimeoutErrors are not admitted.
It checks the same FP32 lighting_v3 protocol and unchanged scientific sources.
It restores generator LoRA, Adam, global RNG and any pending learned-controller
action. New-controller construction follows the original restored-RNG order;
partial-controller construction does not consume the saved generator RNG.
For a partially trained arm, it first settles any interrupted validation and
pending next-round decision, then repeats the actual next source/config update
twice from the same newly saved disk state and requires exact/tolerance agreement before any
remaining arm updates; throwaway witnesses are explicitly excluded from the
100-update count. CPU tests do not substitute for that actual GPU gate.

Remaining updates follow the original34/33/33 source schedule, seeds, sampler,
reward weights and controller decisions. Completed arms are not retrained.
Final checkpoints remain locked before the16-source development assessment;
already completed assessment arms are not regenerated. No sealed evaluation
pool is opened. Check all actual update counts and finite gradients/moments
afterwards. Keep original failure history, old inventory, continuation source,
witnesses, raw outputs and charged GPU time.

Scratch is the stage-specific short path in its submission receipt. After all
jobs finish, resolve and verify that exact path beneath project tmp before
removing only disposable scratch. Preserve all checkpoint and result evidence.
