# Seed42 reward-debug campaign

Canonical root: `/storage/home/ydk297/projects/poisoning`; SSH `utc-login`.
Use `.venv-editor` warm, unchanged dependency specification in
`.aris/compute/utc-editor.md`. No installation/model download for the diagnostic.
All CUDA work must run inside a one-A100 Slurm allocation, never on login.
User authorized a NEW cumulative 72000 GPU-second ceiling, including failures.
Historical closed budgets and sealed evaluation pools remain closed.

CPU documented invocation, from canonical root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python -m unittest tests.test_lighting_reward_v2 -v
```

Diagnostic invocation (inside an allocated GPU, after source review):

```bash
.venv-editor/bin/python scripts/debug_lighting_v2.py --output outputs/reward_debug_20261104T054605Z_s42/diagnostics
```

Submit that diagnostic once, at most45 minutes, through the campaign accountant:

```bash
.venv-editor/bin/python scripts/submit_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --stage diagnostics --minutes 45 -- .venv-editor/bin/python scripts/debug_lighting_v2.py --output outputs/reward_debug_20261104T054605Z_s42/diagnostics
```

The script reserves a unique stage before sbatch. Inspect its receipt after any
ambiguous failure; never repeat a stage name or bypass cumulative accounting.
Current accounting invocation (read-only, no GPU allocation):

```bash
.venv-editor/bin/python scripts/submit_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --accounting-only
```

Diagnostics: first4 exposed train images for identity/exposure+/exposure-/color/
unrelated controls; first4 exposed validation images for four prespecified
DDIM settings; all image IDs/order saved. Eta0 is inference-only. The Gaussian
DDPO transition still requires positive eta. LPIPS remains calibrated [-1,1],
CLIP uses cached ViT-L/14 and its processor. Full-canvas metrics are retained;
brightness/chromaticity use verified original content boxes, excluding padding.
No ASR, VQA correctness, independent semantics or BF16 repair is inferred from
an objective passing these controls. BF16 FP32-adapter recipe is tested on a real
ordinary repeated optimizer update; a failed repeat prevents BF16 training.

The v2 objective is versioned in `src/mini_pilot/lighting_reward_v2.py`.
It subtracts source CLIP baseline AND adds an explicit +3% linear-light exposure
target-progress term and color cost. Pure recentering alone does not change
candidate rankings. Never compare raw rewards across v1 and v2 as improvements.

Keep source snapshots, per-image outputs, checkpoints and numeric witnesses.
Use run-specific project `tmp/` scratch and clean it after jobs stop, checking
resolved paths remain beneath that run's scratch directory. No C: weights.
