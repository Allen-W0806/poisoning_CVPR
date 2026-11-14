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
Current accounting invocation (writes receipts, no GPU allocation):

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

The diagnostic calibration gate passed on real CLIP/LPIPS controls. Internal
pilot CPU preflight (writes one unique directory, never starts CUDA):

```bash
.venv-editor/bin/python train_lighting_v3.py --preflight_only --output_dir outputs/reward_debug_20261104T054605Z_s42/internal_v3_doc_preflight
```

Internal pilot after CPU preflight and real reward calibration pass (one A100,
maximum300 allocated minutes; 100 actual updates per trained arm, 34/33/33):

```bash
.venv-editor/bin/python scripts/submit_reward_campaign.py --campaign outputs/reward_debug_20261104T054605Z_s42 --stage internal_v3 --minutes 300 -- .venv-editor/bin/python train_lighting_v3.py --output_dir outputs/reward_debug_20261104T054605Z_s42/internal_pilot
```

The worker defaults are rank16, alpha32, LR5e-6, clip0.5, DDIM50 eta1,
guidance5/image guidance1.5, full FP32. It performs five ordinary/reload-repeat
numerical witness updates before training. Failed precision/replay gates halt
the worker. Final checkpoints are locked before the 16-image development
assessment. The controller receives v3 validation reward, never ASR. Frozen and
Fixed remain the explicitly user-approved identical-prompt equivalence control.

Keep source snapshots, per-image outputs, checkpoints and numeric witnesses.
Use run-specific project `tmp/` scratch and clean it after jobs stop, checking
resolved paths remain beneath that run's scratch directory. No C: weights.

## Development rejection and v3 admission

The initial lighting-v2 rejection decision observed43 development generations,42 with brightness_progress exactly-1. One additional generation was saved while cancellation took effect: the final preserved observations contain44 generations,43 at-1, with progress SD1.192e-5. The final CPU rescoring uses all44 observations; generated observations and committed optimizer updates are counted separately. Analytic controls had been insufficient. The full rejected run is retained as internal_pilot_v2_saturation_probe; this is not a completed comparison.

V3 uses mean content linear-light target error, with continuous progress x/(1+abs(x)) instead of hard clipping. Semantic term is now CLIP_delta+mean_brightness_progress, weighted by w_sem; common evaluator always fixes w_sem2. Color/LPIPS settings and generator hyperparameters stay fixed. Explicit version in src/mini_pilot/lighting_reward_v3.py. New100-update per-arm experiment starts from the shared initial state, seed42, never resumes across different reward definitions.

## Completed campaign and claim boundary

All fixed stages completed at44166 GPU-seconds (12.2683333h). DDPO, Rule and RL each committed100 updates. RL minus DDPO common development reward was-0.0004266559, with paired-source95% interval[-0.0127251116,0.0121611358]. Color/target failures remain; BF16 is still unrepaired. No internal victim ASR was evaluated.

The separate Shadowcast reference reached189/200 keyword successes; clean0 was0/200. Full8000 VizWiz accuracy changed from0.5569875 to0.55355, a-0.34375-percentage-point difference under26/27 victim updates. These are original-task reference results, not matched evidence for the proposed method.

Fresh result-to-claim review found no material numerical error but rejected the broad attack/self-improvement/preservation claim. Scoped measurements are supported provisionally, with same-family review and no formal integrity audit. Detailed verdict, findings and future evidence gaps are recorded in the campaign reports; no further experiment is automatically launched.

CPU regression (no scratch files or GPU):

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python -m unittest tests.test_lighting_reward_v3 -v
```

CPU real-output rescore must pass lighting_v3_calibration.json admission gates before submitting internal_v3. The rescore is not an observed trained-v3 result. All v2 cancellation time is charged in the same20GPUh campaign.
