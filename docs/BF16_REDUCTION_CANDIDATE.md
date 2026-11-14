# Withdrawn BF16 reduction proposal: zero GPU spend

Admission: WITHDRAWN BEFORE SUBMISSION. Do not allocate a GPU for this proposal.
The original unexecuted proposal source is retained under the campaign's
rejected_proposals/check_bf16_reduction_candidate_UNEXECUTED.py. It is not an
experiment and must not appear as a measured baseline or successful repair.

Fresh source review found that src/rl/diffusion_policy.py:35 already
unconditionally sets allow_bf16_reduced_precision_reduction=False inside the
actual inherited policy constructor. The standalone process default True,
printed before constructing the policy, was not the effective training value.
Thus disabling it again cannot be called a new repair for the earlier failure.
No candidate submission, CUDA kernel, model load or optimizer update occurred.
The existing running FP32 experiment and all protected sources stay unchanged.

Read-only CPU proof, from canonical /storage/home/ydk297/projects/poisoning:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-editor/bin/python scripts/check_bf16_reduction_candidate.py --campaign outputs/reward_debug_20261104T054605Z_s42 --stage preflight
```

The command inspects the unconditional constructor assignment, verifies its
SHA against the active pilot protocol, and reports zero GPU spend. It does not
write files, load weights or images, or submit work. --stage diagnose now
rejects execution before any model work. BF16 consistency remains unresolved;
the passing full-FP32 ordinary/reload tests provide the usable path this round.
