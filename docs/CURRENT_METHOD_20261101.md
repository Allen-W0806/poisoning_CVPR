# Current implementation and evidence boundary — 2026-11-01

This is the current method entry point. The body of METHOD.md is a preserved early infrastructure-stage description, not the source of truth for later diffusion runs.

The benign editor is pinned InstructPix2Pix revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`; `src/editing/pretrained_editor.py` uses RGB, white BICUBIC padding and the cached pipeline. Production diffusion learning in `src/rl/diffusion_rl_trainer.py` is single-pass, on-policy DDPO score-function learning on UNet attention LoRA. It uses detached terminal coefficients and fixed coordinate/time scaling. It has no PPO likelihood-ratio clipping, GRPO group normalization or KL anchor. `src/rl/trainer.py` is the separate categorical REINFORCE implementation.

The frozen evaluation target is Qwen3-VL-2B-Instruct revision `89644892e4d85e24eaac8bacfd4f463576704203`, loaded through `src/models/qwen_loader.py`. `src/evaluation/vqa_official_compatible.py` supplies normalized VQA consensus scores. Independent edited-image validity is separate from inherited-reference agreement. Clean language-attention LoRA checkpoints were trained on original labels in three bounded seeds; no significant clean-task gain has been established on an untouched test.

The legacy preservation reward is unchanged. Its semantic/answer terms are redundant in observed historical data; constant penalties and clipping cause substantial censoring. Identity caller credit and default-vs-YAML wiring gaps remain open. Do not treat this reward as validated for new scientific training.

The new three-way experiment is a tiny randomly initialized UNet test of the actual DDPO trainer, fixed configuration, ReduceLROnPlateau, and RSIController. It verifies full preadaptation state equivalence and equal evaluation/update budgets. It is neither pretrained-editor scientific validation nor evidence of a novel autonomous research mechanism. Actual Slurm GPU seconds and synchronized component wall time are recorded separately.

The generator reward-release guard in `src/rl/phase5_backend.py` remains unchanged and active. Independent human review remains BLOCKED until genuine exports are received. Historical data are source-disjoint within the latest split but not a certified independent assessment. BF16 backward failures remain preserved. No security metric or attack objective enters the benign synthetic optimization.

See the current records:

- [Root causes and remediation](../reports/ROOT_CAUSE_AND_REMEDIATION.md)
- [Scientific gates](../reports/SCIENTIFIC_GATE_MATRIX.md)
- [Executed small comparison and next steps](../reports/SMALL_SCALE_VALIDATION_NEXT.md)
- [Historical audit](../reports/AUDIT_DECISION_MATRIX.md)
- [October31 completed validation](../reports/NEXT_VALIDATION_DECISION_20261031.md)
