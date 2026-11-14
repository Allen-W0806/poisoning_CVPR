# Phase 2 next steps

1. Review `SEMANTIC_VALIDATION_REPORT.md` and the 20-row contact sheet; fill human_review_template.json in a new derived review artifact. Do not overwrite automatic judgments. Natural-image edited accuracy remains gated until independent review.
2. Implement and test a mathematically valid stochastic diffusion trajectory optimizer before any generator-RL claim. Current native-scheduler reward update is explicitly BLOCKED; supervised LoRA is separate.
3. Investigate repeatability of the earlier fused-attention 4.41e-8 failure; final math-backend smoke is scoped to its recorded environment. Retain every failed attempt and metadata-scaling diagnosis.
4. Only after these readiness gates, consider the prepared stability/ablation configs. Freeze equal complete data/compute/proposal/validation budgets; do not auto-launch the grid or reinterpret tiny proxy rewards as scientific advantage.

Measured Phase 2 allocated cost so far: 0.329722 GPU-hours. Use one A100 with 48G host RAM and a 10–15 minute ceiling for another bounded feasibility run; consult actual generator peak and step times in GENERATOR_TRAINING_REPORT.md. These measurements do not validate larger batches or Step1X capacity. Held-out 300 was opened once and must not be rerun for tuning.

Next safe executable checks (CPU, no held-out reuse):

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv-editor/bin/activate; python -m unittest discover -v'
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python scripts/check_generator_rl_contract.py'
```

If another real supervised generator smoke is needed within remaining budget:

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python scripts/submit_phase2.py --task training --minutes 10'
```
