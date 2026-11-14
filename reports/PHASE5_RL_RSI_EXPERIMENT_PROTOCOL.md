# Phase 5 prepared RL/RSI comparison

**NO-GO for scientific launch.** Six reviewed configs are frozen under `configs/phase5_v2`; initial proposals under `configs/phase5` remain retained. All are PREPARED_NOT_RUN with launch_authorized=false. No large comparison was submitted.

| Arm | Intervention |
|---|---|
| M0 | Frozen generator |
| M1 | Standard generator DDPO-SF |
| M2 | Ordinary fixed scheduler: after update4 LR1e-5 ->5e-6 |
| M3 | Traces collected; no harness revision |
| M4 | Diagnosis-guided reusable revisions |
| M5 | Active additional observations plus independently gated revisions |

Planning seeds42/123/2026; eight logical updates for trained arms, batch10 distinct content sources, microbatch1 and accumulation10; LoRA rank8/alpha16. Fixed64-source development observations at updates2/4/6/8 and64 private Gate sources; max2 candidate attempts per M4/M5 seed. No target-model poisoning or security reward. Editor and evaluator must not share per-item Gate feedback. Generator/service snapshots stay frozen within each paired candidate comparison.

Before branching, compare adapters, optimizer, scheduler, step/version, contracts, consumed sources and all Python/NumPy/Torch/CUDA RNG exactly. Report full metadata equality separately; only arm/rsi_mode bookkeeping may be stripped from scientific-contract comparison. No numeric tolerance is relaxed. Exact available fields alone cannot establish causal readiness: frozen weights, content/source order, rollout noise and independent pre-intervention execution require additional proof. Historical strict continuation failures remain unresolved. The checker never auto-clears causal readiness.

Statistical unit is the unique source-image content cluster, not each repeated question or evaluation look. Keep all predeclared seeds and failed runs. Report seed-specific paired candidate/incumbent effects with image-cluster bootstrap 95% intervals (10,000 deterministic resamples, analysis seed20261028), source/record coverage and complete search cost. Any pooling is descriptive; three seeds do not support a broad significance claim. Fixed development looks are descriptive trajectories; no best-look/seed selection. Gate supports bounded operational selection, not a fresh confirmatory efficacy claim. Lock natural-image acceptance criteria using independent labels before opening an unseen evaluation cohort. Existing exposed Gate/test cannot provide that confirmation.

| Readiness item | Status |
|---|---|
| independent_editing_validity | NOT SATISFIED |
| validated_training_reward | NOT SATISFIED |
| content_disjoint_splits | PASS |
| active_harness_tests | PASS |
| conventional_scheduler_baseline | PASS |
| pre_intervention_comparability | NOT SATISFIED |
| private_gate_isolation | NOT SATISFIED |
| frozen_resource_protocol | PASS |

Cost is a planning scenario, not an allocation: prior eight-update GPU seconds3227/3211/2707/3216 and median generation+judging15.77772137sec per observation. Five trained arms x3seeds plus expanded development:27.58953 A100 GPUh; frozen arm0.84148h; paired Gate attempts6.73183h. Subtotal35.16284h; +20% overhead = **42.19541 A100 GPUh**. Human review, a changed independent evaluator and clean-target fine-tuning are unmeasured and excluded. Current authorized next engineering work requires0 GPUh. Resolve labels/reward/fresh private Gate/strict branch comparability first; then a separately authorized small clean feasibility check may precede any campaign.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [scripts/prepare_phase5_protocol.py](<H:/Project/Poisoning/scripts/prepare_phase5_protocol.py>), [configs/phase5_v2/M0.yaml](<H:/Project/Poisoning/configs/phase5_v2/M0.yaml>), [configs/phase5_v2/M1.yaml](<H:/Project/Poisoning/configs/phase5_v2/M1.yaml>), [configs/phase5_v2/M2.yaml](<H:/Project/Poisoning/configs/phase5_v2/M2.yaml>), [configs/phase5_v2/M3.yaml](<H:/Project/Poisoning/configs/phase5_v2/M3.yaml>), [configs/phase5_v2/M4.yaml](<H:/Project/Poisoning/configs/phase5_v2/M4.yaml>), [configs/phase5_v2/M5.yaml](<H:/Project/Poisoning/configs/phase5_v2/M5.yaml>), [src/evaluation/phase5_comparability.py](<H:/Project/Poisoning/src/evaluation/phase5_comparability.py>).

Tests: [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_comparison_prepared_v2.json](<H:/Project/Poisoning/reports/phase5_comparison_prepared_v2.json>), [reports/phase4_results.json](<H:/Project/Poisoning/reports/phase4_results.json>), [reports/phase5_dataset_v2.json](<H:/Project/Poisoning/reports/phase5_dataset_v2.json>).
