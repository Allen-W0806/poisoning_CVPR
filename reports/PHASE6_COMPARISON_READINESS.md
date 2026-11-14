# Phase6 controlled comparison readiness

Static contract checks PASSED. **Scientifically executable comparison remains NO-GO.** Phase5 configs remain unchanged. Corrected prepared proposals live in the new immutable Stage4 run's `prepared_configs/`.

M0 frozen; M1 standard Generator RL; M2 conventional one-cooldown scheduler; M3 trace+diagnosis without revision; M4 diagnosis-guided workflow revision; M5 active investigation with independent gated revision. New proposal corrects M3 prose to match Phase6, without modifying its historical config. Static equalities:

```json
{
  "generator_config": true,
  "split_sha256": true,
  "dataset_version": true,
  "seeds": true,
  "development_sources": true,
  "private_gate_sources": true,
  "training_reward_config": true,
  "backend": true
}
```

Same generator config, backend, data hashes, seeds, reward-config reference and development/Gate sizes verified. Trained arms have80 source exposures and8updates. M0 has no training; its resource accounting differs explicitly. M4/M5 incur bounded candidate search costs; report those separately rather than conceal extra evaluator exposure. Phase5/6 prepared arms are NOT RUN.

Still missing: an actual shared pristine LoRA/optimizer/RNG initialization artifact per matched seed, verified frozen services/source order/noise, accepted benign reward, fresh private independent evaluator and locked natural-image selection thresholds. Checking names/configs is not runtime equality. Final-update selection and all predeclared seeds/failures retained; use source-content clusters and report per-seed paired uncertainty, no best-seed/step choice. Three planning seeds are not a powered efficacy study.

Small benign four-update pilot proposal is saved but not launched. It requires all scientific gates plus a separately explicit GPU budget. Historical42.19541 A100h forecast for the full scenario is not allocation, and omits human/new independent evaluator costs. Stage5 cannot run from a static PASS.

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/progression/tasks.py](<H:/Project/Poisoning/src/progression/tasks.py>), [src/evaluation/phase5_comparability.py](<H:/Project/Poisoning/src/evaluation/phase5_comparability.py>), [configs/phase5_v2/M0.yaml](<H:/Project/Poisoning/configs/phase5_v2/M0.yaml>), [configs/phase5_v2/M5.yaml](<H:/Project/Poisoning/configs/phase5_v2/M5.yaml>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

新增真实历史step0检查seeds42/123观测字段相同；仍不能证明干预前公平执行。shared pristine initialization/frozen weights/source/noise/private Gate/budget仍阻塞。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
