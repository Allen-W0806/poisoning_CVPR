# Phase6 saved reward construct audit

Audit executed without retraining on500 saved observations. **Audit completion is PASS; training reward validation remains BLOCKED.** Coefficients did not change.

Legacy arithmetic: `R=S+0.5A-0.2D-0.1C-U`, clipped[-1,1.5]. U and C are constant; S=0.5A in these observations. Lower-bound outcomes230/500 (46%); upper-bound0. Instruction-success measurement is missing in500/500; independent-label agreement is unavailable (0 labels). Qwen supplies both answer and semantic proxies, so agreement is not independence.

Component distributions and missing counts:

```json
{
  "distributions": {
    "semantic": {
      "n": 500,
      "min": 0.0,
      "max": 0.5,
      "mean": 0.27,
      "population_sd": 0.24919871588754225
    },
    "answer_consistency": {
      "n": 500,
      "min": 0.0,
      "max": 1.0,
      "mean": 0.54,
      "population_sd": 0.4983974317750845
    },
    "distortion": {
      "n": 500,
      "min": 0.02003131667152047,
      "max": 0.4137026108801365,
      "mean": 0.17306057108938694,
      "population_sd": 0.07804845514605824
    },
    "cost": {
      "n": 500,
      "min": 0.3333333333333333,
      "max": 0.3333333333333333,
      "mean": 0.3333333333333333,
      "population_sd": 0.0
    },
    "invalid_penalty": {
      "n": 500,
      "min": 1.0,
      "max": 1.0,
      "mean": 1.0,
      "population_sd": 0.0
    },
    "reward_unclipped": {
      "n": 500,
      "min": -1.115833515425523,
      "max": -0.03733959666763742,
      "mean": -0.5279454475512108,
      "population_sd": 0.5010002812569477
    },
    "reward": {
      "n": 500,
      "min": -1.0,
      "max": -0.03733959666763742,
      "mean": -0.4955117319948971,
      "population_sd": 0.4657612431127833
    }
  },
  "missing": {
    "semantic": 0,
    "answer_consistency": 0,
    "distortion": 0,
    "cost": 0,
    "invalid_penalty": 0,
    "reward_unclipped": 0,
    "reward": 0,
    "instruction_success": 500
  },
  "populations": {
    "updates": {
      "n": 320,
      "lower_clip_count": 165,
      "mean": -0.5475048743720011
    },
    "evaluations": {
      "n": 180,
      "lower_clip_count": 65,
      "mean": -0.4030794788800456
    }
  }
}
```

| Pair | Correlation |
|---|---|
|answer_consistency:semantic|1.0|
|distortion:semantic|-0.15152126652053796|
|distortion:answer_consistency|-0.15152126652053796|
|cost:semantic|2.815668407696543e-16|
|cost:answer_consistency|2.815668407696543e-16|
|cost:distortion|4.6941812934116084e-17|
|invalid_penalty:semantic|undefined: constant component|
|invalid_penalty:answer_consistency|undefined: constant component|
|invalid_penalty:distortion|undefined: constant component|
|invalid_penalty:cost|undefined: constant component|
|reward_unclipped:semantic|0.9995256503119592|
|reward_unclipped:answer_consistency|0.9995256503119592|
|reward_unclipped:distortion|-0.18189111769304442|
|reward_unclipped:cost|-1.1789187974480651e-16|
|reward_unclipped:invalid_penalty|undefined: constant component|
|reward:semantic|0.9997001557580958|
|reward:answer_consistency|0.9997001557580958|
|reward:distortion|-0.16936662851626283|
|reward:cost|8.009145067137381e-17|
|reward:invalid_penalty|undefined: constant component|
|reward:reward_unclipped|0.999783372696136|

Saved-component arithmetic reproduced exactly twice; maximum stored arithmetic discrepancy0.0. This is deterministic calculation stability, not repeated independent judge reliability or construct validity. Across old checkpoints/seeds/outputs the maximum within-source reward range is0.9537942330042521; this is not fixed-output evaluator noise.

All saved observations use one benign lighting instruction. Cross-edit-category performance is unavailable. Color/animal/food/vehicle keyword summaries in the receipt are overlapping original-question strata, not editing categories. No semantic-success rate is inferred from reward strata.

Scientific objective is proposed as independently verified instruction completion under factual/visual-preservation constraints. It is not yet accepted as a scalar training objective. Need independently judged identity/valid SC/valid SI-V/invalid/unknown/disputed cases, inter-reviewer agreement, missingness, construct redundancy and clipping analysis before releasing a fixed reward version. Compatibility reward remains diagnostic_only; trainer refuses unreleased rewards. Security ASR and evasion metrics remain outside optimizer/Gate.

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/rl/reward_v2.py](<H:/Project/Poisoning/src/rl/reward_v2.py>), [src/rl/phase5_backend.py](<H:/Project/Poisoning/src/rl/phase5_backend.py>), [src/rl/diffusion_rl_trainer_v2.py](<H:/Project/Poisoning/src/rl/diffusion_rl_trainer_v2.py>), [src/progression/tasks.py](<H:/Project/Poisoning/src/progression/tasks.py>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

500条raw/scalar已分离；230/500裁剪、C/U常数、S=0.5A、指令成功与校准不确定性缺失；系数不变，奖励不发布。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
