Generated from immutable runs and current Slurm accounting. Development proxies only; no internal held-out selection.

| Arm | Seed | Final reward | Delta vs frozen | Source-paired95CI |
|---|---|---|---|---|
| M0 | 42 | -0.43772059 | reference | reference |
| M0 | 123 | -0.25461367 | reference | reference |
| M1 | 42 | -0.62472904 | -0.18700845511009295 | [-0.4675121701260408, 0.0002261811122297952] |
| M1 | 123 | -0.34785917 | -0.09324549342195194 | [-0.2812885372396559, 0.0011641624625772042] |
| M2 | 42 | -0.53095553 | -0.09323494805643955 | [-0.2801065130634233, 0.0003370220661163277] |
| M2 | 123 | -0.25415147 | 0.0004622069746255897 | [3.3651895821087686e-05, 0.001021090146154174] |

Across training/sampling seeds (sample SD; only2 seeds):

```json
{
  "M0": {
    "seed_count": 2,
    "mean_reward": -0.3461671304454406,
    "sample_sd": 0.12947613924652557,
    "components": {
      "semantic": {
        "mean": 0.35,
        "sample_sd": 0.07071067811865478
      },
      "answer_consistency": {
        "mean": 0.7,
        "sample_sd": 0.14142135623730956
      },
      "vqa_consistency_raw": {
        "mean": 0.7,
        "sample_sd": 0.14142135623730956
      },
      "vqa_same_answer_fraction": {
        "mean": 0.75,
        "sample_sd": 0.21213203435596428
      },
      "distortion": {
        "mean": 0.1764067764393985,
        "sample_sd": 0.0006361786242759092
      },
      "cost": {
        "mean": 0.33333333333333337,
        "sample_sd": 0.0
      },
      "generation_seconds": {
        "mean": 5.862335474009161,
        "sample_sd": 0.006621165735783777
      }
    }
  },
  "M1": {
    "seed_count": 2,
    "mean_reward": -0.48629410471146306,
    "sample_sd": 0.1957765652803446,
    "components": {
      "semantic": {
        "mean": 0.275,
        "sample_sd": 0.1060660171779821
      },
      "answer_consistency": {
        "mean": 0.55,
        "sample_sd": 0.2121320343559642
      },
      "vqa_consistency_raw": {
        "mean": 0.65,
        "sample_sd": 0.21213203435596428
      },
      "vqa_same_answer_fraction": {
        "mean": 0.7,
        "sample_sd": 0.282842712474619
      },
      "distortion": {
        "mean": 0.17394528556615113,
        "sample_sd": 0.00043792682945244444
      },
      "cost": {
        "mean": 0.33333333333333337,
        "sample_sd": 0.0
      },
      "generation_seconds": {
        "mean": 5.424132912408095,
        "sample_sd": 0.5960917216721817
      }
    }
  },
  "M2": {
    "seed_count": 2,
    "mean_reward": -0.3925535009863476,
    "sample_sd": 0.19573003294687905,
    "components": {
      "semantic": {
        "mean": 0.325,
        "sample_sd": 0.10606601717798214
      },
      "answer_consistency": {
        "mean": 0.65,
        "sample_sd": 0.21213203435596428
      },
      "vqa_consistency_raw": {
        "mean": 0.65,
        "sample_sd": 0.21213203435596428
      },
      "vqa_same_answer_fraction": {
        "mean": 0.7,
        "sample_sd": 0.282842712474619
      },
      "distortion": {
        "mean": 0.1746008064597845,
        "sample_sd": 0.00012551937347490803
      },
      "cost": {
        "mean": 0.33333333333333337,
        "sample_sd": 0.0
      },
      "generation_seconds": {
        "mean": 5.892628098139539,
        "sample_sd": 0.01548915052874456
      }
    }
  }
}
```

RSI minus fixed RL, source-paired bootstrap,10000 resamples, seed7:

```json
[
  {
    "seed": 42,
    "rsi_minus_rl": {
      "mean_delta": 0.09377350705365342,
      "ci95": [
        -1.8918514251697884e-05,
        0.28134584888815883
      ],
      "ci_half_width": 0.14068238370120525,
      "paired_deltas": [
        1.671910285994116e-06,
        0.0,
        0.9378291601936023,
        -2.999752759935692e-05,
        0.0,
        -4.959627985945403e-05,
        0.0,
        0.0,
        0.0,
        -1.6167759895324707e-05
      ],
      "independent_source_count": 10,
      "bootstrap_seed": 7,
      "bootstrap_resamples": 10000,
      "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
    }
  },
  {
    "seed": 123,
    "rsi_minus_rl": {
      "mean_delta": 0.09370770039657753,
      "ci95": [
        -0.00027541894093152804,
        0.28152740653976804
      ],
      "ci_half_width": 0.14090141274034979,
      "paired_deltas": [
        -0.00013535022735600144,
        -0.0006669268012046148,
        -0.00015258491039271682,
        -0.00016428232192988723,
        -4.6759843826293945e-05,
        -0.0001057825982569982,
        -0.00034234225749973923,
        0.0,
        0.0,
        0.9386910329262416
      ],
      "independent_source_count": 10,
      "bootstrap_seed": 7,
      "bootstrap_resamples": 10000,
      "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
    }
  }
]
```

Fairness checks:

```json
[
  {
    "seed": 42,
    "initial_adapter_exact": true,
    "training_source_order_exact": true,
    "training_sampling_seeds_exact": true,
    "updates": [
      8,
      8
    ],
    "source_exposures": [
      80,
      80
    ],
    "validation_trajectories": [
      40,
      40
    ],
    "actual_gpu_seconds": [
      3227,
      3211
    ],
    "rsi_decisions": [
      {
        "step": 2,
        "accept_schedule_revision": false,
        "reason": "insufficient plateau evidence",
        "old_lr": 1e-05,
        "new_lr": 1e-05,
        "diagnostic_seconds": 0.0025531838182359934,
        "controller_revision": {
          "revision_number": 1,
          "timestamp": "2026-10-27T18:11:19.992635+00:00",
          "previous_config": {
            "batch_size": 10,
            "learning_rate": 1e-05,
            "curriculum": "uniform"
          },
          "proposed_config": {
            "batch_size": 10,
            "learning_rate": 5e-06,
            "curriculum": "uniform"
          },
          "validation_hash": "14274c3e52e7b835f607d4c526a28c8f7bab1198b1f53d8a632f10cce0023044",
          "validation_score": -0.5310547025750081,
          "cost_gpu_hours": 0.22392495278527752,
          "cumulative_cost_gpu_hours": 0.22705867436442834,
          "accepted_schedule": false,
          "quality_improvement_accepted": false,
          "reason": "insufficient plateau evidence",
          "evidence": [
            {
              "mean_delta": -0.09333411606649558,
              "ci95": [
                -0.2801365571990609,
                0.00014560095965862052
              ],
              "ci_half_width": 0.14014107907935974,
              "paired_deltas": [
                -0.0003425329923629761,
                0.0,
                5.91278076171875e-05,
                0.00010587647557258606,
                0.0,
                4.372000694252698e-06,
                -0.9337307686607043,
                0.0,
                0.0,
                0.0005627647042274475
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            }
          ],
          "policy": "single exploratory LR cooldown; no candidate quality-gain or rollback claim"
        }
      },
      {
        "step": 4,
        "accept_schedule_revision": true,
        "reason": "two unresolved validation gains",
        "old_lr": 1e-05,
        "new_lr": 5e-06,
        "diagnostic_seconds": 0.0029644700698554516,
        "revision_experiment_id": "EXP_20261027T182428_b84e4d1d6803_phase4_PH4_RSI_CONFIG_REVISION",
        "controller_revision": {
          "revision_number": 2,
          "timestamp": "2026-10-27T18:24:28.635930+00:00",
          "previous_config": {
            "batch_size": 10,
            "learning_rate": 1e-05,
            "curriculum": "uniform"
          },
          "proposed_config": {
            "batch_size": 10,
            "learning_rate": 5e-06,
            "curriculum": "uniform"
          },
          "validation_hash": "14274c3e52e7b835f607d4c526a28c8f7bab1198b1f53d8a632f10cce0023044",
          "validation_score": -0.5309081215287248,
          "cost_gpu_hours": 0.21906756450532056,
          "cumulative_cost_gpu_hours": 0.4461262388697489,
          "accepted_schedule": true,
          "quality_improvement_accepted": false,
          "reason": "two unresolved validation gains",
          "evidence": [
            {
              "mean_delta": -0.09333411606649558,
              "ci95": [
                -0.2801365571990609,
                0.00014560095965862052
              ],
              "ci_half_width": 0.14014107907935974,
              "paired_deltas": [
                -0.0003425329923629761,
                0.0,
                5.91278076171875e-05,
                0.00010587647557258606,
                0.0,
                4.372000694252698e-06,
                -0.9337307686607043,
                0.0,
                0.0,
                0.0005627647042274475
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            },
            {
              "mean_delta": -0.09318753502021232,
              "ci95": [
                -0.2800789792397991,
                0.0004350084466859696
              ],
              "ci_half_width": 0.14025699384324253,
              "paired_deltas": [
                0.00017915964126591355,
                0.0,
                0.00019081085920336083,
                0.00012770071625711754,
                0.0,
                2.0950660109542163e-05,
                -0.9337307686607043,
                0.0,
                0.0,
                0.0013367965817451255
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            }
          ],
          "policy": "single exploratory LR cooldown; no candidate quality-gain or rollback claim"
        }
      }
    ],
    "exact_execution_reproducibility": false,
    "pre_schedule_state_difference": {
      "step": 4,
      "adapter_changes": {
        "max_abs_change": 4.343278851592913e-05,
        "mean_abs_change": 1.6294180766974903e-06,
        "updated_tensors": 256,
        "adapter_tensor_count": 256
      },
      "optimizer_exact": false,
      "scope": "checkpoint saved before any schedule revision at this step; same scientific configuration"
    },
    "pre_schedule_paired_evaluations": [
      {
        "step": 2,
        "rsi_minus_rl": {
          "mean_delta": 2.7014985680573477e-05,
          "ci95": [
            -5.698867142200914e-05,
            0.0001543129999190474
          ],
          "ci_half_width": 0.00010565083567052827,
          "paired_deltas": [
            -0.0001337409019470659,
            0.0,
            0.0005470350384711997,
            2.5519728660494678e-05,
            0.0,
            -4.501864314077064e-05,
            0.0,
            0.0,
            0.0,
            -0.00012364536523812308
          ],
          "independent_source_count": 10,
          "bootstrap_seed": 7,
          "bootstrap_resamples": 10000,
          "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
        }
      },
      {
        "step": 4,
        "rsi_minus_rl": {
          "mean_delta": 8.988227695226892e-05,
          "ci95": [
            -1.0466612875459358e-05,
            0.0002104726061224915
          ],
          "ci_half_width": 0.00011046960949897544,
          "paired_deltas": [
            0.0002765983343124834,
            0.0,
            0.0004941672086715698,
            -0.00011895224452018738,
            0.0,
            1.4286115765593799e-05,
            0.0,
            0.0,
            0.0,
            0.00023272335529322952
          ],
          "independent_source_count": 10,
          "bootstrap_seed": 7,
          "bootstrap_resamples": 10000,
          "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
        }
      }
    ],
    "original_normalized_answer_disagreements_with_frozen": {
      "rl": 0,
      "rsi": 0
    }
  },
  {
    "seed": 123,
    "initial_adapter_exact": true,
    "training_source_order_exact": true,
    "training_sampling_seeds_exact": true,
    "updates": [
      8,
      8
    ],
    "source_exposures": [
      80,
      80
    ],
    "validation_trajectories": [
      40,
      40
    ],
    "actual_gpu_seconds": [
      2707,
      3216
    ],
    "rsi_decisions": [
      {
        "step": 2,
        "accept_schedule_revision": false,
        "reason": "insufficient plateau evidence",
        "old_lr": 1e-05,
        "new_lr": 1e-05,
        "diagnostic_seconds": 0.0024302019737660885,
        "controller_revision": {
          "revision_number": 1,
          "timestamp": "2026-10-27T18:11:13.426262+00:00",
          "previous_config": {
            "batch_size": 10,
            "learning_rate": 1e-05,
            "curriculum": "uniform"
          },
          "proposed_config": {
            "batch_size": 10,
            "learning_rate": 5e-06,
            "curriculum": "uniform"
          },
          "validation_hash": "14274c3e52e7b835f607d4c526a28c8f7bab1198b1f53d8a632f10cce0023044",
          "validation_score": -0.25439507024983565,
          "cost_gpu_hours": 0.22102883410918164,
          "cumulative_cost_gpu_hours": 0.22447808664557264,
          "accepted_schedule": false,
          "quality_improvement_accepted": false,
          "reason": "insufficient plateau evidence",
          "evidence": [
            {
              "mean_delta": 0.00021860413253308453,
              "ci95": [
                2.6026595383881748e-05,
                0.00048409993760292373
              ],
              "ci_half_width": 0.000229036671109521,
              "paired_deltas": [
                -6.744116544732393e-05,
                0.0005104720592499223,
                -9.59932804107666e-05,
                0.0002523183822632502,
                0.00024133622646338448,
                7.246062159538269e-05,
                0.0012178704142571384,
                0.0,
                0.0,
                5.50180673598577e-05
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            }
          ],
          "policy": "single exploratory LR cooldown; no candidate quality-gain or rollback claim"
        }
      },
      {
        "step": 4,
        "accept_schedule_revision": true,
        "reason": "two unresolved validation gains",
        "old_lr": 1e-05,
        "new_lr": 5e-06,
        "diagnostic_seconds": 0.0022444629576057196,
        "revision_experiment_id": "EXP_20261027T182417_feee4c573de0_phase4_PH4_RSI_CONFIG_REVISION",
        "controller_revision": {
          "revision_number": 2,
          "timestamp": "2026-10-27T18:24:17.481274+00:00",
          "previous_config": {
            "batch_size": 10,
            "learning_rate": 1e-05,
            "curriculum": "uniform"
          },
          "proposed_config": {
            "batch_size": 10,
            "learning_rate": 5e-06,
            "curriculum": "uniform"
          },
          "validation_hash": "14274c3e52e7b835f607d4c526a28c8f7bab1198b1f53d8a632f10cce0023044",
          "validation_score": -0.2543092974523703,
          "cost_gpu_hours": 0.2177930769552606,
          "cumulative_cost_gpu_hours": 0.44227116360083324,
          "accepted_schedule": true,
          "quality_improvement_accepted": false,
          "reason": "two unresolved validation gains",
          "evidence": [
            {
              "mean_delta": 0.00021860413253308453,
              "ci95": [
                2.6026595383881748e-05,
                0.00048409993760292373
              ],
              "ci_half_width": 0.000229036671109521,
              "paired_deltas": [
                -6.744116544732393e-05,
                0.0005104720592499223,
                -9.59932804107666e-05,
                0.0002523183822632502,
                0.00024133622646338448,
                7.246062159538269e-05,
                0.0012178704142571384,
                0.0,
                0.0,
                5.50180673598577e-05
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            },
            {
              "mean_delta": 0.0003043769299983845,
              "ci95": [
                5.428940057738973e-06,
                0.0006988018415868434
              ],
              "ci_half_width": 0.0003466864507645522,
              "paired_deltas": [
                6.222873926153838e-05,
                0.001208978891372725,
                -9.236931800848946e-05,
                0.00017721056938169166,
                -5.139708518975272e-05,
                -4.735738039018944e-05,
                0.0016259029507637024,
                0.0,
                0.0,
                0.00016057193279261917
              ],
              "independent_source_count": 10,
              "bootstrap_seed": 7,
              "bootstrap_resamples": 10000,
              "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
            }
          ],
          "policy": "single exploratory LR cooldown; no candidate quality-gain or rollback claim"
        }
      }
    ],
    "exact_execution_reproducibility": false,
    "pre_schedule_state_difference": {
      "step": 4,
      "adapter_changes": {
        "max_abs_change": 6.645589019171894e-05,
        "mean_abs_change": 1.305959708963717e-06,
        "updated_tensors": 256,
        "adapter_tensor_count": 256
      },
      "optimizer_exact": false,
      "scope": "checkpoint saved before any schedule revision at this step; same scientific configuration"
    },
    "pre_schedule_paired_evaluations": [
      {
        "step": 2,
        "rsi_minus_rl": {
          "mean_delta": 9.404264390469041e-05,
          "ci95": [
            -1.2947320938105911e-05,
            0.0002187872920185418
          ],
          "ci_half_width": 0.00011586730647832386,
          "paired_deltas": [
            -0.00014075785875322655,
            -2.4124979972839355e-05,
            -2.6121735572370852e-06,
            0.0002257317304611428,
            0.0005335003137588723,
            2.5705993175528796e-05,
            0.00030805543065071106,
            0.0,
            0.0,
            1.4927983283952173e-05
          ],
          "independent_source_count": 10,
          "bootstrap_seed": 7,
          "bootstrap_resamples": 10000,
          "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
        }
      },
      {
        "step": 4,
        "rsi_minus_rl": {
          "mean_delta": 0.09325811261932056,
          "ci95": [
            -0.00014254739880558754,
            0.27993802892789243
          ],
          "ci_half_width": 0.140040288163349,
          "paired_deltas": [
            6.488710641860962e-05,
            -0.00042168200016012847,
            0.9331547419230143,
            -0.00027405768632893235,
            3.888905048377023e-05,
            -2.902299165730149e-05,
            1.2390315532684326e-05,
            0.0,
            0.0,
            3.498047590255737e-05
          ],
          "independent_source_count": 10,
          "bootstrap_seed": 7,
          "bootstrap_resamples": 10000,
          "scope": "source sampling uncertainty conditional on one training seed; repeated validation selection is not confirmatory"
        }
      }
    ],
    "original_normalized_answer_disagreements_with_frozen": {
      "rl": 0,
      "rsi": 0
    }
  }
]
```

Intervals describe ten source images conditional on the fitted seed, not a population of training seeds. Repeated looks are correlated; the RSI development set guides its schedule and is reused for reporting, so the comparison is exploratory and potentially optimistic. Matched exposure and look counts do not imply identical GPU compute. Known backward numerical variability persists across fresh model execution. A favorable proxy difference alone does not establish independently verified semantics or generalization. No held-out300 run was opened.

The pre-schedule checkpoints and evaluations above use the same learning configuration in both arms. Any difference already present before the first LR revision is a numerical/execution replicate difference and cannot be caused by that later revision. These are individual descriptive witnesses, not a calibrated noise distribution. Original-answer disagreements are separately counted to expose evaluator variability.

The fixed cooldown heuristic is mean paired gain <= CI half-width on two successive looks, applied once. The raw reason string calls this unresolved gain; precisely, it is a magnitude-versus-width heuristic and is not equivalent to a percentile interval containing zero. Asymmetric bootstrap intervals may exclude zero while still meeting the heuristic. Schedule application never means an accepted quality gain; the original formula and raw decisions were preserved unchanged.

Evidence-bounded claim flags (source-conditional intervals; not confirmatory):

```json
{
  "two_seed_rl_proxy_gain_resolved": false,
  "two_seed_rsi_proxy_gain_resolved": false,
  "generalization_improvement_established": false,
  "independent_semantic_improvement_established": false,
  "larger_compute_followup_justified_now": false,
  "reason": "Independent semantic review and strict continuation remain incomplete; two-seed repeated-development pilot is exploratory"
}
```
