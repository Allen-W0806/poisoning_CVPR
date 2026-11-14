# 夜间良性实验实际状态

更新时间：2026-11-12T07:56:03.244587+00:00。Run ID：`overnight_benign_20261112T051015Z`。

ALL_CLEAN_VQA_STAGES_COMPLETE。本轮最大同时1个GPU作业，总上限10 GPU-hours，含失败/重试。每个Slurm作业独立wall time；按实际分配墙钟时间计GPU消耗，不用利用率折扣。未运行不是PASS。

本表来自本次实际SSH/Slurm回读。

## 实际作业

|Job ID|配置|Slurm状态|实际GPU-hours|最近进度|结果状态|
|---|---|---|---:|---|---|
|91023|canary / attempt 0|COMPLETED|0.01250|completed |PASS|
|91029|frozen / attempt 0|COMPLETED|0.07306|completed |PASS|
|91043|seed42 / attempt 0|COMPLETED|0.37806|completed |PASS|
|91057|seed123 / attempt 0|COMPLETED|0.37444|completed |PASS|
|91078|seed3407 / attempt 0|COMPLETED|0.37778|completed |PASS|

CPU准备/调度作业：91020 prepare COMPLETED, 91021 dispatch COMPLETED, 91024 dispatch COMPLETED, 91030 dispatch COMPLETED, 91044 dispatch COMPLETED, 91058 dispatch COMPLETED, 91079 dispatch COMPLETED。

所有预设阶段状态：`[{"stage": "canary", "status": "COMPLETED", "job_id": "91023"}, {"stage": "frozen", "status": "COMPLETED", "job_id": "91029"}, {"stage": "seed42", "status": "COMPLETED", "job_id": "91043"}, {"stage": "seed123", "status": "COMPLETED", "job_id": "91057"}, {"stage": "seed3407", "status": "COMPLETED", "job_id": "91078"}]`。P1配对数据检查：`{"status": "PASS_CPU_PAIRED_DATA_CHECK", "CPU_job_id": "91042", "GPU_status": "BLOCKED_NOT_RUN"}`。P1所有GPU比较（Frozen/SFT/正常编辑RL/Fixed/Rule/UCB/Learned/Random/HPO）均为BLOCKED_NOT_RUN。

已消耗1.2158333333333333 GPU-hours；包括活动任务最坏时限的承诺量1.2158333333333333；可再承诺8.784166666666666 GPU-hours。待运行次序canary→Frozen VQA→seed42/123/3407；前置门槛失败即停止其依赖GPU任务。普通节点故障最多原配置安全恢复一次；OOM/NaN/超时/保存失败不重试。

## 数据、指标与证据

正常VQAv2 train2014内部source-disjoint cohort：train1024/dev128/test512/transfer128，每源图一个问题。train共识≥3/10，评估不按答案一致度筛选。固定3 seed；每seed512 AdamW更新、逻辑batch4、同模型/LoRA/分辨率/预算。dev128在128/256/512步评估，最大dev分、同分选最早；最终test仅在选择后评估。Frozen为共享greedy基线。transfer只是独立source，不能宣称跨任务RSI；公开预训练污染未知，非官方测试服务器结果。

Qwen3-VL-2B pinned revision；BF16 frozen base，FP32 language q/v LoRA rank4/alpha8，LR1e-4，max_pixels65536。正常QA response-only交叉熵，不是RL。

数据准备实况：`{"status": "PASS_DATA_PREPARED", "manifest_SHA256": "17d7362a01d7cfc0291825da876b5e8201fcba324d7fd6f526db27b532af76b8", "protocol_SHA256": "51cf457c5dce43daf1bfd3080999862585cc0acc7e405358036ff1ef5ab9a763", "source_images": 1792, "image_bytes": 279061722, "P1_pairs": 256, "P1_GPU_status": "BLOCKED", "elapsed_seconds": 182.9212110042572}`。

完成阶段实际指标：

```json
[
  {
    "stage": "canary",
    "seed": null,
    "metrics": null,
    "canary_resume_exact": true,
    "public_adapter_load_exact": true,
    "dev_before": {
      "n_sources": 8,
      "mean": 0.5,
      "source_SD": 0.5345224838248488,
      "source_bootstrap_95CI": [
        0.125,
        0.875
      ],
      "resamples": 10000,
      "analysis_seed": 20261112,
      "training_seed_uncertainty": "NOT_ESTIMATED"
    },
    "dev_after": {
      "n_sources": 8,
      "mean": 0.5,
      "source_SD": 0.5345224838248488,
      "source_bootstrap_95CI": [
        0.125,
        0.875
      ],
      "resamples": 10000,
      "analysis_seed": 20261112,
      "training_seed_uncertainty": "NOT_ESTIMATED"
    },
    "checkpoint": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/canary/attempt_0/checkpoints/canary_final",
    "dev_history": null
  },
  {
    "stage": "frozen",
    "seed": null,
    "metrics": {
      "dev": {
        "n_sources": 128,
        "mean": 0.7515625,
        "source_SD": 0.4009799757405373,
        "source_bootstrap_95CI": [
          0.67890625,
          0.8179687500000001
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      },
      "test": {
        "n_sources": 512,
        "mean": 0.7220703125000001,
        "source_SD": 0.41588000082429605,
        "source_bootstrap_95CI": [
          0.6865234375,
          0.7576171875
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      },
      "transfer": {
        "n_sources": 128,
        "mean": 0.709375,
        "source_SD": 0.4100004801226303,
        "source_bootstrap_95CI": [
          0.6375,
          0.778125
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      }
    },
    "canary_resume_exact": null,
    "public_adapter_load_exact": null,
    "dev_before": null,
    "dev_after": null,
    "checkpoint": null,
    "dev_history": null
  },
  {
    "stage": "seed42",
    "seed": 42,
    "metrics": {
      "test": {
        "n_sources": 512,
        "mean": 0.732421875,
        "source_SD": 0.4090985672428548,
        "source_bootstrap_95CI": [
          0.6966748046875,
          0.7673828125
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      },
      "transfer": {
        "n_sources": 128,
        "mean": 0.7109375,
        "source_SD": 0.41606222562322476,
        "source_bootstrap_95CI": [
          0.6375,
          0.78125
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      }
    },
    "canary_resume_exact": null,
    "public_adapter_load_exact": null,
    "dev_before": null,
    "dev_after": null,
    "checkpoint": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed42/attempt_0/selected_checkpoint",
    "dev_history": [
      {
        "step": 128,
        "score": 0.7546875000000001,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed42/attempt_0/checkpoints/step128"
      },
      {
        "step": 256,
        "score": 0.7390625,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed42/attempt_0/checkpoints/step256"
      },
      {
        "step": 512,
        "score": 0.75390625,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed42/attempt_0/checkpoints/step512"
      }
    ]
  },
  {
    "stage": "seed123",
    "seed": 123,
    "metrics": {
      "test": {
        "n_sources": 512,
        "mean": 0.75078125,
        "source_SD": 0.40423319571335803,
        "source_bootstrap_95CI": [
          0.7160107421875,
          0.7859375
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      },
      "transfer": {
        "n_sources": 128,
        "mean": 0.734375,
        "source_SD": 0.40672254467444596,
        "source_bootstrap_95CI": [
          0.6625,
          0.803125
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      }
    },
    "canary_resume_exact": null,
    "public_adapter_load_exact": null,
    "dev_before": null,
    "dev_after": null,
    "checkpoint": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed123/attempt_0/selected_checkpoint",
    "dev_history": [
      {
        "step": 128,
        "score": 0.73671875,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed123/attempt_0/checkpoints/step128"
      },
      {
        "step": 256,
        "score": 0.73515625,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed123/attempt_0/checkpoints/step256"
      },
      {
        "step": 512,
        "score": 0.7398437499999999,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed123/attempt_0/checkpoints/step512"
      }
    ]
  },
  {
    "stage": "seed3407",
    "seed": 3407,
    "metrics": {
      "test": {
        "n_sources": 512,
        "mean": 0.738671875,
        "source_SD": 0.40821864195301616,
        "source_bootstrap_95CI": [
          0.7037109375,
          0.7734423828124999
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      },
      "transfer": {
        "n_sources": 128,
        "mean": 0.76015625,
        "source_SD": 0.39211516943668756,
        "source_bootstrap_95CI": [
          0.6921875,
          0.82578125
        ],
        "resamples": 10000,
        "analysis_seed": 20261112,
        "training_seed_uncertainty": "NOT_ESTIMATED"
      }
    },
    "canary_resume_exact": null,
    "public_adapter_load_exact": null,
    "dev_before": null,
    "dev_after": null,
    "checkpoint": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed3407/attempt_0/selected_checkpoint",
    "dev_history": [
      {
        "step": 128,
        "score": 0.75703125,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed3407/attempt_0/checkpoints/step128"
      },
      {
        "step": 256,
        "score": 0.75703125,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed3407/attempt_0/checkpoints/step256"
      },
      {
        "step": 512,
        "score": 0.75859375,
        "path": "/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z/stages/seed3407/attempt_0/checkpoints/step512"
      }
    ]
  }
]
```

每个来源的VQA共识分、源图bootstrap CI及逐条预测保存在各attempt中；3 seed全部完成前不填写均值±SD。统计单位是source，不把3×512当作1536独立源图。任务方向为VQA分数↑。未运行的P1论文表格单元格保持空。

三seed配对汇总（只读，不反馈训练）：

|split|独立source数|Frozen|SFT均值 ± 训练seed SD|配对Δ|source bootstrap 95% CI|
|---|---:|---:|---:|---:|---|
|test|512|0.722070|0.740625 ± 0.009334|+0.018555|[+0.001107, +0.036330]|
|transfer|128|0.709375|0.735156 ± 0.024619|+0.025781|[-0.008854, +0.062760]|

仅为冻结cohort上的正常VQA SFT初步结果；test配对CI为正，transfer CI包含0。CI条件于本轮三个训练seed，非跨训练seed总体CI。预训练污染未知，不能用于支持图像编辑Inner/Outer/RSI价值。逐条评分、训练流、dev-only选择及预算只读复核见本轮final_read_only_audit.json。

```json
{
  "status": "COMPLETE_3_SEED_READ_ONLY_ANALYSIS",
  "statistical_unit": "one question per source; seed-averaged paired deltas resampled over sources",
  "uncertainty_limit": "Source-bootstrap CI conditional on these three seeds. Training-seed SD separately reported; three seeds are not a large-seed population CI.",
  "analysis_seed": 20261112,
  "bootstrap_resamples": 10000,
  "transfer_scope": "source-held-out same VQA family, not independent task-family RSI evidence",
  "splits": {
    "test": {
      "n_sources": 512,
      "seeds": [
        42,
        123,
        3407
      ],
      "frozen": 0.7220703125,
      "SFT_seed_scores": [
        0.732421875,
        0.75078125,
        0.738671875
      ],
      "SFT_mean": 0.740625,
      "SFT_training_seed_SD": 0.00933422120730351,
      "paired_delta_mean": 0.018554687499999997,
      "paired_delta_source_bootstrap_95CI": [
        0.0011067708333333294,
        0.03632975260416664
      ],
      "answer_type": {
        "number": {
          "n": 70,
          "frozen": 0.5514285714285714,
          "SFT_seed_mean": 0.5942857142857143
        },
        "other": {
          "n": 242,
          "frozen": 0.6194214876033057,
          "SFT_seed_mean": 0.6363636363636364
        },
        "yes/no": {
          "n": 200,
          "frozen": 0.906,
          "SFT_seed_mean": 0.918
        }
      }
    },
    "transfer": {
      "n_sources": 128,
      "seeds": [
        42,
        123,
        3407
      ],
      "frozen": 0.709375,
      "SFT_seed_scores": [
        0.7109375,
        0.734375,
        0.76015625
      ],
      "SFT_mean": 0.73515625,
      "SFT_training_seed_SD": 0.02461867383841988,
      "paired_delta_mean": 0.025781249999999995,
      "paired_delta_source_bootstrap_95CI": [
        -0.00885416666666667,
        0.06276041666666667
      ],
      "answer_type": {
        "number": {
          "n": 19,
          "frozen": 0.3526315789473684,
          "SFT_seed_mean": 0.47719298245614034
        },
        "other": {
          "n": 54,
          "frozen": 0.6259259259259259,
          "SFT_seed_mean": 0.6358024691358025
        },
        "yes/no": {
          "n": 55,
          "frozen": 0.9145454545454546,
          "SFT_seed_mean": 0.9218181818181819
        }
      }
    }
  }
}
```

远端根目录：`/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z`。协议`protocol.json`，来源SHA`source_hashes.json`，固定数据`manifest.json`，账本`ledger.json`，Slurm脚本`slurm/`，原始日志`logs/`，逐步训练/预测/恢复检查`stages/<stage>/attempt_<n>/`，完整checkpoint在该目录`checkpoints/`和`selected_checkpoint/`。本地小证据镜像：`outputs/overnight_benign_20261112T051015Z/remote/`。权重和图像不重复下载到C盘。

磁盘检查时点unix=1794470469.7571504，远端保留原图/配对/日志/checkpoint约804.39 MiB。本轮已无活动GPU。 模型共享缓存没有复制。本地最新disk收据记录约5.36 MiB小证据，C盘未创建本轮大模型/数据文件。空本地run scratch已校验路径后移除；远端job trap清理各自临时目录，CPU监控仅rmdir已验证路径的空run根，保留checkpoint和证据。

## 编辑与方法门槛

Frozen图像编辑门槛仍为NOT_CLEARED；既有≥20步/512px视觉诊断见`reports/frozen_quality_gate.md`，未重复采样搜索、未启动下游编辑RL。P1配对数据准备使用原始真实照片作为目标、人工变暗/模糊作为输入，全部对照共用同一split。数据准备不是方法验证。代码映射与阻塞见`reports/OVERNIGHT_METHOD_AUDIT.md`。

Inner是on-policy score-function；Learned Outer是categorical REINFORCE，均不能标成PPO。缺少old-policy模型快照、exp(new-old)重要性ratio与clipped surrogate/clip fraction；梯度裁剪不是PPO。独立任务族迁移/RSI尚无证据。

## 监控任务

```json
{
  "platform": "Codex app",
  "automation_id": "rsi-experiment-monitor",
  "name": "RSI_Experiment_Monitor",
  "kind": "heartbeat",
  "status": "DELETED_COMPLETED",
  "interval_hours": 2,
  "target_thread_id": "01a1cd9e-c068-75f1-8a52-df467c21e633",
  "verification": "Created with automation_update, viewed through app tool, and read back ACTIVE from app automation.toml",
  "Windows_task_enabled": false,
  "scheduled_background_trigger_test": "PASS_OBSERVED_CODEX_HEARTBEAT_2026-11-12T07:55:38.977Z",
  "monitor_command_manual_test": "Previously passed; not presented as proof of a scheduled Codex trigger",
  "last_observed_trigger_UTC": "2026-11-12T07:55:38.977Z",
  "monitor_command_background_result": "PASS_REAL_SSH_SLURM_ACCOUNTING_AND_CHECKPOINT_INTEGRITY",
  "deleted_UTC": "2026-11-12T08:03:24.681208+00:00",
  "deletion_reason": "All five authorized campaign GPU stages and CPU completion callbacks are COMPLETED; no remaining runnable campaign task; Frozen editing quality still blocks P1; no new GPU expansion.",
  "deletion_tool_result": {
    "automationId": "rsi-experiment-monitor",
    "mode": "delete",
    "deleteStatus": "deleted"
  }
}
```

本轮预设GPU阶段均已完成，Codex两小时监控任务已删除，不再重复唤醒；未扩展新GPU实验。 任务名`RSI_Experiment_Monitor`，ID `rsi-experiment-monitor`。之前误建的Windows同名任务已停用，证据`reports/monitor/windows_task_disabled_receipt.json`，旧Windows XML/手动测试收据仅为历史记录。首次Codex后台定时触发已实际观察并成功执行SSH/Slurm与完整性检查，触发时间详见下方收据。 GPU阶段之间的短Slurm CPU完成回调也已全部结束。当前队列/GPU只读探针、checkpoint SHA/更新时间、日志及预算在`reports/monitor/heartbeat.json`。其他用户和其他项目任务不取消、不抢占。

## 异常与下一步

- 本次检查未发现已记录异常；不能据此认证未完成结果。

下一步：ALL_CLEAN_VQA_STAGES_COMPLETE。只按冻结协议推进，不修改LR、reward、训练数据或最终测试集。

修改/新增实现：`src/operations/night_budget.py`、`tests/test_night_budget.py`、`tests/test_night_analysis.py`、`configs/overnight_benign_20261112.json`、`scripts/prepare_night_benign.py`、`scripts/night_vqa_worker.py`、`scripts/night_campaign_remote.py`、`scripts/night_campaign.py`、`scripts/summarize_night_vqa.py`、`scripts/audit_night_vqa_results.py`、`scripts/check_night_p1.py`、`src/training/paired_generator_data.py`及历史Windows监控启动/安装脚本（任务已停用）。新增预算和配对统计CPU测试13项通过（集群快照中预算10项另已实际通过）；命令和故障修复记录在`outputs/overnight_benign_20261112T051015Z/COMMANDS.md`；canary/正式运行是否通过，以本表真实状态和远端日志/checkpoint为准。
