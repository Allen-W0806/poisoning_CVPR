# poisoning_CVPR

私有论文研究代码快照：正常图像编辑、clean VQA、扩散LoRA数值检查、良性Inner/Controller基础设施及真实实验记录。

## 已运行与验证范围

正常VQA使用Qwen3-VL-2B-Instruct固定revision，train1024/dev128/test512/source-transfer128，seeds42/123/3407，每seed512次实际AdamW更新。test Frozen=0.722070，SFT=0.740625 ± 0.009334（训练seed SD）。配对source bootstrap95%CI=[0.001107,0.036330]。transfer差值CI=[-0.008854,0.062760]包含0；不宣称跨任务RSI或预训练无污染。

GPU实际作业91023/91029/91043/91057/91078均COMPLETED，总4377秒=1.215833 GPU-hours。数值恢复、公开adapter与checkpoint哈希检查记录随附。以上是正常VQA结果，不能替代完整生成器RL/Outer方法验证。

Frozen正常图像编辑质量门槛仍未通过；P1的完整三seed编辑对照保持NOT_RUN。Inner为score-function，Learned Outer为REINFORCE，均不是PPO。旧4-step和低样本smoke不能作为完整方法效能结果。

## 目录

- `src/`：模型/数据/编辑/训练/评估、扩散trajectory、controller与checkpoint实现。
- `scripts/`、`configs/`、`tests/`、`slurm/`：良性实验与诊断入口、固定协议和测试。
- `reports/`：实现清单、质量门槛、方法审计和夜间结果。
- `outputs/overnight_benign_20261112T051015Z/`：固定manifest、逐步训练、逐条预测、计费与完整性证据。原始图片和tensor checkpoint保留在原资源中，不上传到GitHub。
- `experiments/p0_foundations_20261111_v1/`：选取的小型P0证据，不等于再次运行P0。
- `SOURCE_RELEASE_MANIFEST.json`：原文件相对路径、字节数和SHA256。
- `EXPORT_SCOPE.json`：打包范围和明确排除项。

## 环境与检查

Python3.11；固定依赖见`requirements.txt`与`requirements-editor.txt`。CUDA版本应匹配实际资源；本仓库没有附模型权重或大型数据。现有训练/Slurm脚本保留真实运行版本和机器路径，复用时需要自己的路径、新run ID、数据/权重及预算检查，不能直接重开已完成的历史run。

无需模型/GPU的轻量检查：

```bash
python -B -m unittest tests.test_night_budget tests.test_night_analysis tests.test_native_preprocessing tests.test_vqa_official tests.test_vizwiz_official -v
```

依赖torch/diffusers或历史artifact的其他测试需要相应环境；本次上传不宣称全部测试都重新执行。GPU工作需先通过对应数值、质量与预算门槛。

## 快照范围

上传的是现有良性实验实现及审计证据。独立ASR后门优化附件、Shadowcast投毒控制、账户认证/同步工具、凭据、模型权重、大型图像数据、缓存和原Git历史未打包。原工作区全部保留；本快照不是原目录的无差别镜像，也不把缺失的方法标为已实现。

原始机器上的精确日志/checkpoint位置和哈希保留在证据JSON中。早期文档是历史记录；结论以最新`reports/OVERNIGHT_EXPERIMENT_STATUS.md`、`reports/METHOD_READINESS_20261113.md`和原始运行证据为准。

第三方VQA评分代码的原license、出处和固定revision保留在`src/evaluation/_vendor/`。本次导出没有给原作者代码新增许可证。
