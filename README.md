# poisoning_CVPR

论文研究代码与审计数据快照：正常图像编辑、clean VQA、扩散LoRA数值检查、良性Inner/Controller基础设施及真实实验记录。

## 已运行与验证范围

正常VQA使用Qwen3-VL-2B-Instruct固定revision，train1024/dev128/test512/source-transfer128，seeds42/123/3407，每seed512次实际AdamW更新。test Frozen=0.722070，SFT=0.740625 ± 0.009334（训练seed SD）。配对source bootstrap95%CI=[0.001107,0.036330]。transfer差值CI=[-0.008854,0.062760]包含0；不宣称跨任务RSI或预训练无污染。

GPU实际作业91023/91029/91043/91057/91078均COMPLETED，总4377秒=1.215833 GPU-hours。数值恢复、公开adapter与checkpoint哈希检查记录随附。以上是正常VQA结果，不能替代完整生成器RL/Outer方法验证。

Frozen正常图像编辑质量门槛仍未通过；P1的完整三seed编辑对照保持NOT_RUN。Inner为score-function，Learned Outer为REINFORCE，均不是PPO。旧4-step和低样本smoke不能作为完整方法效能结果。

## 目录

- `src/`：模型/数据/编辑/训练/评估、扩散trajectory、controller与checkpoint实现。
- `scripts/`、`configs/`、`tests/`、`slurm/`：良性实验与诊断入口、固定协议和测试。
- `reports/`：实现清单、质量门槛、方法审计和夜间结果。
- `outputs/overnight_benign_20261112T051015Z/`：固定manifest、逐步训练、逐条预测、计费与完整性证据。原始图片和完整tensor checkpoint保留在原资源中；本次只另附三个选定的小LoRA。
- `experiments/p0_foundations_20261111_v1/`：选取的小型P0证据，不等于再次运行P0。
- `SOURCE_RELEASE_MANIFEST.json`：原文件相对路径、字节数和SHA256。
- `EXPORT_SCOPE.json`：打包范围和明确排除项。

## 环境与检查

Python3.11；固定依赖见`requirements.txt`与`requirements-editor.txt`。CUDA版本应匹配实际资源；本仓库不附完整基础模型或大型数据；只附下文列出的三个已验证小LoRA。现有训练/Slurm脚本保留真实运行版本和机器路径，复用时需要自己的路径、新run ID、数据/权重及预算检查，不能直接重开已完成的历史run。

无需模型/GPU的轻量检查：

```bash
python -B -m unittest tests.test_night_budget tests.test_night_analysis tests.test_native_preprocessing tests.test_vqa_official tests.test_vizwiz_official -v
```

依赖torch/diffusers或历史artifact的其他测试需要相应环境；本次上传不宣称全部测试都重新执行。GPU工作需先通过对应数值、质量与预算门槛。

## 快照范围

上传的是现有良性实验实现及审计证据。独立ASR后门优化附件、Shadowcast投毒控制、账户认证/同步工具、凭据、完整基础模型权重、大型图像数据、缓存和原Git历史未打包。原工作区全部保留；本快照不是原目录的无差别镜像，也不把缺失的方法标为已实现。

原始机器上的精确日志/checkpoint位置和哈希保留在证据JSON中。早期文档是历史记录；结论以最新`reports/OVERNIGHT_EXPERIMENT_STATUS.md`、`reports/METHOD_READINESS_20261113.md`和原始运行证据为准。

第三方VQA评分代码的原license、出处和固定revision保留在`src/evaluation/_vendor/`。本次导出没有给原作者代码新增许可证。

## 论文结果表与资源复现（2026-11-14）

本节由已核验的`comparison.json`自动生成，对应`reports/PAPER_RESULTS_20261114.json`和CSV。VQA分数范围0–1，越高越好。

|split|source数|Frozen|clean SFT均值 ± seed SD|配对Δ|source bootstrap 95% CI|
|---|---:|---:|---:|---:|---|
|test|512|0.722070|0.740625 ± 0.009334|+0.018555|[+0.001107, +0.036330]|
|transfer|128|0.709375|0.735156 ± 0.024619|+0.025781|[-0.008854, +0.062760]|

每seed实际512次optimizer更新；dev-only选择的最终step分别为128/512/512。transfer CI包含0，不能确认迁移收益。该实验是正常VQA SFT，不能解释为图像编辑RL/Outer/RSI提升。P1未运行单元格仍为空。

三个选定normal-VQA LoRA在`artifacts/clean_vqa_lora/seed42`、`seed123`、`seed3407`，每个tensor文件3,227,648字节。它们是FP32 LoRA，不包含基础模型或optimizer/RNG状态；加载base时必须显式固定上述Qwen revision，不能默认为最新版本。基模型声明见[固定revision模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md)。

模型和数据的已验证绝对路径、来源、revision、大小/哈希在`reports/ASSET_LOCATIONS_20261114.json`。原Windows代码位置为`H:/Project/Poisoning`；真实模型和数据位于该清单所列Linux集群路径。

```bash
# 默认只检查已有资产，不下载或运行训练
bash scripts/download_and_setup.sh --check
# 查看具体路径与来源
bash scripts/download_and_setup.sh --paths-json
# 未来确需重新下载时显式选择此模式（本轮未执行）
# bash scripts/download_and_setup.sh --download-missing
```

--check已在原集群通过：3个model snapshots、1792张固定source、1796个archive/image/metric文件哈希；本轮未重新下载完整模型/原始数据、未运行GPU训练。下载分支未实际执行，不宣称下载重建已验证。

普通Git提交中每个文件均低于100MB；本轮不需要LFS。全球忽略模型/原始数据/outputs缓存，只对三个小adapter显式放行。跳过资产和小文件范围见`reports/SKIPPED_ASSETS_20261114.json`、`DATA_RELEASE_MANIFEST_20261114.json`。`SOURCE_RELEASE_MANIFEST.json`是首次源码快照的历史清单；本次新数据以新manifest为准。
