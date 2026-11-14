# IP2Pix / LoRA 良性验证中文报告

实际修复与短实验已完成。47项测试通过；BF16/FP32 在本轮配置下的重复与续训一致。**Inner 没有优于 Frozen，Outer 也没有额外价值证据。**

|事项|状态|实际结果|
|---|---|---|
|代码修复与 CPU 回归|已实现、已运行、已验证|47/47 通过|
|BF16/FP32 数值可靠性|本配置已验证|普通两步重复及恢复后的下一步均精确一致；不等于历史 BF16 全面修复|
|Frozen vs Inner|已运行|1 seed，train4/dev2/test4/transfer4；4次 Inner 更新；test Δ=-0.000571128，transfer Δ=-0.000711899|
|fixed / rule / UCB Outer|已运行|每种3候选、6次尝试更新、152次 predict；9候选全拒绝；最终相对 Inner Δ=0|
|RSI、跨 seed 和独立 benchmark|未验证|当前仅有小规模 recipe HPO；无效能或 RSI 主张|

一次 A100 作业90945：纽约时间 2026-11-11T21:54:36.221820-05:00 至 2026-11-11T21:59:38.193006-05:00；Slurm实际304 GPU秒（0.084444 GPUh），上限1200秒。未追加长任务。所有原始日志、图像、trajectory、数值 witness 和 checkpoint 已回读到H盘，并生成 SHA 清单。

实际命令与逐图ID：[完整实验状态](MINIMAL_EXPERIMENT_STATUS.md)。

修改的核心文件为 `src/rl/diffusion_policy.py`、`src/rl/diffusion_trajectory.py`、`src/training/generator_adapter.py`、`src/mini_pilot/engine.py`；新增独立良性指标/控制器、配置、运行入口和测试。逐文件问题、修复及验证见 [实现审计](IMPLEMENTATION_AUDIT.md)。没有改动原稿 `E:/main9_revised.tex`，没有覆盖其他实验或启动任何安全指标优化回路。

|任务|方法|源图数|平均 reward proxy|平均亮度目标误差|SSIM（对源图）|相对 Frozen|
|---|---|---:|---:|---:|---:|---:|
|test|Frozen|4|-0.054608103|0.067546787|0.386068|+0.000000000|
|test|Inner|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|fixed_recipe|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|fixed_rule|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|ucb_search|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|transfer|Frozen|4|-0.087634593|0.102039040|0.367535|+0.000000000|
|transfer|Inner|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|fixed_recipe|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|fixed_rule|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|ucb_search|4|-0.088346492|0.102726066|0.367350|-0.000711899|

奖励和SSIM均为图像代理；没有编辑 ground truth 或独立人工语义验证。原始图像配对预览见 `output/benign_ddpo_audit_20261111/paired_editing_examples.png`。低步数/小样本输出不支持实际编辑效能结论。当前暂停扩大训练，具体问题与下一门槛见 [下一实验决策](NEXT_EXPERIMENT_DECISION.md)。

[架构纠正与原始来源](ARCHITECTURE_CORRECTIONS.md)；[概念图 SVG](../output/benign_ddpo_audit_20261111/benign_validation_framework.svg)；[概念图 PDF](../output/benign_ddpo_audit_20261111/benign_validation_framework.pdf)；方法与 Algorithm 1 源码为 `docs/benign_ddpo_method_outline.tex`。

**运行后发现并修复的限制：** 1张训练图存在1像素 padding 裁剪偏差；当前源码已修正并新增回归，原 GPU 日志与 checkpoint 原样保留。评估与迁移的裁剪及上述数字不受该边界错误影响，但修复后的训练结果尚未运行，不能将旧 checkpoint 称作修复后产物。

独立只读完整性复核另见 `EXPERIMENT_AUDIT.md`；该复核为 same-family/provisional，不是跨家族或人工认证。数据和模型未复制到C盘；本轮临时目录在核验后清理，checkpoint和实验源码作为复现证据保留。


## 运行后修复验证（CPU only）

新增 padding 取整与测试驱动 SHA 覆盖回归后，49 项 CPU 测试通过，零失败/跳过。已用修复后的度量重新检查 126 个保存图像；涉及源图 ['574996000']，各 split 最大 reward 绝对变化为 `{'train': 0.0020111316098415655, 'dev': 0.0, 'test': 1.3877787807814457e-17, 'transfer': 2.7755575615628914e-17}`。这只是已保存像素的 CPU 复测，没有新的模型调用、optimizer 更新或 GPU 作业。训练原始奖励/轨迹/checkpoint 不改写；修复后模型学习效能仍未运行。详细证据在 `experiments/benign_ddpo_audit_20261111_v1/postrun_fix/`。


## 保存图像的视觉检查

配对预览中，电脑图与领带图出现明显纹理化和结构破坏，Frozen 与 Inner 都存在该问题。因此4步采样没有建立可用的普通编辑基线；本轮 P3 只能验证等预算提案/拒绝流程，不能作为控制器方法价值的有效效能实验。该观察来自保存图的技术审阅，不冒充独立人工质量评测。下一步应先在独立、固定的少量普通编辑样本上校准合理推理步数与视觉质量，再考虑新的学习实验；本轮不追加 GPU。


## 磁盘与清理

本轮证据和最终图件在H盘保留约 237.97 MiB（含源码、checkpoint、失败/拒绝候选、数值witness与图像）。无模型/数据下载或训练checkpoint写入C盘。C盘可用空间审计点为 59.97 → 58.59 GiB，差额未作归因；未删除Codex会话历史、模型或共享运行时。H盘可用 212.59 GiB。图件中间渲染、本轮临时目录和新建bytecode缓存已清理；集群scratch同样不存在，canonical源码SHA与开始时相同。清理及账本证据见 `LOCAL_CLEANUP.json`、`disk_after.json`、`final_resource_receipt.json`。
