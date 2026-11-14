# 良性 IP2Pix / LoRA 实现审计

本轮为实际修复与运行，不是计划。固定证据目录：`experiments/benign_ddpo_audit_20261111_v1/remote/`。入口包原文在 `docs/benign_starter_original/`；其架构判断按待验证提议处理。

## 文件审查与修复

|文件|观察到的问题 / 本轮改动|验证依据|
|---|---|---|
|`src/rl/diffusion_policy.py`|父类无条件 CUDA/BF16 autocast，可能令声明 FP32 的调用实际执行 BF16。现在按 editor 精度选择 autocast，输入 dtype 与 base 一致；CPU 路径不强制 CUDA 同步。原 `PilotPolicy` 已有部分精度处理，不能把父类问题误当成历史 BF16 分叉的已证实根因。|新 CPU 精度回归；真实 BF16/FP32 数值 gate|
|`src/rl/diffusion_trajectory.py`|原噪声可被广播，timestep/有限性检查不足；现在拒绝错误 shape/dtype/device、非有限 noise/epsilon/action 和不在 schedule 中的步。保留随机 DDIM 数学、action detach 和原 1e-6 回放阈值。|Diffusers 转移对照、Gaussian density、解析终局奖励梯度回归|
|`src/training/generator_adapter.py`|PEFT 加载只拒绝 unexpected keys，缺失 adapter 张量可能静默保留当前权重。现在先完整核对 adapter 键集、shape、dtype、有限性，再加载并逐张量验证精确值。|缺失/多余键、错误 dtype/shape、NaN 均在修改参数前拒绝|
|`src/mini_pilot/engine.py`|统一精度路径；拒绝空 batch 与非有限 replay error；按实际设备搬移轨迹；检查更新后 LoRA 有限性。checkpoint 增加 SHA-256、optimizer/clipping 合约及原子状态写入；支持 NumPy RNG 树比较；记录实际 optimizer 次数和 Adam dtype。|CPU 小 UNet 下一步恢复；真实预训练模型重复/续训 gate|
|`src/mini_pilot/benign_validation.py`|新增只含良性亮度、边缘结构代理的独立 reward；SSIM 与裁剪率仅作评估。新增固定/固定规则/UCB 搜索、源图配对核查。|奖励方向、UCB 冷启动、固定规则、seed 配对测试|
|`scripts/run_benign_ddpo_minimal.py`、`configs/benign_ddpo_minimal.json`|新增严格限时、固定协议、阶段 gate、完整拒绝记录的良性实验入口；不导入 victim 或安全指标优化器。|Slurm 90945 完成；运行源文件以完整 archive SHA 为准；原 CPU receipt 只覆盖部分测试，复核后已补全后续版本|
|`tests/test_benign_ddpo_audit.py`、`scripts/test_benign_ddpo_audit.py`|新增数值/恢复/奖励/搜索回归，与既有专项测试合并。|47 项通过，0 失败、0 error、0 skip|
|`scripts/benign_cluster_io.py`、`scripts/benign_remote_inventory.py`|只向独立 run 目录部署；保留源码快照、环境清单与实际命令；禁止同 run 二次 GPU 提交。|source_upload.json、submission.json、remote_inventory.json|
|`scripts/audit_benign_crop_posthoc.py`|新增运行后保存像素的 crop 复测，单独记录原始/修正奖励，禁止重写训练日志与 checkpoint。|126 个图像 CPU 复测；无新增模型或 optimizer 调用|
|`scripts/report_benign_ddpo_audit.py`、`scripts/render_benign_audit_diagram.py`|生成中文报告、CSV、文件哈希清单、原始图像配对预览与概念 PDF/SVG。|数字从 JSON/日志重算；图像与 PDF 已视觉检查|
|`ARCHITECTURE_DECISION.md`、`CODEX_START_HERE.md`、`README_FOR_CODEX_CN.md`、`docs/benign_starter_original/`|纳入入口资料，完整保留原附件；仅纠正当前架构文件的定义和证据状态。|原 ZIP SHA-256 和原文副本|
|`docs/benign_ddpo_method_outline.tex`|新增独立方法与 Algorithm 1；原稿未改写。|方法 outline，未声称整稿编译通过|

新 checkpoint schema 为 `benign_pilot_checkpoint_v2`。旧 schema 不具备本轮完整合约/sidecar，新的 `PilotTrainer.restore` 会显式拒绝；旧 checkpoint 文件未修改。仅验证项目自建、可信 checkpoint 的恢复，未实现任意不可信 pickle 的安全加载。

## 算法核查

现有路径不是需要重新发明的单标量优化器：随机去噪转移、终局奖励、逐转移 score-function 和重复更新已经存在。本轮核查其实际执行。`eta=1`，epsilon prediction，leading DDIM，`steps_offset=1`、`set_alpha_to_one=False`；末步方差仍为正。Frozen 和 Inner 使用相同 DDIM；原生 Euler 仅单独复现一次冻结推理，不混入对照。

log-prob 在坐标上取平均、loss 再在时间与 batch 上取平均；相对联合转移 log density 的求和只差固定正比例。没有 PPO ratio、当前 batch 标准化或旧轨迹训练复用；因此不能称 PPO/importance-sampling DDPO。终局奖励可用于所有时间步，不需要每步独立奖励。

本轮奖励为 `亮度目标误差减少 - 0.05*(1-边缘方向余弦)`，亮化目标为源图平均亮度 +0.05，暗化为 -0.05，截到 [0,1]。原运行意图排除白色 padding，但审阅发现训练源图 574996000 使用 floor 居中，与 Pillow 的 round 居中相差1像素，纳入一列 padding、遗漏一列内容；不能声称原训练严格排除了 padding。评估/迁移图的 crop 未受影响，修正后训练会产生什么结果未知。当前源码已改用 Pillow 一致的 round，并补奇数 padding 测试；这是运行后修复，未重跑 GPU。它是光度/结构代理，不是 edited-label ground truth，也不是语义身份认证；SSIM 未进入优化。没有 ASR、隐蔽性、victim 反馈、投毒比例或答案重定向信号。

## 历史材料的边界

`train_lighting_v3.py` 与 `src/mini_pilot/engine.py` 证明后续实现存在，不能继续沿用入口附件「现有代码只有一次更新」的概括。历史 `outputs/reward_debug_20261104T054605Z_s42/internal_pilot/numerical_witness/comparison.json` 记录 FP32 五次更新/恢复 PASS；其 `BF16_fix_claim=false`。旧报告标题或语句不替代当前源文件、日志与 checkpoint。本轮不重算或合并历史安全研究结果。

`E:/main9_revised.tex` 的良性 pilot 是另一配置（512px、rank8、两步、20/4/4），不能把本轮值替换进原表。原文保留 SHA-256，未改写。新增独立方法 outline、Algorithm 1 和概念图；没有把设想写成实验结果。

## 验证边界

BF16 与 FP32 都仅在本轮 256px/4步/rank2、batch2 配置通过。历史 BF16 不一致没有在旧配置上复现或修复认证。本机 Anaconda 在实际 import torch 时出现 OpenMP runtime 重复初始化退出，且缺少 Diffusers；未使用 `KMP_DUPLICATE_LIB_OK` 掩盖问题，实际测试在现有集群环境完成。

“FP32”表示 dtype 与计算路径；使用的源权重仍是固定 revision 的 `variant=fp16`，再上转为 FP32，并非另一套原生全精度权重。原运行没有验证 outer loop 的进程崩溃后恢复，也没有实际接受过正向候选；只验证了拒绝/恢复路径。

运行前 CPU receipt 未包含测试驱动自身和其余6个执行测试模块；完整 archive 包含全部文件，复核已逐字节对应。因此原 receipt 的覆盖范围需保留限定。当前测试驱动已将自身与全部执行模块纳入 SHA 清单。原始运行源码完整展开在 `experiments/benign_ddpo_audit_20261111_v1/executed_source/`，与复核后当前源码分开保存。

仓库开始时已有大量未提交修改；`git_status_before.txt` 与 commit 文件记录基线。未 reset、删除原始实验、修改集群 canonical 源码或干扰其他作业。当前新代码只在单独源快照中运行。


## 运行后修复验证（CPU only）

新增 padding 取整与测试驱动 SHA 覆盖回归后，49 项 CPU 测试通过，零失败/跳过。已用修复后的度量重新检查 126 个保存图像；涉及源图 ['574996000']，各 split 最大 reward 绝对变化为 `{'train': 0.0020111316098415655, 'dev': 0.0, 'test': 1.3877787807814457e-17, 'transfer': 2.7755575615628914e-17}`。这只是已保存像素的 CPU 复测，没有新的模型调用、optimizer 更新或 GPU 作业。训练原始奖励/轨迹/checkpoint 不改写；修复后模型学习效能仍未运行。详细证据在 `experiments/benign_ddpo_audit_20261111_v1/postrun_fix/`。


## 保存图像的视觉检查

配对预览中，电脑图与领带图出现明显纹理化和结构破坏，Frozen 与 Inner 都存在该问题。因此4步采样没有建立可用的普通编辑基线；本轮 P3 只能验证等预算提案/拒绝流程，不能作为控制器方法价值的有效效能实验。该观察来自保存图的技术审阅，不冒充独立人工质量评测。下一步应先在独立、固定的少量普通编辑样本上校准合理推理步数与视觉质量，再考虑新的学习实验；本轮不追加 GPU。
