# 发给 Codex 的资料包（中文说明）

**先将本文件与 `ARCHITECTURE_DECISION.md`、`CODEX_START_HERE.md` 放在同一代码仓库或交给 Codex。**

直接发送给 Codex 的简短开场指令：

> 请阅读我提供的 `ARCHITECTURE_DECISION.md` 和 `CODEX_START_HERE.md`，严格按照后者的阶段顺序执行。先审计当前项目，特别是 InstructPix2Pix/LoRA brightness pilot、BF16 梯度及 optimizer、采样/概率计算、数据集划分，并给出真实状态。通过最小 smoke test 后，开始**正常图像编辑任务**上的最小 RL 实验；再对比固定规则与自适应 RSI 流程。请实际修改代码、运行可执行的测试/小实验并记录日志，绝不能虚构结果、随意启动长时间 GPU 任务，也不能以 ASR/隐蔽性为训练目标。最后给我一份中文实验状态报告，说明已经运行了什么、有什么结果、还卡在哪里。

## 建议顺序

- **P0**：核对 repo、依赖、GPU、日志，建立可复现的最小运行入口。
- **P1**：修复 BF16/optimizer、采样器、log-prob 和模型 checkpoint 问题。
- **P2**：正常图像编辑基准上的 Frozen vs RL 训练 LoRA，核实结果。
- **P3**：在无害任务上验证 Outer candidate proposal / accept-reject 和 Bandit 是否有额外收益。
- **P4**：把**验证过**的架构和方法与 LaTeX 对齐，保留差异清单。

## 需要纠正原文的两点

- DDPO 不要求每一步都生成独立奖励；它通常使用终局奖励与去噪转移概率来估计策略梯度。
- UCB/Thompson 等 Bandit 并非自动等价于已训练的长期决策 Outer PPO。只用小规模搜索也不能自动证明 RSI 创新。

原始的 `ARCHITECTURE_DECISION.md` 原样附上作为**待核对的研究决策**，而非可直接执行的真值。
