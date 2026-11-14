# ARCHITECTURE_DECISION.md — RSI 图像生成研究框架:架构决策

> 本文件是研究架构决策，不是运行证据。2026-11-11 已依 `CODEX_START_HERE.md` 纠正 DDPO 定义；原始版本完整保存在 `docs/benign_starter_original/ARCHITECTURE_DECISION.md`。当前实现、实际运行和验证范围分别见 `reports/IMPLEMENTATION_AUDIT.md` 与 `reports/MINIMAL_EXPERIMENT_STATUS.md`。
> 安全约束(贯穿全文):ASR / stealth **只作离线 robustness 评测指标**,**绝不**进入任何优化回路;benign image-editing 任务用于一切方法学验证。

---

## 0. 结论摘要

| 组件 | 决定 | 一句话理由 |
|---|---|---|
| Inner RL | **Option A:在 IP2P LoRA 上做 proper DDPO**(多步 policy-gradient) | 匹配「研究一个 RL-trained image generator」的目标,且是现有 score-function pilot 的最小、正确补全 |
| DRaFT | **可选的可微奖励优化对照**,不作本轮主方法 | 它是 differentiable reward fine-tuning；本仓库尚无受控对照证明其相对样本效率，不称为 DDPO |
| 反馈粒度 | 最小实现**只用 L0**(benign:instruction fidelity + content preservation) | 先把 RL/RSI 方法学在无害任务上跑通;learned scoring model / credit network 一律降级为 optional hypothesis |
| Outer / RSI | **regularized proposal + held-out 验证的 accept/reject**(RRSI 式);**不是 PPO** | 独立 outer 交互次数太少,PPO 不成立;写得出 clipped objective ≠ 用得起 PPO |
| RSI 判据 | 以 **held-out / 跨 benchmark 的改进保留率**为准 | 否则无法与 HPO / online reward calibration / evolutionary search 区分 |

---

## 1. Inner RL:A / B / C 评估与选定

### 1.1 三个候选

- **A — 直接 RL 微调条件生成器的 adapter 参数**(DDPO 或 DRaFT 作用在 IP2P 的 LoRA 上)。
- **B — 独立的 decision-level RL policy + 冻结的 editor**(policy 决定指令/强度/区域,生成器不被训练)。
- **C — decision policy 与 renderer 联合优化。**

### 1.2 对照文献的判断

- **DDPO**(Black et al., ICLR 2024):把去噪过程当作 MDP,用 policy-gradient 微调扩散采样器;reward 可为非可微黑盒;方差高、样本需求大。
- **DRaFT**(Clark et al., ICLR 2024):把**可微** reward 的梯度反传穿过采样链到 LoRA；DRaFT-K 截断到最后 K 步，DRaFT-LV 针对 K=1 降低方差。它采用 reward/pathwise gradient，而非本轮的 likelihood-ratio policy gradient；不能将原论文某些基准中的优势直接外推为本项目结果。
- **InstructPix2Pix**(Brooks et al., CVPR 2023):instruction-conditioned editor;base(UNet/VAE/text-encoder)冻结,LoRA 可训。

### 1.3 决定:选 **A**,实现为 **proper DDPO**;**否决 B、C**

**为什么是 A。** 研究目标明确是「研究一个 **RL-trained image generator**」,而非「在冻结生成器之上学一个控制器」。现有 pilot 已经在更新 LoRA,A 的底座已经存在。B 会让生成器本身不被训练,直接与研究目标相悖;在**无害、良态**的 benign reward 下,当初支持 B 的「攻击 reward 使扩散 RL 脆弱」这一前提已被移除,B 的优势随之消失。C 引入两个相互作用的学习信号,违反「保持最小、不要混合互不相容的定义」,仅列为 future work。

**为什么本轮先检验 DDPO 类估计器。** 当前仓库已经存在 score-function 路径，先检验其转移概率、梯度、重复采集/更新和续训行为，可以保留清晰的实现谱系。DRaFT 是可另外比较的 differentiable reward fine-tuning；是否更有效或更省样本，须由独立受控实验决定，不预设结论。

### 1.4 选定形式(DDPO on IP2P LoRA)的完整定义

- **可训练参数 η**:IP2P UNet attention 上的 LoRA;base UNet/VAE/text-encoder 冻结。
- **policy π_η**:LoRA 适配后的去噪器,定义反向扩散的转移 `p_η(z_{t-1} | z_t, c)`。
- **state s_t**:`(源图编码, conditioning c = 指令/CFG, 时间步 t, 当前潜变量 z_t)`。
- **action a_t**:采样得到的去噪转移(到 z_{t-1})。
- **transition**:给定 a_t 由 scheduler 决定;随机性来自 policy 采样。
- **trajectory** τ=(z_T,…,z_0) → 编辑图 x̃。
- **reward(benign,终局)**:`R(x̃, x, 指令) = instruction fidelity + content preservation`;DDPO 允许非可微 reward;**不含 ASR / stealth**。
- **update**:终局奖励 `R(x̃)` 作为去噪轨迹的回报，乘以各随机转移的 `∇η log pη(z_next | z,c,t)` 并求和。**不要求每个时间步独立奖励，也不要求 learned credit network**。可使用与当前 action 独立的 baseline。当前最小实现使用 baseline=0、单次 on-policy score-function 更新，不使用 PPO ratio 或旧轨迹重放。
- **固定缩放**:现有代码把 log density 在 latent 坐标、时间和 batch 上取均值；坐标与时间均值相对轨迹求和只引入固定正比例因子。该值是平均坐标 log density，不能直接冒充联合轨迹概率或用于未经验证的 PPO importance ratio。
- **采样约束**:当前支持 epsilon prediction、unclipped leading DDIM、`eta>0`，并使用 `steps_offset=1,set_alpha_to_one=False` 保持末步方差为正。它与原生 IP2Pix Euler 推理不是同一 sampler；Frozen/Inner 必须使用同一 DDIM 合约。

### 1.5 关于现有 pilot 的如实定性(非常重要)

原附件描述的「一次更新、2 步、batch 10、单 seed」是某次历史 pilot 的自述，不能代替当前代码和 checkpoint 审计。仓库还保留了后续多次更新实现与历史 FP32 运行证据。

**一次 score-function 更新不能自动认证为已验证 DDPO；使用终局奖励也不会使它自动失去 DDPO 的方法归属。** 需要逐项检查真实随机转移、采样/训练 log-prob 一致性、action detach、有限梯度、optimizer 参数归属、多轮新轨迹采集、checkpoint 后的下一步行为，以及独立评估。Null 结果的原因尚未被辨识，不能仅凭一次小实验断言为高方差、BF16 或模型无法学习。

---

## 2. 反馈与评测粒度(保持最小)

三层反馈,**第一版只实现 L0**:

| 层 | 含义 | 成本 | 本框架中的角色 |
|---|---|---|---|
| **L0** | 图像级指标(instruction fidelity、content preservation、LPIPS/SSIM、area) | 秒级,无 victim | **benign Inner RL 的唯一 reward** |
| **L1** | 缩减规模 downstream probe(= 你原来的 mini-batch ASR) | 分钟级 | **可选**的中保真锚点,用于离线研究 L0↔L2 关系 |
| **L2** | 完整 victim fine-tune 后的行为 | 小时级 | **离线** robustness 评测的 ground truth,**绝不作 reward** |

**原则(严格遵守本轮指令):**
- **不**引入 learned scoring model / credit-assignment network,除非出现可识别的实验需求**并**配套验证策略;在此之前它们是 optional hypothesis,不是既成贡献。
- **condition-level 结果 ≠ per-image credit**:victim 级别的一个标量(ASR/Δclean)**无法**自动归因到单张编辑图;任何 per-edit credit 都需单独的可识别性论证与标定,v1 不做。
- VLM backdoor 部分:ASR、FP-ASR、clean-utility、held-out、defensive robustness 全部作为**离线评测协议**,不进入生成优化。

---

## 3. Outer Controller：先验证有界配置搜索，RSI 为待证假设

Outer Controller 可以提出有界训练 recipe 修改。若仅选择学习率，当前证据应称为配置搜索/HPO；不得预先命名为已训练的 Outer PPO 或已验证 RSI。

- **State**:当前 improvement configuration、相关历史反馈、performance diagnostics、剩余评测预算。
- **Action**:对改进机制的一个**有界修改指令**。
- **Transition**:候选修改 → 独立验证 → accept/reject。
- **Reward**:受控预算下、**独立评测**的 learning-efficiency 改进。
- **Memory**:历史决策、评测结果、成本、拒绝原因。

### 3.1 PPO 是否成立:**否**

每一轮 outer 交互都要跑一次(editor 训练 + 评测)的昂贵周期,独立 transition 数量约 O(10)。**能写下 clipped objective 不等于有足够独立轨迹训练它**;replay buffer **不产生**新的独立 transition。对照 AIDE² 与 RRSI:二者的 outer 都是**proposal + 基于 held-out 的 accept/reject / tree-search**,而非 PPO。

**控制器谱系(就低不就高):**
1. **固定规则**(无学习):提出验证后预测改进最大的修订 → gate。
2. **简单自适应 / 不确定性规则**(UCB/Thompson over 小离散修订臂):加探索,非上下文。← **推荐主选**
3. **contextual bandit**:仅当轮数/动作结构足够时。
4. **sequential policy-gradient(含 PPO)**:排除,除非后续真的积累了足够独立轨迹。

### 3.2 与 HPO / online reward calibration / evolutionary search 的区别(诚实界定)

普通 HPO / 进化搜索是在**固定目标**上搜配置;"RSI" 的主张要求被接受的修改**改变了「下一轮改进如何产生」**,**并且**能迁移到 held-out(RRSI 的核心检验)。若只有 in-distribution 增益而 held-out 不保留,那就是 RRSI 所警告的 **benchmark-specific overfitting / 在线标定**,不是 RSI。因此:
- **采纳 RRSI 式正则**:leakage-critic(拒绝编码了任务特定信息的修改)+ 以 **held-out / 跨 benchmark 改进保留率**作为接受判据;
- **采纳 AIDE² 式冻结层**:held-out 评测划分与预算在整个 RSI 过程中**冻结**。

---

## 4. RSI 传递对象与「是否真 RSI」的判据

- **传递对象**:经 held-out 验证后被接受的「对改进机制的修改」(配置和/或 inner 训练 recipe),连同其证据与成本进入 Memory,供下一轮消费。
- **判据(实证,不预设结论)**:
  - 固定机制 vs 周期性重拟合 vs RSI-transfer,在**等预算**下比较;
  - 看 **held-out / 跨 benchmark 改进保留率**是否提升(RRSI),以及达到目标所需的评测预算是否下降;
  - 若周期性重拟合 ≈ RSI-transfer,则新颖性只是在线标定 → 按命名阶梯下滑。

**命名阶梯(诚实回退):**
1. held-out 可迁移地改进了「改进器」 → **recursive self-improvement**(目标)。
2. 只有 in-distribution 改进 → **iterated configuration / recipe adaptation**(砍掉 RSI 词)。
3. 什么都不迁移 → **two-timescale search**。

---

## 5. 已实现 / 已提议 / 已验证 三分账本

> 本轮已读取当前代码并运行专项测试；下面仅记录技术范围，实际指标和作业终态仍以本轮报告与证据为准。

| 项 | 状态 |
|---|---|
| IP2P pinned base + FP32 LoRA/Adam；随机 DDIM；新轨迹逐转移 score-function 更新 | **已实现并在本轮短配置中运行**；47 项 CPU 测试通过 |
| 正常亮化任务 Frozen vs Inner；重复采集与真实 LoRA 更新 | **已运行**；4 次 Inner 更新，不预设正向学习 |
| 正向学习优势、训练种子泛化 | **未证明**；单 seed、4 张评估图只属开发 smoke |
| 多步去噪转移概率 + 终局奖励的 DDPO 类估计器 | **已实现并作解析/回放/梯度检查**；无 PPO，不要求独立逐步奖励 |
| decision 层以外的 L1/L2、learned scoring、credit network | **已提议 / 可选** |
| fixed / fixed-rule / UCB proposal + dev accept/reject | **已实现**；当前仅学习率搜索/HPO，运行终态见报告 |
| RSI improvement-procedure 跨独立 benchmark 传递 | **未验证**；暗化任务仅为同一开发池上的跨任务 smoke |
| BF16 / FP32 backward、Adam 与 checkpoint 下一步一致性 | **本轮 256px/4步/rank2 短配置通过**；不代表历史 BF16 问题全面修复 |

技术测试通过、真实 LoRA 更新、良性编辑效能和 RSI 增益是四种不同证据层级；本轮实测结果以新报告、逐次日志及 checkpoint 为准。未经测量的效能值保持 TBD。

运行后只读复核发现1张训练图存在1像素 padding 裁剪偏差。当前源码已改为复用 Pillow 的 resize/居中规则，49项 CPU 回归通过；原 GPU 运行保留并注明该缺陷，修正后的训练尚未运行。4步输出也未建立可用的图像编辑质量基线，因此不扩大 GPU 实验。

---

## 6. 参考文献(已核实,勿臆造)

- DDPO — Black, Janner, Du, Kostrikov, Levine. *Training Diffusion Models with Reinforcement Learning.* ICLR 2024.
- DRaFT — Clark, Vicol, Swersky, Fleet. *Directly Fine-Tuning Diffusion Models on Differentiable Rewards.* ICLR 2024. arXiv:2309.17400.
- InstructPix2Pix — Brooks, Holynski, Efros. *InstructPix2Pix: Learning to Follow Image Editing Instructions.* CVPR 2023.
- AIDE² — Srikanth, Zhao, Xu, Wu, Jiang (Weco AI). *Recursive Self-Improvement of AI Research Agents.* arXiv:2609.26457, 2026.
- RRSI — *Regularized Recursive Self-Improvement of Agent Harnesses.* arXiv:2609.24972, 2026.
