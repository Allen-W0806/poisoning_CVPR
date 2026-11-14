# RSI-Poisoning 最新完整实验进度报告

**科学结论：NO-GO。存在真实参数更新和多项工程验证；尚无可信的自然图像任务改进、RSI优势、训练后攻击有效性或真实ASR结论。**

- 项目唯一权威根目录：`/storage/home/ydk297/projects/poisoning`。
- 报告编制时间：2026-11-01T16:42:14.342680+00:00；实时调度/心跳证据时间：2026-11-01T16:33:31.641395+00:00。
- 文件：`/storage/home/ydk297/projects/poisoning/reports/RSI_POISONING_FULL_PROGRESS_LATEST.md`；当前对话附件副本：`H:/Project/Poisoning/reports/RSI_POISONING_FULL_PROGRESS_LATEST.md`。
- 本次工作：只读核查Slurm、现有JSON/CSV/日志、Checkpoint文件清单和代码，并汇总分析。未提交GPU/CPU实验作业，未重新运行88099，未改科学目标、损失、阈值、训练守卫或周期任务。未尝试修复或调用Google Drive/OAuth/rclone。
- 证据范围：直接从集群读取938个小型报告/结果/源码文件，约9.94MB；对大Checkpoint仅现场检查存在性/大小并阅读既有比较与校验回执，没有重新执行全张量数值复现。时间统一使用`TZ=UTC`的Slurm查询，避免11月1日夏令时回拨造成歧义。
- 本文“准确率”若无特别说明为VQAv2原始十答案的官方兼容共识得分均值，非二元exact match；对编辑图则只能称“原参考答案一致度”，直到独立确认标签仍有效。

## 1. 交接摘要与状态定义

项目方法框架是图像编辑生成器的内层RL，与基于实验记忆、诊断、配置修订和回滚的外层RSI相结合；Qwen是受控下游VLM，清洁任务表现和训练后安全行为属于不同评价层级。原研究目标以现有CVPR文稿/Algorithm1为准；当前完成的是良性数值、反馈、语义和对照验证，不替代文稿最终任务，也没有完成BadSem复现。本文不替作者补齐尚未定义的科学损失或dataset-to-trajectory信用分配。

| 状态 | 本报告含义 |
|---|---|
| PASS | 明确命名、已执行的检查通过；不能外推为科学有效 |
| FAIL | 实际执行未满足原定检查；保留原始失败和数值 |
| BLOCKED | 前置条件缺失，不能执行或不能作出该解释 |
| NOT TESTED | 未完成相应实验/观测；缺失数值为未知/null，不能写成0% |
| NOT VALIDATED | 有局部结果，但不足以支持广义可靠性或科学主张 |

| 模块 | 已证实的进展 | 当前边界 |
|---|---|---|
| 数据/隔离/人工审阅 | 最新80源32/16/32内部ID和内容分离；160对与32对盲审材料齐备 | 80源均有历史暴露标记；独立有效人工审阅0/32，160池0/160；BLOCKED |
| IP2Pix生成器 | 固定预训练IP2Pix可生成；一个UNet attention-LoRA参数集合，128模块/256张量/1,594,368参数 | 技术有效不代表语义保留或指令成功；完整BF16可靠性FAIL/NOT VALIDATED |
| DDPO/RL | 执行的是DDPO-SF；分类策略为REINFORCE；合成精确/MC梯度及参数更新有证据 | 非生产GRPO；数学合成通过不保证真实图像奖励正确或BF16执行可靠 |
| Reward/归一化 | 原公式与逐项保存值一致；匹配反馈可追溯 | 46%下限饱和、冗余项、no-op偏好、独立语义缺失；科学构念未通过 |
| RSI | 真实memory/版本/Gate/拒绝/回滚闭环；三分支同前缀对照已跑 | 新三分支CI均跨0；旧真实图像比较在干预前已分叉，不能归因 |
| Qwen clean训练 | 3种子、每种16逻辑步；112张量/1,605,632参数实际更新；恢复通过 | 训练后泛化改进不显著；编辑数据训练和真实攻击条件训练未做 |
| Post-training ASR | 只读接口与谱系要求存在 | 真实有效ASR样本/分母没有，ASR=null/NOT TESTED |
| 对照 | 冻结/变换/几何、旧DDPO/RSI、新fixed/普通scheduler/RSI均有对应范围 | 没有通过全部门槛的自然图像四臂完整科学比较 |
| 资源与监控 | 当前无研究GPU运行；88209是唯一两小时后继 | 验证预算2381/3600 GPU秒，余1219秒；不等于项目全历史费用 |

**最新新增知识是负面定位证据：88099的普通重复精确一致，但插桩重复在裁剪前已分叉；88105进一步定位32个原始梯度张量。没有新的正向科学效能结果，也没有数值修复。**

## 2. 最新真实作业与监控

### 2.1 三个重点Job

| Job | Run ID（位于results/） | 开始UTC | 结束UTC | Slurm状态/Exit | GPU秒 | 科学解释 |
|---|---|---|---|---|---:|---|
| 88099 | overnight_supervisor_20261101_v1_P1_ABBA_v1_a0 | 2026-11-01 05:45:30 | 05:47:10 | FAILED / 1:0 | 100 | 完成预定四worker诊断并检出数值不一致；不是提交/加载失败 |
| 88102 | overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a0 | 2026-11-01 05:49:22 | 05:49:24 | FAILED / 1:0 | 0 | 取证脚本漏`results/`前缀；不是新的模型失败 |
| 88105 | overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a1 | 2026-11-01 05:51:09 | 05:51:11 | COMPLETED / 0:0 | 0 | CPU取证与哈希比较完成；未修复GPU数值问题 |

88102的真实异常是`FileNotFoundError`，尝试读取`.../poisoning/overnight_supervisor_.../summary.json`；正确父目录为`.../poisoning/results/overnight_supervisor_.../summary.json`。修复仅作用于CPU路径构造，旧失败目录仍在。88105结论为`PASS_FORENSIC_EXTRACTION_ONLY`。【E21–E23】

### 2.2 自88097起所有已执行的本项目后续任务

下表列出Slurm当前可见记录。完成回调和同步任务是系统工作，不计为新研究结果；本次没有处理同步。88209是未来后继，另列于监控小节。

| Job | 任务/结果解释 | 状态 | 开始UTC | 结束UTC | 秒 | GPU秒 | Exit |
|---|---|---|---|---|---|---|---|
| 88097 | 32对图像/来源/质量与已有adapter CPU核验；工程PASS，人工审阅0 | COMPLETED | 2026-11-01T05:45:13 | 2026-11-01T05:45:23 | 10 | 0 | 0:0 |
| 88098 | 有限afterany完成回调；非新增周期监控 | COMPLETED | 2026-11-01T05:45:30 | 2026-11-01T05:45:30 | 0 | 0 | 0:0 |
| 88099 | O/I/I/O同卡BF16诊断；数值FAIL，保留首次差异 | FAILED | 2026-11-01T05:45:30 | 2026-11-01T05:47:10 | 100 | 100 | 1:0 |
| 88100 | 有限afterany完成回调；非新增周期监控 | COMPLETED | 2026-11-01T05:47:40 | 2026-11-01T05:47:41 | 1 | 0 | 0:0 |
| 88101 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T05:49:19 | 2026-11-01T05:49:23 | 4 | 0 | 0:0 |
| 88102 | CPU取证脚本缺少results/路径前缀；工程FAIL | FAILED | 2026-11-01T05:49:22 | 2026-11-01T05:49:24 | 2 | 0 | 1:0 |
| 88103 | 有限afterany完成回调；非新增周期监控 | COMPLETED | 2026-11-01T05:49:40 | 2026-11-01T05:49:40 | 0 | 0 | 0:0 |
| 88104 | 有限afterany完成回调；非新增周期监控 | COMPLETED | 2026-11-01T05:51:06 | 2026-11-01T05:51:07 | 1 | 0 | 0:0 |
| 88105 | 路径修复后CPU取证PASS；非数值修复 | COMPLETED | 2026-11-01T05:51:09 | 2026-11-01T05:51:11 | 2 | 0 | 0:0 |
| 88106 | 有限afterany完成回调；非新增周期监控 | COMPLETED | 2026-11-01T05:51:22 | 2026-11-01T05:51:22 | 0 | 0 | 0:0 |
| 88113 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T08:32:50 | 2026-11-01T08:32:53 | 3 | 0 | 0:0 |
| 88125 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T10:32:52 | 2026-11-01T10:32:56 | 4 | 0 | 0:0 |
| 88130 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T12:32:52 | 2026-11-01T12:32:55 | 3 | 0 | 0:0 |
| 88134 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T14:32:53 | 2026-11-01T14:32:56 | 3 | 0 | 0:0 |
| 88172 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T16:32:52 | 2026-11-01T16:32:55 | 3 | 0 | 0:0 |
| 88193 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T14:45:19 | 2026-11-01T14:45:20 | 1 | 0 | 0:0 |
| 88194 | 原两小时巡检/手动核验；非模型实验 | COMPLETED | 2026-11-01T14:47:53 | 2026-11-01T14:47:57 | 4 | 0 | 0:0 |
| 88198 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | FAILED | 2026-11-01T15:06:50 | 2026-11-01T15:06:55 | 5 | 0 | 2:0 |
| 88199 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T15:18:07 | 2026-11-01T15:18:54 | 47 | 0 | 0:0 |
| 88200 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T15:19:24 | 2026-11-01T15:19:45 | 21 | 0 | 0:0 |
| 88201 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | FAILED | 2026-11-01T15:19:54 | 2026-11-01T15:21:09 | 75 | 0 | 2:0 |
| 88202 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T15:26:37 | 2026-11-01T15:26:46 | 9 | 0 | 0:0 |
| 88203 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T15:26:54 | 2026-11-01T15:27:56 | 62 | 0 | 0:0 |
| 88204 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | COMPLETED | 2026-11-01T15:28:22 | 2026-11-01T15:28:46 | 24 | 0 | 0:0 |
| 88205 | 历史CPU同步/授权任务；本报告不处理、不重试，非研究结果 | FAILED | 2026-11-01T15:28:52 | 2026-11-01T15:30:08 | 76 | 0 | 2:0 |


88099之后没有新的研究GPU实验。后续有CPU取证、原监控和历史CPU同步任务；不能将它们的COMPLETED数量当成科学实验成功数。

### 2.3 原两小时监控仍工作

| 项目 | 最新现场证据 |
|---|---|
| 安装方式 | 原`slurm_cpu_successor`，enabled=true，installed=true，interval_hours=2 |
| 最近巡检 | Job 88172；COMPLETED，0:0 |
| 最近成功心跳UTC | 2026-11-01T16:32:55.456718+00:00 |
| 唯一已注册后继 | Job 88209；PENDING |
| 下次计划UTC | 2026-11-01T18:32:35+00:00（纽约EST 2026-11-01T13:32:35-05:00） |
| 连续监控失败数 | 0 |
| 当前研究GPU RUNNING/PENDING | 0 / 0 |
| 重复监控 | 未安装；现场只有一个属于本项目tag的周期后继 |

实时路径：`/storage/home/ydk297/projects/poisoning/reports/monitoring/scheduler.json`；`reports/overnight_supervisor/heartbeat.json`；`reports/overnight_supervisor/history.jsonl`；`experiments/overnight_supervisor_history.jsonl`（后三项均相对于上述项目根目录）。当前阶段仍为`BLOCKED_EVIDENCE_MONITORING`、科学Gate `NO-GO`；巡检成功并不表示科研门槛通过。旧`paused_at`是历史字段，不能覆盖当前enabled、真实COMPLETED巡检和PENDING后继的证据。

## 3. 数据集准备、隔离与Semantic Review

### 已完成/实际执行

使用项目缓存VQAv2 `train2014-2017-04-26`问题、十名标注者答案和COCO train2014图像。题目/原答案保持原始值；训练监督为预先规定的标准化多数答案（至少6/10一致），不能据模型输出改标签。已有官方归档join核验1040行：最新80记录加960个匹配条件记录，原问题、image ID及十答案一致。【E01、E38】

| 数据版本/池 | 数量与用途 | 已发现问题 | 可以声称什么 |
|---|---|---|---|
| synthetic_v1 | train/validation/test各10 | train/test实际内容重复7；train/val重复5；val/test重复5 | 仅工程fixture；不能支持泛化 |
| vqav2_seed42_v1 | train1000、validation200、test300；另fixture10 | train/test ID分离但实际内容SHA重叠1；fixture10均在train | 历史泄漏FAIL保留 |
| vqav2_content_v2 | train999、development104、gate96、test301 | 内部ID/内容零交集；已有开发暴露 | 内部内容隔离PASS，不等于未见测试 |
| 匹配反馈池 | pilot24、main160，6种条件 | 起始选样时排除已知项目暴露；后来已多次分析/复用 | 当时来源筛查有效；如今是已打开开发证据 |
| Oct31最新80源 | train32、development16、assessment32 | 内部ID/SHA零交集，但80/80有历史暴露标记 | source/content-disjoint内部拆分；不是真正独立评估 |

最新assessment32中，4源有已执行早期推理暴露，10源出现在旧已打开test清单，4源与既往训练或训练清单有交集。上述计数可以重叠，不能相加为唯一泄漏数；清单出现也不等于已证实发生optimizer更新。dHash距离筛查不排除所有近重复，基础模型预训练污染未知。【E12、E13】

### 人工与标签有效性

| 项目 | 实际值/状态 |
|---|---|
| 全160对技术图像有效 | 160/160；不是160对语义保留 |
| 全160对独立人工审阅 | 0；语义保留率/标签改变率/歧义率均未知 |
| 预选盲审子集 | 32对；selection seed20261031032 |
| 子集manifest SHA256 | 33b5cce366a2d0a734951ef61b4ea842e91c4f2c08bf4f975553a56e6d227b1e |
| 当前真实双审/裁决 | 未发现有效提交；仅reviewer_1、reviewer_2、adjudication空模板 |
| 最新CPU核验 | 88097；32源图对哈希、官方题答关联、几何重建PASS；new_model_calls=0 |
| 32对对齐SSIM均值 | 0.7659552108；仍不能认证语义 |

盲审工具位于`/storage/home/ydk297/projects/poisoning/results/next_validation_20261101T0040Z_v1/semantic_review32/`；含`INSTRUCTIONS.md`、`blinded_manifest.json`及双审和裁决模板。现有160图界面/队列在`results/matched_feedback_20261028_v2/semantic_review_queue.csv`及其main/contact_sheets目录。盲审32对来自既有C4主池160，**不是Qwen最新assessment32的同一组数据**，不能混用两者样本量或结果。两名真实独立审阅者需分别标记场景、对象/数量/身份/颜色、空间关系、文字、VQA答案保留和指令执行；分歧单独裁决。助手筛图、同Qwen判断和SSIM均不能代替独立标签。【E14–E16】

**方法有效性：BLOCKED。下一步：完成既有32对双审；重新审计并预封存未用于训练/反馈/选择的评估源。旧未见声称不能因ID分离而自动通过。**

## 4. InstructPix2Pix Image Generator

### 模型、参数与真实更新

模型是`Timbrooks/InstructPix2Pix@31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`的项目固定缓存；FP16缓存变体、BF16运行。历史良性生成设置为256×256、10去噪步、guidance7.5/image_guidance1.5、math SDPA。image instruction和VQA question为分离变量；生成器实际编码图像和instruction，不把VQA question/history/budget一并作为已实现条件。

同一editor.pipe/policy.pipe共享UNet。一个ψ是一个LoRA参数集合，不是一个张量，也不是另有独立φ策略模型。rank8、alpha16；UNet自/交叉attention的`to_q/to_k/to_v/to_out.0`全部覆盖；底座UNet、VAE、文本编码器冻结。【E01、E02】

| LoRA目标 | 模块数 | A/B张量数 | 参数元素 |
|---|---:|---:|---:|
| to_q | 32 | 64 | 399,360 |
| to_k | 32 | 64 | 397,824 |
| to_v | 32 | 64 | 397,824 |
| to_out.0 | 32 | 64 | 399,360 |
| 合计 | 128 | 256 | 1,594,368 |

生产optimizer一个参数组：只拥有上述trainable LoRA；AdamW LR1e-5、betas(.9,.999)、eps1e-8、weight_decay0；constant LambdaLR，梯度范数cap1。Qwen的1,605,632参数以及不同tiny-UNet参数数目属于别的模型，不能混写。

### 已执行的生成/训练范围

| 实验 | 实际结果 | 解释 |
|---|---|---|
| 早期冻结编辑sanity | 3源×[42,42,123]共9输出；同seed像素精确、异seed输出变化；修正版平均生成0.903741秒/图，不含加载/哈希 | 有限确定性和技术有效PASS |
| 160源冻结C4 | 160/160可解码RGB；模型参数前后hash相等；无optimizer更新 | 不证明指令/语义保留 |
| Phase4真实预训练LoRA | RL、RSI各2seed、各8步，共32个训练更新；初末LoRA实际改变，冻结权重不变 | 真实参数优化存在，科学奖励/干预可比性未通过 |
| 88084局部VJP | 72次孤立比较，1次常数系数optimizer更新，256张量变化 | 不是自然图像学习验证 |
| 88099四worker | 各1个从同一已保存状态继续的常数系数更新；普通重复精确、插桩重复失败 | 数值Gate FAIL |

历史Phase4 RL42/123初末LoRA最大变化约7.986e-5/7.982e-5；RSI42/123约5.991e-5/5.983e-5。梯度有限、权重移动不等于编辑能力提高。冻结生成时无权重更新，诊断时常数reward=1也不是成功学会任务。

**当前问题：指令成功率与语义保留率未知；完整普通BF16重启未验证。下一步：双审现有图对及定位数值根因，不能开始完整预训练生成器科研训练。**

## 5. DDPO / Reinforcement Learning

### 方法识别与数学检查

执行核心`src/rl/diffusion_rl_trainer.py::DiffusionRLTrainer.update`是单遍on-policy **DDPO_SF_v1（score-function）**。Gaussian DDIM转移为动作，action与terminal reward脱离梯度；坐标/时间/逻辑batch平均。分类`src/rl/trainer.py`是REINFORCE。保存old logp用于1e-6容差的behavior replay检查，不等于PPO ratio或GRPO；生产路径没有组均值/标准差归一化、PPO clipped surrogate、KL/reference anchor。【E01、E03、E37】

保留实现的损失为 `L = -(1/N) Σ_i R_i (1/T) Σ_t mean_coordinate(log pψ(action_it | state_it))`。固定baseline0、scale1；`fixed_advantage`返回reward，不用当前batch统计量。梯度裁剪与reward裁剪是不同操作。没有在本次改变该科学损失。

| 实际测试 | 结果 | 适用范围 |
|---|---|---|
| categorical logits精确枚举 vs autodiff | 最大误差2.7756e-17 | 有限状态解析一致性PASS |
| 现有EditPolicy精确枚举 | 最大误差3.4694e-17 | 相同有限环境PASS |
| 32独立seed MC，N=65,536 | relative RMSE0.006801/0.006102；误差斜率−0.4685/−0.4723 | 噪声随N约N^-1/2下降，支持该fixture估计一致性 |
| DDPO Gaussian transition旧小样本 | N=65,536 relative RMSE约0.227693，原10%门槛FAIL | 保留，不能删去失败 |
| 同定义更大预定样本 | N=1,048,576 relative RMSE0.0575715，z2.4102，斜率−0.4843 | 大样本解析参考PASS，不证明BF16完整网络可靠 |
| 现有EditPolicy更新 | 27个head参数；head.weight/bias约0.01变化；encoder hash不变；checkpoint/下一步恢复PASS | 参数传播工程验证 |
| tiny随机UNet真实DDPO | 4,096 LoRA参数实际更新；冻结参数不变；固定轨迹下一步恢复PASS | 小模型接线，不能替代预训练IP2Pix |

独立group reference有零方差处理、detach reward/advantage/old logp、初始ratio=1、双符号clip、梯度裁剪及KL行为检查，并枚举全部27个三动作组。其surrogate梯度约[-.316157,-.003783,.319940]，原期望reward梯度约[-.177125,-.022902,.200027]；二者目标不同，不能把组标准化估计器称为原始期望reward的无偏梯度。REINFORCE在该合成任务从0.433333升至约0.997；组reference约0.9996，但更新/epoch预算不同，不能据此宣称GRPO优越。【E03、E04】

当前`src/rl/phase5_backend.py::require_training_reward`即使收到`training_eligible=True`也拒绝未释放科学reward；其SHA256仍为`73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712`。【E31】

**方法有效性：合成数学与更新接线PASS；自然图像真实学习效果NOT VALIDATED。下一步：先解决数值与reward构念，复用已有梯度合成结果，不重跑同一smoke。**

## 6. 最新BF16数值问题：88099、88102、88105

### 6.1 受控设置与四worker结果

88099在同一A10080GB `epyc11`上，四个新进程依次ordinary0 / instrumented1 / instrumented2 / ordinary3，共一个100 GPU秒分配。它是B2/T2、常数内容无关reward1的Checkpoint诊断，不是历史B10/T10科学训练；不加载target VLM，不使用攻击/ASR反馈。

同一硬件UUID/driver、Torch2.6.0+cu124、CUDA12.4、cuDNN90100，deterministic开启、TF32关闭、CUDNN benchmark关闭、CUBLAS_WORKSPACE_CONFIG=:4096:8。初始Checkpoint、forward输出、pre-update RNG及数据/随机轨迹控制匹配。插桩模式在clip前复制raw gradients到CPU；这种同步/张量生命周期改变本身可能影响执行，不能当作修复。

| 比较 | 初始/forward/RNG | 裁剪后梯度 | Adapter/Adam | 结论 |
|---|---|---|---|---|
| ordinary0 vs ordinary3 | 完全一致 | bitwise一致 | 完全一致 | 本次普通重复PASS，非一般性可靠证明 |
| instrumented1 vs instrumented2 | 完全一致 | 不一致 | 不一致 | 插桩路径内重复FAIL |
| ordinary0 vs instrumented1 | 完全一致 | 完全一致 | 完全一致 | 本次这一对匹配 |
| ordinary3 vs instrumented2 | 完全一致 | 不一致 | 不一致 | 不能简单归因为ordinary与instrumented标签 |

四worker的loss同为−2.1162105203，reward均1，behavior replay最大差异0，256/256 trainable张量都有变化；这些局部worker检查均PASS，但跨worker数值Gate FAIL。普通0、插桩1、普通3的preclip norm为1.8282486200；插桩2为1.8282474279。【E21】

### 6.2 CPU取证把最早“观测到”的差异推进到clip之前

| 指标 | 88099保存证据/88105 CPU提取值 |
|---|---|
| 最早已观测阶段 | `raw_backward_before_clipping` |
| 原始梯度不同张量 | 32/256，均finite |
| raw最大绝对差 | 1.3751909136772156e-5 |
| raw整体relative L2 | 4.563635686017952e-5 |
| postclip最大绝对差 | 7.517635822296143e-6 |
| postclip整体relative L2 | 4.563268658940317e-5 |
| Adam exp_avg不同张量 | 256 |
| Adam exp_avg_sq不同张量 | 256 |
| 首个注册顺序不同参数 | down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight |
| 该首参数形状/不同元素 | [8,320]，2559元素不同 |
| 该首参数maxabs | 1.8553691916167736e-10；这不是全局最大值 |
| 时间上第一个CUDA primitive | NOT IDENTIFIED |

“参数注册顺序第一个不同张量”不是“反向执行时间上第一个出错算子”。不能从层名断言to_q局部matmul就是根因。整体norm和Adam结果可以传播差异，但尚无完整因果分解。88105只读取已有张量/校验数据，没有新GPU实验。【E22、E23】

### 6.3 已排除与未排除

| 原因/假设 | 证据与当前判断 |
|---|---|
| 不同GPU/软件/源码 | 四worker元数据匹配；不能解释该组已观测差异 |
| 不同初始adapter/optimizer/scheduler/RNG/forward | 对该四worker已保存边界匹配，排除这些边界的差异解释 |
| clip是唯一差异源 | raw在clip前已不同，故排除“仅由clip产生”；不排除其传播/缩放作用 |
| 仅ordinary对instrumented的固定路径效应 | 插桩路径内部也失配，且ordinary0与插桩1相同，不能如此归因 |
| activation checkpointing是唯一根因 | 旧86853中enabled/disabled resumed分支相同；该fixture不支持唯一归因，不代表所有图都排除 |
| 局部LoRA线性VJP错误 | 88084孤立72次局部比较重复一致；FP64 CPU解析参照最大relative error2.3793e-7；只支持局部性质 |
| 上游BF16 attention/fan-in、多次backward累积、kernel调度/allocator/生命周期 | 仍是假设；未定位唯一primitive，无确认因果修复 |

历史86853普通恢复FAIL、86856插桩witness PASS、88016普通/插桩不稳定、新一轮两路径都PASS以及88099再次插桩FAIL均保留；结果呈间歇/上下文依赖，不能挑选一次PASS覆盖历史FAIL。【E25、E21、E24】

### 6.4 修复与训练资格

**实际数值修复：没有。** 已修复的是88102的CPU路径错误；没有换损失、放宽bitwise检查、关闭断言或靠重复直到PASS。局部VJP PASS、普通路径一次精确，都不释放完整BF16/B10/T10训练资格。当前结论为FAIL/NOT VALIDATED，完整生成器训练BLOCKED。

原始证据目录：`/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P1_ABBA_v1_a0/`，四子目录各有`forward.pt`、`gradients.pt`、`restored_before/training_state.pt`、`after/training_state.pt`。8个training_state文件每个19,461,170字节；ordinary gradients各6,446,778字节，instrumented gradients各12,893,498字节；本次现场确认存在，完整比较以原回执为依据。CPU输出在同根`results/overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a1/summary.json`及`named_raw_gradient_comparison.csv`。

## 7. Reward Function、指标与Normalization

### 7.1 实际定义与单位

旧科学代理reward保留为`clip(S + .5*A - .2*D - .1*C - U, -1, 1.5)`；U只在verified且S=1时为0，其余为1。实现默认权重与历史这批数值一致，但旧caller未显式使用YAML配置；“改YAML即可改实际reward”的假设不成立。【E09、E11、E30】

| 指标/量 | 单位和实现 | 解释边界 |
|---|---|---|
| S | 每图-question语义proxy；identity可验证1；同模型暂定.5、不验证 | 不是独立语义标签 |
| A | 原/编辑预测标准化一致，且参考得分均>0 | 不是独立事实保留或ASR |
| D | 每图RGB/255；MAE和SSIM；D=clip((MAE+(1−SSIM)/2)/2,0,1) | 图像相似不等于成功完成指令 |
| C | 每轨迹去噪步数/30；10步为1/3 | 不是实测GPU秒；sequential CFG可每步3个UNet调用 |
| VQA共识分 | 每答案十份leave-one-out参照，后做样本均值 | 编辑后标签有效前只称原参考一致度 |
| exact_any_reference | 标准化整段答案是否匹配任一参照，另行保存 | 不替换共识分，不按正确标签提取答案 |
| logp/terminal coefficient | 每latent转移/每完整trajectory | 坐标平均log density不是VQA概率 |
| batch mean | 固定逻辑batch的日志汇总 | 不是把一个dataset score广播成每图reward |
| normalization | reward分量clip；fixed baseline0/scale1；梯度cap1 | 生产无group mean/std，非GRPO |
| dataset/post-training metrics | 数据集/受训模型谱系/seed层面 | 不能等同逐图或逐trajectory信用分配 |

SSIM可用；C5几何重建160/160精确。C4与C5按256 RGB、白padding、BICUBIC及全画幅对齐计算，160均值0.7340322949；32盲审子集0.7659552108。uint8/data_range255与float32/data_range1可有约1e-6数值差，必须记录dtype，不能混用未对齐原图。lpips/piq/torchmetrics均未安装，校准LPIPS权重未验证，LPIPS=null；AlexNet分类权重不是LPIPS替代品。【E16、E17】

### 7.2 历史500条reward的构念缺陷

| 观察量 | 实际数值 |
|---|---:|
| 总观测 | 500（train320、development180） |
| 原保存算术重算 | 500/500一致 |
| S=.5A | 500/500 |
| U/C | 恒为1 / 1/3 |
| reward均值/样本方差 | −0.495511732 / 0.217368272 |
| reward范围 | [−1, −0.0373395967]，全部负值 |
| 下限饱和 | 230/500=46%；上限0 |
| 12位小数不同reward值 | 260 |
| 成对tie比例 | 0.211190381 |
| 中心化五分量矩阵rank | 2 |

S/A重复同一判断，U/C不能排序，floor censoring吞掉部分D变化。identity即使旧caller丢失A credit仍得0.9666667，高于该批所有changed reward；应有credit的诊断参考为1.4666667。这是no-op偏好和旧接线缺陷，不能通过换权重悄悄解决。现有v2诊断接口对verified pixel identity credit、执行/记录配置一致性、重复逻辑source拒绝等4项PASS，但`training_eligible=False`，旧历史caller与旧reward没有被回写“修正”。科学reward构念仍NOT VALIDATED。【E09、E11、E34】

### 7.3 十个独立原图诊断批：可复现，但不能代表policy比较

160独立源，10批×16、seed42/123/3407共480条原图评估，无missing/invalid；所有160原始答案跨3seed相同。有效独立样本量为160，不是480。下表列seed42，其余seed对应分数相同。【E05、E06】

| 批ID（seed42） | n | 均分 | 批内样本方差 | 描述性95%CI | 无效/缺失 |
|---|---|---|---|---|---|
| 0 | 16 | 0.83750 | 0.0931667 | [0.67500, 0.96875] | 0/0 |
| 1 | 16 | 0.89375 | 0.0672917 | [0.75000, 0.98750] | 0/0 |
| 2 | 16 | 0.51250 | 0.2465000 | [0.28125, 0.74391] | 0/0 |
| 3 | 16 | 0.86875 | 0.0916250 | [0.70625, 1.00000] | 0/0 |
| 4 | 16 | 0.80000 | 0.1226667 | [0.61250, 0.95000] | 0/0 |
| 5 | 16 | 0.84375 | 0.0946250 | [0.68125, 0.97500] | 0/0 |
| 6 | 16 | 0.85625 | 0.0892917 | [0.69375, 0.98125] | 0/0 |
| 7 | 16 | 0.87500 | 0.1166667 | [0.68750, 1.00000] | 0/0 |
| 8 | 16 | 0.82500 | 0.1100000 | [0.65000, 0.95625] | 0/0 |
| 9 | 16 | 0.78125 | 0.1482917 | [0.58750, 0.95000] | 0/0 |


批间均分方差0.0120421，pooled批内方差0.1180125；批均值范围0.5125–0.89375。保留类别构成的composition permutation p=.07179，0/45探索性批对经Holm后显著。同一冻结模型在不同题目批上的差异可能来自难度/构成，不能解释为GRPO候选策略差。原图平均推理延迟0.211964秒；计时合计101.742567秒，不含完整分配开销。

### 7.4 匹配160源×六条件：真实负面描述结果

| 条件 | 变换 | 原参考一致度 | 95%CI（语义clean再分析口径） | 得分1占比 |
|---|---|---:|---|---:|
| C0 | 原图 | 84.875% | [79.62%,89.69%] | 80.00% |
| C1 | 像素identity | 84.875% | 同C0 | 80.00% |
| C2 | brightness×1.03 | 85.0625% | [79.88%,89.88%] | 80.00% |
| C3 | contrast×1.03 | 85.0625% | [80.00%,89.81%] | 80.00% |
| C4 | 冻结IP2Pix良性lighting编辑 | 64.9375% | [58.00%,71.69%] | 58.75% |
| C5 | 仅IP2Pix几何resize/padding | 76.4375% | [70.37%,82.44%] | 70.625% |

全部原始accepted记录1032，主分析960=160×6，重复记录不增加n。C4−C0=−19.9375pp，配对95%CI[−26.56,−13.62]pp；C4−C5=−11.50pp，CI[−17.19,−6.19]pp。当前两比较探索family Holm p均约.000200；父报告四个预注册比较family对C4−C0为.000400。CI端点因bootstrap seed略有不同，原始点估计不变，不能择优混用family。【E08、E10】

按相同source划分的十批如下；它们仍是不同难度source块，不是十个GRPO候选。

| 匹配源批ID | n | C0原图 | C4 IP2Pix | C5几何控制 | C4−C0 | 配对95%CI |
|---|---|---|---|---|---|---|
| 0 | 16 | 0.87500 | 0.68750 | 0.76875 | -0.18750 | [-0.43750, +0.06250] |
| 1 | 16 | 0.86875 | 0.51875 | 0.65000 | -0.35000 | [-0.55000, -0.16875] |
| 2 | 16 | 0.93750 | 0.66250 | 0.76875 | -0.27500 | [-0.50000, -0.07500] |
| 3 | 16 | 0.77500 | 0.61250 | 0.61250 | -0.16250 | [-0.34375, +0.00000] |
| 4 | 16 | 0.93750 | 0.81250 | 0.93750 | -0.12500 | [-0.31250, +0.00000] |
| 5 | 16 | 0.91250 | 0.51875 | 0.78125 | -0.39375 | [-0.64375, -0.18125] |
| 6 | 16 | 0.86250 | 0.75625 | 0.84375 | -0.10625 | [-0.25625, +0.00000] |
| 7 | 16 | 0.72500 | 0.55625 | 0.66250 | -0.16875 | [-0.35625, +0.00000] |
| 8 | 16 | 0.78750 | 0.75000 | 0.81250 | -0.03750 | [-0.18750, +0.07500] |
| 9 | 16 | 0.80625 | 0.61875 | 0.80625 | -0.18750 | [-0.43750, +0.06250] |


C0–C4组内均值方差0.04513125；113/160组（70.625%）五候选全部tie，仅47/160（29.375%）有非零差异。C4对C0为3胜/116平/41负；C4对C5为4胜/129平/27负。选取6源×6条件的36个候选case重复精确，是有限子集复现，不是全160源重复。

自动flag102/160（darkening81、SSIM<.7为52、大亮度变化16；可重叠）仅用于审阅优先级。它们不是102个真实语义失败。C4相对C0的降低包括几何影响与内容影响；C4−C5分离了附加生成影响，但不能区分合法外观变化、物体改变或标签损坏。**这是一项冻结推理的负面描述性结果，不是训练后攻击成功或ASR。**

**下一步：先独立审阅，保留全attempt结果，再在真实有效标签子集上做预先规定的次级分析；先澄清原目标对指令成功、语义、信用分配的定义，不擅改损失。**

## 8. RSI Memory、Diagnosis、Revision与Rollback

| 层次 | 真正实现与跑过的内容 | 尚不能声称 |
|---|---|---|
| 历史自然图像Phase4 | update4后LR1e-5→5e-6；memory/配置记录 | 不代表自主改loss/架构/数据或通用RSI |
| 通用controller API | batch10/20/32、LR范围、uniform/easy_first、版本与拒绝 | API允许不等于这些修订均已被自然图像实验采用 |
| 合成完整性harness | 两次固定/无修订score.5→加content_hash检查score1；无效/回归候选拒绝；rollback H0000 | 是故意构造的integrity oracle，不是自主科研能力 |
| 单seed tiny联合闭环 | 固定与RSI，持久memory、同分拒绝、真实磁盘rollback；Gate差3.5914e-6，CI[−9.6313e-6,1.6935e-5] | 单seed/在Gate上选择、旧rollback借fixed/final设计不足以证明独立优势 |
| 修正版三分支 | fixed、ReduceLROnPlateau、RSI；3seed、各4步、独立重放同前缀、自己的prefix回滚 | 只是受控tiny FP32良性任务，不是预训练自然图像RSI效能 |

最新三分支协议使用与前一tiny实验相同的无害`cos(first latent coordinate)` terminal objective；每arm2步共同前缀+2步后续、batch10、T2。新前缀在参数、optimizer/scheduler、CPU/CUDA/NumPy/Python RNG、梯度、预测、contract/version/consumed IDs上做严格核验；磁盘initial/prefix交叉比较12项相等。不能用新FP32修复回溯宣布旧BF16已可比。【E26、E27】

| seed | 分支 | 选定测试均分 | 候选→保留LR | 4次更新/实际NFE | 验证/测试查询 | 实测持有GPU秒 |
|---|---|---|---|---|---|---|
| 42 | fixed | 0.157166216697 | 1e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.6500 |
| 42 | adaptive_scheduler | 0.157166216697 | 1e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.0253 |
| 42 | rsi | 0.157163351309 | 2e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.0457 |
| 123 | fixed | 0.083919592202 | 1e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.0083 |
| 123 | adaptive_scheduler | 0.083911675261 | 5e-06 → 5e-06 | 4 / 320 | 48 / 32 | 15.0142 |
| 123 | rsi | 0.083911675261 | 5e-06 → 5e-06 | 4 / 320 | 48 / 32 | 15.0429 |
| 3407 | fixed | 0.139420322026 | 1e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.0151 |
| 3407 | adaptive_scheduler | 0.139420322026 | 1e-05 → 1e-05 | 4 / 320 | 48 / 32 | 15.0139 |
| 3407 | rsi | 0.139426963869 | 2e-05 → 2e-05 | 4 / 320 | 48 / 32 | 15.0279 |


RSI seed42提案LR2e-5被拒绝，恢复其自身prefix，retained LR1e-5；seed123提案5e-6接受，结果与普通scheduler相同；seed3407提案2e-5接受。只有明确执行的LR/版本变化有实证，未展示通用自主科学修订。

| 对比 | 3seed均值差 | crossed seed×condition bootstrap95%CI | 结论 |
|---|---:|---|---|
| RSI−fixed | −1.38016185e-6 | [−1.13167674e-5,8.26225732e-6] | CI跨0，不支持优势 |
| RSI−普通scheduler | +1.25881828e-6 | [−5.28001692e-6,9.22750623e-6] | CI跨0，不支持额外贡献 |

每arm相同4 optimizer调用、320实测forward NFE、48验证查询+32测试查询（含相应replay计费口径）；每arm GPU-held约15.0–15.65秒，九arm合计135.8434秒。完整成功Slurm88032分配144秒，含额外开销；初次88029也分配147秒，因scheduler state的−inf sentinel无法严格JSON序列化导致三分支组件不完整，其失败被保存并计费。该job整体COMPLETED不代表内部每个组件PASS。序列化修复未改变梯度、reward、检查门槛或seed。

不同分支实测耗时稍异，且同卡顺序执行；不能用这些小差别宣称RSI加速。拒绝候选消耗也纳入预算。**RSI工程闭环PASS，RSI科学优势NOT VALIDATED。**

## 9. Qwen3-VL Clean / Modified Training

Pinned `Qwen/Qwen3-VL-2B-Instruct@89644892e4d85e24eaac8bacfd4f463576704203`；BF16底座+FP32语言attention LoRA rank8/alpha16/dropout0，28层q_proj/v_proj，112个A/B张量、1,605,632参数；visual与底座冻结。response-only监督交叉熵，不以RL/攻击reward训练。AdamW1e-4、wd0、accumulation2、constant scheduler、norm cap1。每seed16逻辑步，另1次下一步replay调用。【E18、E19】

| 批次/Job | 数据范围 | before→after共识分 | 实际证明 |
|---|---|---|---|
| 早期86807 | 历史development32，1seed | 94.6875%→94.6875%，0答案变化 | 有限clean LoRA工程PASS；早期fresh object未断言next-step梯度 |
| 86846/86848/86851 | 起始按项目暴露排除的共同32原图；3独立初始化seed | 三seed均90%→90%；paired delta0 | 全部112张量更新、frozen hash不变、fresh-object next-step与预测reload PASS |
| 87921 / seed42 | 最新assessment32，历史开发暴露 | 78.4375%→78.4375%；delta0 | 16步、17实际optimizer调用含replay；无答案变化 |
| 87921 / seed123 | 同32源 | 78.4375%→80.625%；+2.1875pp | 改1题；paired95%CI[0,+6.5625]pp |
| 87921 / seed3407 | 同32源 | 78.4375%→80.625%；+2.1875pp | 与seed123改同一题，非另一个独立成功源 |
| 88016后续复评 | 复用已有adapter，新对象原图评估 | 旧结论重现；无新增训练 | integrity/复评PASS，不是新科学实验 |

三seed早期90%结果的before/after95%CI均[81.5625%,96.5625%]；共享图像有效n32，不是96。Oct31的两个改善seed都将question362793004由yes（score.3）改为no（score1），因此差值=.7/32；exact双侧配对sign-flip p=1。零差值seed的经验bootstrap[0,0]也不证明总体等价。Oct31 development16均93.125%→93.125%，无变化；assessment32分类为number3/other11/yes-no18，高共识小样本并不代表总体。【E20】

Oct31 assessment无效响应0%，posttraining/reload复评mean latency约0.319秒/图；早期3seed最终adapter哈希各异，证明权重状态不同，不证明能力提高。早期clean工程失败86804（CPU Adam step state别名）与86805（optimizer状态设备放置）保留；通用保存/恢复修复后86806/86807及后续三seed检查通过。不能把Qwen恢复通过转移为IP2Pix恢复通过。

**Modified/edited-image训练：BLOCKED，未启动；有效编辑标签没有独立认证。真实poisoned/后门目标训练：本轮未执行。清洁LoRA不是full-model训练，也没有可靠的clean-task泛化提升。下一步应先冻结独立原图测试、预注册有统计功效的样本/步数设计，而非挑seed或重复小题集。**

## 10. Post-training ASR与Clean Accuracy

| 量 | 当前真实结果 | 状态 |
|---|---|---|
| clean原图受训Qwen共识得分 | 上节真实三批次结果，带明确adapter谱系 | 工程PASS；提升NOT VALIDATED |
| benign编辑数据受训Qwen的clean accuracy | 没有合格受训artifact/独立标签实验 | BLOCKED / NOT TESTED |
| 真实post-training ASR | 无已验证科学样本及有效分母；值null，不是0% | NOT TESTED |
| 清洁控制false activation、SC/SI-V分层安全率 | 仅接口/合成fixture，没有可报告真实科学数据 | NOT TESTED |
| 冻结Qwen在C4编辑图答案变化 | 已观测，见匹配原参考一致度 | 冻结推理现象，不能标记post-training ASR |

`src/evaluation/security_readonly_v2.py`要求受训artifact谱系、checkpoint/数据manifest与显式事件分母；inference_only与post_training群体不混合。存在接口/测试不等于已获得真实ASR。任何攻击ASR、poisoning outcome或stealth信号均不连接生成器更新；本报告不提出或执行攻击优化任务。

## 11. Baseline对照与旧结果的可比性

### 11.1 历史预训练自然图像Phase4

| seed | frozen M0最终proxy reward | DDPO M1最终 | RSI M2最终 | 实际训练步 |
|---:|---:|---:|---:|---|
| 42 | −0.4377205865 | −0.6247290416 | −0.5309555346 | M1/M2各8 |
| 123 | −0.2546136744 | −0.3478591678 | −0.2541514674 | M1/M2各8 |

这些是反复使用development的旧proxy reward，不是clean accuracy或ASR。历史两seed的初始adapter/optimizer/RNG、训练source顺序与sampling seeds匹配；**在step4 LR干预前**，RL/RSI adapter已分别最大分叉4.34328e-5（42）/6.64559e-5（123），两seed各256个exp_avg和256个exp_avg_sq张量也不同，保存RNG和LR仍匹配。故不能将最终差异因果归于RSI。【E28、E29】

历史Phase4总分配13,122 GPU秒（3.645h），含诊断；四训练臂32次更新、320个训练trajectory/source exposures、跨臂seed唯一训练源154。旧步数成本C恒1/3，不能用它宣称真实compute降低。该阶段是旧独立预算范围，不是本轮1h验证子预算。

### 11.2 当前可以/不可以比较的对象

| 对照 | 是否实际跑过 | 可比性与结论 |
|---|---|---|
| 原图/identity/brightness/contrast/冻结IP2Pix/几何 | 是，同160源 | 冻结参考一致度有配对负面差异；标签有效性未知 |
| 历史真实IP2Pix DDPO vs RSI | 是，2seed | 干预前分叉，因果比较不合格 |
| tiny FP32 fixed/DDPO、普通scheduler、RSI | 是，3seed | 新前缀与query/NFE预算匹配；未见科学优势 |
| 自然图像frozen/DDPO-only/普通scheduler/RSI四臂 | 没有合格完整实验 | 数值、语义、reward与独立评估均有门槛，BLOCKED |
| clean vs modified-target post-training及ASR | 未完成 | NOT TESTED，不能从冻结编辑推理替代 |

## 12. GPU资源、预算与失败记账

### 12.1 必须分开的两个账本

| 口径 | 现场数值 | 限定 |
|---|---:|---|
| 10/26至当前可见本项目历史Slurm GPU分配 | 18,625秒 = 5.173611 GPUh，53个GPU作业 | 按WorkDir属于本项目或项目明确job前缀归属；含旧阶段、失败、加载/写盘；不包括RSI-Video |
| 当前验证阶段已用 | 2,381秒 = 0.661389 A100 GPUh | 3600秒子预算，含本阶段所有失败与诊断 |
| 当前验证已预留/活跃研究GPU | 0秒 / 0个 | 本次不提交任何任务 |
| 当前验证剩余 | 1,219秒 = 20分19秒 = 0.338611 GPUh | 不能当作全项目剩余预算或完整四臂试验授权 |
| 88099实际新增 | 100秒 = 0.027778 GPUh | 已包含于2381，不能重复收费 |
| 88102/88105 | GPU秒均0 | CPU作业 |
| 当前报告编制新增GPU | 0 | 只读整理 |
| 美元费用 | 未提供计价信息 | 不编造 |

旧报告的1344、1676、1955、2102、2246、2281等子总计是不同阶段的时间快照，不能相加。最新从2281加88099的100得到2381。Slurm分配秒不等于GPU kernel忙时；实际利用率未在本次采样，不声称0%或100%。历史窗口5.1736h不是完整生命周期上界，不能据此推出未声明的全项目剩余额度；RSI-Video的48GPUh规则不移用到此项目。

16:28UTC资源快照中epyc-gpu的16节点各报告2×A10080GB，其中epyc05/06/07/12为idle（合计8张完整idle节点GPU），其余mixed/allocated。资源可用性会变，不能据此越预算提交；本项目当前未占研究GPU。

验证子账本（对照实时sacct，所有失败保留）：

| 验证账本Job | 状态 | Slurm GPU秒 |
|---|---|---|
| 86736 | FAILED | 138 |
| 86738 | COMPLETED | 0 |
| 86739 | FAILED | 6 |
| 86741 | COMPLETED | 20 |
| 86742 | COMPLETED | 24 |
| 86745 | COMPLETED | 10 |
| 86793 | COMPLETED | 129 |
| 86794 | COMPLETED | 0 |
| 86795 | FAILED | 436 |
| 86796 | COMPLETED | 19 |
| 86804 | FAILED | 33 |
| 86805 | FAILED | 31 |
| 86806 | COMPLETED | 60 |
| 86807 | COMPLETED | 96 |
| 86846 | COMPLETED | 79 |
| 86848 | COMPLETED | 79 |
| 86851 | COMPLETED | 78 |
| 86853 | FAILED | 53 |
| 86856 | COMPLETED | 53 |
| 87921 | COMPLETED | 332 |
| 88016 | COMPLETED | 279 |
| 88029 | COMPLETED | 147 |
| 88032 | COMPLETED | 144 |
| 88084 | COMPLETED | 35 |
| 88099 | FAILED | 100 |


## 13. 历史审计项及当前修复边界

下面保留`SCIENTIFIC_GATE_MATRIX.md`对23项历史问题的状态。PASS只指该行命名的新范围；旧失败不回写。后来的method-v2诊断接口对identity credit和配置绑定有局部PASS，不能将旧F12/F19或reward科学构念整体升级为PASS。

| 历史条目 | 该历史审计状态 | 已有证据 | 剩余风险 |
|---|---|---|---|
| F01 Claim/estimand mismatch | PASS | Protocol names cosine engineering task and no VLM/security reward. | No BadSem reproduction or security efficacy has been established. |
| F02 Same-model semantic verification | BLOCKED | 0 genuine reviewed pairs; no human/model error rate available. | Independent reviewer agreement and semantic validity remain unknown. |
| F03 Reward redundancy and floor censoring | FAIL | 500/500 S=.5A;230/500 floor; centered component rank2. | Feedback construct is not validated for scientific learning. |
| F04 Instruction success absent | BLOCKED | C0/C1 scores identical for160; identity exceeds all historical changed rewards. | No human-confirmed lighting success rate; no new objective introduced. |
| F05 RSI versus ordinary scheduling | BLOCKED | New comparison PASS; modest LR/rollback mechanism explicitly described. | Providing a scheduler baseline does not prove novel RSI or autonomous superiority. |
| F06 Preadaptation causal confounding | PASS | Disk initial/prefix comparisons: 12; all equal=True. | Only the new FP32 tiny-model comparison is repaired; historical BF16 causal claims remain invalid. |
| F07 Reused data and weak replication | BLOCKED | Newest32 assessment sources:4 prior inference,10 in old opened-test manifest. | No real-image generalization claim; three seeds cannot substitute independent data. |
| F08 Exact-content split leakage | PASS | Zero ID/hash overlaps within latest32/16/32; all80 file hashes valid. | PASS for new internal split only, not historical v1 or exposure independence. |
| F09 Pretrained stochastic continuation | FAIL | Both new paths PASS; earlier ordinary and instrumented failures preserved. | Intermittent path-dependent history is unresolved; batch10 continuation is unaccepted. |
| F10 Question versus global preservation | BLOCKED | Forms prepared; zero human labels. | Rubric availability is not semantic remediation. |
| F11 Step proxy versus measured compute | PASS | Both GPU attempts charged; per-branch actual GPU-held seconds retained. | No claim that old C is measured cost or has become a validated cost objective. |
| F12 Identity loses consistency credit | FAIL | Caller gives A0 and reward.9666667 versus1.4666667 with declared credit. | Bug still present in legacy caller; no historical reward overwritten. |
| F13 Exposure versus realized compute | PASS | Each branch4 updates,320 forward NFEs,48 validation+32 test queries. | Measured times differ; sequential same-node timings are not a latency benchmark. |
| F14 Suspected editing/padding failures | BLOCKED | Geometry reconstruction160/160; aligned SSIM mean.7340322949; LPIPS absent. | No human-confirmed edit/instruction failure rates. |
| F15 Active-contract reconstruction | BLOCKED | New config/optimizer and rollback checks PASS. | General production pretrained resume across a historical revision remains unvalidated. |
| F16 Obsolete method entry | PASS | Original body byte text preserved after removing banner; code paths checked. | Documentation does not certify scientific benefit. |
| F17 Backward primitive not localized | BLOCKED | Previous instrumented raw error2.834317e-5; new pair exact; no stable path effect. | No specific CUDA primitive or unique causal mechanism identified. |
| F18 Independent validity unknown | BLOCKED | 0 reviewed pairs; disagreement and validity rates remain unknown. | Human-dependent task cannot be replaced by model flags. |
| F19 Reward config/default mismatch | FAIL | Values match historical campaign; wiring gap persists. | Editing YAML alone would not change coefficients; future version needs explicit contract. |
| F20 Runtime data guards weaker | FAIL | All80 latest image hashes PASS; historical official join receipt retained. | Audit success does not make runtime loader a corruption detector. |
| F21 Near-duplicates/pretraining exposure | BLOCKED | All80 prior-exposure flags true; latest split only source-disjoint. | Foundation-model contamination and unrecorded prior exposure unknown. |
| F22 Real post-training scientific/security evidence absent | BLOCKED | Absent denominators remain missing; scientific gate NO-GO. | Natural-image efficacy and training-induced security behavior untested. |
| F23 Within-batch duplicate freshness gap | PASS | Prepatch duplicate accepted;5 regression tests PASS including real trainer early rejection. | Distinct trajectories for the same condition remain explicitly permitted; no historical duplicate was inferred. |

关键未决问题不仅是BF16：独立语义/指令成功、奖励冗余与no-op、真正独立评估、历史干预前分叉、真实post-training行为证据均未完成。文稿、Algorithm1和最新Figure1还存在单ψ/双φψ与DDPO/GRPO术语不一致；应忠实厘清并由方法定义确认，不能用重命名实现来隐去差异。【E01、E32、E33】

## 14. 下一步实验决策：P0 / P1 / P2

以下是基于现有证据的决策，不表示本次已执行；资源为未来上限建议/估计而非实测。没有任何一项在本次触发新GPU任务。

| 优先级/任务 | 前置条件 | 要验证的问题 | 可复用证据 | 预期资源 | 成功/失败标准 | 现在是否可执行 |
|---|---|---|---|---|---|---|
| P0-A：既有32对独立双审与裁决 | 两名真实独立reviewer，按固定盲审rubric提交 | 原答案是否仍正确、全局语义与指令是否达成 | 全部现有图、hash、盲审选择、C5几何控制；不重生成 | 0GPU；64份独立case审阅+分歧裁决，人工时间未测 | PASS=真实双审完整且绑定manifest、按原rubric处理分歧；FAIL/BLOCKED=缺字段/冲突未裁决/来源不独立；不能事后改有效性阈值 | 材料就绪，等人；BLOCKED，不能由AI假标 |
| P0-B：冻结方法/单位/信用分配定义 | 方法作者确认文稿含义；不更换研究问题或loss | 单一生成器参数集合、DDPO-SF、q/instruction、dataset反馈如何合法归到trajectory | 本报告、代码审计、Figure1/Algorithm1、旧loss | CPU文档核对，0GPU | PASS=概念与执行路径一一对应，未定义处明列；FAIL=仍用GRPO名覆盖DDPO或把ASR广播成逐图reward | 可做CPU定义核对；本次仅报告未改稿 |
| P0-C：重建独立评估资格 | 完整项目暴露清单及可用原始来源 | source/content隔离之外是否已有训练、反馈、选择或测试暴露 | 既有官方题答/归档、1514独特文件hash核验与暴露表 | 0GPU；小规模CPU/新图来源下载如需另行执行 | PASS=先封存名单及hash、排除已知暴露/重复、报告采样分布；FAIL=旧80/160被换名为新test；模型预训练暴露仍记未知 | 暴露审计可复用；新测试尚未封存/未执行 |
| P1-A：BF16最早primitive因果定位与真实修复验证 | 先用已有raw张量形成单一可证伪假设；预注册模式/次数/bitwise标准 | 上游attention、fan-in累积或生命周期哪个机制导致同Checkpoint差异 | 88099四worker张量、88105命名差异、88084局部VJP、所有旧失败 | 先0GPU CPU图/张量分析；如仍需要新GPU，建议独立300秒A100硬上限且从1219扣除，须另行执行 | PASS=定位可复现机制、修复普通路径并按预注册重复全部精确；FAIL=任一原检查不等、仅插桩偶然PASS、未定位原因；B2/T2不替代B10/T10 | CPU假设整理可做；GPU本次禁止且尚无已证明修复 |
| P1-B：reward构念/配置/no-op验证 | P0-A真实标签；P0-B定义先确认 | 保留原目标下信号是否对应有效编辑；legacy与v2的配置/identity行为 | 500条reward、160源六条件、现有v2诊断、精确算术 | 首选0GPU重算与审阅后配对分析；不安装新损失 | PASS=原先规定的语义/指令/配置条件满足且排序可解释；FAIL=只提高proxy、identity胜出却未编辑、labels未知；缺少科学接受阈值需先定义，不能看到结果后补阈值 | 部分CPU可做；科学释放BLOCKED |
| P1-C：LPIPS完整可用性与预处理对齐 | 合法可用的实现与校准权重；明确reference/dtype/crop/padding | 能否测得标准LPIPS而非AlexNet替代值 | 已有160对C4/C5、SSIM和geometry hash | 可CPU、小规模；暂无可信实测耗时，不预占GPU | PASS=版本/校准权重hash齐、固定预处理与参照；FAIL=缺包/缺权重/尺寸不匹配/借其他指标冒名 | 当前依赖缺失；本次不安装，BLOCKED |
| P2-A：自然图像四臂良性DDPO/RSI | P0/P1数值、语义、reward及独立eval全部通过，且新资源估算/预算明确 | frozen、DDPO-only、普通调度与RSI是否有独立增益 | 已有前缀等价检查、完整state回滚、计费方法；不复用已打开final分数选择 | 当前1219秒不足以保证有功效的多seed四臂；旧Phase4已用13122秒，需另定预算且不得超原授权 | PASS=干预前全状态精确、相同评估和实际预算、独立测试的预定义增益CI满足门槛；FAIL=前缀分叉、查询不等/选择泄漏或CI不支持 | BLOCKED；不开始大型训练 |
| P2-B：有功效的clean Qwen原图学习比较 | 独立封存原图，预声明最小有意义效应与样本数/训练步数 | 是否有真实clean-task能力提高，而非同一道题改变 | 完全复用112张量ownership与fresh-object恢复证据；保留现有3seed负结果 | 新步数/样本需用已有每图约.319秒和79秒级小训练估算，尚未锁定总额，不承诺在余量内 | PASS=预定义独立配对效应及不确定性支持；FAIL=CI跨0/重复同源当样本/按结果挑seed | 可设计，尚不具备独立测试前置条件；本次未执行 |

**当前最有价值的先行工作是P0-A既有32对真实双审；下一项GPU工作应是P1-A的单一假设数值定位，而不是再次smoke、更多单提示搜索或扩大预训练生成器训练。** 人工审阅与CPU方法/暴露审计可并行推进。增加prompts或更新数应排在数值/语义/独立评估之后，并预先定功效和总预算。

Post-training安全指标只保留独立只读评估要求，不安排ASR驱动优化、攻击生成、poisoning或stealth训练。现有reward-release guard、数值阈值、负面结果留存及总预算门槛均继续有效。

## 15. 给其他AI的不可越界结论

1. 代码/optimizer确实能更新特定LoRA，不等于学习到有价值任务能力。
2. DDPO-SF解析有限测试通过，不等于完整预训练BF16程序或科学reward已经通过。
3. 88105是CPU取证PASS，不是BF16修复PASS；最早观测到raw backward差异，具体primitive未识别。
4. 冻结编辑后VQA原参考一致度下降是可复现负面描述性结果，不是后门植入或post-training ASR。
5. 三seed中两个+2.1875pp来自同一道题；CI包含0且p=1，不支持clean泛化增益。
6. 新三臂控制同前缀和相同NFE/query通过，但RSI效应CI跨0；旧自然图像RL/RSI干预前分叉。
7. 人工语义标签0，LPIPS缺失，旧开发数据暴露不能靠重新命名消除。
8. 当前完整自然图像科学训练保持NO-GO；没有新的科学有效性成功结果。

## 附录A. 关键证据文件与内容指纹

下列SHA来自本次读取的集群小文件原始字节，用于追溯；不是重新运行实验的结果。动态scheduler/heartbeat以第2节实时快照为准。

| 证据号 | 集群绝对路径 | 读取内容SHA-256 |
|---|---|---|
| E01 | `/storage/home/ydk297/projects/poisoning/reports/METHOD_IMPLEMENTATION_AUDIT.md` | 0d8badb184e4bf73a712b6ff7a3a202a6959b7712945f6749a50d2e1b54972b1 |
| E02 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/generator_ownership.json` | 6898a41a60611f8a118574418416eecdd972e36cd1eb36db1618bf16ec062232 |
| E03 | `/storage/home/ydk297/projects/poisoning/results/gradient_feedback_20261028_remote_v2/synthetic_checks.json` | 22f7313d0317ef9e9936f77f21ad0ed5c1304510aef7897a0322237bc1c838d9 |
| E04 | `/storage/home/ydk297/projects/poisoning/results/group_reference_20261028_v3/checks.json` | ae058c5418af7041eaa7d1cec37e0d6d0f37b503901d5fa18617c0f9573e30b1 |
| E05 | `/storage/home/ydk297/projects/poisoning/results/gradient_feedback_summary_20261028/batch_feedback.csv` | 4ccd7a1e7d734516ce4d470122d51d68d7057bf3dc2592ff70b1f50128f47827 |
| E06 | `/storage/home/ydk297/projects/poisoning/reports/EXPERIMENT_RESULTS.md` | 2a97c9936cdfa8f5efb18470d19c897ee0d3b4ed06c16ff2f38c815ddd3d2f8e |
| E07 | `/storage/home/ydk297/projects/poisoning/results/matched_feedback_20261028_v2/main_manifest.json` | 9decfa7a0bb21ed4572f1710fdbde8c6feacec80cd6bed85f7ca8c2124bcaa2b |
| E08 | `/storage/home/ydk297/projects/poisoning/reports/VALIDATED_MATCHED_FEEDBACK.md` | bba3ac558cd9586d6fa6823ecad3d19d1021ac77b68d39cc17348590fe17abde |
| E09 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/feedback_distribution.json` | 0366dd31c8ac97bdeb832804fb608b3c100aca61ebeb533970e7d0e54b0cf7f9 |
| E10 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/matched_feedback.json` | ffba922e251d2f84eeeb0e9911b7f0dd86709220238739f9efb5025118aa612d |
| E11 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/identity_and_config_audit.json` | 82fdb78f4e5244b9be63028e02247ca551d007d46317cdbd9ce07543ef68a3cf |
| E12 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json` | 93244b96592c7f6b5ea949df18084852e948a11cd65849a35f6a7d52fc0b79c8 |
| E13 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/data_versions_verified.json` | ab842dfc99426aa999c8cdcac55437d7bdbd6d2c86298b2d43932602f237b622 |
| E14 | `/storage/home/ydk297/projects/poisoning/results/next_validation_20261101T0040Z_v1/semantic_review32/selection_receipt.json` | a5653643ba56894b778e5020fbe4059cde0876a96b0e19d84d2c8759850916d4 |
| E15 | `/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P0_e81c1f3e7320_a0/review32_verified.json` | b0dc457944d2298bb4d650df4f35479f5a1cf553b2ee75f8d9b551091edb17ad |
| E16 | `/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P0_e81c1f3e7320_a0/quality32_verified.json` | 24c41fe4da09cd79e2825c528b630412e986715075530f6c7c75e9780db1fa86 |
| E17 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/metric_availability.json` | 232c81a3957d32771709f99c402d323df413a565c275e0ad0929f358cb222b18 |
| E18 | `/storage/home/ydk297/projects/poisoning/results/replication_monitor_20261029_v1/replication_statistics.json` | f1f682ca78cca4e83abf4a3395a0e706ec2232a5f2503d330c133b1def8ce4df |
| E19 | `/storage/home/ydk297/projects/poisoning/results/small_scale_validation_20261031T1928Z_v1/clean_statistics.json` | baa64c1db531ad62bdbb64e83792773a783e492ec55bf8434e2c0f49085be4c6 |
| E20 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/cpu_audit/qwen_statistical_conclusion.json` | a1cf29083caee705de9f5136e831422cc826b5bf25ebdb591ce2a3e62f223fa4 |
| E21 | `/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P1_ABBA_v1_a0/summary.json` | 270d4ca90ac2b46bb686ec855e67ac71c15925dfbc75a489b86684ad68121da2 |
| E22 | `/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a1/summary.json` | f25762b6b003be262976ab6276839d24b0ee7f6a5714bb28a548b2199f21b6eb |
| E23 | `/storage/home/ydk297/projects/poisoning/results/overnight_supervisor_20261101_v1_P1_FORENSICS_v1_a1/named_raw_gradient_comparison.csv` | b95ccc9aa0562640d2995c50b9623481f044e8132e8c6e214e3ad54f10d2875a |
| E24 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/vjp_probe/summary.json` | 4feb55b16c2b0d79def16e8d48343941065ba27bada51b6eb12b437d07383ef8 |
| E25 | `/storage/home/ydk297/projects/poisoning/results/replication_monitor_20261029_v1_ip2pix_a0/diagnostic/summary.json` | 4197a9f3f13eb64d8a37b4f18fa5d13641feb4c0aac34c0cbb0ca801e6ed19dc |
| E26 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/threeway/results.json` | 343a61ac69ef8abd0c2234760fff2a95a3ed3302b5ba28fe23ca55566a97ca94 |
| E27 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1_threeway_repair_v1/protocol.json` | 9b9c5da40310505380c8d82192cb1da003964c8d60db2fb237e8b8b9ae7d30d1 |
| E28 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/historical_prefix_state_comparison.json` | 9e4531cdf1e525a36b01d219f7d2dc188c0efb9a55ac535147fbb43b984394f2 |
| E29 | `/storage/home/ydk297/projects/poisoning/reports/phase4_results.json` | 7f87e84962d66966c687f79b263e1b8c045c31a372744afa91bd677162cb3b35 |
| E30 | `/storage/home/ydk297/projects/poisoning/reports/METRIC_AND_NORMALIZATION_AUDIT.md` | 8e6c1e8c4186a17dfc31742649a4ab2b40dc427d372e8b93534c5ac5b8284c09 |
| E31 | `/storage/home/ydk297/projects/poisoning/src/rl/phase5_backend.py` | 73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712 |
| E32 | `/storage/home/ydk297/projects/poisoning/reports/AUDIT_DECISION_MATRIX.md` | 64f9814fdc08c78518c5020d1fb1d9cc276c2a7939d7b09f5da206cfa6d67c88 |
| E33 | `/storage/home/ydk297/projects/poisoning/reports/SCIENTIFIC_GATE_MATRIX.md` | 4e490b5ad7df66f7eef79a0187635acd0611647125a209142c7f5f245b40adad |
| E34 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/versioned_contract_checks.json` | 7c739398ebb35c7fb730e11b2cf6677a12b420ae8ccad40902dee965c78dbb0c |
| E35 | `/storage/home/ydk297/projects/poisoning/results/small_scale_validation_20261031T1928Z_v1/synthetic_rsi/results.json` | bbb2f6ef05838ca3110e2acb9c00215e0d8e5f01bfbb27377f8c97b47f9d975c |
| E36 | `/storage/home/ydk297/projects/poisoning/results/audit_remediation_20261101T0120Z_v1/audit/F23_after.json` | 9b6528901818e807e8bf0a612e6cff3e2fa51f1a9e0efc0a15a013c0f1ec13af |
| E37 | `/storage/home/ydk297/projects/poisoning/results/ddpo_wiring_20261028_v3/wiring_checks.json` | cf45113ddf0a908f48f789abd8e66052d318325be4b811d846376db27a2b0215 |
| E38 | `/storage/home/ydk297/projects/poisoning/results/method_code_audit_20261101T0510Z_v1/original_fields_verified.json` | 024cd94ed9797a70e83063fac1d3dfac1eac6862b8b4e52d60496a162d0224b3 |


关键实现路径另包括：`/storage/home/ydk297/projects/poisoning/src/training/generator_adapter.py`、`src/rl/diffusion_policy.py`、`src/rl/diffusion_rl_trainer.py`、`src/rl/reward.py`、`src/rl/reward_normalization.py`、`src/rsi/controller.py`、`src/rsi/memory.py`与`src/rsi/harness/`（后续相对项同属于该绝对项目根目录）。当前文稿来源为`output/cvpr2027_visuals/source/manuscript_original.tex`/`manuscript_integrated.tex`，最新Figure1在`output/rsi_figure1_final_20261031/RSI_Figure1_Final.pdf`；正式审计快照在`results/method_code_audit_20261101T0510Z_v1/audit/supplied_method_snapshot/`。

## 附录B. 可见历史窗口全部53个GPU作业（失败不删除）

窗口起点2026-10-26；按Slurm JobID去重，以`ElapsedRaw×gres/gpu`计。包含失败与加载/写盘，全部属于本项目可识别WorkDir/命名范围。这里不是重定义当前验证子预算。

| Job | JobName | 状态 | GPU秒 | Exit |
|---|---|---|---|---|
| 86254 | poison-smoke | COMPLETED | 59 | 0:0 |
| 86256 | poison-smoke | COMPLETED | 90 | 0:0 |
| 86259 | poison-smoke | COMPLETED | 117 | 0:0 |
| 86260 | poison-smoke | COMPLETED | 59 | 0:0 |
| 86292 | p2-editor | COMPLETED | 41 | 0:0 |
| 86293 | p2-training | FAILED | 46 | 1:0 |
| 86299 | p2-training | FAILED | 77 | 1:0 |
| 86300 | p2-vqa_dev | COMPLETED | 63 | 0:0 |
| 86301 | p2-training | FAILED | 44 | 1:0 |
| 86302 | p2-training | FAILED | 42 | 1:0 |
| 86303 | p2-training | FAILED | 15 | 1:0 |
| 86304 | p2-training | FAILED | 49 | 1:0 |
| 86305 | p2-vqa_test | COMPLETED | 73 | 0:0 |
| 86306 | p2-training | COMPLETED | 48 | 0:0 |
| 86307 | p2-semantics | COMPLETED | 146 | 0:0 |
| 86308 | p2-controller | COMPLETED | 543 | 0:0 |
| 86403 | p3-one | FAILED | 88 | 1:0 |
| 86404 | p3-one | FAILED | 73 | 1:0 |
| 86405 | p3-one | FAILED | 119 | 1:0 |
| 86408 | p3-backward | COMPLETED | 131 | 0:0 |
| 86409 | p3-one | COMPLETED | 90 | 0:0 |
| 86411 | p3-smoke | FAILED | 953 | 1:0 |
| 86412 | p3-semantics | COMPLETED | 156 | 0:0 |
| 86430 | p4-diagnostic | COMPLETED | 314 | 0:0 |
| 86433 | p4-frozen | COMPLETED | 388 | 0:0 |
| 86434 | p4-rl42 | COMPLETED | 3227 | 0:0 |
| 86435 | p4-rl123 | COMPLETED | 2707 | 0:0 |
| 86444 | p4-diagnostic_outcomes | COMPLETED | 59 | 0:0 |
| 86445 | p4-rsi42 | COMPLETED | 3211 | 0:0 |
| 86446 | p4-rsi123 | COMPLETED | 3216 | 0:0 |
| 86736 | benign-gradient-feedback | FAILED | 138 | 1:0 |
| 86739 | ddpo-tiny-validation | FAILED | 6 | 1:0 |
| 86741 | benign-editor-recheck | COMPLETED | 20 | 0:0 |
| 86742 | ddpo-and-edit-recheck | COMPLETED | 24 | 0:0 |
| 86745 | ddpo-math-sdpa-recheck | COMPLETED | 10 | 0:0 |
| 86793 | matched-benign-feedback | COMPLETED | 129 | 0:0 |
| 86795 | matched-benign-feedback | FAILED | 436 | 1:0 |
| 86796 | matched-RGB-control-repair | COMPLETED | 19 | 0:0 |
| 86804 | clean-target-engineering | FAILED | 33 | 1:0 |
| 86805 | clean-target-engineering | FAILED | 31 | 1:0 |
| 86806 | clean-target-engineering | COMPLETED | 60 | 0:0 |
| 86807 | clean-target-engineering | COMPLETED | 96 | 0:0 |
| 86846 | rsi-P1_seed42 | COMPLETED | 79 | 0:0 |
| 86848 | rsi-P1_seed123 | COMPLETED | 79 | 0:0 |
| 86851 | rsi-P1_seed3407 | COMPLETED | 78 | 0:0 |
| 86853 | rsi-P2_IP2Pix | FAILED | 53 | 1:0 |
| 86856 | rsi-P2_IP2Pix | COMPLETED | 53 | 0:0 |
| 87921 | poisoning-small-benign | COMPLETED | 332 | 0:0 |
| 88016 | poisoning-next-benign | COMPLETED | 279 | 0:0 |
| 88029 | poisoning-audit-benign | COMPLETED | 147 | 0:0 |
| 88032 | poisoning-threeway-repair | COMPLETED | 144 | 0:0 |
| 88084 | poisoning-method-vjp | COMPLETED | 35 | 0:0 |
| 88099 | poisoning-overnight-gpu | FAILED | 100 | 1:0 |


## 附录C. 本次只读核查命令和保留范围

```bash
cd /storage/home/ydk297/projects/poisoning
date -Is
TZ=UTC squeue -u "$USER" -o '%i|%j|%T|%S|%k'
TZ=UTC sacct -u "$USER" --starttime 2026-10-26 -X -n -P \
  --format=JobIDRaw,JobName%80,State,Start,End,ElapsedRaw,ExitCode,AllocTRES%150,WorkDir%300
sinfo -p epyc-gpu -N -o '%N|%T|%G'
```

本报告没有删除、覆盖或修改旧实验、失败、Checksum、Checkpoint、人工模板、训练代码或监控定义。仅生成本Markdown，并在H盘使用约10MB临时只读材料；交付后清理本次临时目录，原始科学证据仍在上述集群路径。未复制模型权重到C盘，未产生大型训练文件。Google Drive恢复不作为本报告交付或后续独立分析的依赖。
