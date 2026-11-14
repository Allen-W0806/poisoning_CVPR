# Reward / Editing / Numerical Debug Report

更新时间 UTC：2026-11-04T11:45:45.170348+00:00

本轮只跑 seed=42，累计授权20GPU小时，历史关闭额度没有重开。内部状态：COMPLETED。

## 本轮实测结论

内部最终16图：RL-RSI−DDPO共同奖励差-0.000427，配对源图95%区间[-0.012725, +0.012161]；本轮未证实RL-RSI优于DDPO。

RL-RSI平均亮度目标进展-0.160752，色度MAE0.041347，平均目标线性亮度误差0.012377；像素不编辑参考误差0.007235，其v3 Reward为0。当前不支持只改目标属性、其他语义完整保持的主张。BF16仍未修复，采用通过实际数值门的FP32替代路径。

## Reward 的实际问题

旧验证集原图CLIP均值0.136749707628，LPIPS自比较0，因此2×CLIP−LPIPS=0.273499415256，算术正确。用户列出的−0.014295来自另一组development assessment，而非这16张validation；相同validation上的旧RL为−0.0154535，DDPO为−0.0271254，Frozen为−0.0959000。训练提高了旧代理Reward，却仍使内容亮度下降，显示代理目标和任务方向不一致。CLIP cosine不是成功概率或指令完成率；原图不要求得分为零。源图已低于提问中的0.2阈值。真实CLIP输出embedding已经归一化；本轮四张训练图上cosine与归一化点积最大差为0。LPIPS使用校准AlexNet和[-1,1]输入，CLIP使用RGB PIL与固定processor，未发现这些实现错误。

实际问题是目标错配：旧奖励在校准控制上偏好不编辑，轻微提亮反而降低得分。加大LPIPS惩罚会更偏好不编辑。仅减去源图CLIP能让原图得分归零，但不会改变同一源图的候选排序，也不能单凭归零宣称学习方向修好。

对实际PilotTrainer更新代码的静态复查未发现策略梯度符号反转：loss=−R×mean(logp)，使用原始终局奖励和零基线，未用会在B=1时消除梯度的批内归一化。这说明损失符号与最大化期望奖励一致，不等于有限样本Adam更新必然提高任务质量；实际效果仍以固定checkpoint的开发评估为准。

初版lighting_v2通过了解析控制，但在44个实际开发训练输出中43个亮度项硬截断为−1，标准差仅1.192e-5。该开发实验被拒绝并取消，全部日志、图片和checkpoint保留在`internal_pilot_v2_saturation_probe`；它不是完成的对比实验。2840秒取消作业照常计费。

采用显式新版本lighting_v3_mean_progress_20261104：

`R = w_sem*(CLIP_edit-CLIP_source + mean_brightness_progress) - color_cost - w_lpips*LPIPS`。

目标是本次良性任务的+3%线性光曝光对应的内容区域平均亮度。令x=(原图到目标的平均亮度误差−编辑图到目标的平均亮度误差)/max(原图误差,0.001)，progress=x/(1+abs(x))，连续保留方向和偏差排序；色度误差/0.02仍截断至5。空间保持仍由独立LPIPS和色度项约束；平均属性本身不能证明对象完整保持。亮度/色度只看原图内容区域；CLIP/LPIPS/SSIM仍覆盖完整画布。这增加了明确的任务项，属于科学目标修订，不能冒充通用后门奖励修复；不能跨版本比较原始Reward大小。同44个已保存开发输出的v3重评分亮度项标准差为0.47148127556241953，不再硬饱和。该重评分未调用新模型，也未冒充v3训练结果。新pilot从相同初始状态重新训练。

| Control | n | legacy_reward | rejected_v2_reward | lighting_v3_reward | LPIPS |
|---|---|---|---|---|---|
| identity | 4 | 0.288984 | 0.000000 | 0.000000 | 0.000000 |
| exposure_plus3pct | 4 | 0.287551 | 0.853534 | 0.945792 | 0.000279 |
| exposure_minus3pct | 4 | 0.288482 | -1.042489 | -1.052642 | 0.000281 |
| channel_swap | 4 | 0.191857 | -4.870937 | -4.661448 | 0.102562 |
| unrelated_image | 4 | -0.370566 | -5.876251 | -6.607443 | 0.640159 |

校准图为预先选定的前4张训练图，解析曝光仅是正/负控制，未替代InstructPix2Pix生成器。校准通过不代表生成器已经学会提亮、保留全部语义或提高ASR。

## 编辑质量

旧pilot实际是DDIM10步，而非2步。原图被白色填充到512方形。旧RL验证输出全画布编码亮度差−0.1093863，内容区域−0.0406355；白边放大负变化，内容仍变暗。不能把边框变化当作目标属性变化。

| setting | n | LPIPS | SSIM | content_encoded_luma_delta | content_chromaticity_MAE |
|---|---|---|---|---|---|
| legacy10_g75_eta1 | 4 | 0.373171 | 0.480115 | -0.081860 | 0.118074 |
| steps50_g75_eta1 | 4 | 0.158819 | 0.641421 | -0.029210 | 0.059815 |
| steps50_g5_eta1 | 4 | 0.101354 | 0.677580 | -0.011535 | 0.034320 |
| steps50_g5_eta0 | 4 | 0.348638 | 0.477540 | -0.048169 | 0.107288 |

这四张验证图由固定顺序选定。50步/guidance5/eta1改善LPIPS和色度，仍未实现平均提亮。eta0明显改变部分场景；它只用于推理诊断，不能直接用于要求正Gaussian方差的现有DDPO训练。新pilot采用50步、guidance5、imageguidance1.5、eta1、rank16/alpha32、LR5e-6、clip0.5。这些参数共同改变，不能据此单独归因rank或学习率。Frozen已出现色偏，故rank不是其原因。

## BF16 与 FP32

在本轮v2目标诊断下，FP32 LoRA/linear计算/Adam已实际应用，且梯度和Adam状态有限；但BF16普通同一步重复仍失败：1216720个元素不同，最大绝对差9.456717180000851e-06，exact=False，tolerance=False。Torch/CUDA RNG均一致。差异已出现在raw gradients，不能归因于随机种子或简单地用梯度裁剪解释。

进一步源码复查发现，现有DiffusionPolicy构造器已经关闭BF16低精度归约；进程在构造前的默认True不代表训练时实际设置。因此这项候选在提交前撤销，新增GPU消耗0，不能当作新修复。

内部新pilot全FP32数值门：PASS。它上采样现有FP16-variant权重到FP32，不恢复权重已经丢失的精度。普通第一/第二更新和磁盘重载后的下一更新分别比较1538个张量；通过这一局部门不等于BF16路径修复或普遍数值认证。

## 内部pilot与外部复现

| Method | status | optimizer_steps | lighting_reward | LPIPS | SSIM | ASR | clean_accuracy |
|---|---|---|---|---|---|---|---|
| Frozen | MEASURED | 0 | -2.483343 | 0.085933 | 0.752509 | NOT TESTED | NOT TESTED |
| Fixed (equivalent) | MEASURED_EQUIVALENT_CONTROL | 0 | -2.483343 | 0.085933 | 0.752509 | NOT TESTED | NOT TESTED |
| DDPO-only | MEASURED | 100 | -2.344747 | 0.084585 | 0.753064 | NOT TESTED | NOT TESTED |
| Rule RSI | MEASURED | 100 | -2.364122 | 0.084915 | 0.752834 | NOT TESTED | NOT TESTED |
| RL-RSI | MEASURED | 100 | -2.345174 | 0.084783 | 0.752923 | NOT TESTED | NOT TESTED |
| Shadowcast clean0 control | MEASURED | 26 | NOT TESTED | NOT TESTED | NOT TESTED | 0.000000 | 0.556988 |
| Shadowcast poison50 | MEASURED | 27 | NOT TESTED | 0.201819 | 0.812616 | 0.945000 | 0.553550 |

内部为32/16/16良性开发实验，使用v3共同评价器；最终16图来自历史暴露的训练池。Frozen=Fixed为用户指定的冻结生成器、相同prompt等价性对照，仍执行图像编辑采样；像素不编辑对照另列，v3 Reward为0。内部没有训练poisoned victim，ASR/ΔClean不填代理值。三次controller更新不是独立种子，也不是正式accepted revision。

最终内容区域属性如下：亮度差为编码域平均变化，目标误差为线性光亮度的绝对误差。平均亮度上升本身不能证明每张图接近1.03倍曝光目标，也不能证明颜色和对象保持。

| Method | content_encoded_luma_delta | content_chromaticity_MAE | content_brightness_progress | mean_target_linear_luma_error |
|---|---|---|---|---|
| Frozen | 0.005918 | 0.041251 | -0.226385 | 0.013420 |
| Fixed (equivalent) | 0.005918 | 0.041251 | -0.226385 | 0.013420 |
| DDPO-only | 0.003982 | 0.041282 | -0.161619 | 0.012463 |
| Rule RSI | 0.003836 | 0.041271 | -0.169380 | 0.012702 |
| RL-RSI | 0.004061 | 0.041347 | -0.160752 | 0.012377 |

额外冻结Qwen3-VL-2B检查使用全部16张development assessment的原问题、原10人答案和greedy回答；原图及四个物理编辑条件共80次查询，Fixed共享Frozen观测。这是预先登记的事后质量审查，未用于controller或模型选择。单个原问题不能覆盖整图所有颜色、对象、数量或证明完整语义保持；冻结模型也可能答错。

| Group | n | source_answer_accuracy | source_prediction_agreement |
|---|---|---|---|
| pixel_source | 16 | 0.812500 | 1.000000 |
| frozen | 16 | 0.687500 | 0.750000 |
| ddpo_only | 16 | 0.687500 | 0.750000 |
| rule_rsi | 16 | 0.687500 | 0.750000 |
| rl_rsi | 16 | 0.687500 | 0.812500 |
| fixed | 16 | 0.687500 | 0.750000 |

Shadowcast采用官方代码固定提交b13f877a，完整Trump-to-Biden标签任务，3439干净+50投毒。PGD采用已发布代码的4000步，与论文Sec4.1所述2000步不同；该差异不为追求结果而调整。模型LLaVA1.5-7B、CLIP-L/14@336均固定revision。ASR使用全部200官方测试图，成功要求整词Biden存在且Trump不存在；错误不得静默跳过。干净对照也评估相同200图，记录自然目标响应率。277张源图及277张guidance构造图与200张ASR图的文件/解码RGB精确重复审计：overlap_found=False；这不覆盖近重复或预训练污染。主要清洁效用为VizWiz公开完整8000题，使用原始LLaVA问题和prompt、2026年4月官方公开的10人参考答案；这是完整test自评分；原论文Table3未明确EvalAI分区，是否相同分区尚未核实。参考答案不进入模型，输出全部锁定后才评分。补充效用为104条已暴露VQAv2 development，两个数据集分别报告，GQA未测。两模型均配置one epoch，实际optimizer更新数为clean0=26、poison50=27；描述性差值不是等更新数的独立因果估计。对选定投毒文本的官方预处理静态重放：50/50有非零监督token，整条忽略数0。这仅验证标签资格，未证明采样器实际消费了每个样本。软件Torch2.1.2、单卡eager和4×32微批差异均披露，不宣称逐位复现或与内部良性实验匹配。

BadVLM按用户最新决定跳过。提供的BadVLA仓库是机器人项目。BadSem/CBV仅列已核实文献边界，未用自行编写的重着色函数冒充论文复现。

## 统计与资源

实际/含在途预留GPU小时：12.2683 / 12.2683，上限20。失败、取消、烟测及干净模型均计费；详见GPU_ACCOUNTING.json。单训练种子不能估计seed均值±std或seed级95%CI。配对源图bootstrap区间只描述这个固定训练种子的源图不确定性，不能证明方法泛化显著性。

ASR的经验源图bootstrap在全成功/全失败时可能退化；这种退化不表示总体错误率不确定性为零。

```json
{
  "RL_RSI_minus_DDPO_lighting_v3": {
    "difference": -0.00042665592432652305,
    "CI95_low": -0.012725111644544203,
    "CI95_high": 0.012161135779698972,
    "resampling_unit": "paired source image, one trained seed; NOT seed uncertainty"
  },
  "RL_RSI_minus_DDPO_content_luma": {
    "difference": 7.863308107190284e-05,
    "CI95_low": -2.350545702239752e-05,
    "CI95_high": 0.00020156600001295275,
    "resampling_unit": "paired source image, one trained seed; NOT seed uncertainty"
  },
  "ddpo_only_minus_Frozen_lighting_v3": {
    "difference": 0.13859604620749447,
    "CI95_low": 0.04724550356830845,
    "CI95_high": 0.22210736684632426,
    "resampling_unit": "paired source image, one trained seed; NOT seed uncertainty"
  },
  "rule_rsi_minus_Frozen_lighting_v3": {
    "difference": 0.11922185204388615,
    "CI95_low": 0.04230393387842685,
    "CI95_high": 0.19265870387487635,
    "resampling_unit": "paired source image, one trained seed; NOT seed uncertainty"
  },
  "rl_rsi_minus_Frozen_lighting_v3": {
    "difference": 0.13816939028316794,
    "CI95_low": 0.042206782968955865,
    "CI95_high": 0.22598077433127894,
    "resampling_unit": "paired source image, one trained seed; NOT seed uncertainty"
  },
  "comparison_scope": "RL-minus-DDPO is the requested comparison; additional frozen comparisons are descriptive, one fixed seed, no multiplicity-adjusted efficacy claim"
}
```

外部原任务ASR与清洁效用的固定模型源图区间分别保存在`shadowcast/verified_statistics.json`和`vizwiz/verified_statistics.json`。Shadowcast poison50 ASR：{'mean': 0.945, 'source_CI95_low': 0.91, 'source_CI95_high': 0.975, 'cases': 200, 'unique_source_hashes': 200, 'scope': 'conditional on these fixed trained victims; not training-seed uncertainty'}。

VizWiz poison50−clean0 配对差值：{'mean': -0.003437499999999998, 'source_CI95_low': -0.0062148711068735416, 'source_CI95_high': -0.0006749135269613419, 'cases': 8000, 'unique_source_hashes': 7928, 'scope': 'conditional on these fixed trained victims; not training-seed uncertainty'}。

所有模型与大文件均在集群，H盘仅镜像精选结果；C盘无本轮模型下载。保留源码、参数/优化器见证、checkpoint、逐图结果及失败记录。运行结束清理专属tmp目录。
