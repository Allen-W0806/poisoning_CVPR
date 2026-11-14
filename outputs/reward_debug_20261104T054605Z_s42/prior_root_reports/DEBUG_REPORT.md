# Reward / Editing / Numerical Debug Report

更新时间 UTC：2026-11-04T06:39:41.104550+00:00

本轮只跑 seed=42，累计授权20GPU小时，历史关闭额度没有重开。内部状态：TRAINING。

## Reward 的实际问题

旧验证集原图CLIP均值0.136749707628，LPIPS自比较0，因此2×CLIP−LPIPS=0.273499415256，算术正确。用户列出的−0.014295来自另一组development assessment，而非这16张validation；相同validation上的旧RL为−0.0154535，DDPO为−0.0271254，Frozen为−0.0959000。训练提高了旧代理Reward，却仍使内容亮度下降，显示代理目标和任务方向不一致。CLIP cosine不是成功概率或指令完成率；原图不要求得分为零。源图已低于提问中的0.2阈值。真实CLIP输出embedding已经归一化；本轮四张训练图上cosine与归一化点积最大差为0。LPIPS使用校准AlexNet和[-1,1]输入，CLIP使用RGB PIL与固定processor，未发现这些实现错误。

实际问题是目标错配：旧奖励在校准控制上偏好不编辑，轻微提亮反而降低得分。加大LPIPS惩罚会更偏好不编辑。仅减去源图CLIP能让原图得分归零，但不会改变同一源图的候选排序，也不能单凭归零宣称学习方向修好。

版本化修订lighting_v2_20261104：

`R = w_sem*(CLIP_edit-CLIP_source) + brightness_progress - color_cost - w_lpips*LPIPS`。

目标是本次良性任务的+3%线性光曝光；progress为相对原图的逐像素亮度目标误差改善，用max(原图误差,0.001)归一化并截断至[-1,1]；色度误差/0.02截断至5。亮度/色度只看原图内容区域；CLIP/LPIPS/SSIM仍覆盖完整画布。这增加了明确的任务项，属于科学目标修订，不能冒充通用后门奖励修复；不能跨版本比较原始Reward大小。

| Control | n | legacy_reward | lighting_v2_reward | LPIPS |
|---|---|---|---|---|
| identity | 4 | 0.288984 | 0.000000 | 0.000000 |
| exposure_plus3pct | 4 | 0.287551 | 0.853534 | 0.000279 |
| exposure_minus3pct | 4 | 0.288482 | -1.042489 | 0.000281 |
| channel_swap | 4 | 0.191857 | -4.870937 | 0.102562 |
| unrelated_image | 4 | -0.370566 | -5.876251 | 0.640159 |

校准图为预先选定的前4张训练图，解析曝光仅是正/负控制，未替代InstructPix2Pix生成器。校准通过不代表生成器已经学会提亮、保留全部语义或提高ASR。

## 编辑质量

旧pilot实际是DDIM10步，而非2步。原图被白色填充到512方形。旧RL验证输出全画布编码亮度差−0.1093863，内容区域−0.0406355；白边放大负变化，内容仍变暗。不能把边框变化当作目标属性变化。

| setting | n | LPIPS | SSIM | content_encoded_luma_delta | content_chromaticity_MAE | common_reward |
|---|---|---|---|---|---|---|
| legacy10_g75_eta1 | 4 | 0.373171 | 0.480115 | -0.081860 | 0.118074 | -5.361262 |
| steps50_g75_eta1 | 4 | 0.158819 | 0.641421 | -0.029210 | 0.059815 | -4.139101 |
| steps50_g5_eta1 | 4 | 0.101354 | 0.677580 | -0.011535 | 0.034320 | -2.822837 |
| steps50_g5_eta0 | 4 | 0.348638 | 0.477540 | -0.048169 | 0.107288 | -5.729627 |

这四张验证图由固定顺序选定。50步/guidance5/eta1改善LPIPS和色度，仍未实现平均提亮。eta0明显改变部分场景；它只用于推理诊断，不能直接用于要求正Gaussian方差的现有DDPO训练。新pilot采用50步、guidance5、imageguidance1.5、eta1、rank16/alpha32、LR5e-6、clip0.5。这些参数共同改变，不能据此单独归因rank或学习率。Frozen已出现色偏，故rank不是其原因。

## BF16 与 FP32

FP32 LoRA/linear计算/Adam已实际应用，且梯度和Adam状态有限；但BF16普通同一步重复仍失败：1216720个元素不同，最大绝对差9.456717180000851e-06，exact=False，tolerance=False。Torch/CUDA RNG均一致。差异已出现在raw gradients，不能归因于随机种子或简单地用梯度裁剪解释。

内部新pilot全FP32数值门：PASS。它上采样现有FP16-variant权重到FP32，不恢复权重已经丢失的精度。普通第一/第二更新和磁盘重载后的下一更新分别比较1538个张量；通过这一局部门不等于BF16路径修复或普遍数值认证。

## 内部pilot与外部复现

| Method | status | optimizer_steps | lighting_v2_reward | LPIPS | SSIM | ASR | clean_accuracy |
|---|---|---|---|---|---|---|---|
| Frozen | GENERATOR_READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Fixed (equivalent) | GENERATOR_READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| DDPO-only | RUNNING | 30 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Rule RSI | READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| RL-RSI | READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Shadowcast clean0 control | IN PROGRESS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Shadowcast poison50 | IN PROGRESS | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |

内部为32/16/16良性开发实验，最终16图来自历史暴露的训练池。Frozen=Fixed为用户指定等价性对照。内部没有训练poisoned victim，ASR/ΔClean不填代理值。三次controller更新不是独立种子，也不是正式accepted revision。

Shadowcast采用官方代码固定提交b13f877a，完整Trump-to-Biden标签任务，3439干净+50投毒。模型LLaVA1.5-7B、CLIP-L/14@336均固定revision。ASR使用全部200官方测试图，成功要求整词Biden存在且Trump不存在；错误不得静默跳过。干净对照也评估相同200图，记录自然目标响应率。补充清洁效用使用104条已暴露VQAv2 development及原10参考答案、官方评分；GQA/VizWiz未测。软件Torch2.1.2、单卡eager和4×32微批差异均披露，不宣称逐位复现或与内部良性实验匹配。

BadVLM按用户最新决定跳过。提供的BadVLA仓库是机器人项目。BadSem/CBV仅列已核实文献边界，未用自行编写的重着色函数冒充论文复现。

## 统计与资源

实际/含在途预留GPU小时：1.2214 / 12.1550，上限20。失败、取消、烟测及干净模型均计费；详见GPU_ACCOUNTING.json。单训练种子不能估计seed均值±std或seed级95%CI。配对源图bootstrap区间只描述这个固定训练种子的源图不确定性，不能证明方法泛化显著性。

所有模型与大文件均在集群，H盘仅镜像精选结果；C盘无本轮模型下载。保留源码、参数/优化器见证、checkpoint、逐图结果及失败记录。运行结束清理专属tmp目录。
