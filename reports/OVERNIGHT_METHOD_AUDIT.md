# 夜间方法审计与P1准备（2026-11-12）

本轮执行正常VQA统计扩展，与图像生成方法比较分开。最新P0报告不再重复运行。`reports/frozen_quality_gate.md`已复核，且实际重新查看`outputs/p0_foundations_20261111_v1/paired_preprocessing.png`：默认guidance的斑马明显偏黄、棒球场偏饱和；去padding没有消除，5/3出现颗粒与锐化感。不能把某项SSIM/LPIPS降低当作任务符合证明。旧4-step结果不能支撑方法价值；今晚不追加大批采样搜索。

|能力|实际路径/函数/行号|现有机制|本轮状态|
|---|---|---|---|
|Inner|`src/mini_pilot/engine.py:98` `PilotTrainer.update`；loss第127行|fresh on-policy trajectory终局R乘transition logp；梯度范数裁剪|实现存在，旧低分辨率smoke有证据；512px/有效编辑质量下的方法比较NOT_RUN|
|随机扩散log probability|`src/rl/diffusion_trajectory.py:10` `transition`；`src/rl/diffusion_policy.py:79` `sample`|DDIM随机transition，detached action的Gaussian logp；freshness/replay guards|既有数值检查通过范围不外推新分辨率；今晚未重跑|
|SFT|`src/training/generator_trainer.py:12` `GeneratorTrainer.__init__`；第46行`update`|监督epsilon MSE；旧target是源图亮度×1.02，不读取本轮真实配对target|不是本轮公允paired SFT；只可作为历史可行性实现|
|固定/规则/UCB|`scripts/run_benign_ddpo_minimal.py:259`起候选循环|每候选同更新次数、dev accept/reject；规则/HPO机制|既有4-step小样本验证不能作为可靠效能；新paired任务GPU NOT_RUN|
|Learned Outer|`src/mini_pilot/controller_v3.py:48` `choose`，第56行`learn`，第62–65行|Categorical REINFORCE、validation gain−EMA baseline、独立AdamW参数更新|真实可学习实现；跨独立任务保留收益NOT_RUN；仍是recipe controller|
|Random/HPO新对照|新paired manifest arms已统一列入|尚未接入同预算完整paired训练入口|PARTIAL / GPU NOT_RUN|
|显式paired SFT数据接口|`src/training/paired_generator_data.py:9` `load_training_pair`|仅train，检查输入/真实目标SHA，同几何预处理，同条指令|实际读取4条train pair且拒绝dev通过；未接入GPU训练，不能标完整SFT已验证|

旧接口仍可读历史checkpoint，未将它们接入夜间优化。现有controller可修改LR/reward recipe，但今晚自动监控/调度器不会调用它们，也不会改LR/reward/数据。三个VQA seed只是固定监督学习，均不是Inner RL或Outer实验。

PPO事实：没有old-policy模型快照；没有`exp(logp_new−logp_old)`ratio；没有`min(ratio*A, clip(ratio)*A)`clipped surrogate；没有clip-fraction/KL日志。Inner原始终局R/零baseline不等于PPO advantage估计。Outer具有advantage，但仍是REINFORCE。`clip_grad_norm_`与sampler `clip_sample=False`均不是PPO clipping。论文若要求PPO，以上为缺失机制，不能靠重命名解决。

新的P1 CPU准备固定train128/dev32/test64/transfer32真实照片。前三split输入为已知0.7亮度退化，transfer为Gaussian blur1.5；target是未退化原始真实图缩放到≤512，而非编辑器生成伪标签。全部Frozen/SFT/NormalEditingRL/Fixed/Rule/UCB/Learned/Random/HPO共用manifest与seeds42/123/3407。它是受控退化恢复任务，不代表开放域编辑效果。不得将同一原图/不同退化放到不同split。

下一步必须先清Frozen质量门槛，补显式paired SFT与公共正常任务reward/成功判定，再做512px梯度/replay/checkpoint canary。各训练臂使用相同初始化、训练样本、调用预算上限和完整消耗账本；outer候选、拒绝和恢复消耗全部计入。Frozen有零训练成本，不通过空跑伪装等预算。Learned须与fixed/random/rule/UCB/HPO比较，独立任务族transfer必须保留，缺少时只能叫recipe controller。

实际CPU命令：`python -B scripts/check_night_p1.py --campaign <remote_run>`，Slurm job91042，已生成`p1_preparation/CPU_check.json`，256组几何/SHA/source互斥检查通过。`analysis_code/data_adapter_smoke.json`记录4条真实train读取、dev拒绝通过。没有GPU运行的单元格保持空。
