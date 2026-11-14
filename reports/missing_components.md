# 当前缺口（不是新的执行计划）

- PPO未实现：无old-policy模型、ratio、PPO surrogate clipping；Inner为score-function，learned Outer为REINFORCE。
- Frozen有>=20步运行证据，但亮化指令成功与独立视觉质量门槛尚缺；SSIM/LPIPS不能替代。
- 普通编辑的配对SFT benchmark与DRaFT/reward-gradient comparator未形成完整可比实现；当前GeneratorTrainer是epsilon-denoising，不可冒充这些结果。
- >=3训练seed的Inner完整对照NOT_RUN；fixed/random/rule/UCB/HPO/learned Outer的当前任务同预算对照NOT_RUN。
- Learned Outer有真实CPU参数学习和历史3次开发更新，不是未实现；但独立任务族的经验改善和RSI没有证据。
- VLM闭环是LoRA工程smoke，未做全模型训练、多VLM、真实编辑后重标注、独立总体效能验证。
