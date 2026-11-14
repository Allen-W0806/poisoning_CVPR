# Outer事实核查

`src/mini_pilot/controller_v3.py:LearnedController.choose/learn` 为categorical REINFORCE，含`validation_gain-baseline`、log_prob和AdamW；确实能修改policy参数。历史`outputs/reward_debug_20261104T054605Z_s42/internal_pilot/controller_updates.jsonl`有3次更新，max change约0.0010000/0.0010014/0.0010024。该历史小样本不是RSI收益证据。

本轮补全`LearnedController.load_state_dict`，CPU测试了policy/Adam/RNG/baseline/memory恢复及下一动作、下一更新精确一致。本轮没有运行learned Outer的生成器训练对比。v1/v2旧副本仍没有这一新增恢复接口；不将v3测试外推给旧副本。

`src/mini_pilot/benign_validation.py:RecipeSelector` 的fixed/rule/UCB与learned controller是不同实现；旧短作业9候选全拒绝，仅证明fallback。当前不存在Outer PPO的old-policy/ratio/surrogate clipping。现有memory保存和下一轮配置使用不等于改进程序在独立任务族上的改进。

fixed/random/rule/UCB/HPO/learned需要同数据、初始化、候选数、optimizer尝试数、dev查询和总GPU预算。当前完整组合为NOT_RUN，论文数值空置。独立transfer任务族尚无已验证数据/结果，因此只能称recipe controller，不称已验证RSI。
