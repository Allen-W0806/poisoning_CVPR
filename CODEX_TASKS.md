# CODEX_TASKS — 本次用户消息确定的执行门槛

本文件在 2026-11-11 根据用户当前消息保存。此前仓库没有独立 CODEX_TASKS.md。附件是研究背景，当前消息优先；以下只执行正常图像编辑、干净 VLM 和良性方法基础设施。

## P0（不完成则禁止 P1 GPU 学习试验）

P0.1 仓库事实核查：产出 reports/code_inventory.md，每个能力附路径、函数、行号、实际测试命令及证据。无运行痕迹必须标 NOT_RUN。核查 diffusion_policy、diffusion_trajectory、generator_adapter、mini_pilot/engine 中 PPO old_policy、ratio、advantage、clipping 的实际计算与日志。

P0.2 固定独立普通编辑 dev 样本；Frozen 比较 >=20 steps 与合理分辨率，保存源图、编辑指令、结果、SSIM/LPIPS、配对可视化及日志。若仍结构破坏，先定位模型加载/预处理/采样/解码，不调 RL reward；此门槛决定 P1 是否启动。

P0.3 正常 image-QA 上用开源 VLM 完成小规模训练、验证、checkpoint、fresh-object resume consistency、独立 source-held-out 评估。只用正常数据；产出 reports/clean_vlm_pipeline.md、outputs/clean_vlm_metrics.json，保留命令、预测及训练日志。此 P0 技术闭环允许有界训练 smoke，不是 P1 方法学习实验。

P0.4 固定 train/dev/test/transfer manifest，自动检查 ID/content 泄漏；统一 seed、预算、指标方向、不确定性和运行记录；未运行论文单元格空置。

## P1（P0 通过后）

Inner 同初始化同预算比较 Frozen/SFT/reward-finetuning/RL，至少3训练seeds，报告任务指标、SSIM、LPIPS及不确定性。Outer先核实真实参数学习；fixed/random/rule/UCB/HPO为完整对照，独立良性transfer任务族是RSI必要条件。未通过门槛只记录 NOT_RUN，不启动对照。

## P2

多任务族、多VLM正常稳健性、预注册统计和方法/代码一致性；不属于当前自动扩展范围。

## 交付与保留

最终 reports/IMPLEMENTATION_STATUS.md；每步保存源码、命令、日志、结果；不覆盖旧证据，不虚构PASS，不构造后门/投毒样本，不把ASR/隐蔽性等接入优化；所有GPU工作须先检查资源并有硬上限。
