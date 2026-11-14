# 架构与方法纠正

1. 原 ZIP 三份文档已纳入仓库，原文另存 `docs/benign_starter_original/`。`ARCHITECTURE_DECISION.md` 是待验证设计，不能当作结果。`CODEX_START_HERE.md` 与本次用户请求确定良性执行范围。
2. 纠正 §1.4–1.5：DDPO 可以只使用终局奖励；各随机去噪转移的 score-function 累加，不需要独立逐步奖励或 learned credit network。一轮 LoRA 更新也不能自动认证 DDPO。现有路径应精确称 on-policy score-function DDPO 类实现，无 PPO ratio。
3. 明确实际 sampler：原生 IP2Pix Euler 先单独复现；受控训练与 Frozen 对照改用同一随机 DDIM 合约。坐标平均 log density 是固定缩放，不能未经验证当成联合概率用于 importance ratio。
4. 纠正“DDPO/DRaFT 谁更省样本”的预设。DRaFT 是可微奖励梯度优化的不同对照；本轮未运行，不声称优劣。
5. 当前 Outer 为有限学习率臂的固定 recipe、固定规则和非上下文 UCB，再加 dev accept/reject。它是 HPO/recipe adaptation，不是已训练 Outer PPO，也不是已验证 RSI。暗化 smoke 未提供独立 benchmark 或改进程序迁移的证据。
6. BF16/FP32 短配置通过与历史 BF16 全面修复严格分开。技术通路可复现、真实参数有变化，不等价于任务效能变好。
7. 原 `E:/main9_revised.tex` 未覆盖；独立方法/Algorithm 1 在 `docs/benign_ddpo_method_outline.tex`（LaTeX outline，未进行整稿编译）。概念流程图在 `output/benign_ddpo_audit_20261111/`，提供 SVG 可编辑对象与 PDF。图是框架示意，不是经验效果图。

核对过的原始来源：

- [DDPO 原论文，§3/§4](https://proceedings.iclr.cc/paper_files/paper/2024/file/14f75513f0f1ca01de1e826b52e6b840-Paper-Conference.pdf)：终局 reward 与 diffusion MDP / policy gradient。
- [DDPO 官方 PyTorch 实现](https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py)：随机 DDIM 与 transition log-prob。
- [IP2Pix 官方仓库](https://github.com/timothybrooks/instruct-pix2pix)：图像条件编辑模型；作者主页本次抓取404，改核官方仓库。
- [DRaFT 原始发布](https://deepmind.google/research/publications/51081/)：通过采样链反传 reward gradient。
- [RRSI 官方仓库](https://github.com/google-research/rrsi)：将 evolve selection 与 held-out/OOD 评估区分。
- [AIDE² 原文](https://arxiv.org/abs/2609.26457)：作为研究背景，未在本仓库复现其算法或结果。

这些文献说明方法定义，不替代本仓库的实测证据。
