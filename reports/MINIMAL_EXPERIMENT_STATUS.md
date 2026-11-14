# 最小良性实验实际状态

**终态：COMPLETED_BOUNDED_DEVELOPMENT_SMOKE；Slurm 90945 COMPLETED，运行中作业 0。**

|对象|实现|运行|验证结论|
|---|---|---|---|
|随机采样 / log-prob / FP32 LoRA-Adam / checkpoint|已实现|已运行|47 项 CPU 测试通过；本轮 BF16、FP32 精确重复/续训通过|
|Frozen vs Inner|已实现|已完成|4 次 Inner 更新；本轮代理效能无正向收益|
|fixed / rule / UCB Outer|已实现|已完成|9 个候选全拒绝；额外收益 0|
|可迁移 RSI improvement procedure|未建立|未运行独立 benchmark 验证|不能宣称 RSI；当前仅 HPO/recipe search|

## 实际命令、环境和时间

本地执行 `python scripts/benign_cluster_io.py stage`、`python scripts/benign_cluster_io.py cpu`、`python scripts/benign_cluster_io.py submit`；仅一次 GPU 提交。`status` 与 `fetch` 只读已有作业/结果。

实际 Slurm 内命令（外层 `timeout --signal=TERM 1080s`）：

```bash
/storage/home/ydk297/projects/poisoning/.venv-editor/bin/python /storage/home/ydk297/projects/poisoning/experiments/benign_ddpo_audit_20261111_v1/code/scripts/run_benign_ddpo_minimal.py --assets-root /storage/home/ydk297/projects/poisoning --output /storage/home/ydk297/projects/poisoning/experiments/benign_ddpo_audit_20261111_v1/gpu --config /storage/home/ydk297/projects/poisoning/experiments/benign_ddpo_audit_20261111_v1/code/configs/benign_ddpo_minimal.json --cpu-receipt /storage/home/ydk297/projects/poisoning/experiments/benign_ddpo_audit_20261111_v1/cpu/cpu_tests.json
```

- Slurm ID：90945；PID：2927808；GPU：NVIDIA A100 80GB PCIe，1 块；seed：42。
- 开始（纽约）：2026-11-11T21:54:36.221820-05:00；结束：2026-11-11T21:59:38.193006-05:00。
- 开始 UTC：2026-11-12T02:54:36.221820+00:00；结束 UTC：2026-11-12T02:59:38.193006+00:00。
- Slurm 实际 reservation：304 GPU 秒 = 0.084444 GPUh；上限 1200 秒 = 0.333333 GPUh。包含模型加载、数值 gate、失败/拒绝候选和评估；没有追加 GPU 作业。
- 软件：现有 `.venv-editor`；Torch 2.6.0+cu124 / Diffusers 0.35.1 / PEFT 0.17.1 / Accelerate 1.11.0 / Transformers 4.57.1。详细环境见 `remote_inventory.json`。
- 模型：IP2Pix revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`、`variant=fp16`，复用缓存并上转为对应计算 dtype；256×256、4 个随机 DDIM 步、rank2/alpha4、FP32 对照、AdamW lr1e-5、batch2、clip0.5。
- 记录到的 peak GPU allocation：5010178048 bytes；FP32 路径 UNet predict 调用 680 次。该计数不包含 activation-checkpoint 内部重算或 VAE/text encoder，因此不冒充全部 FLOPs。

## 数据划分

总计 14 个 source ID / content hash 全部互斥，来自历史可见 COCO/VQAv2 开发训练池。`test` 仅为本 run 内 source-held-out assessment，不是全项目封存的独立最终测试。历史 test 与 prospective reservation 均未打开。transfer 使用另 4 张源图与暗化指令；这是同一开发池的跨任务 smoke，不是独立 benchmark。

|划分|sample_id|image_id|源文件 SHA-256|
|---|---|---|---|
|train|564978001|564978|`6cb0d62a553f46c3337134365f651c40249bc10f201ed1b33ecc125efbd3d692`|
|train|36445002|36445|`24720e7dc0d068d8cdf54a94f228a1b46ac47f989510145924cd97fef2d8e8fb`|
|train|574996000|574996|`cd696eae998978fd5591a2ac791cafae8d91c5e47219ba721bd0d4c513f601fa`|
|train|293215034|293215|`45c49a38ac9e5500b5bd8df89ae4723996d8255a2f81adfbebd2f651f4e048bc`|
|dev|118227001|118227|`d9096db6fd300e6bae48da52f4c20acf396b643536e7d7ee37287a24c2c8e145`|
|dev|48261002|48261|`988deff1b1800c0504731b14c3fb968c0d14ce38d9e9e7b5739a5499f1517b93`|
|test|143096006|143096|`cea24f7b944de23685d5ad4da2ba5b6957357873cb0b42201bb802951aa5841e`|
|test|66753005|66753|`7062447126fdaa45e2fefa50352f325011a396e5373a49368fceff6351c65924`|
|test|233938002|233938|`94a449812a2508334ac516b29e180d4f2066535b85df9e8e8cebfdaa52948e05`|
|test|29716003|29716|`ff57df85ff7d87340f5dd8c65c1b43fc6c9e7fe04d9c4910d42ce9977be3df01`|
|transfer|450102000|450102|`15a92c34f98a88446f269544087ba55b19ae924e9ddda687adc5b5b876267842`|
|transfer|79077005|79077|`07911aea6f9302bf40fa6b0068ca43d7408648e1ab2d72319e4347d6af1314ac`|
|transfer|459001006|459001|`8f00140dc7952a894527e5b7ffb7da7c9cc5f296824d84f02524245025e44b2a`|
|transfer|40466000|40466|`849238b18684cf7d8d4e4ba068fe44c43d6abf6bf464f56c51361984bd8b5c29`|


## 实测结果

|任务|方法|源图数|平均 reward proxy|平均亮度目标误差|SSIM（对源图）|相对 Frozen|
|---|---|---:|---:|---:|---:|---:|
|test|Frozen|4|-0.054608103|0.067546787|0.386068|+0.000000000|
|test|Inner|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|fixed_recipe|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|fixed_rule|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|test|ucb_search|4|-0.055179231|0.067926676|0.383597|-0.000571128|
|transfer|Frozen|4|-0.087634593|0.102039040|0.367535|+0.000000000|
|transfer|Inner|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|fixed_recipe|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|fixed_rule|4|-0.088346492|0.102726066|0.367350|-0.000711899|
|transfer|ucb_search|4|-0.088346492|0.102726066|0.367350|-0.000711899|

Inner−Frozen 的亮化评估差为 -0.000571128，源图 bootstrap 95% 区间 [-0.0009981820448105062, -0.0001440748900108646]；暗化迁移差为 -0.000711899，区间 [-0.0010922673480028879, -0.0003315308974946411]。每个任务只有 4 张图、1 个训练 seed，区间只是探索性的源图重采样，不估计训练 seed 变异。不将负值包装成学习成功，也不据此宣称所有配置均无法学习。

Frozen 与 Inner 的评估 source、seed、sampler、精度和 query 数匹配。Frozen 没有训练成本；Inner 多出的训练成本单独记录，不声称两者总训练计算量相同。

## Outer 预算与拒绝记录

|方法|候选数|尝试 optimizer 更新|训练 rollout|dev 查询|UNet predict 调用|接受|相对 Inner 的 test / transfer 差|
|---|---:|---:|---:|---:|---:|---:|---|
|fixed_recipe|3|6|12|6|152|0|+0.000000000 / +0.000000000|
|fixed_rule|3|6|12|6|152|0|+0.000000000 / +0.000000000|
|ucb_search|3|6|12|6|152|0|+0.000000000 / +0.000000000|


每种方法从同一 Inner checkpoint 恢复；3 个候选 × 每候选 2 次更新 × batch2，另各 6 次 dev query、4 次 test 和4 次 transfer query。每候选 40 次 predict 调用，每种方法合计 152 次（含最终评估）。规则、UCB 与固定 recipe 均用同一个 dev 门槛。三种方法均接受 0/3，恢复到同一 Inner 状态，最终 test/transfer 图像与 Inner 相同；不能把零差解释为控制器收益。UCB 只有3轮，仍是小规模非上下文搜索。

本轮一共执行 32 次 optimizer 调用：BF16 数值诊断5 + FP32 数值诊断5 + Inner4 + Outer尝试18。被拒绝更新的成本保留，不把“尝试次数”混同为最终 checkpoint 的 retained step。

## 证据与恢复

- 原始事件：`remote/gpu/events.jsonl`；图像观测：`remote/gpu/fp32/observations.jsonl`；完整 stdout/stderr：`remote/slurm-90945.log`。
- 数值 gate：`remote/gpu/{bf16,fp32}/numerical_gate.json`；保留普通 A/B 更新及 resumed 下一步的张量 witness。
- Inner checkpoint：`remote/gpu/fp32/checkpoints/inner_final/training_state.pt`；SHA-256 `48e2c4574b29fd0bcbe01a5cba21f479b03bc035cf6ae4b086b95f7dd71677fb`。
- 9 个 candidate 与3个 selected checkpoint 均保留；目录名、哈希、dev 变化和拒绝理由见 `outer_comparison.json` 与 events。
- 本地已重算 126 个图像哈希和 17 个 checkpoint sidecar；完整文件 SHA 清单在 `LOCAL_EVIDENCE_VERIFICATION.json`。其中原 status.json 的 retained_bytes 在写入最终状态前计数，比实际 GPU 文件总量小85字节；以最终文件清单为准。
- Slurm 为 COMPLETED，无训练异常 trace；本地 OpenMP 启动失败与初次受限 DNS 检查属于环境审计问题，未启动本地 GPU 训练。独立实验效能、语义人工审核和跨 seed 验证仍未完成。


## 运行后修复验证（CPU only）

新增 padding 取整与测试驱动 SHA 覆盖回归后，49 项 CPU 测试通过，零失败/跳过。已用修复后的度量重新检查 126 个保存图像；涉及源图 ['574996000']，各 split 最大 reward 绝对变化为 `{'train': 0.0020111316098415655, 'dev': 0.0, 'test': 1.3877787807814457e-17, 'transfer': 2.7755575615628914e-17}`。这只是已保存像素的 CPU 复测，没有新的模型调用、optimizer 更新或 GPU 作业。训练原始奖励/轨迹/checkpoint 不改写；修复后模型学习效能仍未运行。详细证据在 `experiments/benign_ddpo_audit_20261111_v1/postrun_fix/`。


## 保存图像的视觉检查

配对预览中，电脑图与领带图出现明显纹理化和结构破坏，Frozen 与 Inner 都存在该问题。因此4步采样没有建立可用的普通编辑基线；本轮 P3 只能验证等预算提案/拒绝流程，不能作为控制器方法价值的有效效能实验。该观察来自保存图的技术审阅，不冒充独立人工质量评测。下一步应先在独立、固定的少量普通编辑样本上校准合理推理步数与视觉质量，再考虑新的学习实验；本轮不追加 GPU。


## 磁盘与清理

本轮证据和最终图件在H盘保留约 237.97 MiB（含源码、checkpoint、失败/拒绝候选、数值witness与图像）。无模型/数据下载或训练checkpoint写入C盘。C盘可用空间审计点为 59.97 → 58.59 GiB，差额未作归因；未删除Codex会话历史、模型或共享运行时。H盘可用 212.59 GiB。图件中间渲染、本轮临时目录和新建bytecode缓存已清理；集群scratch同样不存在，canonical源码SHA与开始时相同。清理及账本证据见 `LOCAL_CLEANUP.json`、`disk_after.json`、`final_resource_receipt.json`。
