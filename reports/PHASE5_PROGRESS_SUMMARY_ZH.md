# Phase 5 实现、实测结果与下一阶段判断

**结论：限定范围的 CPU 工程验证 GO；下一轮大规模 Generator RL/RSI 科学比较 NO-GO。** 本阶段已完成版本化代码、真实 harness 演化执行、数据重分组与回归测试；没有提交科学训练或安全攻击实验。

## 上一阶段的问题与本次修复

过去把普通学习率调整称为 RSI；保存的 reward 权重未完整接入执行；identity 的一致性项被错误排除；不同 ID 仍可能来自同一图像内容；原始答案与编辑后事实没有充分分离；旧验证集已暴露，不能作为新独立 Gate；分支在干预前已不完全相同。旧 reward 还存在常量/冗余项和截断饱和，因此旧增益不能作为方法有效证据。

本次新增 v2 接口修复配置接线、identity 判定、语义 UNKNOWN/DISPUTED、编辑答案校验、逐次内容校验和重复内容 batch 阻断。原 Phase1–4.5 代码、失败及数值均保留。真实 Generator RL 仍以 IP2Pix UNet attention LoRA 为学习对象，保留原 DDPO-SF 后端；没有改成外部分类控制器。由于尚无经独立验证的新训练奖励，Phase5 trainer 明确阻断优化器更新。

## RSI 现在是否真的修改可复用流程

是，但已验证范围仅为无害数据完整性任务。采用 Video-RSI 的追加观察、竞争原因诊断、可复用执行流程修改、独立成本 Gate、不可变版本与回滚。实现是有限诊断规则和声明式工作流解释器，没有宣称复现其完整 LLM Editor。

实际执行 `EXP_20261028T025518_c497cfd041ee_phase5_harness_evolution`：32个独立构造图像，准确率50%→100%；无效修改被拒，文件内容校验修改获接受，退化修改被拒，回滚成功。总搜索259次诊断、25,389字节、0.09015秒墙钟、0.07947秒CPU、0模型调用、0GPU。获选流程诊断数32→64，故不能称为提速。

对真实旧训练 episode225164001 另执行4项新 CPU 检查：哈希匹配、旧质量数值可重算、10步状态有限；语义/指令/答案原因仍未知，未强行提出修改。数值重算不是指标正确性证明。

## 数据与独立评价是否可靠

保留1500条记录，重新核对官方问题/答案及图片字节。train/dev/Gate/test为999/104/96/301；唯一内容组988/103/94/298。旧 train/test 同字节组整体进入 test，旧 test 成员全部保留。近重复候选检索为0，不能解释为不存在近重复。

**独立可用语义样本0，确认SC/SI-V均为0。** 关键词只是候选；编辑后事实缺失时保持UNKNOWN。dev/Gate来自已暴露旧验证集；旧test也曾打开。当前独立自然图像评价不可靠，需真实独立标注、验证奖励及新隔离评价方案。程序配对或检查点字段相等也不能替代严格分支可比性证据。

## 方法与证据边界

扩散策略学习与 harness 外层演化在数学上可以分离；DDPO提供策略梯度依据，BadSem提供语义/训练后评价区分，Video-RSI提供程序级修订机制。联合流程的语义有效性、收益和效率仍是新研究假设。冻结Qwen推理变化不能称为训练后后门ASR。本阶段只实现固定授权工件的只读安全评价接口，未产生真实安全结果。

## 测试、保留与监控

集群完整测试81/81通过；本地新增22/22通过。本地早期完整测试75通过、3项因缺Diffusers报错，失败日志保留。两轮同族假设复核发现的问题已记录和修复；第二轮后的改动仅执行器测试，没有第三轮独立复核或“审查无问题”结论。

本地3418、集群3344份旧证据文件未改变；两端32项受保护源码哈希一致。已有两小时监控仍启用，快照只有一个自有待运行作业86589，未新建重复监控。快照下次到期2026-10-28T04:32:35+00:00，实际开始受排队影响。

## 下一步与资源

M0冻结、M1普通Generator RL、M2普通scheduler、M3仅trace、M4诊断修订、M5主动独立Gate修订已准备并冻结，均PREPARED_NOT_RUN。先完成独立编辑事实/有效性审核、奖励构念验证、新私有Gate隔离和干预前严格可比性；这些CPU/标注工作可继续，当前GPU预算0小时。科学门槛满足后，再单独授权小规模干净闭环验证。

三固定seed、每训练臂8步、64图dev/Gate及限定候选次数的完整准备方案，按旧实测估计35.16 A100 GPU小时，含20%余量约 **42.20小时**；不是获批分配。人工标注、未来独立评价器和clean-target微调费用尚未知，未计入。未自动提交大规模比较。

全部10份技术报告见同目录；方法提案与LaTeX替换片段在docs。原始HTML未改，未发现原始LaTeX，因此不存在已编译修订论文。

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [docs/PHASE5_METHOD_PROPOSAL_v1.md](<H:/Project/Poisoning/docs/PHASE5_METHOD_PROPOSAL_v1.md>), [docs/phase5_method_patch.tex](<H:/Project/Poisoning/docs/phase5_method_patch.tex>), [src/rl/phase5_backend.py](<H:/Project/Poisoning/src/rl/phase5_backend.py>), [src/rsi/harness/harness_executor.py](<H:/Project/Poisoning/src/rsi/harness/harness_executor.py>).

Tests: [tests/test_phase5_contracts.py](<H:/Project/Poisoning/tests/test_phase5_contracts.py>), [tests/test_phase5_harness.py](<H:/Project/Poisoning/tests/test_phase5_harness.py>), [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_harness_smoke.json](<H:/Project/Poisoning/reports/phase5_harness_smoke.json>), [reports/phase5_dataset_v2.json](<H:/Project/Poisoning/reports/phase5_dataset_v2.json>), [reports/phase5_comparison_prepared_v2.json](<H:/Project/Poisoning/reports/phase5_comparison_prepared_v2.json>), [reports/phase5_canonical_cpu_tests_final.log](<H:/Project/Poisoning/reports/phase5_canonical_cpu_tests_final.log>), [reports/phase5_preservation_verified.json](<H:/Project/Poisoning/reports/phase5_preservation_verified.json>), [reports/phase5_monitor_snapshot.json](<H:/Project/Poisoning/reports/phase5_monitor_snapshot.json>).
