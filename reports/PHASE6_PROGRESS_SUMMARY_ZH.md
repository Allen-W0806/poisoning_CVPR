# Phase6 连续推进、验证和监控总结

**当前结论：六项 CPU 数据/审核准备/方法工程阶段已执行；独立自然图像评价及科学训练仍 NO-GO。** 科学 GPU预算0，未启动模型推理、Generator RL训练或安全攻击实验。

## 执行结果

| 阶段 | 当前状态 | 最终完成证据 | 实验ID |
|---|---|---|---|
|S1_DATA|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044752_b0572261ea61_phase6_s1_data|
|S1_REVIEW_PACKAGE|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package|
|S1_REWARD_AUDIT|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044804_7ad7c946a999_phase6_s1_reward_audit|
|S1_SEMANTIC_REVIEW|NEEDS_HUMAN_REVIEW|N/A|—|
|S2_REAL_HARNESS|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044805_f12f313ebb56_phase6_s2_real_harness|
|S3_RL_ENGINEERING|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044811_00449775da3b_phase6_s3_rl_engineering|
|S3_REWARD_RELEASE|BLOCKED|N/A|—|
|S4_STATIC_COMPARISON|PASSED|VERIFIED_SLURM_AND_OUTPUTS|EXP_20261028T044816_33120182732f_phase6_s4_static_comparison|
|S4_SCIENTIFIC_READY|BLOCKED|N/A|—|
|S5_PILOT|BLOCKED|N/A|—|

重新核对1500条官方图片—问题—答案及字节，保留全部v1/v2。train/dev/Gate/test仍999/104/96/301；训练抽样按内容分组，固定10个不同源图。独立确认的编辑事实、SC/SI-V标签均0，因此没有把数据工程PASS写成语义有效。

已创建24对真实历史训练图像的七维独立审核包：[experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/index.html](<H:/Project/Poisoning/experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/index.html>)。包含原图/编辑图、原问题与10条原答案、指令、代理判断、优先原因及空白审核字段。先固定每源首个历史观察，再按不确定/截断/失真排序；这不是全部编辑对的最坏案例或独立测试集。尚无人工标签，不自动认证导出表单。

Reward审计覆盖500条旧观察：230条落在下界（46%）；C与U常量、S与A冗余；500条均缺显式指令完成测量。固定分量算术可重复，不等于奖励科学有效。所有编辑使用同一lighting指令，无法比较多种编辑类别；question关键词只作为问题内容分层。未调整系数，训练奖励继续阻断。

真实harness：8个训练episode追加32次CPU检查，24个不重叠训练源图作结构Gate，另有48个明确注入的元数据负例。总准确率按72行等权计算；原流程/新流程均100%，诊断操作216→144，减少33.3%。无效重排和退化修改被拒，回滚验证通过。只支持原图文件完整性流程效率，不支持编辑图语义改进、诊断正确率或安全ASR提升。全部搜索成本在记录中，单次墙钟差不能作为普遍提速结论。

Generator RL保留真实IP2Pix UNet LoRA及DDPO-SF。CPU代数/似然/条件/新鲜度/批次契约通过；旧32次实际LoRA更新证据保留。10源图＝10轨迹＝100去噪转移＝10次累积＝1次优化器更新。严格随机续训失败仍未修复，未重复耗GPU试图凑位级相等。

M0–M5静态同配置检查通过，新增准备版本修正M3为trace+诊断但不修订；均未运行。实际同初始化、独立评价、奖励发布和干预前可比性未满足。小pilot只准备，42.20 A100小时仅历史完整方案估计，未获批。

## 自动推进、修复和审查

控制器自动执行全部可用CPU阶段，人工审核阻塞不妨碍独立任务。任务内PASS为同一CPU分配中的临时工程断言；完成后还须Slurm与输出验证，失败会降低当前阶段就绪状态。监控只处理已验证的目录/log修复及最多两次瞬时IO重试，新尝试是不可变子实验。科学失败不当作工程重试。

已修复监控将标量观察计数当图像列表的实际错误；旧core/integrity及32项保护源码保留，使用v6入口。修复PowerShell原子替换兼容问题并实际运行同步成功。浏览器发现的横向溢出已修复，保留前后测试证据。首轮假设复核指出4项未声明语义和2项陈旧声明；已记录并修复，原首轮审计保留，新生成纠正子实验，未改写旧结果。两轮同族复核仅provisional，不能作为独立学术认可。

当前集群回归日志：[reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>)（92项通过；最终测试收据见stage证据）。旧Phase1–5文件3440份未改变。所有实验原始结果位于canonical `/storage/home/ydk297/projects/poisoning/experiments/`；本地新输出在 `H:/Project/Poisoning/experiments/`，准确ID见上表及各receipt。

## 资源、监控与下一步

Phase6 CPU作业86601/86602完成，Slurm elapsed33+26＝59秒，均1CPU/12GiB、0GPU。登录节点CPU测试/检查另记，不能当作GPU费用。现在没有运行或排队的本项目科学实验；无关项目作业只作队列上下文。

原两小时监控最近定时成功UTC2026-10-28T04:32:58.710285+00:00，最新额外核验UTC2026-10-28T04:56:30.690039+00:00；自有后继86595，下次到期UTC2026-10-28T06:32:35+00:00，实际受队列影响。仅一个监控链，退出聊天后由Slurm继续。下一个自动动作是轻量完整性/完成检查与双语页面刷新，不会自动启动GPU。

中文页面[dashboard/index_zh.html](<H:/Project/Poisoning/dashboard/index_zh.html>)；英文页面[dashboard/index_en.html](<H:/Project/Poisoning/dashboard/index_en.html>)；共享JSON[dashboard/state.json](<H:/Project/Poisoning/dashboard/state.json>)。本地需执行同步脚本并刷新，不能宣称静态文件自动跨机器更新。找不到已有实验双语package，已保留找到的旧概念讲解页面并新增最小实验视图。

确切缺口：独立七维审核及编辑后事实→奖励构念验证→新私有评价与严格分支可比性→单独GPU预算。全部依赖阶段保持阻塞；其余当前可执行任务已完成。

## 完整语义与接口假设

# Phase6 assumptions (declared before dependent implementation)

| ID | Choice | Decision | Class | Source |
|---|---|---|---|---|
| P01 | Stage granularity | Separate engineering sub-stages from semantic/scientific gates; blocked semantic work cannot prevent independent CPU work. PASS only means the stated sub-stage outputs passed. | semantic | user |
| P02 | Human truth unavailable | Prepare blank seven-dimension reviews from historical TRAIN images; no automatic promotion of submitted forms to truth or reward release. NEEDS_HUMAN_REVIEW persists pending independent adjudication. | semantic | user |
| P03 | Historical workflow Gate | Predetermine eight train investigation sources and24 disjoint train validation sources. Use exact recorded PNG identity/dimensions plus explicitly injected metadata-negative controls. Evaluate redundant-decode removal; no natural-image or security accuracy claim. | semantic | default |
| P04 | Performance-cost threshold | Reuse Phase5 engineering policy: no source regression; equal task accuracy requires >=20% diagnostic-operation saving. Report wall/CPU/bytes and all rejected attempts separately. Not a semantic gain or guaranteed wall-time speedup. | semantic | default |
| P05 | Stage retries | At most two explicit bounded engineering retries per task/signature; scientific blockers never retried as engineering. Immutable descendant runs only. No arbitrary generated code or GPU launch in controller. | interface | user |
| P06 | Monitor integration | Add a conditional Phase6 hook to the existing monitor, not a second scheduler. Disable legacy automatic GPU recovery while Phase6 zero-budget policy is active. Recurrence performs lightweight receipts/integrity/completion/dashboard work. | interface | user |
| P07 | Reward stability | Deterministically recompute saved component arithmetic twice, report repeat equality separately from variation across old checkpoints/seeds. Same-code replay is not independent reward validity. | semantic | default |
| P08 | Review sampling | Prioritize unknown/disagreement/clipped/high-distortion TRAIN pairs, at most24 unique source contents, preserving queue reasons. Seven separate dimensions plus edited factual answer, reviewer identity, uncertainty and disagreement remain blank. | semantic | default |
| P09 | Pilot authorization | Prepare only a four-update matched benign pilot proposal. Current authorization is zero GPU seconds; no automatic pilot launcher exists. Future budget and accepted reward require a new explicit versioned approval. | interface | user |
| P10 | Dashboard reuse | Search repository, canonical files and relevant local delivery folders first. Reuse any discovered bilingual package. If none exists, document missing baseline and add a minimal bilingual view of shared JSON. Local files require explicit sync/reload; no claim of autonomous cluster-to-laptop updates. | interface | default |
| P11 | Completion validity | Completed Slurm status plus valid immutable output manifest and finite required metrics establishes engineering completion only. Unknown accounting stays blocked. Registry observations are append-only; finalized scientific metrics are never rewritten. | semantic | user |
| P12 | Configuration comparability | Check prepared static equality while keeping runtime initialization, fresh independent evaluation and pre-intervention execution unresolved. Resolve M3 prose conflict in a new config version, not by rewriting Phase5. | semantic | default |
| P13 | Observed monitor schema collision | Phase6 reward-audit observations is a scalar count, whereas historical monitor expects an image-record list. Preserve historical core/integrity modules and route live entry points to versioned v6 monitor modules. Scalar count is not image evidence; validate list item schema before association checks. | interface | default |
| P14 | Structural Gate denominator | Score72 rows:24 recorded original-source PNGs plus48 injected wrong-size/wrong-digest metadata controls, weighting each row equally (1:2 positive/control mixture). Report row errors separately from affected unique sources. This does not score edited images or episode diagnoses. | semantic | sweep |
| P15 | Historical representative selection | Select first saved observation per raw source-content digest in fixed parent/step order before priority ranking or8+24 partitioning. Later edits of that source do not compete. Queue is diagnostic first-occurrence representatives, not all-pair worst-case or representative efficacy. | semantic | sweep |
| P16 | Edit category coverage | Report the sole observed editing-instruction category (benign lighting) separately. Original-question keyword strata may overlap and are question-content strata, not edit categories. Cross-edit-category performance is unavailable. | semantic | sweep |
| P17 | Audit versus generating provenance | Record audit-context v2 versions separately from all source manifests/configs; historical observations retain their legacy dataset/reward identity. A corrected audit is a new descendant of its previous audit; original experiment IDs form a separate source list. No relabeling old observations as v2 execution. | semantic | sweep |
| P18 | In-process versus final completion | Stage PASSED inside a shared CPU allocation means provisional task assertions and output integrity only; permits other CPU tasks, never scientific launch. Completed Slurm accounting and output validation later confirm completion; terminal invalid results fail the stage and block dependent readiness. Unknown accounting stays pending. | semantic | sweep |
| P19 | Automatic CPU continuation | Existing monitor may dispatch a single bounded, non-recurring CPU worker (1core12GiB20min) for ready CPU tasks or at most two retryable transient/output-directory failures. No GPU launcher. CPU worker is experiment execution, not a second monitoring scheduler. Unclassified failures require a verified code repair; no arbitrary autonomous patch generation. | interface | default |
| P20 | Review availability monitoring | Monitor reports JSON submission-file presence under reviews/incoming without accepting labels or changing human/reward readiness. Files are candidate submissions requiring independent provenance/adjudication, not verified truth. | interface | default |

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/progression/controller.py](<H:/Project/Poisoning/src/progression/controller.py>), [src/progression/tasks.py](<H:/Project/Poisoning/src/progression/tasks.py>), [src/progression/real_harness.py](<H:/Project/Poisoning/src/progression/real_harness.py>), [src/progression/monitor.py](<H:/Project/Poisoning/src/progression/monitor.py>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

新增CPU审计和完整版部署已完成；科学NO-GO。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
