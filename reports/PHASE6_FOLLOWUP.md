# Phase 6 后续进度与部署报告

更新时间：2026-10-28T16:05:14.234629+00:00。本报告记录 CPU 后续审计和 Dashboard 部署；已完成的 Phase 6 测试和六项阶段没有重跑，未启动新科学 GPU 训练。

## 已完成及实际新增

新增不可变实验：`EXP_20261028T154458_eefb8fb1fe7a_phase6_review_dashboard_followup`。

- 核对 24/24 图片对、48/48 图片文件，24 项精确历史 episode 匹配和 24 项原始 COCO 到预处理图的像素关联；原图、编辑图、问题、10 个原始参考答案、指令、模型版本、种子和轨迹来源均匹配。原始答案不会被当作编辑后答案。
- 新独立审查工作台：`reviews/workflow/index.html`；七个维度全部默认 UNKNOWN，编辑后事实答案 UNKNOWN。可保存本地草稿并导出独立审查 JSON。用 `scripts/submit_independent_review.ps1` 提交，必须有真实来源证明和 operator 核实才能计为独立证据。不得以文件元数据自动证明人工身份。
- `reviews/incoming/*.json` 每次监控会被读取；无效文件隔离记录。按 reviewer 和图片哈希核验，保留冲突与分歧，输出维度覆盖、配对一致率、Cohen kappa（常数/无重叠时不计算）、编辑答案一致性及自动代理分组分析。样本是优先诊断队列，不是总体无偏抽样。
- 500 条旧记录已分离为 `raw_measurements.jsonl` 与历史标量。230/500=46% 下限裁剪；训练子集历史 51.56% 采用不同分母。C=1/3、U=1 为常数，S=0.5A；全部缺少指令成功度及校准语义不确定性。真实生成/判别耗时另外保留。未调系数、未发布新训练奖励。安全 ASR 不纳入良性编辑反馈。
- 新测两项真实历史图片结构工作流修订：改变检查顺序、摘要记忆。沿用固定 24 来源/72 行 Gate、零回退且至少节省20%诊断操作门槛；两项均 144→144，准确率1→1，没有降低操作数，故均拒绝，接受数0。原已接受工作流不变。不是语义效果提升。新run的本地父版本别名H0000，对应source_accepted_harness中的原实验H0002；候选版本只在各自run内有意义，不应混作全局新版本。
- 两轮 Gate 全部新成本：576 次诊断操作、174140736 字节读取、1.988536s wall、1.952700s CPU。原调查成本5.820236s wall/5.723810s CPU、1040操作、332716553字节作为复用证据单列，未再次执行或隐藏。
- 新审查历史 seeds42/123 的真实 step0 checkpoint：LoRA、优化器、scheduler、步数、记录到的各 RNG 状态相同，adapter最大差0；科学 contract 相同，全元数据不同。这不能证明 frozen base、source order、rollout noise 和后续执行可比性，也不是新 pilot 初始化已验证。
- 本次实验 wall=4.185370s、CPU=2.538867s、GPU=0、模型调用=0。单独的新接口测试7/7通过；原92/92是历史 Phase6结果。

## 独立证据及 NO-GO

当前独立审查提交0、经独立来源核实的共识图片对0。没有新增独立语义证据；人工一致性和自动代理对人工的有效性尚无法实证估计。UNKNOWN未转为人工标签。

M0–M5 保持 PREPARED_NOT_RUN / BLOCKED。沿用已有相同模型与 LoRA配置、split哈希、seed、评估定义等静态检查，未重复阶段测试。仍需真实 shared pristine initialization、frozen服务权重身份、训练源/噪声顺序及干预前匹配证据。现有 development/Gate 已暴露，不是新隔离评估。统计单位为 seed 内独立源内容簇，重复transition、问题和72行元数据控制不能充当72个独立图像；未来按seed报告配对不确定性，不挑最佳seed/step。

阻塞项：独立编辑后事实与语义审查；有证据支持的良性奖励；新隔离 Gate；干预前可比性；明确 GPU 预算。授权 GPU=0。全比较42.1954 A100h含20%余量仅为历史预测，未含人工与新独立评估成本，不能称完整获批成本。下一项合格工作是独立人工审查和来源核实；CPU即可，人工时间未知。收到标签后可以自动做来源检查、分歧和代理一致性分析，仍不能自动放行科学训练。

## Dashboard 和持续更新

使用用户完整包 UI、SVG图表与实验解释；原包SHA见 `dashboard_package/phase6_supplied_v1/PROVENANCE.json`，完整ZIP保留。原minimal代码与快照保留。唯一权威输入 `dashboard/dashboard_data.json`；`state.json` 仅为同内容兼容导出，index_zh/en为完整页面别名。旧实验记录没有覆盖。

Windows：`H:\Project\Poisoning\dashboard\rsi_poisoning_zh.html` 和 `H:\Project\Poisoning\dashboard\rsi_poisoning_en.html`。

集群：`/storage/home/ydk297/projects/poisoning/dashboard/rsi_poisoning_zh.html` 和 `/storage/home/ydk297/projects/poisoning/dashboard/rsi_poisoning_en.html`。

Experiment.finish 完成事件和已有两小时 Slurm monitor 的 cycle 调用同一刷新入口。首页、当前阶段、Stage状态/ID、已完成实验与原始指标、奖励审计条图、独立审查数、问题修复记录、active jobs和时间一起重建，不只追加registry。发布采用互斥锁与逐文件原子替换，JSON最后发布；同步会核验四页与JSON时间一致。

实际 Chrome QA：`reports/phase6_followup/browser/browser_qa.json`；当前 PASS=True。测试真实 HTTP 服务、真实集群刷新、Windows同步、两页25秒自动轮询、中文→英文、file手动重载及审查图片/defaults；未伪造科学指标。源代码 `scripts/qa_phase6_followup_dashboard.cjs` 可复查。HTTP每25秒读取JSON；file模式每25秒重新读本地HTML，无法跨机器自行拉取数据。

关闭 Codex：集群 Slurm successor 链仍运行，无重复调度器。Windows本地副本需要运行 `powershell -NoProfile -File scripts/sync_phase6.ps1` 才会接收集群新导出，没有偷偷建立第二个Windows定时任务。浏览器刷新与跨机数据同步是两件事。

## 当前监控证据

现有 Slurm monitor 的最后成功：2026-10-28T14:32:42.006754+00:00；最后tick作业 86649。下一唯一作业 86661，资格时间 2026-10-28T16:32:35+00:00；实际队列与sacct证据见 `reports/phase6_followup/monitor_verification.json`。这是本次查询的快照，作业号会随链条变化。

监控仅检查jobs、完成/输出完整性、科学阻塞、review提交及Dashboard同步；每个可复现工程失败签名最多2次修复。科学目标、奖励定义、标签、标准不自动修改。不得自动发起新GPU训练/attack optimization。不会将结构性文件测试当作语义提升。
