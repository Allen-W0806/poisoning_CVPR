# 良性最小实验完整性复核

**结论：WARN。** 审阅由全新只读 `gpt-6-astra` / ultra agent 完成；`review_independence=same-family`，`acceptance_status=provisional`。它直接读取原始代码、日志、图像、二进制 witness、checkpoint 与报告，不是跨模型家族或人工认证。复核全程未进行训练或 GPU 工作。

|检查|状态|审阅结论与证据|
|---|---|---|
|A. Ground truth 谱系|PASS|14 个 source ID/content hash 互斥，但来自历史开发训练池。没有 target edit 或人工标签；图像评估明确为源图参考的光度/结构代理。`remote/gpu/source_manifest.json:96`、`src/mini_pilot/benign_validation.py:7`|
|B. 归一化|PASS|固定 /255、原始亮度误差和边缘余弦，没有按方法自身最大值放大分数。原始9个数值图像字段均可从PNG复算，最大误差小于8e-16。padding 问题属于区域定义误差，已另列勘误。|
|C. 文件与数字|PASS|独立核验原始283个文件、126个图像哈希、17个checkpoint sidecar、163个实际执行源码文件。二进制 witness 的6个精确重复/恢复比较均可重现。10行CSV与原始观察一致。原47项测试与运行后49项测试是不同执行。|
|D. 实际调用|PASS（限已报告路径）|数值诊断、逐转移score-function、reward、SSIM、clipping、配对统计和9次拒绝均真实执行。候选正向晋升分支、Outer进程崩溃后恢复未验证，未声称已验证。|
|E. 范围与预算|WARN|1个训练seed，4张assessment、4张暗化图；同一历史开发池、低步数。每种Outer为3候选/6次尝试更新/12次训练rollout/6次dev query/8次最终评估/152次显式predict调用；不是全部FLOPs。所有selected相同来自共同fallback，不能证明控制器等效。|
|F. 评估类型|PASS|真实图像指标为 `self_supervised_proxy`；Gaussian/随机UNet/ramp CPU fixture 为 `simulation_only`；GPU重复/续训是工程不变量。没有 `real_gt` 或 `human_eval` 编辑效能结果。|

## 核查到的实际行为

- 64条持久化轨迹、256个随机转移的相邻状态精确衔接；std均为正（约0.020648–0.782443）。用已记录noise独立重算平均坐标Gaussian log-prob，与保存值差异不超过1.81e-7。
- 32次实际optimizer调用均记录stored/replay log-prob差为0；4次Inner更新之外，数值诊断10次、Outer尝试18次的成本都保留。
- 两种计算精度的普通重复、原始/裁剪梯度、Adam、RNG及checkpoint后下一步更新精确一致，限同一进程/设备/shape。FP32为固定fp16权重variant的上转计算，不是另一套全精度源权重。
- 亮化与暗化全部8个逐源图Inner−Frozen差为负，均值分别为−0.000571128与−0.000711899；9个Outer候选全部拒绝，3种selected输出均与Inner字节相同。
- 12张配对预览tile与对应原始PNG逐像素相同。电脑/领带图的结构损失支持报告中的技术视觉观察，不能冒充独立人工评估。

## 发现、修复与最终复核

原始GPU运行的训练图 `574996000` 因centering取整不同，多计一列padding、遗漏一列内容。运行后只在已保存像素上重测，涉及17条训练/诊断观察，训练reward最大绝对变化0.00201113；dev无变化，assessment/transfer变化低于3e-17。原运行与checkpoint完整保留。**修复后的训练尚未运行**，反事实训练结果未知。

后续源码改用 `ImageOps.contain` 得到与Pillow相同的resize尺寸，再按相同round规则居中；回归包括奇数padding及343×25088极端纵横比。最后49项CPU测试通过，131个receipt哈希与最终源码匹配。修复后只读审阅确认以下三项均已解决：完整列出相对实际执行archive变化的3个文件；receipt键统一 `Path.as_posix()`；crop与Pillow resize/居中实现一致。

原CPU receipt未覆盖驱动自身及其余6个测试模块；完整原始archive包含它们且已独立核验。新receipt已补全。原status的retained_bytes在最终写入前计数，小于实际GPU文件总量85字节，最终文件清单优先。以上均已写入中文报告，不覆盖原证据。

## 主张影响与后续事项

支持：本轮有界score-function执行、短配置数值恢复、实际参数更新、原始样本上的负代理结果、9次拒绝与完整成本。

需限定：matched Outer调用预算、同开发池跨一条暗化指令的迁移观察、同进程恢复。源图bootstrap不估计训练seed不确定性。

不支持：实用编辑优势、语义保留、人类质量提升、一般性学习利弊、历史BF16全面修复、独立benchmark迁移、控制器优劣、Outer PPO或RSI。数值通路通过并没有建立可用的普通编辑质量基线。

当前没有未处理的上述代码修复项；余下是实证边界：先校准普通编辑基线、构建独立质量/指令遵循评估，再决定新短pilot。没有追加GPU任务。

原始证据目录：`experiments/benign_ddpo_audit_20261111_v1/remote/`；运行后CPU证据：`experiments/benign_ddpo_audit_20261111_v1/postrun_fix/`。机器可读结论为 `EXPERIMENT_AUDIT.json`。
