# Next Steps / Actual Status

- Reward算术、归一化、输入范围：审计完成；旧公式未发现所述cosine计算Bug。
- 任务目标修订：v2控制通过，identity=0；是良性曝光任务的新目标，不能泛化为ASR修复。
- Seed42新pilot：TRAINING。
- BF16：未修复；全FP32路径通过实际重复/恢复局部门，采用此路径完成pilot。
- Shadowcast干净/投毒模型及真实ASR：执行中；只报告已经完成的测量。
- BadVLM：按用户决定跳过；无正确作者代码/全文时不把BadVLA或猜测实现当作BadVLM。
- 多seed：本轮只seed42；不生成不存在的跨seed方差或显著性。
- Tables5/6/7：提供真实测量的插入稿，未测ASR/调度器/PBT/匹配外部比较保留NOT TESTED；不为80%覆盖率造数。
- 后续科学工作：先验证目标属性和独立语义保持，再预注册内部victim训练、触发定义与ASR分母；新增独立训练seed及真正matched baseline之后才评估方法优势。
