# P0 完整性复核

**Overall WARN；A PASS / B PASS / C PASS / D PASS / E WARN / F PASS。** 使用 `experiment-audit` 的新只读 gpt-6-astra / ultra 审阅。`review_independence=same-family`、`acceptance_status=provisional`；不是跨模型家族或人工认证。审阅没有训练、推理或启动GPU。

|检查|最终状态|直接核验的证据与边界|
|---|---|---|
|A Ground truth|PASS|16张源JPEG哈希、全部ID/问题、160个人类答案顺序、共识数和训练回答都与manifest及原始archive所提取记录匹配；两个官方archive SHA一致。编辑图仍只用原始标签的一致性proxy。|
|B Normalization|PASS|24条clean/edited QA score按vendored scorer独立复算；SSIM、亮度、clipping逐图复算；所有配置均值与LPIPS行平均一致。固定/255及LPIPS[-1,1]输入，不按模型最好结果缩放。没有独立再做LPIPS网络推理，不伪称所有像素指标均由另一模型重新运行。|
|C Artifacts|PASS|96个文件SHA/bytes全部匹配；28条编辑/重建观察有保存图。两个next-update witness的562个tensor和NumPy及其他RNG/metadata精确相同；两个checkpoint各112个public adapter张量与training_state逐字节相同。仅adapter文件推理加载路径未独立验证。|
|D Execution|PASS|主CPU64项、预处理CPU另3项实际通过。clean训练3个物理optimizer调用、2 retained steps；独立source test真正解码评分。Learned Controller本轮是合成CPU更新/恢复测试，不是新Outer效能试验。|
|E Scope/Budget|WARN|1训练seed、每split4源图、共识筛选偏倚；dev前后0.975不变。transfer是另一组clean QA，非独立编辑任务族。两作业170+57=227 GPU秒。P1表30行NOT_RUN、120个outcome cell为空。|
|F Evaluation type|PASS|clean QA为dataset human-label `real_gt`（仍受标注共识限制）；编辑QA和源参考图像质量为proxy；resume为工程等价；Controller合成fixture为`simulation_only`。无正式`human_eval`。|

Frozen原4个diffusion配置平均均变暗；对照guidance5/3平均略变亮，但SSIM下降、LPIPS升高，且配对图出现纹理。质量门槛未清除，P1不启动与证据一致。原生比例/指导参数诊断是同dev探索，不是独立确认，也不定位唯一根因：输入网格改变噪声几何，5/3同时改变两项guidance。

报告复核最初指出3个小的出处范围问题，最终均确认已修正：dHash为16源图的120-pair筛查、min Hamming19、0 flags，全面/历史感知去重仍NOT_RUN；8.009GiB峰值仅属于90965，90978峰值未记录，联合峰值未知；最终普查为479个文件。

支持：本轮clean VLM LoRA技术闭环、full-state fresh-object next-step恢复、实际GPU/CPU计数、固定split和有限Frozen质量测量。未支持：正常编辑学习优势、跨seed泛化、controller优越性、PPO、独立任务族RSI、完整公平baseline或论文方法成熟度。

证据：`experiments/p0_foundations_20261111_v1/LOCAL_VERIFICATION.json`、`original_annotation_extract.json`、`final_resource_receipt.json`及`remote/`。结论限定在实际执行源码和数据；旧实验保持独立。
