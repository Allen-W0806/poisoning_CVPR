# Phase 3 实验进度与结果

**真实 Generator RL 的单步机制已验证；batch=10 三轮更新已执行，但断点后严格续训复现未通过。**

方法是显式 DDIM 随机采样上的 DDPO score-function：保留 InstructPix2Pix 原图、文字指令和三分支 guidance，用单条图像轨迹的良性奖励计算真实 UNet LoRA 梯度。它区别于此前监督 MSE 和外部分类控制器。

完整单步通过 1 次；三轮通过 0 次。主要证据 `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`。

| 更新 | 奖励均值 | 奖励方差 | 梯度范数 | 单轮LoRA最大变化 | 参数更新秒数 |
|---|---:|---:|---:|---:|---:|
| 1 | -0.529893 | 0.221066 | 0.020314 | 9.99995e-06 | 106.60 |
| 2 | -0.439160 | 0.209795 | 0.018450 | 1.00133e-05 | 106.81 |
| 3 | -0.340771 | 0.186267 | 0.016661 | 1.00347e-05 | 106.83 |

实际可训练参数 1,594,368；冻结张量 1130 个，前后哈希一致=True。断点后下一步的适配器最大差异 2.8720805858029053e-06；参数精确一致=False，AdamW精确一致=False，随机状态一致=True，下一步loss一致=True。


本阶段实际累计 **1610 GPU 秒（0.4472 GPU 小时）**，包含全部失败与诊断，新增预算2小时。

奖励固定 S1、A0.5、D0.2、C0.1、U1，按个体保存，不做每批重标定。未核实的语义变化持续受罚；奖励有差异或梯度非零只支持“反馈已接通”，不能证明图像质量改善。表中的均值上升来自重新采样，缺少匹配冻结对照，不能解释为有效性提升。跨来源奖励方差也不能单独证明同一来源编辑有可控的学习信号。单个来源重复生成是相关观测，不能当作独立样本。

已生成10对原图/编辑图复核材料、CSV/JSONL空白表、逐项自动诊断和联系图。**人工独立核实仍为0，10对都保持 UNVERIFIED。** 新单图描述与训练奖励的成对判断使用不同输出，但仍是同一 Qwen 模型，不能称为独立模型或人工真值。

语义程序曾因整数计数不符合字符串 schema 而把10对全部标成解析失败；旧输出与旧表已保留。单独CPU审计修正计数表示，不新增模型调用、不改人工标签；详见语义报告中的v2派生表。

Qwen 保持冻结：既有开发集81.15%、一次性内部测试77.6333%未更改、未重跑。Phase3 不训练 Qwen，不实施后门/ASR/规避优化。

最终配置为 BF16 基础计算、FP32 LoRA 运算、物理单来源顺序 guidance；关闭梯度检查点后单步严格续训验证通过，但batch10仍未通过，未放宽容差。集群编辑器环境完整52项测试通过。三次早期单步失败、batch10完整运行失败和一次固定状态诊断全部保留。

M0–M4 和有限稳定性研究仅已准备配置，没有执行性能对照或 Generator RSI，因此 **RSI 增益尚无证据**。下一步先定位batch10梯度/AdamW数值差异、补独立语义标注，再按相同暴露量与完整成本执行小规模对照；不自动运行全网格。

原有两小时 Slurm CPU 监控链保留，扩展检查轨迹概率、条件、LoRA变化、梯度、奖励方差、断点完整性与独立Phase3预算；空闲时不启动研究作业。监控不会改变奖励/数据/测试标准。

证据位置：canonical `/storage/home/ydk297/projects/poisoning/experiments/EXP_*phase3_*`，本地镜像 `H:/Project/Poisoning/experiments/`；每个运行保留 manifest/config/环境/git/Slurm日志/轨迹/适配器与AdamW状态。

- `EXP_20261027T141314_7f2c3dc26636_phase3_slurm_one`: **FAIL**; `00950716195e20f9a08b2175b75d16221e1a231d`; Slurm `None`; config SHA256 `c258b5e71131a7e90253b025b6c69e49129f073e0d68e5134c7caca02c6472d6`.
- `EXP_20261027T141335_bc288d994eca_phase3_generator_rl_one`: **FAIL**; `00950716195e20f9a08b2175b75d16221e1a231d`; Slurm `86403`; config SHA256 `36388e7b5ed912fd6285b06dcbecff6b71ab945e912d6bef78ba32adfdc63832`.
- `EXP_20261027T141845_9b4a88c37076_phase3_slurm_one`: **FAIL**; `3dcf5947d32673a1a7075fdd9be8c099420372aa`; Slurm `None`; config SHA256 `7cacbbd0ff299647979d54908d05b755520b8a1a7f611b8055971d49b183d7e0`.
- `EXP_20261027T141857_470631a915ce_phase3_generator_rl_one`: **FAIL**; `3dcf5947d32673a1a7075fdd9be8c099420372aa`; Slurm `86404`; config SHA256 `21cdb9a46b19722adc02ec3a4e252aefc786ec1adf35015e6daca284278b036f`.
- `EXP_20261027T142201_648c36328684_phase3_slurm_one`: **FAIL**; `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`; Slurm `None`; config SHA256 `6bdb90d51116c1d9a2b61cf9b3836be57399df07c5a81e873107c78a0deddbeb`.
- `EXP_20261027T142212_e407797b1bef_phase3_generator_rl_one`: **FAIL**; `28b29ca7f4c605bc29082647d6be7c7f6f33aa05`; Slurm `86405`; config SHA256 `6672be0a9e91b191361ec76dd496baf617a73c5d1ef75d5e615c53683e71fd2e`.
- `EXP_20261027T142708_33e331a1db43_phase3_slurm_backward`: **PASS**; `bf29d2e530bf89e7255104d6a340c2f3c4cafb40`; Slurm `None`; config SHA256 `b75e7249855760a2630a5496b9bec03a2879f66fe157fdf678157d1e3d99a586`.
- `EXP_20261027T142716_ee26d8788a67_phase3_backward_diagnostic`: **PASS**; `bf29d2e530bf89e7255104d6a340c2f3c4cafb40`; Slurm `86408`; config SHA256 `10350b3b71aed00b60207a8fcc11b4d4594d1e2d5bf9439adb78cc8fcce91bde`.
- `EXP_20261027T143117_b4453cea2d66_phase3_slurm_one`: **PASS**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `None`; config SHA256 `a05bacfdef50c57b61521283d0a8da130916bf01fd584e7dde22e6636c0e56bf`.
- `EXP_20261027T143134_4f17668ff54a_phase3_generator_rl_one`: **PASS**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `86409`; config SHA256 `77eb5528c4ef433a3642a7b4a22acf76a6d50d3caae5146ae913f49fc136a7a4`.
- `EXP_20261027T143457_03808d628b67_phase3_slurm_smoke`: **FAIL**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `None`; config SHA256 `776352cf5ab2668d993db235f6d2cf5fcefe58cc7318447a0d5d65f24d304274`.
- `EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke`: **FAIL**; `7d548a34efa2558c9d5b61f8a559d06a953019e6`; Slurm `86411`; config SHA256 `95ed27bdd7157147c1298c74e632dad0796f5b17fa1dd2e757ad608ac4f6acdb`.
- `EXP_20261027T145338_d0c6788762ba_phase3_slurm_semantics`: **PASS**; `643b63c10f634635cda412462ff4e0588a2a39db`; Slurm `None`; config SHA256 `9cbbaed8e01b3aa6583b425e5285b1e5a30ee0e323e4b0ef698cc04fce6c13ef`.
- `EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review`: **PASS**; `643b63c10f634635cda412462ff4e0588a2a39db`; Slurm `86412`; config SHA256 `fc9565f863054222fbe7c60a0b879cd6a3e14c2d9f9beafa30ed1152890c3137`.
- `EXP_20261027T150420_0c1cc77f77e2_phase3_semantic_schema_audit`: **PASS**; `9e203d6012d8d0690c1505b9e99420def82ffb3d`; Slurm `None`; config SHA256 `c030a7caf3e2a50adf5fa03d9bfbffb34fda77edab2bf65c43b26e93b7d25231`.

详见同目录算法审查、实现、smoke、语义验证、RL/RSI对照、资源与失败报告。审查为 same-family/provisional，不构成独立科学认证。
