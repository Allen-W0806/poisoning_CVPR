# 实际实施状态：P0 foundations

**P0工程工作已实际运行；P1未启动。** 主CPU suite64项通过，另3项预处理测试通过。A100作业90965/90978分别170/57秒，合计227 GPU秒。Clean VLM闭环成功；Frozen已校准到512px/20–50步并做有界预处理对照，但当前任务质量门槛为PARTIAL_NOT_CLEARED_FOR_P1。

|任务|状态|实际证据与限制|
|---|---|---|
|P0.1代码事实核查|已完成|24能力，函数/文件逐项定位，PPO不存在；64项CPU回归。不是所有入口都跑过，无运行证据标NOT_RUN|
|P0.2 Frozen校准|已执行，门槛未完全通过|4张固定dev、VAE重建+20/30/50原生步+30随机DDIM；完整图像/指令/SSIM/LPIPS保存。512px/20–50步下主要场景可辨认，未见上一轮部分4步样本那样的全面纹理崩坏；但斑马图明显偏黄、棒球场饱和度改变，固定亮化指令未稳定满足。去除白色方形padding使部分感知距离下降，但不能单独消除偏色；guidance 5/3减轻偏色同时引入颗粒纹理。没有找到可同时认证指令符合与内容保留的配置，保留全部负结果。|
|P0.3 clean VLM|PASS_ENGINEERING_ONLY|train4，2 retained updates/3 actual calls，fresh-object next-step及最终预测精确恢复；test consensus72.5%（n4）|
|P0.4协议|固定并验证（有限范围）|4/4/4/4 ID与content互斥、配置/manifest/hash、源图bootstrap及预算账本；独立任务族、pretraining与完整感知去重未验证|
|P1/P2方法价值|NOT_RUN|没有>=3seeds完整对照，未将smoke当成效能或RSI证据|

## 修改文件

- `src/mini_pilot/controller_v3.py`：补全learned controller恢复与严格字段/张量检查。
- 新增`src/training/clean_vlm_pipeline.py`：response-only正常QA、FP32语言LoRA/Adam、严格checkpoint、fresh-object恢复、真实解码评估。
- 新增`src/data/p0_protocol.py`：固定split校验、内容SHA、源图统计；`src/evaluation/benign_editing_quality.py`：仅评估SSIM/校准LPIPS/content crop。
- 新增`src/editing/native_preprocessing.py`及`tests/test_native_preprocessing.py`：无白色方形padding的等比例网格resize；另3项测试。原有预处理默认不改写；在相同4-dev上真实测量了2个30步配置，效果仍有限。
- 新增`configs/p0_foundations_20261111.json`、`CODEX_TASKS.md`、`tests/test_p0_foundations.py`。
- 新增执行/复现脚本：`scripts/p0_cluster_io.py`、`prepare_p0_manifest.py`、`run_p0_foundations.py`、`test_p0_foundations.py`、`inventory_p0_code.py`、`report_p0_foundations.py`。
- 新增`scripts/run_p0_quality_followup.py`：最多5分钟的同dev推理诊断，拒绝覆盖和重复提交，保留全部输出，不使用test或reward优化。
- 未改写旧checkpoint、旧论文、原始数据或上一轮结论；本轮在独立run目录保存来源与结果。

## 关键实测结果

|配置|n|SSIM ↑|LPIPS ↓|平均亮度变化|方向变亮的源图|
|---|---:|---:|---:|---:|---:|
|vae_roundtrip|4|0.652760|0.082231|-0.006544|0/4|
|native_20|4|0.552083|0.174997|-0.028025|3/4|
|native_30|4|0.565553|0.167602|-0.025375|2/4|
|native_50|4|0.570467|0.155478|-0.026174|2/4|
|ddim_30|4|0.552687|0.196627|-0.009602|2/4|
|native_aspect_30|4|0.564799|0.135883|-0.030945|2/4|
|native_aspect_preserve_30|4|0.506705|0.226840|+0.001357|3/4|


clean VLM：dev前后97.5%→97.5%（n4）；独立source test72.5%（n4，源图bootstrap95%CI[25%,100%]）；clean-QA transfer100%（n4，退化经验区间不能证明总体确定）。没有训练收益结论。编辑dev使用原始QA答案的一致性为97.5%，不是编辑后正确答案认证。

## 当前能力回答

- Can run clean VLM end-to-end? **YES，bounded LoRA engineering**。
- Can run reliable benign Inner RL? **PARTIAL**：更新/数值技术路径存在且有smoke，正常编辑学习优势未验证。
- Is learned Outer RL implemented and trained? **YES，REINFORCE的CPU与历史3次开发更新；不是PPO，当前任务效能未验证**。
- Do we have cross-task RSI evidence? **NO**；当前transfer仅独立source的clean QA。
- Do we have fair baseline results? **NO（完整当前任务组合）**；只有Frozen校准结果与旧有限smoke，P1表格未运行值空置。

## 实际命令、证据与下一步

执行：`python scripts/p0_cluster_io.py stage`、`cpu`、`prepare`、`submit`、`status`、`fetch`，另有`quality_followup`和`quality_status`。Slurm内真实命令在`reproduction_ledger.md`，主进程PID 2754066。90965峰值CUDA allocation约8.009GiB；90978峰值未记录，不声称两作业总体峰值。源图/指令/seed/输出PNG/metrics、训练、预测、checkpoint及resume比较在`experiments/p0_foundations_20261111_v1/remote/`，规范指标在`outputs/clean_vlm_metrics.json`。

下一条可立即执行的无GPU复核命令：

```powershell
python scripts/inventory_p0_code.py
```

CPU回读/可视化命令`python scripts/report_p0_foundations.py --preview-only`已在本机默认Python实际运行；模型工作仍使用现有集群环境，因为本机Anaconda缺Diffusers且实际import torch有已知OpenMP冲突。新的GPU重现必须使用新run标识和固定协议，不可覆盖本轮目录。当前不追加P1训练；先完成当前普通编辑任务的指令符合/质量审阅，并具备完整SFT与reward-finetuning对照、独立任务族和>=3seeds协议。

详细材料：`code_inventory.md`、`frozen_quality_gate.md`、`clean_vlm_pipeline.md`、`experiment_protocol_p0.md`、`outer_controller_audit.md`、`missing_components.md`。所有结论以本轮日志和checkpoint为准。


## 原始标注复核

已从缓存的官方questions/annotations ZIP直接提取所选16个原始记录，核对全部问题、image/question IDs、160个人类答案及其顺序、共识数和训练回答。原始提取与archive SHA在`experiments/p0_foundations_20261111_v1/original_annotation_extract.json`；没有复制完整大archive或打开历史封存test。


## 保留与清理

本轮源码、日志、checkpoint、原始/生成图与最终预览在H盘保留约70.24MiB。模型权重与数据未下载到C盘，未安装新运行时。C盘可用57.42→57.20GiB（差额未归因）；H盘可用212.49GiB。本轮空scratch已验证路径后移除；会话历史、共享运行时、模型和旧证据未删除。final_resource_receipt.json确认两作业已COMPLETED且本轮active jobs为空。


## 新只读完整性复核

`experiment-audit`的全新gpt-6-astra/ultra审阅得到A/B/C/D/F PASS、E WARN，总体WARN（same-family/provisional）。实测数值/文件和恢复一致性通过，科学范围与Frozen任务质量仍有限。详见`P0_EXPERIMENT_AUDIT.md`及JSON；不外推为完整论文方法验证。

## 2026-11-12 夜间执行补充

未重复已完成P0。新固定1792-source clean VQA cohort与256组P1正常配对已准备、CPU校验；新canary91023通过，正式Frozen VQA91029完成，三个seed按账本串行执行。已真实创建/启用/手动测试Windows `RSI_Experiment_Monitor`，每2小时更新心跳与夜间报告；当前Interactive权限需保持用户登录，锁屏和关闭显示器不影响。实时job ID、实际GPU-hours、完成/运行中/未提交与checkpoint/日志以`reports/OVERNIGHT_EXPERIMENT_STATUS.md`和`reports/monitor/heartbeat.json`为准。Frozen图像编辑质量未过门槛，因此P1 GPU仍NOT_RUN，未将Inner score-function/Outer REINFORCE改称PPO。

用户随后明确纠正要求Codex内置定时任务。现已通过Codex应用工具创建并回读 `rsi-experiment-monitor`（heartbeat，ACTIVE，每2小时，当前聊天）；误建Windows同名任务已停用。上段Windows创建和测试是历史事实，不再代表当前调度方式。`reports/monitor/codex_automation_receipt.json`记录应用任务状态；Windows停用回执另存。首次Codex后台定时触发尚未观察到，不能把旧Windows测试称为Codex触发验证。

## 夜间完成与首次Codex触发验证（2026-11-12）

07:55:38 UTC真实Codex heartbeat已触发并成功执行SSH/Slurm监控。全部五个本轮GPU作业COMPLETED：canary91023（45秒）、Frozen91029（263秒）、seed42 91043（1361秒）、seed123 91057（1348秒）、seed3407 91078（1360秒）；总4377 GPU秒=1.215833 GPU-hours，零重试，作业时间不重叠，本轮无活动GPU。三个seed各实际512次optimizer更新、1024训练source各出现两次；dev-only最终选择分别128/512/512步，未用test挑选。全部selected checkpoint和adapter SHA已回读通过。

正常VQA test512：Frozen0.722070；SFT三seed均值0.740625、训练seed SD0.009334；配对Δ0.018555，固定三seed条件下source bootstrap95%CI [0.001107,0.036330]。transfer128为同任务族独立source：Frozen0.709375、SFT0.735156±0.024619；ΔCI [-0.008854,0.062760]包含0，不能确认迁移收益。仅支持本固定正常VQA cohort的初步SFT表现，不支持生成器RL/Outer/RSI或完整论文方法主张。图像编辑Frozen质量门槛仍未过，P1 GPU均NOT_RUN。

只读复核重新计算3840条预测评分、三条训练流/梯度与更新日志、dev-only选择和预算；集群再次核验全部1792张原图及256组配对512文件字节哈希。证据在本轮`final_read_only_audit.json`和`remote/final_resource_receipt.json`。本地协议副本为Windows CRLF而remote为LF，原始远端byte SHA仍匹配锁、JSON值完全相同，未修改协议或数据。全面历史感知近重复与预训练曝光仍未验证。

本轮全部阶段已完成且没有可继续的已授权GPU实验，已通过Codex工具删除`rsi-experiment-monitor`，停止空轮询；误建Windows任务继续保持Disabled。后台实际触发及删除证据在`reports/monitor/codex_automation_receipt.json`。源图、配对、全部checkpoints和日志继续保留在远端约804.4MiB，run scratch已不存在；本地仅保留小证据。完整结果见`reports/OVERNIGHT_EXPERIMENT_STATUS.md`。
