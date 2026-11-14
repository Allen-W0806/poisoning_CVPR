# 完整方法可运行性审计（2026-11-13，美东时间）

结论：**还没有形成可启动完整方法实验的实现。附件是未接线的参考骨架，不能把语法通过、已有clean VQA结果或旧Inner smoke当作它的端到端验证。**

本次读取`E:/asr_poison_rl.py`、`E:/README_RUN.md`、`E:/run.slurm`，并检查当前仓库核心源码与已有证据。附件说明中的“接hook、运行preflight、开始RL”是文档内容，不等于用户授权执行这些步骤。该附件将ASR用作生成器训练反馈，与此前正常任务实验边界不同；本次仅进行结构完整性审计，未接通、修复或运行后门攻击优化回路。

## 附件实际情况

|能力|实际位置|代码状态|运行/验证状态|
|---|---|---|---|
|Python语法|`E:/asr_poison_rl.py`|AST parse与compile通过|STATIC_PASS；没有import或执行目标模块|
|图像采样及实际trajectory log-prob|`generate_poison_batch`，第72行，raise在90行|未实现接口|NOT_RUN；没有可验证的采样轨迹|
|实际模型训练|`finetune_victim`，第93行，raise在100行|未实现接口|NOT_RUN；fresh-from-checkpoint只是接口约定|
|实际评估|`eval_asr`，第103行，raise在113行|未实现接口|NOT_RUN|
|clean/eval/source数据|`load_clean_fixed`第116行、`load_eval_set`第121行、`load_source_pool`第127行|3个未实现接口|NOT_RUN；没有实际manifest或互斥校验结果|
|policy与optimizer构造|`build_policy`，第132行，raise在134行|未实现接口|NOT_RUN；LoRA范围、dtype和optimizer配置未落地|
|Inner更新框架|`rl_train_loop`，第190行；loss在230行|部分实现：REINFORCE形式的循环骨架，依赖上述空接口|NOT_RUN；不能认证DDPO实现或训练效果|
|preflight|`preflight_check`，第175行|函数定义存在，依赖空接口|NOT_RUN；main里的调用被注释|
|BF16/FP32数值一致性|README、Slurm第17–19行；`build_policy`为空|说明与注释不构成精度修复实现|NOT_RUN / NOT_VERIFIED|
|完整checkpoint/恢复|全文静态调用检查|未发现模型/optimizer/scheduler/RNG状态保存恢复调用|未实现于此文件；NOT_RUN|
|持久化实验结果|`history`第202、243、248行|history仅在内存中返回；有控制台logging|未写出结构化结果或checkpoint；Slurm日志也没有本次运行证据|
|Outer Controller / RSI|此附件函数清单、README“没做”段|没有独立Outer/RSI实现|NOT_RUN|
|PPO|此附件实际更新代码|有advantage和梯度裁剪；没有old-policy/importance ratio/PPO surrogate clipping|不是PPO|

**实际有7个直接抛出`NotImplementedError`的函数，不是“已经接好两个hook”。**

主入口第269–275行只实际调用`Config`与`log.info`；第271行checkpoint路径仍是`/path/to/theta0`。preflight与训练调用均为注释。因此即使import依赖成功、脚本打印提示后正常退出，也不是完成一次实验。本轮没有通过执行这个模块来验证退出码。

## Slurm脚本并非当前可用实验入口

`E:/run.slurm:26`只调用上述占位入口；第14–15行环境设置仍为注释，没有固定当前集群环境、绝对工作目录、真实权重/数据路径、唯一run目录或本轮预算账本。集群适配与shell运行均为NOT_RUN。第6行请求24小时、1个GPU，其最大分配上限为24 GPU-hours，不符合此前夜间10 GPU-hours约束；不能以本次“检查可运行性”视为授权24小时任务。

本次没有调用sbatch、没有占用GPU，没有新建定时任务，也没有覆盖原实验、checkpoint或附件。

## 当前仓库已经具备什么

|模块|实际代码|可支持的结论|仍缺的证据|
|---|---|---|---|
|正常VQA LoRA训练、保存恢复、评估|`src/training/clean_vlm_pipeline.py:55` `CleanVLMTrainer`；update80、save109、restore116、evaluate129|已实现，已有三seed实际运行与只读复核|不能转移为本附件或完整图像编辑方法的验证|
|正常VQA对照|`outputs/overnight_benign_20261112T051015Z/comparison.json`与`final_read_only_audit.json`|test512：Frozen0.722070，SFT0.740625±0.009334；实际1.215833 GPU-hours|transfer配对区间包含0；仅独立source，不是跨任务RSI|
|扩散采样/log-prob与Inner更新|`src/rl/diffusion_policy.py:79` `sample`；`src/rl/diffusion_trajectory.py:10` `transition`；`src/mini_pilot/engine.py:98` `update`|已有实际代码与有界旧smoke证据|当前正常编辑质量下的有效学习与完整公平对照NOT_RUN；没有与本附件接通的证据|
|显式配对数据读取|`src/training/paired_generator_data.py:9` `load_training_pair`|数据接口及实际CPU读取/拒绝dev校验已有证据|完整paired SFT训练仍未验证；旧GeneratorTrainer不能替代|
|Learned Controller|`src/mini_pilot/controller_v3.py:48` `choose`、第56行`learn`|实际存在参数学习的REINFORCE实现|与当前paired任务的联调、等预算完整对照及跨任务改善未验证|
|Frozen图像编辑质量|`reports/frozen_quality_gate.md`及保存PNG|≥20步/512px已经运行；保留偏色、纹理问题等负结果|质量门槛仍NOT_CLEARED，不能开始依赖它的正式Inner/Outer GPU对比|

**可运行clean VQA，不等于完整Inner+Outer+RSI已经齐备。** 当前方法缺口还包括可靠编辑质量、完整显式配对SFT与合理对照、同split同预算的三seed Inner/Outer比较、独立任务族迁移证据。已有Learned Controller不能被说成完全没有实现，也不能把它的存在当成RSI有效性证明。

## 本次实际检查与证据

执行命令（实际运行）：

```powershell
H:\Anaconda\python.exe -B scripts/audit_external_scaffold.py --source E:/asr_poison_rl.py --readme E:/README_RUN.md --slurm E:/run.slurm --output outputs/method_readiness_20261114T013135Z/static_audit.json
```

结果：Python静态语法PASS，空接口7，main实际调用仅Config/log.info，目标模块执行NOT_RUN，GPU作业0。另实际parse当前6个核心源码，回读三seed正常VQA的既有核验结果。没有重复已完成P0模型测试。

机器可读证据、输入SHA、函数行号、命令输出在`outputs/method_readiness_20261114T013135Z/static_audit.json`、`static_audit.log`和`repository_facts.json`。原始附件不改写；本轮只新增静态审计脚本与审计记录，没有将ASR脚本安装为训练入口。

当前准入决定：**完整方法GPU实验NO-GO；已有clean VQA路径可作为正常任务基础。** 后续可继续完成正常图像编辑/QA代码和验证；ASR驱动的后门生成优化不接入该良性实验框架。
