# 本轮实际命令与证据

Run ID `overnight_benign_20261112T051015Z`。固定协议和来源副本在本目录；远端为`/storage/home/ydk297/projects/poisoning/outputs/overnight_benign_20261112T051015Z`。

实际本地执行（Windows PowerShell，项目根目录）：

```powershell
python -B -m unittest tests.test_night_budget tests.test_night_analysis -v
python -B scripts/night_campaign.py stage
python -B scripts/night_campaign.py start
python -B scripts/night_campaign.py monitor
python -B scripts/night_campaign.py fetch
powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File scripts/Install-RSIExperimentMonitor.ps1
Start-ScheduledTask -TaskName RSI_Experiment_Monitor
Get-ScheduledTask -TaskName RSI_Experiment_Monitor
Get-ScheduledTaskInfo -TaskName RSI_Experiment_Monitor
```

`stage`是一次性操作，不可重复覆盖目录；`start`有持久账本去重，后续使用`monitor`。SSH均使用BatchMode/IdentityAgent=none/IdentitiesOnly=yes，已实际成功，无交互密码/agent依赖。新增13项CPU测试原始输出`new_cpu_tests.log`；cluster source快照中10项预算测试输出`remote/new_CPU_guard_tests.log`。

Slurm实际提交的脚本与回读保存在`remote/slurm/`与`remote/ledger.json`。数据准备91020/完成CPU回调91021；canary91023/回调91024；Frozen VQA91029/回调91030；seed42 91043/回调91044。实际后续job ID以账本为准，不预造seed123/3407的job ID。P1 CPU配对检查91042（零GPU），证据`remote/analysis_code/submission.json`、`remote/p1_preparation/CPU_check.json`。

实际GPU工作命令（每个stage/attempt都在唯一输出目录）：

```bash
python -B code/scripts/night_vqa_worker.py --campaign <run> --stage canary --attempt 0
python -B code/scripts/night_vqa_worker.py --campaign <run> --stage frozen --attempt 0
python -B code/scripts/night_vqa_worker.py --campaign <run> --stage seed42 --attempt 0
```

可复查命令（只读，不重复提交）：

```bash
squeue -u ydk297
sacct -X -j 91023,91029,91043 --format=JobID,State,ElapsedRaw,AllocTRES,Start,End
```

故障/修复如实保留：首次stage前Windows命令行过长，未上传/提交作业，改为stdin打包传输；首次remote start因工作目录不正确导致新CPU测试找不到，未提交GPU，修复前源码已保留。控制平面后续加严`COMPLETED+PASS`联合门槛与显式`--no-requeue`，修订前内容和SHA在`remote/controlplane_revisions/`。没有改动worker、训练协议、数据、LR、reward或test；已有活动job的Requeue已由owner设置为0。首次分析脚本拼装的换行转义语法错误在提交前失败，修复后实际提交CPU job91042。以上失败均未产生GPU占用。

Windows S4U注册被系统权限拒绝（不是自动审批拒绝）；Interactive注册、启用和实际手动触发均成功，LastTaskResult=0。需保持当前用户登录，锁屏/关显示器可继续；不声称已验证注销后运行。集群完成回调与Windows会话独立。完整权限、电源、SSH与计划时间证据`reports/monitor/scheduler_receipt.json`及导出的任务XML。

## Codex首次后台触发后的实际完成复核

2026-11-12T07:55:38.977Z收到真实Codex heartbeat，执行`python -B scripts/night_campaign.py monitor`，全部五个GPU作业COMPLETED，实际4377 GPU秒。本轮末两个正式seed实际job ID为91057、91078；对应CPU完成回调91058、91079亦COMPLETED，无重试。

只读命令`python -B scripts/audit_night_vqa_results.py --run outputs/overnight_benign_20261112T051015Z`已通过，输出`final_read_only_audit.json`及日志；重新核对3840条官方兼容评分、三seed固定训练流和各512次有限真实更新、dev-only选择、job不重叠与总预算。第一次审计误将本地CRLF JSON副本要求为远端LF完全相同byte SHA，因此assertion失败；实际remote SHA匹配锁，JSON值相同。改为校验远端原始byte SHA并对本地副本做语义一致检查后通过；两个原始副本均保留，协议没有改动。

另外通过已有SSH做当前1792张原图与512个配对文件的完整哈希复核、最终sacct、du/df、scratch存在性检查，回执`remote/final_resource_receipt.json`。全部验证为CPU操作，没有新GPU作业、模型forward或optimizer调用。

用户纠正后Windows任务已停用；真正的Codex heartbeat已观察到上述后台执行。本轮无剩余任务，已调用`automation_update(mode=delete,id=rsi-experiment-monitor)`成功删除，删除/触发收据`reports/monitor/codex_automation_receipt.json`。未重新创建Windows任务、cron或其他替代定时器。
