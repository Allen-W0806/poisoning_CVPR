# 本轮实际执行账本

|阶段|实际命令|结果|证据|
|---|---|---|---|
|源码语法/普查|python scripts/inventory_p0_code.py|已执行，479个源文件/配置；完整函数计数见CSV|code_inventory.md/CSV|
|CPU gate|python scripts/p0_cluster_io.py cpu|64 passed，0 skip|remote/cpu.log、cpu_tests.json|
|冻结数据|python scripts/p0_cluster_io.py prepare|16源图4/4/4/4互斥；新公开原图2392145 bytes|remote/manifest.json、manifest_lock.json|
|P0 GPU|python scripts/p0_cluster_io.py submit|job90965 COMPLETED；170 GPU秒|submission.json、slurm-90965.log|
|预处理CPU测试|quality_followup内部python -m unittest tests.test_native_preprocessing -v|3项通过；与64项suite分开|remote/quality_followup/cpu.log|
|有界预处理推理|python scripts/p0_cluster_io.py quality_followup|job90978 COMPLETED；57 GPU秒；8个dev输出、0更新|remote/quality_followup/|
|结果回读|python scripts/p0_cluster_io.py fetch|模型日志、图片、checkpoint回读并哈希核验|LOCAL_VERIFICATION.json|
|P1/P2|NOT_RUN|门槛未开|P1_results_unrun.csv空值|

纽约时间：2026-11-11T23:27:21.840544-05:00 → 2026-11-11T23:30:08.643951-05:00。CPU与GPU源码SHA由receipt绑定，配置、数据、editor权重文件、Qwen实际base tensor哈希都保留；依赖版本见remote_inventory.json。旧结果没有覆盖。本轮完整命令：`/storage/home/ydk297/projects/poisoning/.venv-editor/bin/python /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1/code/scripts/run_p0_foundations.py --assets-root /storage/home/ydk297/projects/poisoning --run /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1`。


## 保留与清理

本轮源码、日志、checkpoint、原始/生成图与最终预览在H盘保留约70.24MiB。模型权重与数据未下载到C盘，未安装新运行时。C盘可用57.42→57.20GiB（差额未归因）；H盘可用212.49GiB。本轮空scratch已验证路径后移除；会话历史、共享运行时、模型和旧证据未删除。final_resource_receipt.json确认两作业已COMPLETED且本轮active jobs为空。
