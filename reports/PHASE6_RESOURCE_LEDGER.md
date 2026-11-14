# Phase6 resource ledger

Scientific GPU allocation, model inference calls and new generator optimizer updates are all0. The historical42.20h forecast remains unapproved.

Actual Slurm CPU accounting (not GPU time):

```json
{
  "sacct_stdout": "JobIDRaw|State|ExitCode|ElapsedRaw|ReqTRES|AllocTRES\n86601|COMPLETED|0:0|33|billing=1,cpu=1,mem=12G,node=1|billing=1,cpu=1,mem=12G,node=1\n86602|COMPLETED|0:0|26|billing=1,cpu=1,mem=12G,node=1|billing=1,cpu=1,mem=12G,node=1\n",
  "exit_code": 0,
  "science_gpu_seconds": 0,
  "scope": "CPU-only allocations; login CPU checks excluded from Slurm elapsed accounting"
}
```

Initial CPU job86601 completed0:0 in33 elapsed seconds; corrected audit descendant job86602 completed0:0 in26 seconds. Both request1CPU/12GiB/20min and no GPU; elapsed59 CPU-allocation seconds combined. CPU process timings in each stage receipt exclude some startup/scheduler overhead and must not be presented as Slurm elapsed time. Additional login-node regression/monitor checks are CPU-only and separately logged; no fabricated total CPU utilization.

Existing recurring monitor remains1CPU/2GiB/5min every two-hour eligibility interval. No duplicate monitor scheduler. Any future ready engineering task uses a one-shot bounded CPU worker, while scientific GPU launch has no implementation in this controller. Include all unsuccessful submissions, repairs and Gate attempts in future accounting.

Review image copies, immutable audit outputs and browser QA evidence are retained on H:, not C:. No new model download or multi-GB C: artifact. Run-specific patch/profile/server scratch is removed at closure; source, final dashboards, review images and negative evidence remain. Final exact disk and retained-size receipt: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [configs/phase6.json](<H:/Project/Poisoning/configs/phase6.json>), [src/progression/monitor.py](<H:/Project/Poisoning/src/progression/monitor.py>), [src/progression/real_harness.py](<H:/Project/Poisoning/src/progression/real_harness.py>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

新增CPU实验wall 4.185370s、CPU 2.538867s；GPU和model calls均0。后续文档/浏览器/监控CPU成本未虚构为完整合计。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
