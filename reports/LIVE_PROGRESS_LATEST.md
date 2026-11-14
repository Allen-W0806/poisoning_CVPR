# Live RSI-Poisoning execution status

PROJECT_ID: RSI-POISONING. **Live checked UTC: 2026-11-02T16:47:03.238246+00:00**. Canonical root: `/storage/home/ydk297/projects/poisoning`.

**No RSI-Poisoning GPU job is currently running or pending** (0 found). Current numerical status: **ROOT_CAUSE_PARTIALLY_LOCALIZED; no verified production fix**. The 10-hour authorization expired at **2026-11-02T16:17:57.636501+00:00**. All unused allowances are accounting balances, not current authorization. This check submitted/restarted no jobs and changed no controller/budget state.

## Live Slurm results after 88433

Queried all user allocation records since 2026-11-02 with an end bound of 2026-11-04 to include future-eligible pending jobs, then selected numeric job IDs > 88433 with canonical project WorkDir or SubmitLine. Also inspected current squeue jobs with scontrol; excluded other projects by command/workdir. Job88433 is the reference anchor and is not charged again. All Slurm times below were requested with TZ=UTC; the raw anchor was also queried in America/New_York to cross-check timezone conversion. Allocation rows only (`sacct -X`), no batch/extern double counting.

| Job | Name | Type | State / exit | Submit UTC | Start UTC | End UTC | Elapsed s | GPU s |
|---|---|---|---|---|---|---|---:|---:|
|88434|poisoning-night-cpu|CPU/monitor|COMPLETED / 0:0|2026-11-02T06:18:06|2026-11-02T06:21:08|2026-11-02T06:21:13|5|0|
|88437|poisoning-monitor|CPU/monitor|COMPLETED / 0:0|2026-11-02T06:32:56|2026-11-02T08:32:55|2026-11-02T08:32:59|4|0|
|88438|poisoning-monitor|CPU/monitor|COMPLETED / 0:0|2026-11-02T08:32:55|2026-11-02T10:32:56|2026-11-02T10:33:00|4|0|
|88440|poisoning-monitor|CPU/monitor|COMPLETED / 0:0|2026-11-02T10:32:56|2026-11-02T12:32:57|2026-11-02T12:33:03|6|0|
|88483|poisoning-monitor|CPU/monitor|COMPLETED / 0:0|2026-11-02T12:32:58|2026-11-02T14:32:56|2026-11-02T14:33:00|4|0|
|88487|poisoning-monitor|CPU/monitor|COMPLETED / 0:0|2026-11-02T14:32:56|2026-11-02T16:32:56|2026-11-02T16:33:06|10|0|
|88495|poisoning-fix-localization|GPU diagnostic|FAILED / 1:0|2026-11-02T15:12:47|2026-11-02T15:12:47|2026-11-02T15:13:34|47|47|
|88497|poisoning-fix-localization_bounded|GPU diagnostic|COMPLETED / 0:0|2026-11-02T15:18:42|2026-11-02T15:18:42|2026-11-02T15:19:56|74|74|
|88499|poisoning-fix-attention_bridge|GPU diagnostic|COMPLETED / 0:0|2026-11-02T15:29:10|2026-11-02T15:29:10|2026-11-02T15:30:37|87|87|
|88502|poisoning-fix-h1_order|GPU diagnostic|COMPLETED / 0:0|2026-11-02T15:32:30|2026-11-02T15:32:30|2026-11-02T15:34:49|139|139|
|88503|poisoning-fix-h3_launch|GPU diagnostic|COMPLETED / 0:0|2026-11-02T15:43:13|2026-11-02T15:43:14|2026-11-02T15:45:59|165|165|
|88504|poisoning-fix-verification|GPU diagnostic|FAILED / 124:0|2026-11-02T15:50:57|2026-11-02T15:50:57|2026-11-02T15:56:37|340|340|
|88512|poisoning-monitor|CPU/monitor|PENDING / 0:0|2026-11-02T16:32:56|Unknown|Unknown|0|0|


Post88433 project GPU allocations: 6, states{"FAILED": 2, "COMPLETED": 4}, actual allocated GPU-seconds852. These GPU jobs are numerical debugging/verification fixtures containing optimizer updates, not newly trained benign-efficacy arms. No later project GPU allocation was found. Unregistered project GPU IDs: [].

Other active user jobs excluded from RSI-Poisoning accounting:

- 88507 / flash_v1_train / /storage/home/ydk297/projects/RSI_Video_Branching_20261027_v1/experiments/rsi_video_stage3_minimal_24gpuh_20261101/flash_adapter_20261102_v1/flash_v1_train_resource_v2.sbatch
- 88492 / rsi-night-v1-pair / /storage/home/ydk297/projects/RSI_Video_Branching_20261027_v1/experiments/rsi_video_stage3_minimal_24gpuh_20261101/night_20261102_v1/night_v1_pair.sbatch

These other projects were read only at scheduler-metadata level; their jobs were not modified, canceled or charged here.

## Current code and implementation

Git HEAD `06bd889d9392c4d53621c3e4af15ad718e84f6c5`. Tracked working-tree changes currently comprise 11 paths:

- ` M docs/METHOD.md`
- ` M scripts/manage_monitor.py`
- ` M scripts/run_phase4.py`
- ` M scripts/run_phase4_budgeted.py`
- ` M scripts/run_phase4_rsi.py`
- ` M scripts/smoke_generator_rl.py`
- ` M src/models/pretrained_editor_loader.py`
- ` M src/progression/dashboard.py`
- ` M src/rl/diffusion_rl_trainer.py`
- ` M src/rl/diffusion_trajectory.py`
- ` M src/rl/reward.py`

Git dirtiness is not proof of a numerical fix. All 8 preserved model-policy/trainer/trajectory/phase5-guard/config reference hashes still match; the core backward/loss implementation has not been corrected since the preserved failing numerical reference. The current loader has `strict_backward=False` by default and an optional call to `configure_strict_diffusion_backward`; the helper only sets CUDA_LAUNCH_BLOCKING=1/checks initialization and explicitly calls itself experimental after ordinary verification failed. AST comparison excluding docstrings shows current helper/loader functionality matches the failed candidate: {'cuda_execution_guard.py': True, 'pretrained_editor_loader.py': True}.

Current actual benign training caller loads the editor as:

- line 34: `editor=load_pretrained_editor(ROOT, device='cuda'); p=DiffusionPolicy(editor, cfg['diffusion_config']); t=DiffusionRLTrainer(p); p.contract['pilot_task']=cfg['engineering_proxy']; p.contract['source_manifest_sha256']=sha(run/'source_manifest.json')`

It does not opt in. The explicit opt-in exists in the diagnostic verification harness. No new verified fix or production enablement is present. Untracked entries and exact diffs/current hashes are retained in `git_evidence.json`; archived pre-fix and failed-candidate code remains in the numerical-debug evidence directory. This report-generation script is inspection tooling, not a numerical repair.

## Latest ordinary-path and resume evidence, re-read now

Latest relevant GPU verification is 88504, Slurm FAILED, exit124:0, 340 GPU-seconds. Its process-exit receipt records the bounded timeout at 2026-11-02T15:56:37.576574+00:00. CPU comparisons below were recomputed in this live check from the existing .pt files; no model/smoke/training experiment was rerun.

| Live CPU re-readback | Exact | Tolerance | Differing elements | Max absolute | Relative L2 |
|---|---|---|---:|---:|---:|
|captured_raw_pair|True|True|0|0|0|
|captured_postclip_pair|True|True|0|0|0|
|ordinary_before|True|True|0|0|0|
|after 1_adapter|False|False|744, 464|9.97831193672e-07|1.95410990013e-08|
|after 1_exp_avg|False|False|1, 593, 772|5.12326369062e-06|0.000383509823649|
|after 1_exp_avg_sq|False|True|1, 589, 926|1.6572982986e-09|0.000370291487399|
|after 2_adapter|False|False|1, 387, 812|6.40054304313e-06|7.35980197994e-07|
|ordinary_postclip_update2|False|False|1, 594, 366|0.000983255915344|0.0183695009218|
|saved_resumed_next_trajectory|True|True|0|0|0|


Ordinary recorded loss/norm by update:

- ordinary0: [{"step": 2, "loss": -2.1148666068911552, "gradient_norm": 1.033760666847229}, {"step": 3, "loss": -2.1152764037251472, "gradient_norm": 1.020628571510315}]
- ordinary1: [{"step": 2, "loss": -2.1148666068911552, "gradient_norm": 1.0337653160095215}, {"step": 3, "loss": -2.115276411175728, "gradient_norm": 1.0187238454818726}]

The captured raw pair passes, but the ordinary uninstrumented first and second updates fail. Candidate acceptance therefore fails regardless of capture-enabled results. Some tiny moment differences pass tolerance while failing exact equality; they are not relabeled bitwise equal. Resume actually restored a checkpoint and saved an exact 92-tensor next trajectory. Current existence checks still show no `resume/after 2/training_state.pt`, no completed update receipt, no postclip and no final guard receipt: **next optimizer update after resume remains NOT VERIFIED/TIMEOUT**. No newer project GPU run was found to supersede this result.

## Controllers, monitor and authorization

- Overnight run `overnight_benign_20261102_v2`: STOPPED_NUMERICAL_OR_EXECUTION_FAILURE; admissions_closed=True; init2026-11-02T06:17:57.636501+00:00; deadline2026-11-02T16:17:57.636501+00:00; EXPIRED. Its sealed ledger records 172/36000 GPU-seconds and 35828 unused. Do not interpret that stale unused balance as permission.
- Separate urgent-debug run `numerical_fix_20261102_v1`: ROOT_CAUSE_PARTIALLY_LOCALIZED; admissions_closed=True; 852/960 GPU-seconds, reserved 0, 108 unused; deadline2026-11-02T16:17:57.636501+00:00; EXPIRED. Shared authorization-envelope spend is 1024=172+852; unspent34976 is also expired. Neither finalized controller was reopened.
- Older validation ledger remains separate: cap 3600, consumed 2899, reserved 0, unused 701. Its last action is{'task': 'INSPECT_ONLY', 'reason': 'No eligible independent task; retain scientific blockers and use zero GPU.'}; this read-only request grants no new GPU allowance.
- Existing two-hour monitor: enabled=True, installed=True, interval2h; last tick **88487**, last successful check **2026-11-02T16:33:06.335240+00:00**; registered successor **88512**, due **2026-11-02T18:32:35+00:00**. Current squeue confirms one matching monitor. No duplicate recurring monitor was installed. Latest fallback read the finalized controller and returned TERMINAL_NOOP at 2026-11-02T16:33:05.740604+00:00.

**Currently spendable GPU budget:0 under the expired/closed authorizations.** No silent extension, restart or new admission was made.

## Blockers and next eligible action

The responsible CUDA primitive is still unresolved. Selected-module upstream adjoint divergence was localized, but launch-blocking failed ordinary verification; checkpoint-next-update proof is missing. Scientific reward release and independent development-evaluation admission also remain unresolved; 160 prospective final sources remain sealed. No new learned-model efficacy test is eligible; ASR NOT TESTED.

The next eligible work is CPU-only analysis/preparation of minimal whole-UNet prediction, loss-input and output-adjoint witnesses and a single-factor pre-import versus late-bootstrap launch test. Any new GPU diagnostic needs an explicitly renewed budget/window and preregistered discriminating hypothesis; do not reuse the expired ledger. After a genuine repair, ordinary uninstrumented/raw/postclip/Adam/adapter/ownership and completed resume-next-update checks must all pass before a matched Frozen-versus-DDPO benign learning test can be admitted.

Evidence for this live check: `results/live_progress_20261102T164701Z/live_snapshot.json`, `raw_readonly_queries.json`, `git_evidence.json`, `live_tensor_readback.json`. Slurm and artifact observations are current at the stated timestamp; Drive upload/readback verification is recorded separately. No raw tensors, images, weights, prompts, source or logs are uploaded with this aggregate report.
