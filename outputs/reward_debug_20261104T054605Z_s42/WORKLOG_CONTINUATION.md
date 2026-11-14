# Campaign continuation notes, 2026-11-04

User scope: seed42 only, Reward/editing/numerical debug plus code-available
Shadowcast reproduction; BadVLM skipped. Frozen=Fixed identical-prompt
equivalence control. New cumulative cap72000 GPU-seconds includes failures.
Canonical cluster root `/storage/home/ydk297/projects/poisoning`, SSH utc-login.
Do not charge or change other projects' jobs. No sealed internal pool opened.

## Active work, not completion claims

-88953 Shadowcast official PGD,277pairs,4000iterations,8/255. At08:13UTC140
pairs complete; projected completion around10:03UTC. Cap240minutes.
-88957 full-FP32 lighting-v3 pilot,100 actual updates per trained arm,
34/33/33 validation schedule. At08:28UTC DDPO92; Rule/RL not yet started.
Cap300minutes, cooperative17700seconds. Original5-update numerical witness
passed three1538-tensor exact/tolerance comparisons, frozen base unchanged.
-Foreground supervisor exec session72998, script supervise_reward_campaign.py,
started07:31:56UTC, bounded21600seconds. Poll existing session, never duplicate.
It will CPU-audit PGD, submit poison50 training, lock final clean0/poison50,
evaluate original-task ASR+104 development VQAs, audit raw scores, and run
preregistered frozen-Qwen editing quality after pilot completion. No auto retry.
-Actual campaign spend5.12556GPUh at the08:30-ish report refresh, NOT a final
cost. Refresh accountant for current actual/committed values.

Completed allocations include diagnostic88948, cancelled saturation-v2
88949, smoke88950, failed import88951, cancelled long-TMP88952, failed
argument collision88954, and clean0 trained88955 (1649seconds,26optimizer
steps, epoch0.96744). All costs remain included. Rejected v2 evidence is in
internal_pilot_v2_saturation_probe; do not treat its44generations as a completed
100-step comparison. V3 task/calibration version is lighting_v3_mean_progress_20261104.

## Next required actions

1. Keep watching both spines. After final two victim checkpoints are locked
and supervisor has reserved both original-task eval stages, manually submit
vizwiz_clean0 and vizwiz_poison50, each maximum120minutes, through the same
submit_reward_campaign.py accountant. Check unique receipts/queue before any
submission; no duplicate or retry. Avoid simultaneous accountant mutations.
Command uses .venv-shadowcast-20261104/bin/python scripts/evaluate_shadowcast_vizwiz.py
--campaign outputs/reward_debug_20261104T054605Z_s42 --stage evaluate --poison-count0/50.

2. VizWiz inputs READY:8000images, exact original LLaVA prompts;2026 public
references (10each). Data on cluster, no C: download. preregistration version4
locks worker/scorer hashes before any predictions. Official original API SHA
d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694.
Dedicated vizwiz_official.py executes original class/accuracy AST through
setAccuracy, excluding unrelated caption metrics. It keeps literal references
and unconditionally normalizes predictions. Historical VQAv2 scoring is not
changed. Original VQA scorer was NOT equivalent; counterexamples retained.
The official model loader needs llava AND lora in its model-path basename;
worker uses symlink aliases to identical locked checkpoints. Actual CPU
builder routing test passed for both conditions, before any weight load.

3. If88957 completes, no continuation. If it reaches the exact cooperative
TimeoutError and fully exits as FAILED, only then consider the prepared
internal_v3_continuation60-minute unique stage, same budget. Candidate verifies
original source SHA, primary receipt/terminal status and squeue absence. It
restores pending controller/Adam/RNG correctly, settles any interrupted
validation/action, then gates the actual next source/config with two exact
disk-reloaded GPU updates. CPU4tests PASS; actual continuation still UNUSED.
Do not resume numerical/objective/BF16 failure. Documented invocation is in
docs/LIGHTING_V3_TIMECAP_CONTINUATION.md.

4. After all required results exist: run CPU audit_completed_lighting_v3.py,
audit_vizwiz_results.py, then evaluate_shadowcast_vizwiz.py --stage statistics.
Supervisor already handles audit_shadowcast_results.py --stage statistics.
Pixel area definitions were fixed before assessment in pixel_area_audit_preregistration.json.
Mean source CIs are conditional on one fixed training seed, not seed CIs.

5. Refresh accounting and summarize_reward_campaign.py. Final reports must
retain negative/uncertain findings: CLIP arithmetic was correct, source0.2735
need not be0; scientific objective changed to explicit lighting-v3. BF16
casting stayed finite but failed ordinary-repeat equality; FP32 fallback is
not a BF16 repair. No internal poisoned-victim ASR, scheduler or PBT experiment
exists. Shadowcast original task is not a matched comparison to benign Ours.
Paper Sec4.1 describes2000 PGD steps; released code/run4000. Sec4.2 says>95%
for50-poison Trump-to-Biden, not universal92.1%. Table3 EvalAI partition is
unspecified; do NOT assert it was the4000-case test-dev split.

6. Mirror final source results/images and run local Matplotlib exports plus
native editable figures. Then fresh result-to-claim review per that skill
(explicitly requires gpt-6-astra ultra, fork none). No global evidence_check.py
was found in declared locations; record WARN/unavailable, do not invent PASS.
Semantic review remains same-family/provisional. Narrow unsupported claims;
do not automatically add seeds or tune for positive outcomes.

7. Final selected evidence packaging: package_reward_campaign.py requires
complete pilot, Qwen quality, external statistics and raw audits, all GPU jobs
settled within cap. Retain checkpoint/tensor evidence on cluster; selected
archive carries hashes and raw logs/responses/images. Verify archive/file SHA
after local mirror and publish the3root reports only after detecting any
independent edits. Clean only resolved run-specific scratch after all jobs
stop. Keep raw evidence, models and datasets. Report retained large artifacts.

## Local editable figure preparation

The presentation operation marker was ALREADY RUN ONCE (create,expected1PPTX)
before first authoring around08:18UTC; DO NOT repeat it.
Source scripts/export_reward_campaign_figures.mjs, bundled Node runtime from
Codex primary runtime. Runtime paths are explicit in script invocation.

Diagnostics-only QA successfully exported/validated a native bar chart with
embedded workbook and20real byte-embedded photos. Use blob ArrayBuffer with
contentType=image/png, not standalone Windows path references (placeholders).
Full export requires COMPLETED pilot and creates7slides: controls, validation,
frozen parameter audit, four pages covering all16final assessment sources.
All7preview PNGs go to run scratch for visual review, then cleanup. Final PPTX
and scientific Matplotlib PDFs/PNGs remain in campaign/figures. Full16 PDF is
four pages; first4also have a standalone preview.

Required Windows execution details: set TEMP/TMP to
H:\Project\Poisoning\tmp\reward_debug_20261104T054605Z and PYTHONDONTWRITEBYTECODE1.
Node MJS sets RUNTIME_NODE_MODULES internally. Native helper os.link requires
approved require_escalated execution (default sandbox denies hardlinks); all
targets are verified H: run scratch. No outstanding approval block.

The bundled literal-workbook adapter executes before finalizer in run scratch.
Plot values are rounded to9decimal display places for Excel15digit limits;
full raw JSON/CSV is retained. Workbooks are NEW literal snapshots, not original
source formulas. Do not set requiredEmbeddedWorkbookChartOwnerSlides, which
means preserving an original source workbook and conflicts with snapshots.
The accidental empty root .chart-data-Wchfac directory was verified and removed.
Diagnostics QA PPTX/previews/failed candidates remain only under run tmp and
must be removed at completion. Finalizer receipt stays outside final figures.

## Current reports and validation

Local reports/DEBUG_REPORT.md,BASELINE_COMPARISON.csv,NEXT_STEPS.md were mirrored
from campaign/reports and full SHA matched remote at the latest phase refresh.
They explicitly say IN PROGRESS. Before replacing them, compare root files to
the current local campaign copies to detect external edits.

Fresh final_doc_invocation_check passed all CPU documented commands: official
preflight/import; VizWiz preflight+3scorer+1actual-loader tests;4continuation tests.
It found a writing-preflight bug, now fixed to persist=False for --stage preflight;
GPU stage behavior unchanged. It verified protocol.json byte SHA/size/mtime did
not change, all25 original pilot source SHA still match, primary still RUNNING.
This is CPU doc verification, not GPU/full scientific reproduction validation.

## Update at09:39UTC

DDPO100, Rule68, RL0 committed updates; PGD240/277; actual GPU7.411h.
Rule round2 actually halved LR to2.5e-6; raw gradients/Adam finite and replay0.
No new failure. Foreground supervisor session72998 remains the sole core
orchestrator. Temporary functions monitor cells have finished; do not restart
or duplicate the supervisor. Primary still needs remaining Rule/RL and final
locked assessment, and all external victim/evaluation stages still pending.

The considered BF16 reduced-precision flag proposal was WITHDRAWN before any
GPU submission. Fresh review found the protected DiffusionPolicy constructor
already sets that flag False atline35. Process default True before construction
was not the actual policy value. Original proposed code is preserved under
rejected_proposals as UNEXECUTED. Current check_bf16_reduction_candidate.py is
only an AST/source-SHA CPU audit; diagnose explicitly rejects. Decision JSON
records0GPU seconds and submitted=false. No candidate stage/output exists.
Actual editor environment is Torch2.6.0+cu124; Shadowcast has its isolated
Torch2.1.2+cu121 environment. No BF16 repair is claimed. All25 primary source
hashes still match. Fresh CPU doc follow-up verified the withdrawal evidence.

Packaging helper now supports --phase review (REVIEW_INPUT_MANIFEST.json and
review_selected_evidence.zip) before semantic review/figures, and default
--phase final for final selected delivery. Both require complete raw audits,
measurements and settled GPU jobs. Use review mirror for local figures and
fresh result-to-claim assessment, then upload final deliverables and create
the final package. Keep both manifest receipts; remove intermediate ZIPs only
after checked local copies exist. Cleanup script
scripts/cleanup_reward_campaign_scratch.py defaults to dry-run and refuses
active GPU/CPU allocations; short scratch hashes must bind to this campaign.
Review its verified targets before --execute. Windows local cleanup must use
native PowerShell with an absolute resolved run-directory check.
