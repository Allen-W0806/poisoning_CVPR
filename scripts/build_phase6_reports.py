"""Render Phase6 reports from actual stage receipts, not estimated experiment outcomes."""
import json
from pathlib import Path
import _bootstrap
from src.utils.logging import git_commit,write_json

ROOT=_bootstrap.ROOT;REPORTS=ROOT/'reports'
def load(name,default=None):
    p=REPORTS/name
    return json.loads(p.read_text(encoding='utf-8-sig')) if p.exists() else default
def link(p):return f'[{p}](<{(ROOT/p).as_posix()}>)'
def main():
    state=load('PHASE6_STAGE_STATUS.json');receipts={s:load('phase6/'+s+'.json',{}) for s in state['stages']}
    results={s:r.get('results',{}) for s,r in receipts.items()}
    data=results['S1_DATA'];reward=results['S1_REWARD_AUDIT'];harness=results['S2_REAL_HARNESS']
    gate=harness['selected_result'];review=receipts['S1_REVIEW_PACKAGE'];rl=results['S3_RL_ENGINEERING'];comparison=results['S4_STATIC_COMPARISON']
    live=load('phase6/live.json',{});scheduler=live.get('monitor_scheduler_state',{})
    resource=load('phase6/resource_snapshot.json',{})
    tests=load('phase6/final_test_receipt.json',{'canonical_count':92,'canonical_exit':0,'scope':'review1 recorded run, final closure pending'})
    reviewpath='experiments/'+review['experiment_id']+'/index.html'
    stage_table='\n'.join(f"|{s}|{v['state']}|{v.get('completion_evidence_state','N/A')}|{v.get('experiment_id','—')}|" for s,v in state['stages'].items())
    def save(name,title,body,code):
        footer='\n\n## Source and evidence\n\n'
        footer+='Report build source: `'+git_commit(ROOT)+'`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: '+link('reports/phase6_completion.json')+'.\n\n'
        footer+='Implementation: '+', '.join(link(p) for p in code)+'.\n\n'
        footer+='Regression: '+link('tests/test_phase6.py')+', '+link('reports/phase6_final_tests.log')+'. Stage evidence: '+link('reports/PHASE6_STAGE_STATUS.json')+', '+link('reports/phase6/live.json')+'.\n'
        (REPORTS/name).write_text('# '+title+'\n\n'+body.strip()+footer,encoding='utf8')
    save('PHASE6_EVALUATION_READINESS.md','Phase6 data and independent evaluation readiness',f'''Dataset engineering checks PASSED; independent natural-image semantic evaluation NEEDS_HUMAN_REVIEW. No independent edited factual answers or SC/SI-V labels are certified.

Official question/annotation/image joins and bytes were reverified for{data['official_joins_reverified']} records. Exact-content splits remain disjoint. Source-content sampling is deterministic,10 distinct contents per logical batch. Old test membership is preserved. All v1/v2 files are unchanged, so no corrected manifest was needed. Duplicate records within a split are reported rather than silently deleted:

```json
{json.dumps(data['counts'],indent=2)}
```

Original dataset answers are source facts, not edited-image answers. Edited facts remain UNKNOWN without independently supplied evidence. Runtime v2 contracts, exact-content batch guards and identity/unknown/disputed tests remain active. Development/Gate and old test were historically exposed; they are not new confirmatory evaluation.

Created24 pair review queue: {link(reviewpath)} and {link('experiments/'+review['experiment_id']+'/review_queue.json')}. Includes original/edited images, question,10 original answers, fixed instruction, automatic proxies, priority reasons and blank reviewer fields. First-occurrence representatives per source are selected in fixed historical parent/update order, then ranked by proxy disagreement, lower clipping and distortion. Later edits cannot compete. This is diagnostic triage, not a representative or all-pair worst-case estimate.

Seven independent dimensions: instruction completion, realism, content preservation, factual QA validity, image-question relationship, visible artifacts and uncertainty. Edited factual answer, reviewer identity, second reviewer, disagreement, reason and adjudication are separate fields. Reviewers work independently; automatic proxies are collapsed to reduce anchoring. JSON export is UNADJUDICATED and never promotes labels to truth. No reviewer was fabricated, contacted or assumed. A supplied form still requires identity/provenance and independent adjudication before any reward release. Metadata checks cannot establish human truth.''',
      ['src/data/dataset_v2.py','src/data/semantic_contract_v2.py','src/progression/tasks.py'])
    correlations='\n'.join(f'|{k}|{v if v is not None else "undefined: constant component"}|' for k,v in reward['correlations'].items())
    save('PHASE6_REWARD_VALIDATION.md','Phase6 saved reward construct audit',f'''Audit executed without retraining on{reward['observations']} saved observations. **Audit completion is PASS; training reward validation remains BLOCKED.** Coefficients did not change.

Legacy arithmetic: `R=S+0.5A-0.2D-0.1C-U`, clipped[-1,1.5]. U and C are constant; S=0.5A in these observations. Lower-bound outcomes{reward['clip_lower_count']}/500 (46%); upper-bound{reward['clip_upper_count']}. Instruction-success measurement is missing in500/500; independent-label agreement is unavailable (0 labels). Qwen supplies both answer and semantic proxies, so agreement is not independence.

Component distributions and missing counts:

```json
{json.dumps({'distributions':reward['component_distributions'],'missing':reward['missing_counts'],'populations':reward['by_population']},indent=2)}
```

| Pair | Correlation |
|---|---|
{correlations}

Saved-component arithmetic reproduced exactly twice; maximum stored arithmetic discrepancy{reward['max_saved_arithmetic_error']}. This is deterministic calculation stability, not repeated independent judge reliability or construct validity. Across old checkpoints/seeds/outputs the maximum within-source reward range is{reward['repeat_source_variation_diagnostic']['max_within_source_reward_range']}; this is not fixed-output evaluator noise.

All saved observations use one benign lighting instruction. Cross-edit-category performance is unavailable. Color/animal/food/vehicle keyword summaries in the receipt are overlapping original-question strata, not editing categories. No semantic-success rate is inferred from reward strata.

Scientific objective is proposed as independently verified instruction completion under factual/visual-preservation constraints. It is not yet accepted as a scalar training objective. Need independently judged identity/valid SC/valid SI-V/invalid/unknown/disputed cases, inter-reviewer agreement, missingness, construct redundancy and clipping analysis before releasing a fixed reward version. Compatibility reward remains diagnostic_only; trainer refuses unreleased rewards. Security ASR and evasion metrics remain outside optimizer/Gate.''',
      ['src/rl/reward_v2.py','src/rl/phase5_backend.py','src/rl/diffusion_rl_trainer_v2.py','src/progression/tasks.py'])
    save('PHASE6_REAL_HARNESS_VALIDATION.md','Phase6 recorded-artifact harness validation',f'''Eight distinct TRAIN-source historical episodes were investigated using32 new CPU queries (headers, content digest, pair quality and saved trajectory). Context includes source/edited images, question, source answers, instruction, generation settings, quality/semantic proxies, reward components and execution costs. Source identities and finite saved states were verified. Same-implementation quality arithmetic is reproducible; instruction following, edited factual answers, independent semantic validity and strict continuation remain unresolved.

Fixed source order was written before observations. Eight investigation sources and24 other validation sources do not overlap; all come from historical training, not scientific Gate/test. They are original-source PNGs; Gate does not score the edited images or diagnosis accuracy. Trusted structural oracle uses recorded file bytes and dimensions. Two explicitly injected metadata controls per source create wrong-size and wrong-digest cases. No original/edited image or historical metric was modified.

The accuracy denominator is **72 equally weighted rows**,24 recorded-positive PNGs and48 controls (1:2 mixture), clustered in24 source groups. Incumbent checks geometry+decode+hash; geometry already fully decodes. Proposed reusable revision removes redundant decode. An ineffective reorder was rejected. The revision was accepted, a content-check-dropping regression rejected, and rollback/restore verified. Version programs, proposal evidence, private paired results and all decisions are immutable.

Accepted result: accuracy{gate['parent_accuracy']}→{gate['candidate_accuracy']}, diagnostic operations{gate['parent_resources']['diagnostic_operations']}→{gate['candidate_resources']['diagnostic_operations']} (33.3% reduction); zero new row/source errors. This passes the predeclared equal-accuracy/zero-regression/>=20%-operation-saving rule. One-run wall/CPU measurements are in the receipt and may reflect ordering/cache effects; no general speedup or semantic-editing gain is claimed.

Total investigation plus all accepted/rejected Gate attempts:

```json
{json.dumps(harness['total_search_resources'],indent=2)}
```

No model calls, no optimizer update, no security evaluation. Independent acceptance here means a separate exact structural oracle with aggregate Editor feedback, not human semantic truth or same-UID filesystem secrecy. Scope is recorded-artifact workflow efficiency only.''',
      ['src/progression/real_harness.py','src/rsi/harness/harness_executor.py','src/rsi/harness/version_manager.py'])
    save('PHASE6_RL_READINESS.md','Phase6 actual Generator RL readiness',f'''CPU engineering contracts supported; new scientific training remains BLOCKED. Existing IP2Pix UNet attention LoRA DDPO-SF backend is retained; no demonstrated mathematical defect justified replacement. No categorical controller substitution or new estimator.

Executed CPU regression for stochastic DDIM transitions against Diffusers, analytic Normal log likelihood, score-gradient direction, image/text conditioning layout, on-policy freshness/consumption, fixed advantage scaling, runtime data/reward guards and protocol cases. Exit{rl['cpu_regression_exit']}. Historical witnesses cover{len(rl['historical_actual_update_witnesses'])} real optimizer updates with finite gradients and positive LoRA changes; these are Phase4 observations, not new Phase6 training. Frozen parameters/model revisions are historical integrity evidence, not newly rerun GPU forward passes.

Logical batch10 means10 distinct source contents/trajectories;10 denoising steps each produce100 transitions/backward calls; physical microbatch1 accumulated10 times yields1 optimizer update. Freshness and config contracts must match behavior-policy version; consumed trajectories cannot replay as new on-policy data.

Strict stochastic continuation remains UNRESOLVED. Exact likelihood replay or CPU component equality does not clear the old GPU gradient/adapter divergence. No tolerance relaxation and no repeated GPU equality search. A future falsifiable diagnostic must isolate one numerical hypothesis and carry separate budget authorization.

Reward construction and independent instruction/factual validity are absent. The guarded trainer blocks unreleased reward receipts before the preserved optimizer path. Stage engineering PASS is explicitly distinct from scientific training readiness. Current new updates0, GPU budget0.''',
      ['src/rl/diffusion_trajectory.py','src/rl/diffusion_policy_v2.py','src/rl/diffusion_rl_trainer_v2.py','tests/test_diffusion_rl.py'])
    save('PHASE6_COMPARISON_READINESS.md','Phase6 controlled comparison readiness',f'''Static contract checks PASSED. **Scientifically executable comparison remains NO-GO.** Phase5 configs remain unchanged. Corrected prepared proposals live in the new immutable Stage4 run's `prepared_configs/`.

M0 frozen; M1 standard Generator RL; M2 conventional one-cooldown scheduler; M3 trace+diagnosis without revision; M4 diagnosis-guided workflow revision; M5 active investigation with independent gated revision. New proposal corrects M3 prose to match Phase6, without modifying its historical config. Static equalities:

```json
{json.dumps(comparison['static_equalities'],indent=2)}
```

Same generator config, backend, data hashes, seeds, reward-config reference and development/Gate sizes verified. Trained arms have80 source exposures and8updates. M0 has no training; its resource accounting differs explicitly. M4/M5 incur bounded candidate search costs; report those separately rather than conceal extra evaluator exposure. Phase5/6 prepared arms are NOT RUN.

Still missing: an actual shared pristine LoRA/optimizer/RNG initialization artifact per matched seed, verified frozen services/source order/noise, accepted benign reward, fresh private independent evaluator and locked natural-image selection thresholds. Checking names/configs is not runtime equality. Final-update selection and all predeclared seeds/failures retained; use source-content clusters and report per-seed paired uncertainty, no best-seed/step choice. Three planning seeds are not a powered efficacy study.

Small benign four-update pilot proposal is saved but not launched. It requires all scientific gates plus a separately explicit GPU budget. Historical42.19541 A100h forecast for the full scenario is not allocation, and omits human/new independent evaluator costs. Stage5 cannot run from a static PASS.''',
      ['src/progression/tasks.py','src/evaluation/phase5_comparability.py','configs/phase5_v2/M0.yaml','configs/phase5_v2/M5.yaml'])
    failures=load('phase6/failure_summary.json',{})
    save('PHASE6_FAILURES.md','Phase6 failures, corrections and claim limits',f'''All original Phase1–5 finalized evidence and all first Phase6 audit runs remain. Corrected audits are new descendants with explicit source provenance; no overwritten metric or selected favorable seed.

Observed engineering issues and bounded fixes:

```json
{json.dumps(failures,ensure_ascii=False,indent=2)}
```

First review found4 undeclared semantic assumptions and2 stale rows: Gate denominator/control weighting, first-occurrence source selection, question-strata versus edit-category coverage, generating versus audit provenance, retry coverage, and provisional versus final completion. These were declared/repaired in code and corrected descendant runs. Raw two-round same-family reviews remain provisional; no cross-family acquittal. Later executor-only fixes are identified in BUILD_NOTE.

Automatic repair is limited to known active output directories/logs and retryable transient IO, at most2 retries. Active-run repair cannot rewrite a finalized run. Failed/transient task retry creates a new descendant; no arbitrary source patching or reward change. Unknown scientific failures remain blocked. Checkpoint continuation and poor reward/semantic evidence are scientific blockers, not ordinary retry targets.''',
      ['src/progression/completion.py','src/progression/controller.py','src/monitoring/integrity_v6.py','scripts/sync_phase6.ps1'])
    save('PHASE6_RESOURCE_LEDGER.md','Phase6 resource ledger',f'''Scientific GPU allocation, model inference calls and new generator optimizer updates are all0. The historical42.20h forecast remains unapproved.

Actual Slurm CPU accounting (not GPU time):

```json
{json.dumps(resource,indent=2)}
```

Initial CPU job86601 completed0:0 in33 elapsed seconds; corrected audit descendant job86602 completed0:0 in26 seconds. Both request1CPU/12GiB/20min and no GPU; elapsed59 CPU-allocation seconds combined. CPU process timings in each stage receipt exclude some startup/scheduler overhead and must not be presented as Slurm elapsed time. Additional login-node regression/monitor checks are CPU-only and separately logged; no fabricated total CPU utilization.

Existing recurring monitor remains1CPU/2GiB/5min every two-hour eligibility interval. No duplicate monitor scheduler. Any future ready engineering task uses a one-shot bounded CPU worker, while scientific GPU launch has no implementation in this controller. Include all unsuccessful submissions, repairs and Gate attempts in future accounting.

Review image copies, immutable audit outputs and browser QA evidence are retained on H:, not C:. No new model download or multi-GB C: artifact. Run-specific patch/profile/server scratch is removed at closure; source, final dashboards, review images and negative evidence remain. Final exact disk and retained-size receipt: {link('reports/phase6_completion.json')}.''',
      ['configs/phase6.json','src/progression/monitor.py','src/progression/real_harness.py'])
    save('PHASE6_MONITORING_STATUS.md','Phase6 recurring monitoring and synchronization',f'''Existing Slurm CPU successor chain reused. Latest live monitor snapshot UTC{live.get('timestamp_utc')}; last scheduled-chain success{scheduler.get('last_success_utc')}; next eligibility{scheduler.get('next_due_utc')}, owned successor{scheduler.get('next_job_id')}. Normal queue delay applies. Native Slurm persists after this chat; no sleeping login process or second Codex automation was created.

Versioned v6 monitor entry points preserve protected historical core/integrity source. Existing checks cover queue/accounting, failures/OOM, checkpoints/progress, historical gradients/LoRA/learning rate/data exposure and scientific protection. Idle Phase6 hook hashes only small v2 manifests, verifies newly finalized output contracts, records stage/completion state, reports review-submission file presence without label acceptance, checks disk/process inventory and refreshes both dashboards. No model load or tensor replay on idle cycles; no fabricated live GPU utilization.

Completion validation requires actual Slurm completion0:0, output manifest/file identity, finite required metrics and immutable run contract. In-process PASSED is provisional and can only advance other zero-GPU CPU tasks; final accounting marks VERIFIED_SLURM_AND_OUTPUTS. Invalid outputs/terminal failed allocations fail current readiness, preserving run metrics. Unknown accounting remains pending. Append-only completion observations update the registry view; finalized metrics never change.

Ready/transient-failed CPU tasks can dispatch at most one non-recurring CPU worker; bounded retries do not override scientific gates. No authorized scientific GPU pilot exists, so both old-campaign automatic GPU recovery and new GPU submission are blocked.

Dashboard inventory found no existing experimental bilingual package in the project/canonical paths or searched delivery folders. Located older conceptual research briefing pages, which are retained and not relabeled as experiment dashboards. Added minimal bilingual views from shared `dashboard/state.json`, with links to review evidence. Both files are atomically replaced; sync checks matching timestamps. Browser rendering checks include desktop/mobile overflow, reciprocal links, image loading and blank review fields.

Local pages require explicit `powershell -File scripts/sync_phase6.ps1` and reload; a downloaded HTML cannot receive future cluster updates by itself. This is an on-demand sync workflow, not another scheduler. Cluster cycle updates served/local source files; no hosted publication was found or claimed.''',
      ['scripts/manage_monitor.py','scripts/monitor_experiments.py','src/monitoring/core_v6.py','src/progression/monitor.py','src/progression/dashboard.py','scripts/sync_phase6.ps1'])
    save('PHASE6_NEXT_ACTIONS.md','Phase6 eligible next actions','''All currently eligible CPU stages have executed and their outputs were inspected. No scientific job is active. The next automatic action is the existing monitor's lightweight integrity/completion/dashboard cycle; transient engineering recovery is bounded and cannot change scientific objectives.

1. Independent reviewers assess the24 queued original/edited pairs on seven dimensions, provide edited factual answers, reasons and provenance, then independently adjudicate disagreement. Do not reuse original answers as edited truth.
2. Validate and explicitly release a benign instruction-success/preservation reward using independently judged cases; retain failed cases, constant terms and clipping evidence. Current compatibility reward cannot train.
3. Establish fresh private natural-image evaluation and matched pristine initialization/pre-intervention execution. Previously exposed dev/Gate/test do not become unseen through renaming.
4. Obtain explicit scientific scope and GPU budget, then consider a small matched benign pilot. Evaluate verified quality, repeatability and all diagnostic costs before any larger comparison.

Current dependent states: NEEDS_HUMAN_REVIEW for independent semantics; BLOCKED for reward/scientific readiness and pilot prerequisites. Explicit budget is also absent. No autonomous attack, target-label generation, poisoned-target training, ASR maximization or evasion optimization is eligible.''',
      ['src/progression/controller.py','src/rl/phase5_backend.py','configs/phase6.json'])
    summary=f'''**当前结论：六项 CPU 数据/审核准备/方法工程阶段已执行；独立自然图像评价及科学训练仍 NO-GO。** 科学 GPU预算0，未启动模型推理、Generator RL训练或安全攻击实验。

## 执行结果

| 阶段 | 当前状态 | 最终完成证据 | 实验ID |
|---|---|---|---|
{stage_table}

重新核对1500条官方图片—问题—答案及字节，保留全部v1/v2。train/dev/Gate/test仍999/104/96/301；训练抽样按内容分组，固定10个不同源图。独立确认的编辑事实、SC/SI-V标签均0，因此没有把数据工程PASS写成语义有效。

已创建24对真实历史训练图像的七维独立审核包：{link(reviewpath)}。包含原图/编辑图、原问题与10条原答案、指令、代理判断、优先原因及空白审核字段。先固定每源首个历史观察，再按不确定/截断/失真排序；这不是全部编辑对的最坏案例或独立测试集。尚无人工标签，不自动认证导出表单。

Reward审计覆盖500条旧观察：230条落在下界（46%）；C与U常量、S与A冗余；500条均缺显式指令完成测量。固定分量算术可重复，不等于奖励科学有效。所有编辑使用同一lighting指令，无法比较多种编辑类别；question关键词只作为问题内容分层。未调整系数，训练奖励继续阻断。

真实harness：8个训练episode追加32次CPU检查，24个不重叠训练源图作结构Gate，另有48个明确注入的元数据负例。总准确率按72行等权计算；原流程/新流程均100%，诊断操作216→144，减少33.3%。无效重排和退化修改被拒，回滚验证通过。只支持原图文件完整性流程效率，不支持编辑图语义改进、诊断正确率或安全ASR提升。全部搜索成本在记录中，单次墙钟差不能作为普遍提速结论。

Generator RL保留真实IP2Pix UNet LoRA及DDPO-SF。CPU代数/似然/条件/新鲜度/批次契约通过；旧32次实际LoRA更新证据保留。10源图＝10轨迹＝100去噪转移＝10次累积＝1次优化器更新。严格随机续训失败仍未修复，未重复耗GPU试图凑位级相等。

M0–M5静态同配置检查通过，新增准备版本修正M3为trace+诊断但不修订；均未运行。实际同初始化、独立评价、奖励发布和干预前可比性未满足。小pilot只准备，42.20 A100小时仅历史完整方案估计，未获批。

## 自动推进、修复和审查

控制器自动执行全部可用CPU阶段，人工审核阻塞不妨碍独立任务。任务内PASS为同一CPU分配中的临时工程断言；完成后还须Slurm与输出验证，失败会降低当前阶段就绪状态。监控只处理已验证的目录/log修复及最多两次瞬时IO重试，新尝试是不可变子实验。科学失败不当作工程重试。

已修复监控将标量观察计数当图像列表的实际错误；旧core/integrity及32项保护源码保留，使用v6入口。修复PowerShell原子替换兼容问题并实际运行同步成功。浏览器发现的横向溢出已修复，保留前后测试证据。首轮假设复核指出4项未声明语义和2项陈旧声明；已记录并修复，原首轮审计保留，新生成纠正子实验，未改写旧结果。两轮同族复核仅provisional，不能作为独立学术认可。

当前集群回归日志：{link('reports/phase6_final_tests.log')}（{tests['canonical_count']}项通过；最终测试收据见stage证据）。旧Phase1–5文件3440份未改变。所有实验原始结果位于canonical `/storage/home/ydk297/projects/poisoning/experiments/`；本地新输出在 `H:/Project/Poisoning/experiments/`，准确ID见上表及各receipt。

## 资源、监控与下一步

Phase6 CPU作业86601/86602完成，Slurm elapsed33+26＝59秒，均1CPU/12GiB、0GPU。登录节点CPU测试/检查另记，不能当作GPU费用。现在没有运行或排队的本项目科学实验；无关项目作业只作队列上下文。

原两小时监控最近定时成功UTC{scheduler.get('last_success_utc')}，最新额外核验UTC{live.get('timestamp_utc')}；自有后继{scheduler.get('next_job_id')}，下次到期UTC{scheduler.get('next_due_utc')}，实际受队列影响。仅一个监控链，退出聊天后由Slurm继续。下一个自动动作是轻量完整性/完成检查与双语页面刷新，不会自动启动GPU。

中文页面{link('dashboard/index_zh.html')}；英文页面{link('dashboard/index_en.html')}；共享JSON{link('dashboard/state.json')}。本地需执行同步脚本并刷新，不能宣称静态文件自动跨机器更新。找不到已有实验双语package，已保留找到的旧概念讲解页面并新增最小实验视图。

确切缺口：独立七维审核及编辑后事实→奖励构念验证→新私有评价与严格分支可比性→单独GPU预算。全部依赖阶段保持阻塞；其余当前可执行任务已完成。

## 完整语义与接口假设

'''+(ROOT/'implement-stage/phase6/ASSUMPTIONS.md').read_text(encoding='utf8')
    save('PHASE6_PROGRESS_SUMMARY_ZH.md','Phase6 连续推进、验证和监控总结',summary,
      ['src/progression/controller.py','src/progression/tasks.py','src/progression/real_harness.py','src/progression/monitor.py'])
    print(json.dumps({'required_reports':10,'stage_status_json':True,'scientific_decision':'NO-GO'}))

if __name__=='__main__':main()
