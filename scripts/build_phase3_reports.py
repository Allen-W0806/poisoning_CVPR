"""Build evidence-bounded Phase3 reports from completed immutable runs."""
import json
from pathlib import Path
import _bootstrap

def dump(value):return '```json\n'+json.dumps(value,indent=2,ensure_ascii=False)+'\n```\n'

if __name__=='__main__':
    root=_bootstrap.ROOT; evidence=json.loads((root/'reports/phase3_evidence.json').read_text());runs=evidence['runs']
    passed=lambda suffix:[r for r in runs if r['id'].endswith(suffix) and r['status']['status']=='PASS']
    one=passed('generator_rl_one'); smoke=passed('generator_rl_smoke'); semantics=passed('semantic_review')
    schema=passed('semantic_schema_audit')
    full=smoke[-1] if smoke else None
    attempts=[r for r in runs if r['id'].endswith('generator_rl_smoke')]
    selected=attempts[-1] if attempts else (one[-1] if one else None)
    m=selected['metrics'] if selected else {}
    if selected:
        m=dict(m);audit=selected.get('checkpoint_audit',{})
        m.setdefault('parameters',{'trainable_parameters':audit.get('trainable_parameters')})
        m.setdefault('adapter_changes',audit.get('adapter_changes'))
        m.setdefault('frozen_tensor_count',audit.get('frozen_tensor_count'))
        m.setdefault('frozen_weights_unchanged',audit.get('frozen_hashes_equal'))
        m.setdefault('restart',selected.get('restart_diagnostic'))
        m.setdefault('source_images',len({o['sample_id'] for o in m.get('observations',[])}))
        m.setdefault('training_trajectories',len(m.get('observations',[])))
        restart_path=root/'experiments'/selected['id']/'restart_observations.json'
        replay=json.loads(restart_path.read_text()) if restart_path.is_file() else []
        m.setdefault('restart_diagnostic_trajectories',len(replay))
        observed=m.get('observations',[])+replay
        m.setdefault('qwen_calls',max((o.get('qwen_inference',{}).get('model_calls',0) for o in observed),default=0))
    failures=[r for r in runs if r['status']['status']=='FAIL']
    rows=[line.split('|') for line in evidence['accounting']['stdout'].strip().splitlines() if line.strip()]
    terminals={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED'}
    actual=sum(int(r[2]) for r in rows if r[1].split()[0] in terminals)
    pending=[r[0] for r in rows if r[1].split()[0] not in terminals]
    def write(name,text):(root/'reports'/name).write_text(text,encoding='utf-8')
    runlinks='\n'.join(f"- `{r['id']}`: **{r['status']['status']}**; `{r['manifest']['git_commit']}`; Slurm `{r['manifest'].get('slurm_job_id')}`; config SHA256 `{r['config_file_sha256']}`." for r in runs)
    smoke_report='# Generator RL smoke\n\n'+('PASS: three actual reward-driven generator updates, logical batch10.\n\n' if full else 'BLOCKED: no complete three-update logical-batch10 generator RL witness.\n\n')
    smoke_report+='One-update full witnesses: '+str([r['id'] for r in one])+'. Three-update witnesses: '+str([r['id'] for r in smoke])+'.\n\n'
    if attempts:
        smoke_report+='Three-update batch10 training was executed; the full integration run status is '+attempts[-1]['status']['status']+'. Completed optimizer updates must not be confused with full restart acceptance.\n\n'
    smoke_report+='Actual terminal-reward updates are distinct from supervised MSE and external controller RL. Failed exact-restart attempts retain their real updates and all negative evidence but are not full PASS.\n\n'
    if selected:
        smoke_report+=dump({'experiment':selected['id'],'updates':m.get('updates'),'parameters':m.get('parameters'),'adapter_changes':m.get('adapter_changes'),
            'frozen_tensor_count':m.get('frozen_tensor_count'),'frozen_weights_unchanged':m.get('frozen_weights_unchanged'),
            'implementation_tests':m.get('implementation_tests'),'restart':m.get('restart'),
            'restart_diagnostic':selected.get('restart_diagnostic'),'checkpoint_audit':selected.get('checkpoint_audit'),
            'source_images':m.get('source_images'),'training_trajectories':m.get('training_trajectories'),
            'restart_diagnostic_trajectories':m.get('restart_diagnostic_trajectories'),'unet_calls':m.get('unet_calls_including_diagnostics'),'qwen_calls':m.get('qwen_calls')})
    smoke_report+='\nTests A–G are actual runtime assertions in smoke_generator_rl.py plus analytical/official-transition CPU tests. H requires the separate three-update run; one-source witnesses never substitute for batch10. Full raw trajectories, per-step adapter/optimizer/RNG states, pre-update conditions, gradients, observations and restart diagnostics are under each run directory. Model call counter excludes internal checkpoint recomputations; allocation costs include them.\n\n'+runlinks+'\n'
    write('PHASE3_GENERATOR_RL_SMOKE.md',smoke_report)
    semantic='# Semantic validation\n\n'
    if semantics:
        s=semantics[-1];semantic+=f"Review package: `{s['id']}`.\n\n"+dump(s['metrics'])
        semantic+='`review_form.csv` and `review_form.jsonl` include exact original/edited images, question, all reference answers, editing instruction, blank independent validity judgment, reason and reviewer fields. `semantic_validity_table.json` retains per-sample checks, raw separate single-image inventories, uncertainty categories and disagreements with prior proxy feedback. `images/review_contact_sheet.png` is a 10-row original/edited sheet.\n\n'
    else:semantic+='Review package execution not complete; no claim that files are available.\n\n'
    semantic+='All ten changed Phase2 natural pairs remain **UNVERIFIED**, human verified count0. Object presence/identity, color, count, spatial, attributes, plausibility and artifact descriptors are obtained in separate single-image calls, not reused training pair judgments. Exact description matches/differences and independent numerical RGB/saturation/roughness checks are diagnostics. Wording differences are not proven semantic changes; unsupported/uncertain fields remain UNKNOWN. These calls use the same frozen Qwen family, so they are separate inference outputs but **not independent model-family or human truth**. Human–automatic agreement is N/A because labels do not exist. No accuracy-on-semantically-preserved-edits claim. The previously observed green-shift snow scene remains an instruction-fidelity warning, not a new human label.\n'
    if schema:
        semantic+='\nThe original v1 parser required all strings; actual model output used integer counts and all10 pairs were marked INVENTORY_PARSE_FAILURE. This engineering failure is retained. A separate CPU-derived audit normalizes only explicit nonnegative integer count representation without inference or human labels. Use the v2 table for automatic review flags, and the original immutable package for images/blank forms.\n\n'+dump({'derived_run':schema[-1]['id'],'metrics':schema[-1]['metrics']})
    write('PHASE3_SEMANTIC_VALIDATION.md',semantic)
    comparison='# RL/RSI comparison\n\n**PREPARED_NOT_RUN.** No Phase3 M0–M4 performance comparison or generator-RSI gain is claimed. `configs/phase3_comparison.yaml` distinguishes frozen reference, instruction-selection controller, supervised paired-target fitting, reward-based diffusion transitions and configuration adaptation. Prior Phase2 Euler runs are historical evidence, not scheduler-matched controls for new DDIM runs.\n\nFixed M1/M2/M3 arms plan six updates of batch10=60 training exposures and two validation looks. M4 plans three base plus three candidate updates from a common initial state, total60 exposures/two looks. M0 has no fictitious optimization exposures; its inference count/cost are matched and reported. All generation/judging/selection/rejected-trial/failure costs count toward separate actual GPU caps; a cap may prevent matching exposures, which must be reported.\n\nThe first smoke trains one fixed instruction. Three-task controller comparisons need a separately validated instruction/trajectory contract. Human semantic truth and meaningful independent performance evaluation are not established. Therefore the configurations are study specifications, not an automatic launch-ready sweep. RSI may change LR1e-5 to5e-6 only, using fixed reward/splits and independent validation; tiny single-batch gains cannot accept a revision. Accept/reject/reason/cost/rollback records are required when run. No Phase3 proposal/acceptance/rollback trial was executed; Phase2 rejected controller candidate remains unchanged.\n\nLimited stability candidates are batch10/20, LR1e-5/5e-6, rank8/16 and seeds42/123/3407; `configs/phase3_stability.yaml` disables the full grid. Next meaningful single-change run would use seed123 after the remaining gates.\n'
    comparison+='\n| Method | Implemented/executed evidence | Matched Phase3 performance comparison |\n|---|---|---|\n| M0 | Phase2 frozen native editor outputs | NOT_RUN |\n| M1 | Phase2 external controller, frozen native editor | NOT_RUN |\n| M2 | Phase2 three supervised LoRA updates | NOT_RUN |\n| M3 | '+('Phase3 full DDPO-SF smoke PASS' if full else 'Phase3 implementation; full smoke not PASS')+' | NOT_RUN |\n| M4 | Phase2 controller-RSI rejection/rollback only; Phase3 generator-RSI NOT_RUN | NOT_RUN |\n'
    write('PHASE3_RL_RSI_COMPARISON.md',comparison)
    accounting_table='| Job | State | GPU seconds | Exit |\n|---|---|---:|---|\n'+'\n'.join(f'| {r[0]} | {r[1]} | {r[2]} | {r[3]} |' for r in rows)+'\n'
    resources=f'# Resource report\n\nActual terminal one-A100 allocation: **{actual} GPU seconds = {actual/3600:.6f} GPU-hours**, against a separate Phase3 cap7200 seconds. Pending/unconfirmed jobs: {pending}. All failures and diagnostics are charged; Phase2 historical1187 seconds are not subtracted from the new budget.\n\n'+accounting_table+'\n'
    allobs=[]
    for r in runs:
        for p in (root/'experiments'/r['id']).glob('*_observations.json'):allobs+=json.loads(p.read_text())
    resources+=dump({'recorded_rewarded_trajectories_including_restart_and_failures':len(allobs),
        'sampling_seconds_recorded':sum(o['generation_seconds'] for o in allobs),
        'reward_judging_seconds_recorded':sum(o['reward_seconds'] for o in allobs),
        'retained_phase3_experiment_bytes':sum(r['bytes'] for r in runs),
        'selected_run':selected['id'] if selected else None,
        'training_seconds_by_update':[u['step_seconds'] for u in m.get('updates',[])],
        'peak_allocated_bytes_by_update':[u['peak_gpu_memory_bytes'] for u in m.get('updates',[])],
        'generation_mean_seconds':sum(o['generation_seconds'] for o in allobs)/len(allobs) if allobs else None})
    resources+='\nSampling subtotals exclude seed-control extra generations unless explicitly saved as observations; total allocation includes all generation, model loading, numerical tests, judging, hashing and replay. Peak values are measured CUDA allocation with co-resident frozen Qwen, not total device reservation. No larger comparison was launched. The first batch10 measured ~4.55 seconds/generation, ~7.55 seconds/reward audit and ~106.6 seconds/optimizer update. Six batch10 updates plus two10-source validation looks therefore extrapolate to ~80*(4.55+7.55)+6*106.6 =1608 seconds/arm, with loading/failures/selection overhead additional. Prepared per-arm cap was revised from600 to1800 seconds based on this measured cost; it does not authorize a sweep or expand the global7200-second budget. Multi-seed M0–M4 evaluation is not covered by one successful feasibility check. No new model download/environment installation or multi-GB C artifact was required.\n'
    write('PHASE3_RESOURCE_REPORT.md',resources)
    fail='# Failures and numerical limits\n\n'
    for r in failures:
        fail+=f"## {r['id']}\n\nSlurm `{r['manifest'].get('slurm_job_id')}`; source `{r['manifest']['git_commit']}`.\n\n"+dump({'restart':r['restart_diagnostic'],'inputs':r['restart_inputs'],'real_updates_before_failure':r['metrics'].get('updates')})
    fail+='\nInitial batched-CFG/BF16-autocast-adapter exact continuation failed; FP32 A/B isolation also failed; sequential CFG with BF16 reduced-precision reductions disabled also failed. No precise CUDA-kernel root cause is established. All original forward losses, rewards and RNG could match while backward gradients/AdamW updates differed. Fixed-state job86408 retained comparisons with/without gradient checkpointing: disabling it matched one fresh-instance gradient exactly, but same-instance repeat still differed. That diagnostic is not universal determinism proof and performs no optimizer update. Any subsequent full witness has its own run and configuration; no failed run is relabeled.\n\nLocal system Python lacks Diffusers; three numerical test dependency errors were not hidden by installing into a shared runtime. Canonical editor environment is the declared acceptance environment. No OOM has been observed unless a later raw run explicitly reports it. Tests/thresholds remain fixed; no reward, split, answer annotation or heldout protocol was changed to turn a failure into PASS.\n'
    fail+='\nSemantic inventory schema v1 rejected valid integer-count JSON output and marked all10 pairs unparseable. Package PASS meant files were created, never semantic certification. A separate CPU v2 reparse preserves all raw JSON and original table/forms, accepts only the observed integer representation, and still leaves every human label blank. No GPU re-evaluation or post-hoc semantic truth was fabricated.\n'
    write('PHASE3_FAILURES.md',fail)
    zh=f'# Phase 3 实验进度与结果\n\n'+('**真实生成器 RL 已通过三轮、逻辑 batch=10 的工程验证。**\n\n' if full else '**真实 Generator RL 的单步机制已验证；batch=10 三轮更新已执行，但断点后严格续训复现未通过。**\n\n' if one else '**尚未取得完整 Generator RL 验收通过记录。**\n\n')
    zh+='方法是显式 DDIM 随机采样上的 DDPO score-function：保留 InstructPix2Pix 原图、文字指令和三分支 guidance，用单条图像轨迹的良性奖励计算真实 UNet LoRA 梯度。它区别于此前监督 MSE 和外部分类控制器。\n\n'
    if selected:
        zh+=f"完整单步通过 {len(one)} 次；三轮通过 {len(smoke)} 次。主要证据 `{selected['id']}`。\n\n"
        zh+='| 更新 | 奖励均值 | 奖励方差 | 梯度范数 | 单轮LoRA最大变化 | 参数更新秒数 |\n|---|---:|---:|---:|---:|---:|\n'
        zh+='\n'.join(f"| {u['step']} | {u['reward_mean']:.6f} | {u['reward_variance']:.6f} | {u['gradient_norm']:.6f} | {u['trainable_max_change']:.6g} | {u['step_seconds']:.2f} |" for u in m.get('updates',[]))+'\n\n'
        audit=selected.get('checkpoint_audit',{});restart=selected.get('restart_diagnostic') or m.get('restart',{})
        zh+=f"实际可训练参数 {audit.get('trainable_parameters',0):,}；冻结张量 {audit.get('frozen_tensor_count')} 个，前后哈希一致={audit.get('frozen_hashes_equal')}。断点后下一步的适配器最大差异 {restart.get('adapter_max_difference')}；参数精确一致={restart.get('adapter_exact')}，AdamW精确一致={restart.get('optimizer_exact')}，随机状态一致={restart.get('rng_exact')}，下一步loss一致={restart.get('next_step_loss_exact')}。\n\n"
    zh+=f'\n本阶段实际累计 **{actual} GPU 秒（{actual/3600:.4f} GPU 小时）**，包含全部失败与诊断，新增预算2小时。\n\n'
    zh+='奖励固定 S1、A0.5、D0.2、C0.1、U1，按个体保存，不做每批重标定。未核实的语义变化持续受罚；奖励有差异或梯度非零只支持“反馈已接通”，不能证明图像质量改善。表中的均值上升来自重新采样，缺少匹配冻结对照，不能解释为有效性提升。跨来源奖励方差也不能单独证明同一来源编辑有可控的学习信号。单个来源重复生成是相关观测，不能当作独立样本。\n\n'
    zh+=('已生成10对原图/编辑图复核材料、CSV/JSONL空白表、逐项自动诊断和联系图。' if semantics else '语义复核包执行仍未完成。')+'**人工独立核实仍为0，10对都保持 UNVERIFIED。** 新单图描述与训练奖励的成对判断使用不同输出，但仍是同一 Qwen 模型，不能称为独立模型或人工真值。\n\n'
    zh+='语义程序曾因整数计数不符合字符串 schema 而把10对全部标成解析失败；旧输出与旧表已保留。单独CPU审计修正计数表示，不新增模型调用、不改人工标签；详见语义报告中的v2派生表。\n\nQwen 保持冻结：既有开发集81.15%、一次性内部测试77.6333%未更改、未重跑。Phase3 不训练 Qwen，不实施后门/ASR/规避优化。\n\n最终配置为 BF16 基础计算、FP32 LoRA 运算、物理单来源顺序 guidance；关闭梯度检查点后单步严格续训验证通过，但batch10仍未通过，未放宽容差。集群编辑器环境完整52项测试通过。三次早期单步失败、batch10完整运行失败和一次固定状态诊断全部保留。\n\nM0–M4 和有限稳定性研究仅已准备配置，没有执行性能对照或 Generator RSI，因此 **RSI 增益尚无证据**。下一步先定位batch10梯度/AdamW数值差异、补独立语义标注，再按相同暴露量与完整成本执行小规模对照；不自动运行全网格。\n\n原有两小时 Slurm CPU 监控链保留，扩展检查轨迹概率、条件、LoRA变化、梯度、奖励方差、断点完整性与独立Phase3预算；空闲时不启动研究作业。监控不会改变奖励/数据/测试标准。\n\n证据位置：canonical `/storage/home/ydk297/projects/poisoning/experiments/EXP_*phase3_*`，本地镜像 `H:/Project/Poisoning/experiments/`；每个运行保留 manifest/config/环境/git/Slurm日志/轨迹/适配器与AdamW状态。\n\n'+runlinks+'\n\n详见同目录算法审查、实现、smoke、语义验证、RL/RSI对照、资源与失败报告。审查为 same-family/provisional，不构成独立科学认证。\n'
    write('PHASE3_PROGRESS_SUMMARY_ZH.md',zh)
    print('Built six result reports; algorithm/implementation reports maintained separately.')
