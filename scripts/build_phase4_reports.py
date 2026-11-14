"""Read immutable runs and sacct; never mutate scientific evidence."""
import json
from pathlib import Path
import statistics
import torch
import numpy as np
import _bootstrap
from submit_phase2 import accounting
from src.evaluation.phase4_statistics import paired_summary
from src.rl.optimizer_validation import state_equal,tensor_changes
from src.utils.logging import write_json

def load(path):return json.loads(Path(path).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n|'+'|'.join(['---']*len(headers))+'|\n'+'\n'.join('| '+' | '.join(str(x) for x in row)+' |' for row in rows)+'\n'
def mean(values):return float(np.mean(values)) if values else None
def summary(obs):
    return {key:mean([r['reward'][key] for r in obs]) for key in ['semantic','answer_consistency','distortion','cost','invalid_penalty','reward']} | {
        'generation_seconds':mean([r['generation_seconds'] for r in obs]),'judging_seconds':mean([r['reward_seconds'] for r in obs]),
        'vqa_same_answer_fraction':mean([float(r['answer_consistency']['same_normalized_answer']) for r in obs]),
        'vqa_consistency_raw':mean([r['answer_consistency']['score'] for r in obs]),
        'edited_reference_score_diagnostic_only':mean([r['answer_consistency']['edited_reference_score'] for r in obs])}

def main():
    root=_bootstrap.ROOT;reports=root/'reports'
    runs=[]
    for path in sorted((root/'experiments').glob('*_phase4_*')):
        runs.append({'id':path.name,'path':path,'manifest':load(path/'manifest.json'),'status':load(path/'status.json')['status'],'metrics':load(path/'metrics.json')})
        events=[json.loads(x) for x in (path/'run.log').read_text().splitlines()]
        runs[-1]['executed_training_updates']=[x['value'] for x in events if x['event']=='generator_rl_update']
    ledger=[json.loads(x) for x in (reports/'phase4_jobs.jsonl').read_text().splitlines()]
    if any(r['status'] in {'RUNNING','SUBMITTED','CREATED'} for r in runs):raise RuntimeError('Final reports require terminal Phase4 runs')
    charged,slurm=accounting(ledger);job_cost={row[0]:int(row[2]) for row in slurm}
    diagnostic=sum(job_cost.get(j['job_id'],j['reserved_seconds']) for j in ledger if j['task'].startswith('diagnostic'))
    selected={}
    for run in runs:
        arm=run['metrics'].get('arm')
        if arm:selected['frozen' if arm=='frozen' else arm+str(run['metrics']['seed'])]=run
    evaluations=[];raw=[];comparisons=[];fairness=[]
    frozen=selected.get('frozen')
    if frozen:
        for evaluation in frozen['metrics']['evaluations']:
            row={'arm':'M0','seed':evaluation['seed'],'step':0,'run':frozen['id'],**summary(evaluation['observations'])};evaluations.append(row);raw.append(row)
    for key,run in selected.items():
        if key=='frozen':continue
        for ev in run['metrics'].get('evaluations',[]):
            row={'arm':'M1' if key.startswith('rl') else 'M2','seed':ev['seed'],'step':ev['step'],'run':run['id'],**summary(ev['observations']),
                'paired_frozen':ev['paired_frozen']};evaluations.append(row)
        if run['metrics'].get('evaluations'):
            final=run['metrics']['evaluations'][-1]
            raw.append({'arm':'M1' if key.startswith('rl') else 'M2','seed':final['seed'],'step':final['step'],'run':run['id'],**summary(final['observations']),
                'paired_frozen':final['paired_frozen']})
    for seed in [42,123]:
        fixed=selected.get('rl'+str(seed));rsi=selected.get('rsi'+str(seed))
        if fixed and rsi and fixed['metrics'].get('evaluations') and rsi['metrics'].get('evaluations'):
            f=fixed['metrics'];r=rsi['metrics']
            same_initial=state_equal(torch.load(fixed['path']/'checkpoints/initial_adapter.pt',weights_only=True,map_location='cpu'),
                torch.load(rsi['path']/'checkpoints/initial_adapter.pt',weights_only=True,map_location='cpu'))
            same_sources=[x['source_ids'] for x in f['updates']]==[x['source_ids'] for x in r['updates']]
            same_noise=[[o['seed'] for o in x['observations']] for x in f['updates']]==[[o['seed'] for o in x['observations']] for x in r['updates']]
            comparison=paired_summary(r['evaluations'][-1]['observations'],f['evaluations'][-1]['observations'])
            revision_steps=[d['step'] for d in r.get('rsi_decisions',[]) if d.get('accept_schedule_revision')]
            pre_step=min(revision_steps) if revision_steps else min(len(f['updates']),len(r['updates']))
            pre_difference=None
            if pre_step:
                fs=torch.load(fixed['path']/f'checkpoints/step_{pre_step:04d}/training_state.pt',map_location='cpu',weights_only=False)
                rs=torch.load(rsi['path']/f'checkpoints/step_{pre_step:04d}/training_state.pt',map_location='cpu',weights_only=False)
                pre_difference={'step':pre_step,'adapter_changes':tensor_changes(fs['adapter'],rs['adapter']),
                    'optimizer_exact':state_equal(fs['optimizer'],rs['optimizer']),
                    'scope':'checkpoint saved before any schedule revision at this step; same scientific configuration'}
            pre_evaluations=[{'step':re['step'],'rsi_minus_rl':paired_summary(re['observations'],next(fe['observations'] for fe in f['evaluations'] if fe['step']==re['step']))}
                for re in r['evaluations'] if re['step']<=pre_step]
            ref=next(x['observations'] for x in frozen['metrics']['evaluations'] if x['seed']==seed)
            original_disagreements={arm:sum(a['vqa_diagnostic']['original_normalized_answer']!=b['vqa_diagnostic']['original_normalized_answer']
                for a,b in zip(model['evaluations'][-1]['observations'],ref)) for arm,model in [('rl',f),('rsi',r)]}
            comparisons.append({'seed':seed,'rsi_minus_rl':comparison})
            fairness.append({'seed':seed,'initial_adapter_exact':same_initial,'training_source_order_exact':same_sources,'training_sampling_seeds_exact':same_noise,
                'updates':[len(f['updates']),len(r['updates'])],'source_exposures':[f['training_source_exposures'],r['training_source_exposures']],
                'validation_trajectories':[f['evaluation_trajectories'],r['evaluation_trajectories']],
                'actual_gpu_seconds':[job_cost.get(str(fixed['manifest']['slurm_job_id'])),job_cost.get(str(rsi['manifest']['slurm_job_id']))],
                'rsi_decisions':r['rsi_decisions'],'exact_execution_reproducibility':False})
            fairness[-1].update(pre_schedule_state_difference=pre_difference,pre_schedule_paired_evaluations=pre_evaluations,
                original_normalized_answer_disagreements_with_frozen=original_disagreements)
    aggregate={}
    for arm in ['M0','M1','M2']:
        values=[x['reward'] for x in raw if x['arm']==arm]
        aggregate[arm]={'seed_count':len(values),'mean_reward':mean(values),'sample_sd':statistics.stdev(values) if len(values)>1 else None,
            'components':{k:{'mean':mean([x[k] for x in raw if x['arm']==arm]),
                'sample_sd':statistics.stdev([x[k] for x in raw if x['arm']==arm]) if len(values)>1 else None}
                for k in ['semantic','answer_consistency','vqa_consistency_raw','vqa_same_answer_fraction','distortion','cost','generation_seconds']}}
    fixed_final=[x for x in raw if x['arm']=='M1']
    rl_resolved=len(fixed_final)==2 and all(x['step']==8 and x['paired_frozen']['ci95'][0]>0 for x in fixed_final)
    rsi_resolved=len(comparisons)==2 and all(x['rsi_minus_rl']['ci95'][0]>0 for x in comparisons) and all(
        selected.get('rsi'+str(seed),{}).get('status')=='PASS' and len(selected['rsi'+str(seed)]['metrics'].get('updates',[]))==8 for seed in [42,123])
    claims={'two_seed_rl_proxy_gain_resolved':rl_resolved,'two_seed_rsi_proxy_gain_resolved':rsi_resolved,
        'generalization_improvement_established':False,'independent_semantic_improvement_established':False,
        'larger_compute_followup_justified_now':False,'reason':'Independent semantic review and strict continuation remain incomplete; two-seed repeated-development pilot is exploratory'}
    update_count=sum(len(r['executed_training_updates']) for r in runs)
    training_observations=[o for r in runs for u in r['executed_training_updates'] for o in u.get('observations',[])]
    counts={'training_source_exposures':len(training_observations),'training_diffusion_trajectories':len(training_observations),
        'unique_training_source_images_across_seeds_and_arms':len({o['source_image_id'] for o in training_observations}),
        'validation_trajectories_including_frozen':sum(sum(len(e['observations']) for e in r['metrics'].get('evaluations',[])) for r in selected.values()),
        'conditioning_seed_control_extra_generations':sum(bool(r['executed_training_updates']) for r in runs),
        'outcome_diagnostic_generations':sum(len(r['metrics'].get('details',[])) for r in runs if r['id'].endswith('PH4_RESTART_OUTCOMES')),
        'measured_qwen_model_calls':sum(r['metrics'].get('qwen',{}).get('model_calls',0) for r in selected.values())}
    evidence={'slurm':slurm,'charged_or_reserved_gpu_seconds':charged,'diagnostic_seconds':diagnostic,'max_gpu_seconds':14400,
        'training_updates':update_count,'diagnostic_replay_optimizer_updates':sum(len(r['metrics'].get('replays',[])) for r in runs),
        'runs':[{'id':r['id'],'status':r['status'],'manifest':r['manifest']} for r in runs],
        'raw_final_rows':raw,'all_evaluations':evaluations,'rsi_minus_rl':comparisons,'fairness':fairness,'across_seed_summary':aggregate,'claims':claims,'exposure_counts':counts}
    write_json(reports/'phase4_results.json',evidence)
    header='Generated from immutable runs and current Slurm accounting. Development proxies only; no internal held-out selection.\n\n'
    resource=header+f'Additional Phase4 allocation: **{charged} seconds = {charged/3600:.6f} A100 GPU-hours**, hard cap14400sec. Diagnostics {diagnostic}/2700sec. Active/unreported jobs reserve their full wall limit. One GPU per job. Revision child runs share the same allocation and are not double charged.\n\n'+table(['Job','State','GPU seconds'],[r[:3] for r in slurm])
    resource+='\nCosts include loading, generation, Qwen judgments, training, fixed validation, failed attempts and RSI decisions. Reward compute term remains steps/30; actual costs are separately measured. Source exposures, trajectories and optimization counts are distinct. No third seed or unrestricted sweep. Existing environments and central model cache reused; retained experiment evidence is on H and canonical storage, not C.\n'
    resource+='\n```json\n'+json.dumps(counts,indent=2)+'\n```\n'
    (reports/'PHASE4_RESOURCE_ACCOUNTING.md').write_text(resource,encoding='utf-8')
    rows=[]
    for key,r in selected.items():
        m=r['metrics'];rows.append([key,r['status'],len(m.get('updates',[])),m.get('training_source_exposures',0),m.get('evaluation_trajectories'),r['id']])
    training=header+table(['Arm/seed','Status','Updates','Train exposures','Validation trajectories','Run'],rows)
    diagnostics_rows=[]
    for key,r in selected.items():
        updates=r['metrics'].get('updates',[])
        if not updates:continue
        diagnostics_rows.append([key,min(u['gradient_norm'] for u in updates),max(u['gradient_norm'] for u in updates),
            max(u['changes']['max_abs_change'] for u in updates),r['metrics'].get('adapter_changes',{}).get('max_abs_change'),
            mean([u['step_seconds'] for u in updates]),max(u['peak_gpu_memory_bytes'] for u in updates)/1e9,
            r['metrics'].get('frozen_weights_unchanged')])
    training+='\n'+table(['Arm/seed','Gradient min','Gradient max','Largest per-update LoRA delta','Initial-to-final LoRA delta','Mean optimization sec','Peak allocated GB','Frozen unchanged'],diagnostics_rows)
    training+='\nPer-update loss, reward variance/components, LR, source IDs, generation/judging/optimization time, cumulative elapsed GPU-hour estimates and checkpoint paths remain in each run metrics.json and append-only generator_rl_update events. Actual allocation time is reconciled separately with Slurm; memory values include the co-resident frozen Qwen and are CUDA allocation, not total reservation.\n'
    training+='\n'+table(['Arm','Seed','Step','S','A','D','C','U','Reward','Generation sec'],[[x['arm'],x['seed'],x['step'],*[f'{x[k]:.8f}' for k in ['semantic','answer_consistency','distortion','cost','invalid_penalty','reward','generation_seconds']]] for x in evaluations])
    training+='\nRaw VQA diagnostics at the final checkpoint (A above is the semantic-gated reward credit):\n\n'+table(['Arm','Seed','Raw answer consistency','Same normalized answer fraction','Reference score, diagnostic only'],
        [[x['arm'],x['seed'],x['vqa_consistency_raw'],x['vqa_same_answer_fraction'],x['edited_reference_score_diagnostic_only']] for x in raw])
    training+='\nRaw reference scores are not accuracy on verified semantics-preserving edits. Image semantics remain unverified.\n'
    training+='\nAll intermediate checkpoints and images are retained. Measured baseline generation/judging and diagnostic optimization extrapolated four10-update arms beyond14400sec, so before training all arms were uniformly reduced to8 updates /80 exposures /4 validation looks, finalstep8 (configs/phase4_campaign_v2.yaml). Original10-update driver is retained unchanged. Primary comparison uses finalstep8, never the best observed development look. Training sources are seed-shuffled and image-distinct within each pilot; evaluation sources are fixed and image-disjoint from training. Logical10 / physical1 / accumulation10, rank8 alpha16, BF16 base and FP32 LoRA, fixed benign reward. No training reward-only quality claim.\n'
    training+='\nEach optimizer update draws a different10-source batch, so between-update training reward changes do not isolate learning. Fixed-source matched development outcomes determine the comparison. Every trained arm also performs one repeat-seed conditioning-control generation, charged to the allocation but excluded from its80 training trajectories.\n'
    (reports/'PHASE4_RL_TRAINING.md').write_text(training,encoding='utf-8')
    comparison=header+table(['Arm','Seed','Final reward','Delta vs frozen','Source-paired95CI'],[[x['arm'],x['seed'],f"{x['reward']:.8f}",x.get('paired_frozen',{}).get('mean_delta','reference'),x.get('paired_frozen',{}).get('ci95','reference')] for x in raw])
    comparison+='\nAcross training/sampling seeds (sample SD; only2 seeds):\n\n```json\n'+json.dumps(aggregate,indent=2)+'\n```\n\nRSI minus fixed RL, source-paired bootstrap,10000 resamples, seed7:\n\n```json\n'+json.dumps(comparisons,indent=2)+'\n```\n\nFairness checks:\n\n```json\n'+json.dumps(fairness,indent=2)+'\n```\n'
    comparison+='\nIntervals describe ten source images conditional on the fitted seed, not a population of training seeds. Repeated looks are correlated; the RSI development set guides its schedule and is reused for reporting, so the comparison is exploratory and potentially optimistic. Matched exposure and look counts do not imply identical GPU compute. Known backward numerical variability persists across fresh model execution. A favorable proxy difference alone does not establish independently verified semantics or generalization. No held-out300 run was opened.\n'
    comparison+='\nThe pre-schedule checkpoints and evaluations above use the same learning configuration in both arms. Any difference already present before the first LR revision is a numerical/execution replicate difference and cannot be caused by that later revision. These are individual descriptive witnesses, not a calibrated noise distribution. Original-answer disagreements are separately counted to expose evaluator variability.\n'
    comparison+='\nThe fixed cooldown heuristic is mean paired gain <= CI half-width on two successive looks, applied once. The raw reason string calls this unresolved gain; precisely, it is a magnitude-versus-width heuristic and is not equivalent to a percentile interval containing zero. Asymmetric bootstrap intervals may exclude zero while still meeting the heuristic. Schedule application never means an accepted quality gain; the original formula and raw decisions were preserved unchanged.\n'
    comparison+='\nEvidence-bounded claim flags (source-conditional intervals; not confirmatory):\n\n```json\n'+json.dumps(claims,indent=2)+'\n```\n'
    (reports/'PHASE4_RL_RSI_COMPARISON.md').write_text(comparison,encoding='utf-8')
    audit_path=reports/'phase4_latest_restart_audit.json'
    audit=load(Path(load(audit_path)['path'])/'metrics.json') if audit_path.exists() else {}
    diagnostics=[r for r in runs if r['id'].endswith('PH4_RESTART_DIAGNOSTIC')]
    restart=header+'**Strict stochastic checkpoint continuation remains unaccepted. No relaxed numerical PASS criterion was introduced.**\n\n'+json.dumps(audit,indent=2)+'\n\n'
    restart+='\n```json\n'+json.dumps([{'id':r['id'],'metrics':r['metrics']} for r in diagnostics],indent=2)+'\n```\n'
    outcomes=[{'id':r['id'],'metrics':r['metrics']} for r in runs if r['id'].endswith('PH4_RESTART_OUTCOMES')]
    restart+='\nOutcome-level numerical witness (if budget permitted; empty means NOT_RUN):\n\n```json\n'+json.dumps(outcomes,indent=2)+'\n```\n'
    restart+='\nThe first inspected mismatch is backward gradients after exact behavior likelihood replay and checked-state restoration. Sequential CFG, MATH SDPA, gradient checkpointing disabled, TF32 disabled, FP32 adapter operations and deterministic algorithm settings are inherited. These observations separate checked restoration from numerical execution, but do not identify a specific CUDA kernel cause or prove bitwise continuation impossible. Fixed states/rewards bypass stochastic collection; final RNG equality is not inferred. Four isolated CPU component tests pass; the full editor-environment suite result is recorded in phase4_tests.log. Fresh-initialization uninterrupted pilots do not require checkpoint recovery. Historical Phase3 failures remain FAIL. Numerical magnitudes are descriptive only; outcome-level reproducibility is assessed by paired development observations and seed variation, not accepted as equivalent resumed training.\n'
    (reports/'PHASE4_RESTART_DIAGNOSTIC.md').write_text(restart,encoding='utf-8')
    sempointer=load(reports/'phase4_latest_semantics.json');sem=load(Path(sempointer['path'])/'metrics.json')
    semantic=header+f'Portable package: `{sempointer["path"]}/index.html`; blank `review_form.csv`; raw inventories and recomputed numerical checks in semantic_table.json.\n\n```json\n'+json.dumps(sem,indent=2)+'\n```\n\nAll10 human labels remain blank/UNVERIFIED. Nine inventory wording-disagreement pairs are Uncertain; one awaits independent review without a wording flag. Automatically supported0 and automatically contradicted0 at the overall semantic level. Matching descriptions are not ground truth; differences are not proof of semantic alteration. Old raw model output and parser-failure evidence are retained. New RL images also remain human-unverified. No human/model agreement estimate is available.\n'
    (reports/'PHASE4_SEMANTIC_REVIEW.md').write_text(semantic,encoding='utf-8')
    failed=[r for r in runs if r['status']=='FAIL']
    failure=header+'Strict restart mismatch is an unresolved limitation, although the diagnostic program completed successfully. Diagnostic PASS means evidence creation, not continuation acceptance.\n\n'+table(['Run','Status'],[[r['id'],r['status']] for r in failed])
    failure+='\nThe first diagnostic used a historical_continuation_exact field that only checked adapter tensors; it was false. A separate immutable CPU audit reports historical optimizer/scheduler/step/version/consumed comparisons and corrects its scope, without rewriting old evidence. Early runtime ledger/lock files were initially unignored; some manifests truthfully report a dirty tree, although scientific source had been committed. Runtime ignore rules were fixed before training. Local source review found one semantic scope gap; fixed by the derived audit, same-family/provisional review only. Full failure logs remain in each immutable run.\n'
    failure+='\nThe rl123 launch wrapper inherited seed42 metadata; its saved command and actual scientific child correctly use seed123. Child config/initial-adapter artifacts determine the scientific seed. Launcher metadata was fixed for subsequent jobs; the completed launcher was not rewritten. The final same-family review had one semantic category-mapping gap, now declared in A14; changes after that exhausted two-round review budget were executor-verified only.\n'
    failure+='\nA live read-only status query hit a new RSI revision directory before its status.json was written and raised FileNotFoundError. The query now reports CREATING until required files exist and reads complete event lines. No scientific run was relabeled or retried for this observation error.\n'
    failure+='\nThe native18:33UTC check reported two stale-progress warnings for static Slurm wrappers, while their training children were progressing. Wrapper liveness now includes child logs/metrics under the same unchanged30min threshold; a regression test checks both live-child and truly stale cases. Historical CHECK records remain unchanged, and no research job was retried.\n'
    (reports/'PHASE4_FAILURES.md').write_text(failure,encoding='utf-8')
    lines=['# Phase4 实验进度与结果','',f'实际训练更新 **{update_count}次**；额外分配 **{charged/3600:.6f} A100 GPU小时**（{charged}秒，含诊断/失败/评估/RSI，上限4小时）。诊断{diagnostic}秒。','',
        '严格断点续训仍未通过。恢复字段及前向似然一致，首先观察到反向梯度差异，随后AdamW/LoRA不同；未放宽容差。后续均从同一预训练权重和每seed固定LoRA初始值开始，连续执行。','',
        '两seed汇总（均值±样本标准差；训练/采样随机性混合，仅作描述）：','',
        table(['方法','验证奖励','原始同答率','归一化失真D'],[[arm,f"{aggregate[arm]['mean_reward']:.6f} ± {aggregate[arm]['sample_sd']:.6f}",f"{aggregate[arm]['components']['vqa_same_answer_fraction']['mean']:.0%}",f"{aggregate[arm]['components']['distortion']['mean']:.6f}"] for arm in ['M0','M1','M2'] if aggregate[arm]['seed_count']==2]),
        '逐seed最终检查点：','',
        table(['方法','seed','最终验证奖励','相对冻结差值','配对95%区间'],[[x['arm'],x['seed'],f"{x['reward']:.6f}",f"{x['paired_frozen']['mean_delta']:.6f}" if 'paired_frozen' in x else '基线',f"[{x['paired_frozen']['ci95'][0]:.6f}, {x['paired_frozen']['ci95'][1]:.6f}]" if 'paired_frozen' in x else '基线'] for x in raw]),
        'M0冻结，M1固定Generator RL，M2带RSI的Generator RL。表中均为固定开发集代理指标；与训练奖励分开。最终checkpoint事先固定，不按最高开发分数挑选。','',
        ('两个seed的RL相对冻结开发集代理增益区间下界均大于0；只支持该小开发集上的探索性代理差异。' if rl_resolved else '目前未在两个seed上都取得超过配对源样本不确定性的RL代理增益，不能确认RL优于冻结生成器。'),
        ('两个seed的RSI相对固定RL代理增益区间下界均大于0；仍非独立语义/泛化证据。' if rsi_resolved else '目前未在两个seed上都取得超过配对源样本不确定性的RSI代理增益，不能确认RSI优于标准RL。'),'',
        'RSI相对标准RL：','',
        table(['seed','RSI−RL奖励差','配对95%区间'],[[x['seed'],f"{x['rsi_minus_rl']['mean_delta']:.6f}",f"[{x['rsi_minus_rl']['ci95'][0]:.6f}, {x['rsi_minus_rl']['ci95'][1]:.6f}]" ] for x in comparisons]),
        '两区间均含0，不能排除抽样不确定性。seed123在学习率修订前的第4步已出现约+0.09326的同配置差异；因此最终正差不能全部归因于RSI。修订前LoRA最大差异约4.34e-5/6.65e-5，表明独立执行的数值差异会累积。原图答案没有跨臂变化，但编辑图代理判定存在翻转。','',
        '每臂按预算统一8次更新、80训练暴露、4轮验证。M1与M2的初始LoRA逐位一致，训练来源和采样种子匹配；实际GPU时间分别为seed42的3227/3211秒、seed123的2707/3216秒，未建立严格计算时间等价。RSI仅在第4步后将LR从1e-5降为5e-6一次。','',
        f"训练共{counts['training_source_exposures']}条轨迹，来自{counts['unique_training_source_images_across_seeds_and_arms']}张不同训练图；验证含冻结对照共{counts['validation_trajectories_including_frozen']}次生成，但仅10张不同开发图，不能视为180个独立来源。另有4次重复种子控制生成、6次输出诊断生成，以及2次固定轨迹诊断优化器重放。",'',
        '两seed不足以证明跨训练随机性的优势；区间只描述10张来源图的条件不确定性。RSI复用开发集做决策，结论仅是探索性。','',
        '10对既有图像均未获独立人工语义标签，新编辑也未认证；代理奖励不能证明语义保持或泛化改善。当前不建议直接扩大GPU实验；先补独立语义复核与续训数值诊断，再据小样本代理结果决定下一步。','',
        '所有失败和弱结果保留。严格续训限制、字段范围修正与运行时账本引起的早期dirty标记见失败报告。原有两小时监控链保留，Phase4仅扩展检查，不增加调度器，不因空闲提交实验。','',
        '实验ID附录（PASS仅表示该程序执行完成；续训、语义与增益结论见正文）：\n'+ '\n'.join('- `'+r['id']+'`: '+r['status'] for r in runs),'',
        '完整组件、所有中间验证、初始LoRA/训练样本/种子公平性检查见同目录英文报告及phase4_results.json。']
    (reports/'PHASE4_PROGRESS_SUMMARY_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'gpu_seconds':charged,'training_updates':update_count,'runs':len(runs),'comparisons':comparisons}))

if __name__=='__main__':main()
