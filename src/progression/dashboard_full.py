"""Single authoritative export for the supplied complete bilingual dashboard.

Historical report charts remain historical. Finalized experiment measurements,
current stages, independent-review coverage and scheduler evidence are rebuilt
on each event/monitor refresh. state.json is a derived compatibility export only.
"""
import json
import shutil
from pathlib import Path
from .dashboard import atomic_text
from ..monitoring.common import read_json


def normalize(root,state):
    root=Path(root)
    data=read_json(root/'dashboard_package/phase6_supplied_v1/dashboard_data.json')
    stamp=state['updated_utc']; review=state['review_analysis']; follow=state['followup']
    scheduler=read_json(root/'reports/monitoring/scheduler.json',{})
    data.update(schema_version='phase6_full_live_v3',updated_utc=stamp,source_commit=state['source_commit'])
    data['meta'].update(updated_at=stamp,as_of=stamp,last_verified_phase='Phase 6 follow-up',
        source_kind='canonical_registry_and_verified_receipts',data_mode='event_and_two_hour_monitor',
        verification_note='Registry and receipt export; scheduler observation time is shown separately. Historical charts are not new trials.')
    cpu_names={'S1_DATA','S1_REVIEW_PACKAGE','S1_REWARD_AUDIT','S2_REAL_HARNESS','S3_RL_ENGINEERING','S4_STATIC_COMPARISON'}
    cpu_states=[v['state'] for k,v in state['stage_status'].get('stages',{}).items() if k in cpu_names]
    engineering='PASS' if len(cpu_states)==6 and all(s=='PASSED' for s in cpu_states) else 'NEEDS_VERIFICATION'
    data['current'].update(phase='Phase 6 follow-up',science_gate='NO-GO',engineering_gate=engineering,
        activity='24/24 review pairs verified; full dashboard deployed; two new structural revisions rejected; no new GPU work.',
        active_scientific_jobs=[j for j in state['jobs'] if j['scope']=='this_project' and 'monitor' not in j['name'] and 'cpu' not in j['name']],
        monitor={**scheduler,'reported_job_id':scheduler.get('next_job_id'),
            'reported_job_state':next((j['status'] for j in state['jobs'] if j['job_id']==scheduler.get('next_job_id')),'UNKNOWN'),
            'next_eligible_at':scheduler.get('next_due_utc'),
            'scheduler_observed_at':state['live'].get('timestamp_utc')},
        next_gate=state['next_eligible_task'],recommended_action=state['next_eligible_task'])
    data['headline_metrics']['verified_semantic_samples']=review.get('independently_reviewed_consensus_pairs',0)
    data['dataset']['independent_verified_semantic']=data['headline_metrics']['verified_semantic_samples']
    data['dataset'].update(confirmed_sc=0,confirmed_si_v=0)
    # Keep independently reviewed consensus separate from per-dimension coverage.
    data['phase6_stages']=[{'name':sid,'status':v['state'],
        'evidence':f"{v.get('experiment_id','—')} | {v.get('completion_evidence_state',v.get('reason',v.get('evidence','')))}",
        'evidence_en':f"{v.get('experiment_id','—')} | {v.get('completion_evidence_state',v.get('reason',v.get('evidence','')))}"}
        for sid,v in state['stage_status'].get('stages',{}).items()]
    baseline_ids={r['id'] for r in data['experiments']}
    rows=[{'id':r['experiment_id'],'name':r['experiment_id'].split('_',3)[-1],
        'phase':'P6' if 'phase6' in r['experiment_id'] else 'Historical',
        'status':r['status'],'created_at':r['created_at'],'ended_at':r['ended_at'],
        'metrics_present':r['metrics_present'],'metric_summary':r.get('metric_summary',{}),
        'metrics_path':r.get('metrics_path'),'note':r['scope']} for r in state['experiments']]
    data['live']={'mode':'canonical_export','last_scan_at':stamp,'total_runs_seen':len(rows),
        'recent_experiments':rows,'new_experiments_since_snapshot':sum(r['id'] not in baseline_ids for r in rows),
        'active_experiments':[r for r in rows if r['status'] in ['CREATED','RUNNING','SUBMITTED']],
        'verified_results':[],'new_reports':[],
        'notice':'Finalized engineering receipts are shown separately; no independent semantic efficacy established.'}
    data['followup']=follow;data['review_analysis']=review;data['jobs']=state['jobs']
    data['scientific_decision']='NO-GO';data['authorized_budget']=state['authorized_budget']
    data['blockers']=state['blockers'];data['stage_status']=state['stage_status']
    data['stage_receipts']=state['stage_receipts'];data['current_phase']=state['current_phase']
    data['actual']=state['actual'];data['package_status']=state['package_status']
    # Each finalized follow-up receipt refreshes the issue/status content as well as the registry.
    data['issues'] += [dict(id='FOLLOWUP_DEPLOY',title='完整版与实时状态统一 / Complete UI and live state integrated',
        severity='low',solved=True,status='DEPLOYED',why='旧同步只追加 registry 会留下过期首页。',
        action='Finalization hook + existing two-hour Slurm monitor rebuild all views from dashboard_data.json.',
        evidence='PHASE6_FOLLOWUP.md'),
        dict(id='FOLLOWUP_REVIEW',title='独立标签仍为 UNKNOWN / Independent labels remain UNKNOWN',
        severity='critical',solved=False,status=f"{review.get('submission_files',0)} submissions; {review.get('independently_reviewed_consensus_pairs',0)} independently reviewed consensus pairs",
        why='Automatic scores and file integrity cannot establish edited-image facts.',
        action='Use independent review form, verify human provenance, retain disagreements; reward and pilot stay blocked.',evidence='PHASE6_FOLLOWUP.md')]
    return data


def publish(root,state):
    from ..monitoring.common import FileLock
    with FileLock(Path(root)/'reports/phase6_followup/dashboard.lock') as acquired:
        if not acquired:raise BlockingIOError(11,'Dashboard export already in progress')
        return _publish(root,state)


def _publish(root,state):
    from .dashboard_full_ui import render_pages
    root=Path(root);base=root/'dashboard';base.mkdir(exist_ok=True)
    data=normalize(root,state)
    extra=(root/'src/progression/dashboard_full_extension.js').read_text(encoding='utf8')
    for name,page in render_pages(data).items():
        # Wrap the supplied renderer: every JSON poll also updates all live panels.
        page=page.replace('initUI();renderPage();refreshData(false)',extra+'\ninitUI();renderPage();refreshData(false)')
        page=page.replace('</head>','<style>main,section,article,.grid>*{min-width:0}p,strong,summary,pre{overflow-wrap:anywhere}.stat-row{gap:8px}.stat-row>*{min-width:0}.stat-row strong{text-align:right}.tag{white-space:normal}pre{max-width:100%;white-space:pre-wrap}@media(max-width:700px){.cols-2,.cols-2-even{grid-template-columns:minmax(0,1fr)}.section-head,.en-header,.stat-row{flex-wrap:wrap}.hero-visual{min-width:0}}</style></head>')
        page=page.replace('51.56% of training rewards clipped to −1','46% of all 500 observations clipped to −1 (230/500); training-only subset 51.56%')
        page=page.replace('正在运行 / 排队（最近报告快照）','正在运行 / 排队（调度器观测）').replace('非 Slurm 实时连接','带时间戳的 Slurm 导出')
        page=page.replace('以下状态来自 Phase 6 上传报告，并非实时 Slurm 查询。','以下状态来自当前阶段记录与已核验回执；调度器观测时间单独显示。')
        atomic_text(base/name,page)
        atomic_text(base/('index_zh.html' if '_zh' in name else 'index_en.html'),page)
    for name in ['build_dashboard.py','dashboard_sync.py']:
        atomic_text(base/name,(root/'scripts/dashboard_export_entry.py').read_text(encoding='utf8'))
    sources=base/'sources';sources.mkdir(exist_ok=True)
    for p in (root/'dashboard_package/phase6_supplied_v1/sources').glob('*'):
        if p.is_file():shutil.copy2(p,sources/p.name)
    for p in (root/'reports').glob('PHASE6_*.md'):
        shutil.copy2(p,sources/p.name)
    report=root/'reports/PHASE6_FOLLOWUP.md'
    if report.exists():shutil.copy2(report,sources/report.name)
    encoded=json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2)+'\n'
    atomic_text(base/'state.json',encoded) # compatibility projection, never an input
    atomic_text(base/'dashboard_data.json',encoded) # authoritative commit marker published last
    return {'json':str(base/'dashboard_data.json'),'zh':str(base/'rsi_poisoning_zh.html'),
        'en':str(base/'rsi_poisoning_en.html'),'snapshot_utc':state['updated_utc']}
