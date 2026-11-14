"""Registry-backed bilingual dashboard with one authoritative JSON state."""
import html
import json
from pathlib import Path
from ..monitoring.common import read_json,now
from ..utils.logging import write_json,git_commit

def experiment_inventory(root):
    rows=[]
    for path in sorted((root/'experiments').glob('EXP_*'),reverse=True):
        status=read_json(path/'status.json',{});manifest=read_json(path/'manifest.json',{})
        metrics=read_json(path/'metrics.json',{})
        rows.append({'experiment_id':path.name,'status':status.get('status','UNKNOWN'),
          'created_at':manifest.get('created_at'),'ended_at':status.get('updated_at'),
          'slurm_job_id':manifest.get('slurm_job_id'),'git_commit':manifest.get('git_commit'),
          'stage':manifest.get('stage',manifest.get('hyperparameters',{}).get('stage')),
          'metrics_present':bool(metrics),'scientific_success':False,
          'metric_summary':{k:v for k,v in metrics.items() if v is None or isinstance(v,(str,int,float,bool))},
          'metrics_path':'../experiments/'+path.name+'/metrics.json',
          'scope':'engineering/audit' if 'phase6' in path.name else 'historical, see fixed experiment report'})
    return rows

def build_state(root,live=None):
    root=Path(root);receipts={p.stem:read_json(p) for p in (root/'reports/phase6').glob('S*.json')}
    state=read_json(root/'reports/PHASE6_STAGE_STATUS.json',{});live=live or read_json(root/'reports/phase6/live.json',{})
    review=read_json(root/'reports/phase6_followup/review_analysis.json',{})
    runs=experiment_inventory(root)
    followup=read_json(root/'reports/phase6_followup/latest.json',{})
    queue=live.get('scheduler',{}).get('queue',{})
    jobs=[]
    for line in queue.get('stdout','').splitlines():
        fields=line.split('|')
        if len(fields)>=3:
            jobs.append({'job_id':fields[0],'status':fields[1],'name':fields[2],
              'scope':'this_project' if fields[2].startswith('poisoning-') else 'other_project'})
    return {'schema_version':'phase6_dashboard_v2','updated_utc':now(),'source_commit':git_commit(root),
       'current_phase':'Phase6 follow-up: independent review readiness and live dashboard integration',
       'stage_status':state,'stage_receipts':receipts,'live':live,'experiments':runs,'jobs':jobs,
       'followup':followup,'review_analysis':review,
       'actual':{'new_gpu_hours':0,'new_generator_updates':0,'human_verified_labels':review.get('independently_reviewed_consensus_pairs',0)},
       'authorized_budget':{'scientific_gpu_seconds':0},
       'historical':{'phase4_updates':32,'phase4_gpu_hours':13122/3600,'scientific_RSI_advantage':'not established','strict_continuation':'unresolved'},
       'synthetic':{'phase5_integrity_accuracy_before':.5,'phase5_integrity_accuracy_after':1.,'sources':32},
       'prepared':{'arms':['M0','M1','M2','M3','M4','M5'],'status':'PREPARED_NOT_RUN','historical_planning_gpu_hours':42.19540658619441},
       'models':{'VLM':'Qwen/Qwen3-VL-2B-Instruct','generator':'timbrooks/instruct-pix2pix','trainable':'UNet attention LoRA rank8 alpha16'},
       'failure_history':read_json(root/'reports/phase6/failure_summary.json',{}),
       'scientific_decision':'NO-GO','blockers':['independent edited facts and semantic validation','accepted benign training reward','fresh isolated evaluation Gate','pre-intervention comparison integrity','explicit scientific GPU budget'],
       'next_eligible_task':'Independent24-pair review and provenance/consensus analysis; CPU-only monitor and dashboard synchronization',
       'review_workflow':'../reviews/workflow/index.html',
       'package_status':read_json(root/'reports/phase6_followup/dashboard_package.json',{'provided_complete_package':'NOT_LOCATED','deployed_view':'existing minimal view upgraded with full live state; supplied-package integration pending its location'}),
       'sync_notice':'HTTP pages poll state.json; local files require scripts/sync_phase6.ps1 then reload. File refresh rereads only the local synchronized copy.'}

def refresh(root,live=None):
    from .dashboard_full import publish
    return publish(Path(root),build_state(root,live))
