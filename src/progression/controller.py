import json
import time
from pathlib import Path
from ..monitoring.common import FileLock, append_json, now, read_json
from ..utils.logging import write_json

STATES={'NOT_STARTED','READY','RUNNING','PASSED','FAILED','BLOCKED','NEEDS_HUMAN_REVIEW','NEEDS_AUTHORIZATION'}
DEFINITIONS=[
 ('S1_DATA',[], 'validate_data'),
 ('S1_REVIEW_PACKAGE',['S1_DATA'],'review_package'),
 ('S1_REWARD_AUDIT',[],'reward_audit'),
 ('S1_SEMANTIC_REVIEW',['S1_REVIEW_PACKAGE'],'human_gate'),
 ('S2_REAL_HARNESS',['S1_DATA'],'real_harness'),
 ('S3_RL_ENGINEERING',[],'rl_checks'),
 ('S3_REWARD_RELEASE',['S1_SEMANTIC_REVIEW','S1_REWARD_AUDIT'],'reward_gate'),
 ('S4_STATIC_COMPARISON',['S1_DATA'],'comparison'),
 ('S4_SCIENTIFIC_READY',['S4_STATIC_COMPARISON','S3_REWARD_RELEASE'],'scientific_gate'),
 ('S5_PILOT',['S4_SCIENTIFIC_READY'],'pilot_gate')]

def policy(root):
    value=read_json(Path(root)/'configs/phase6.json')
    if value['scientific_gpu_budget_seconds']!=0 or value['allow_gpu_submission'] is not False:
        raise ValueError('Phase6 has zero authorized scientific GPU budget')
    if value['maximum_retries_per_signature']!=2 or value['logical_batch_size']!=10:
        raise ValueError('Retry and logical batch policy cannot change implicitly')
    return value

def run_controller(root,handlers,selected=None,state_name='PHASE6_STAGE_STATUS.json'):
    root=Path(root).resolve();config=policy(root);base=root/'reports/phase6';base.mkdir(parents=True,exist_ok=True)
    with FileLock(base/'controller.lock') as acquired:
        if not acquired:return {'status':'SKIPPED_LOCKED'}
        state_path=root/'reports'/state_name
        state=read_json(state_path,{'schema_version':'phase6_stage_state_v1','stages':{},'retry_counts':{}})
        for sid,deps,task in DEFINITIONS:
            item=state['stages'].setdefault(sid,{'stage_id':sid,'state':'NOT_STARTED'})
            item.update(required_inputs=deps,readiness_conditions='all required stages PASSED; task-specific evidence validated',
              allowed_actions='bounded CPU '+task,required_tests='task assertions and regression receipts',
              maximum_resource_budget={'wall_seconds':config['cpu_stage_wall_budget_seconds'],'gpu_seconds':0},
              next_eligible_stage=None)
            if item['state']=='RUNNING':
                item.update(state='FAILED',reason='Interrupted prior execution retained; explicit recovery required')
            if item['state']=='FAILED' and item.get('retryable_engineering_failure'):
                signature=sid+':'+item.get('failure_signature','unknown')
                count=state['retry_counts'].get(signature,0)
                if count<2:
                    state['retry_counts'][signature]=count+1;item['state']='NOT_STARTED'
                    append_json(base/'transitions.jsonl',{'timestamp':now(),'stage':sid,'event':'bounded_cpu_retry','attempt':count+1,'signature':signature})
            if item['state'] in {'PASSED','FAILED'} or (selected is not None and sid not in selected):continue
            if any(state['stages'].get(dep,{}).get('state')!='PASSED' for dep in deps):
                item.update(state='BLOCKED',decision='BLOCKED',reason='Unresolved prerequisite: '+', '.join(deps));continue
            if task not in handlers:continue
            item.update(state='READY',decision=None)
            append_json(base/'transitions.jsonl',{'timestamp':now(),'stage':sid,'state':'READY'})
            item.pop('experiment_id',None)
            item.update(state='RUNNING',started_at=now());write_json(state_path,state)
            start=time.perf_counter()
            try:
                output=handlers[task](root,sid)
                if output.get('state') not in STATES or output['state'] in {'NOT_STARTED','READY','RUNNING'}:
                    raise ValueError('Task returned an invalid terminal decision')
                elapsed=time.perf_counter()-start
                if elapsed>config['cpu_stage_wall_budget_seconds']:raise TimeoutError('CPU stage exceeded wall budget')
                item.update(output,elapsed_wall_seconds=elapsed)
                item['decision']='PASS' if item['state']=='PASSED' else 'FAIL' if item['state']=='FAILED' else 'BLOCKED'
            except Exception as error:
                current_id=getattr(error,'phase6_experiment_id',None)
                attempt=read_json(base/'attempts'/f'{sid}.json',{})
                if not current_id and attempt.get('created_at','')>=item['started_at']:current_id=attempt.get('experiment_id')
                if current_id:item['experiment_id']=current_id
                item.update(state='FAILED',decision='FAIL',reason=type(error).__name__+': '+str(error),elapsed_wall_seconds=time.perf_counter()-start)
                item.update(failure_signature=type(error).__name__+':'+str(getattr(error,'errno',None)),
                  retryable_engineering_failure=isinstance(error,OSError) and getattr(error,'errno',None) in {5,11,116})
            item['ended_at']=now();append_json(base/'transitions.jsonl',dict(item));write_json(state_path,state)
            if state_name=='PHASE6_STAGE_STATUS.json':
                from .dashboard import refresh
                refresh(root)
        pending=[sid for sid,deps,task in DEFINITIONS if state['stages'][sid]['state']=='NOT_STARTED' and all(state['stages'].get(d,{}).get('state')=='PASSED' for d in deps)]
        state.update(updated_at=now(),scientific_decision='NO-GO',authorized_gpu_seconds=0,next_eligible_tasks=pending,
          next_automatic_action=pending[0] if pending else 'lightweight integrity/completion/review-availability check at existing monitor cycle')
        for item in state['stages'].values():item['next_eligible_stage']=pending[0] if pending else None
        write_json(state_path,state)
        return state
