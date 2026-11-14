import argparse
import json
import _bootstrap
from src.progression.controller import run_controller
from src.utils.logging import write_json

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spine',action='store_true');p.add_argument('--corrected-audits',action='store_true');a=p.parse_args()
    if a.spine:
        def spine(root,stage):
            path=root/'reports/phase6/spine.json';write_json(path,{'stage':stage,'actual_cpu_execution':True,'gpu_seconds':0})
            return {'state':'PASSED','evidence':str(path),'scope':'entry-point/persistence engineering spine only'}
        result=run_controller(_bootstrap.ROOT,{'reward_audit':spine},selected={'S1_REWARD_AUDIT'},state_name='phase6/spine_state.json')
    else:
        if a.corrected_audits:
            import uuid
            from src.progression.monitor import CPU_STAGES
            from src.monitoring.common import read_json
            path=_bootstrap.ROOT/'reports/PHASE6_STAGE_STATUS.json';prior=read_json(path)
            write_json(_bootstrap.ROOT/'reports/phase6/stage_history'/('prior_'+uuid.uuid4().hex+'.json'),prior)
            for sid in CPU_STAGES:
                item=prior['stages'][sid]
                item.update(state='NOT_STARTED',correction_parent_experiment=item.get('experiment_id'),
                  correction_reason='Versioned provenance/aggregation/completion fixes; original audit retained')
            write_json(path,prior)
        from src.progression.tasks import handlers
        result=run_controller(_bootstrap.ROOT,handlers())
        from src.progression.monitor import cycle
        cycle(_bootstrap.ROOT)
    print(json.dumps(result,indent=2))
