"""CPU-only engineering validation. Never submits jobs or loads a model."""
import argparse
import json
import copy
import time
from dataclasses import asdict
import _bootstrap
from src.utils.logging import Experiment,write_json
from src.rsi.harness.trace_collector import TraceCollector,Resources
from src.rsi.harness.harness_spec import HarnessSpec
from src.rsi.harness.harness_executor import HarnessExecutor
from src.rsi.harness.failure_investigator import FailureInvestigator
from src.rsi.harness.revision_proposer import propose
from src.rsi.harness.version_manager import VersionManager
from src.rsi.harness.cost_aware_gate import IndependentGate,GatePolicy
from src.rsi.harness.engineering_fixture import create_cases

def full_smoke(root):
    with Experiment(root,'phase5_harness_evolution',{'phase':5,'scope':'constructed_harmless_integrity_task',
            'new_gpu_seconds':0,'scientific_training':False,'gate_policy':asdict(GatePolicy())}) as run:
        started=time.perf_counter();cpu=time.process_time()
        train,truth=create_cases(run.path,'train',8);gate_cases,oracle=create_cases(run.path,'gate',32)
        trace=TraceCollector(run.path/'training_trace.jsonl');executor=HarnessExecutor(run.path)
        incumbent=HarnessSpec('geometry_only');manager=VersionManager(run.path/'harness_versions',incumbent)
        failed=executor.run(incumbent,train[0]);assert failed['predicted_valid']!=truth[train[0]['case_id']]
        trace.append('failed_incumbent_execution',failed)
        investigator=FailureInvestigator(executor,trace,train)
        diagnosis=investigator.investigate_integrity_failure(train[0])
        proposal=propose(incumbent,diagnosis,manager.incumbent);assert proposal is not None
        gate=IndependentGate(executor,gate_cases,oracle)
        ineffective=copy.deepcopy(proposal);ineffective['candidate']=HarnessSpec('decode_only',('decode',)).to_dict()
        rejected=manager.evaluate(ineffective,gate);assert not rejected['accepted'] and manager.incumbent=='H0000'
        accepted=manager.evaluate(proposal,gate);assert accepted['accepted'] and manager.incumbent=='H0002'
        regression=copy.deepcopy(proposal);regression['parent_version']=manager.incumbent;regression['candidate']=incumbent.to_dict()
        rejected_regression=manager.evaluate(regression,gate);assert not rejected_regression['accepted'] and manager.incumbent=='H0002'
        manager.rollback('H0000','Explicit restoration demonstration; rejected attempts never replaced incumbent')
        assert manager.program().checks==('geometry',)
        manager.rollback('H0002','Restore accepted workflow for independent future inputs')
        search=Resources();search.add(Resources(**failed['resources']));search.add(investigator.resources)
        for revision in [rejected,accepted,rejected_regression]:
            for key in ['parent','candidate']:search.add(Resources(**revision['actual_evaluation_resources'][key]))
        search.wall_seconds=time.perf_counter()-started;search.cpu_seconds=time.process_time()-cpu
        result={'diagnosis':diagnosis,'ineffective_rejected':not rejected['accepted'],'useful_revision_accepted':accepted['accepted'],
            'regression_rejected':not rejected_regression['accepted'],'rollback_verified':True,'final_incumbent':manager.incumbent,
            'accepted_validation_aggregate':accepted['measured_validation'],'total_search_resources':asdict(search),
            'selected_harness_execution_resources':accepted['measured_validation']['candidate_resources'],
            'trace_events':trace.count,'new_scientific_updates':0,'new_model_calls':0,
            'scope':'Deterministic engineering oracle only; natural-image and security validity not established'}
        run.metrics(result);run.finish(summary='Real active investigation and reusable executable workflow: reject, accept, regression reject, rollback; all costs retained.')
        write_json(root/'reports/phase5_harness_smoke.json',{'id':run.id,'path':str(run.path),'results':result})
        print(json.dumps({'id':run.id,'status':'PASS','accepted_version':manager.incumbent,'cost':asdict(search)}))

def main():
    p=argparse.ArgumentParser();p.add_argument('--spine',action='store_true');args=p.parse_args()
    if not args.spine:return full_smoke(_bootstrap.ROOT)
    with Experiment(_bootstrap.ROOT,'phase5_cpu_spine',{'phase':5,'cpu_only':True,'scientific_training':False}) as run:
        trace=TraceCollector(run.path/'trace.jsonl')
        trace.append('entrypoint',{'artifact_contract':'phase5_engineering_v1','model_calls':0})
        run.metrics({'trace_events':trace.count,'resources':trace.resources().__dict__,'scope':'engineering_spine_only'})
        run.finish(summary='Real entry point, append-only trace and exclusive experiment artifact; no scientific result.')
        write_json(_bootstrap.ROOT/'reports/phase5_latest_spine.json',{'id':run.id,'path':str(run.path)})
        print(json.dumps({'id':run.id,'status':'PASS','scope':'engineering_spine_only'}))

if __name__=='__main__':main()
