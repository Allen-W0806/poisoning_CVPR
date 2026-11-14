"""Only new Phase6 follow-up work; no old test suite, controller or GPU launch."""
import json
import time
from pathlib import Path
import _bootstrap
from src.utils.logging import Experiment,write_json,utcnow
from src.data.dataset import sha256
from src.progression.followup import prepare_review_workflow,separated_measurements
from src.progression.review_workflow import refresh_review_analysis
from src.progression.real_harness import RecordedArtifactGate
from src.rsi.harness.harness_spec import HarnessSpec
from src.rsi.harness.harness_executor import HarnessExecutor
from src.rsi.harness.trace_collector import Resources
from src.rsi.harness.version_manager import VersionManager
from src.evaluation.phase5_comparability import compare_checkpoints

def new_harness_candidates(root,run):
    prior=json.loads((root/'reports/phase6/S2_REAL_HARNESS.json').read_text())
    old=root/'experiments'/prior['experiment_id'];frozen=json.loads((old/'predeclared_cases.json').read_text())
    evidence=json.loads((root/'reports/phase4_evidence.json').read_text())
    expected_by_parent={v['id']:{p['path']:p['sha256'] for p in v['files']} for v in evidence['phase4_runs']}
    cases=[];parents={}
    for selected in frozen:
        if selected['role']!='workflow_gate':continue
        parent=selected['parent']
        if parent not in parents:parents[parent]=json.loads((root/'experiments'/parent/'metrics.json').read_text())
        obs=next(o for u in parents[parent]['updates'] for o in u['observations'] if o['sample_id']==selected['sample_id'])
        source=Path(obs['source_path']);relative=source.relative_to(root/'experiments'/parent).as_posix()
        digest=expected_by_parent[parent][relative];assert sha256(source)==digest
        common={'case_id':selected['sample_id'],'source_group':selected['sample_id'],'image_path':str(source),
          'expected_size':[512,512],'expected_sha256':digest,'split':'workflow_gate','expected_valid':True,'population':'recorded_train_artifact'}
        cases.extend([common,{**common,'case_id':common['case_id']+'_wrong_size','expected_size':[1,1],'expected_valid':False,'population':'injected_metadata_control'},
          {**common,'case_id':common['case_id']+'_wrong_digest','expected_sha256':'0'*64,'expected_valid':False,'population':'injected_metadata_control'}])
    assert len(cases)==72
    incumbent=HarnessSpec.from_dict(json.loads((old/'harness_versions/H0002.json').read_text()))
    gate=RecordedArtifactGate(HarnessExecutor(root),cases);manager=VersionManager(run.path/'candidate_versions',incumbent)
    decisions=[];resources=Resources()
    proposals=[(HarnessSpec('content_first',('content_hash','geometry')),'evidence_acquisition_and_validation_orchestration',['workflow.checks'],'Test whether reordering reduces frozen diagnostic cost'),
      (HarnessSpec('summary_memory',incumbent.checks,memory_mode='summary'),'experience_memory',['workflow.memory_mode'],'Test summary memory without altering fixed acceptance criterion')]
    for program,category,modules,reason in proposals:
        proposal={'parent_version':manager.incumbent,'candidate':program.to_dict(),'change':reason,
          'supporting_evidence':{'source_experiment':prior['experiment_id'],'fixed_gate':'same24 sources/72 rows; same >=20% operation saving and zero regression',
            'hypothesis':'check order or evidence representation may alter resources; improvement not assumed'},
          'expected_effect':'measured cost reduction, otherwise reject','affected_modules':modules,
          'regression_tests':['unchanged positive geometry/content decisions','unchanged48 metadata-negative controls'],'category':category}
        result=manager.evaluate(proposal,gate);decisions.append(result)
        for cost in result['actual_evaluation_resources'].values():resources.add(Resources(**cost))
    assert not any(r['accepted'] for r in decisions)
    manager.rollback('H0000','Both new candidates rejected; original accepted workflow retained')
    return {'source_accepted_harness':{'experiment_id':prior['experiment_id'],'version':'H0002'},'decisions':decisions,
      'newly_accepted_revisions':0,'unchanged_validation_criterion':True,'new_cost':resources.__dict__,
      'reused_prior_investigation_cost':prior['results']['total_search_resources'],'prior_evidence_not_reexecuted':True,
      'scope':'unchanged structural task; no semantic improvement claim; old evidence cost not charged twice'}

def initialization_check(root):
    pairs=[('42','EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL','EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI'),
      ('123','EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL','EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI')]
    results=[]
    for seed,a,b in pairs:
        left=root/'experiments'/a/'checkpoints/step_0000/training_state.pt';right=root/'experiments'/b/'checkpoints/step_0000/training_state.pt'
        result=compare_checkpoints(left,right)
        results.append({'seed':seed,'left':str(left.relative_to(root)),'right':str(right.relative_to(root)),'comparison':result})
    return {'actual_historical_step0_cpu_checks':results,'new_pilot_initialization_verified':False,
      'fresh_isolated_semantic_gate_available':False,'scientific_gpu_budget_seconds':0,
      'full_comparison_forecast_gpu_hours':42.19540658619441,'forecast_is_authorization':False,
      'human_and_future_independent_evaluator_cost':'UNKNOWN; excluded from historical forecast',
      'statistical_unit':'unique source-content cluster within fixed seed, not transitions/questions; limited diagnostic review queue',
      'pilot_status':'BLOCKED','strict_continuation':'UNRESOLVED; not rerun'}

def main():
    root=_bootstrap.ROOT;start=time.perf_counter();cpu=time.process_time()
    config={'phase':6,'stage':'F6_REVIEW_DASHBOARD_FOLLOWUP','seed':42,'new_model_calls':0,'gpu_seconds':0,'completed_phase6_tests_repeated':False}
    with Experiment(root,'phase6_review_dashboard_followup',config,{'stage':config['stage'],'dataset_version':'vqav2_content_v2',
        'reward_version':'legacy_saved_scalar_read_only','harness_version':'prior_H0002_plus_rejected_new_candidates',
        'evaluation_protocol':'independent_review_preparation_v1'},parent=json.loads((root/'reports/phase6/S1_REVIEW_PACKAGE.json').read_text())['experiment_id']) as run:
        result={'packet_audit':prepare_review_workflow(root),'reward_separation':separated_measurements(root,run.path),
           'new_harness_candidates':new_harness_candidates(root,run),'pilot_comparison':initialization_check(root),
           'independent_review_analysis':refresh_review_analysis(root)}
        result['resources']={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'gpu_seconds':0,'model_calls':0}
        run.metrics(result)
        write_json(run.path/'phase6_run_contract.json',{'experiment_id':run.id,'stage':config['stage'],
          'parent_experiment_id':run.manifest['parent_experiment'],'start_time':run.manifest['created_at'],'end_time':utcnow(),'status':'PASS',
          'model_revision':'fixed historical source manifests','dataset_version':'vqav2_content_v2','reward_version':'legacy_saved_scalar_read_only',
          'harness_version':'prior_H0002_plus_rejected_candidates','git_commit':run.manifest['git_commit'],'seed':42,
          'slurm_job_id':run.manifest['slurm_job_id'],'gpu_seconds':0,'configuration':config,'metrics':'metrics.json',
          'failure_reason':None,'checkpoint_path':None,'evaluation_protocol':'independent_review_preparation_v1'})
        outputs={p.relative_to(run.path).as_posix():sha256(p) for p in run.path.rglob('*') if p.is_file() and p.name not in {'run.log','train.log','summary.md','status.json'}}
        write_json(run.path/'output_manifest.json',{'experiment_id':run.id,'files':outputs,'required_metrics':list(result)})
        run.finish(summary='Only new review/proxy/provenance/dashboard/candidate work; no Phase6 suite repetition or scientific training.')
        write_json(root/'reports/phase6_followup/latest.json',{'experiment_id':run.id,'path':str(run.path),'results':result})
        from src.progression.dashboard import refresh
        refresh(root)
    print(json.dumps({'id':run.id,'packet_pairs':24,'new_accepted_revisions':0,'scientific_decision':'NO-GO'}))

if __name__=='__main__':main()
