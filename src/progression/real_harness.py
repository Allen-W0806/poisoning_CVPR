"""Workflow efficiency experiment on recorded train artifacts and labeled metadata controls."""
import copy
import json
from pathlib import Path
from PIL import Image
from ..rsi.harness.harness_spec import HarnessSpec
from ..rsi.harness.harness_executor import HarnessExecutor
from ..rsi.harness.trace_collector import TraceCollector,Resources
from ..rsi.harness.failure_investigator import FailureInvestigator
from ..rsi.harness.version_manager import VersionManager
from ..utils.logging import write_json
from ..data.dataset import sha256

class RecordedArtifactGate:
    """Exact integrity oracle only. Controls perturb metadata, never original images."""
    def __init__(self,executor,cases):
        self.executor=executor;self.cases=copy.deepcopy(cases);self.last_private_artifact=None
        if len({c['source_group'] for c in cases})<24:raise ValueError('At least24 paired source groups')
        if any(c['population'] not in {'recorded_train_artifact','injected_metadata_control'} for c in cases):raise ValueError('Unknown gate population')

    def compare(self,parent,candidate):
        costs=[Resources(),Resources()];correct=[0,0];new_errors=0;error_groups=set();rows=[]
        for case in self.cases:
            a,b=[self.executor.run(p,case) for p in [parent,candidate]];expected=case['expected_valid']
            for i,r in enumerate([a,b]):costs[i].add(Resources(**r['resources']));correct[i]+=r['predicted_valid']==expected
            new_errors+=(a['predicted_valid']==expected and b['predicted_valid']!=expected)
            if a['predicted_valid']==expected and b['predicted_valid']!=expected:error_groups.add(case['source_group'])
            rows.append({'source_group':case['source_group'],'population':case['population'],'expected':expected,'parent':a,'candidate':b})
        saving=1-costs[1].diagnostic_operations/costs[0].diagnostic_operations
        accepted=correct[0]==correct[1] and new_errors==0 and saving>=.2
        result={'accepted':accepted,'n_sources':len({c['source_group'] for c in self.cases}),'n_case_rows':len(rows),
          'parent_accuracy':correct[0]/len(rows),'candidate_accuracy':correct[1]/len(rows),
          'new_errors_on_previously_correct_rows':new_errors,'unique_sources_with_new_errors':len(error_groups),
          'accuracy_denominator':'72 equally weighted rows, one recorded-positive plus two metadata-negative controls per source',
          'diagnostic_operation_saving_fraction':saving,
          'parent_resources':costs[0].__dict__,'candidate_resources':costs[1].__dict__,
          'scope':'recorded file-integrity workflow with explicitly injected metadata controls; no semantic editing claim',
          'policy':'same accuracy, zero regression, >=20% diagnostic-operation reduction, fixed before execution',
          'evaluator_definitions_changed':False,'matched_exposure':True}
        self.last_private_artifact={'paired_rows':rows,'aggregate':result};return result

def real_harness(root,run):
    from .tasks import training_selection
    selection=training_selection(root)[:32]
    if len(selection)<32:raise ValueError('Insufficient distinct train content for8+24 partition')
    frozen=[{'parent':o['parent_experiment_id'],'sample_id':r['sample_id'],'raw_source_sha256':r['image_sha256'],
             'role':'investigation' if i<8 else 'workflow_gate'} for i,(o,r) in enumerate(selection)]
    write_json(run.path/'predeclared_cases.json',frozen)
    evidence=json.loads((root/'reports/phase4_evidence.json').read_text())
    historical={v['id']:{r['path']:r['sha256'] for r in v['files']} for v in evidence['phase4_runs']}
    cases=[]
    for i,(o,r) in enumerate(selection):
        source=Path(o['source_path']);relative=source.relative_to(root/'experiments'/o['parent_experiment_id']).as_posix()
        expected=historical[o['parent_experiment_id']][relative];assert sha256(source)==expected
        with Image.open(source) as im:size=list(im.size)
        trajectory=Path(o['output_path']).parents[1]/'checkpoints'/f"train{o['update']}_trajectory_{o['episode_index']}.pt"
        case={'case_id':o['sample_id'],'split':'train','image_path':str(source),'expected_size':size,
          'expected_sha256':expected,'source_path':str(source),'edited_path':o['output_path'],'trajectory_path':str(trajectory),
          'question':o['question'],'original_answers':o['answers'],'instruction':o['instruction'],
          'generation_metadata':o['settings'],'stored_quality':o['quality'],'stored_semantic_proxy':o['semantic'],
          'stored_reward':o['reward'],'execution_cost':{'generation_seconds':o['generation_seconds'],'reward_seconds':o['reward_seconds']},
          'edited_factual_answer':'UNKNOWN'}
        cases.append(case)
    trace=TraceCollector(run.path/'trace.jsonl');executor=HarnessExecutor(root);resources=Resources();diagnoses=[]
    for case in cases[:8]:
        trace.append('retrieved_training_context',case)
        investigator=FailureInvestigator(executor,trace,[case],max_queries=4)
        for query in ['image_headers','content_hash','pair_quality','trajectory_summary']:investigator.observe(case,query)
        obs={q['query']:q['observation'] for q in investigator.queries}
        matched=all(abs(obs['pair_quality'][k]-case['stored_quality'][k])<1e-8 for k in ['mse','mae','ssim','distortion'])
        assert obs['content_hash']['actual_sha256']==case['expected_sha256'] and obs['trajectory_summary']['finite_states']
        diagnoses.append({'case_id':case['case_id'],'numerical_quality_reproduced':matched,
          'observed':['source identity matches historical evidence','finite saved trajectory'],
          'workflow_observation':'geometry tool already fully decodes; an additional decode check duplicates this work',
          'unresolved':['instruction following','edited factual answer','independent semantic validity','strict stochastic continuation'],
          'scope':'same metric recomputation is arithmetic reproducibility, not construct validation'})
        resources.add(investigator.resources)
    # Operator constructs exact structural truth from historical bytes/header, not human semantic labels.
    private=[]
    for case in cases[8:]:
        common={**case,'split':'workflow_gate','source_group':case['case_id'],'expected_valid':True,'population':'recorded_train_artifact'}
        private.append(common)
        wrong_size={**common,'case_id':case['case_id']+'_wrong_size','expected_size':[1,1],'expected_valid':False,'population':'injected_metadata_control'}
        wrong_hash={**common,'case_id':case['case_id']+'_wrong_digest','expected_sha256':'0'*64,'expected_valid':False,'population':'injected_metadata_control'}
        private.extend([wrong_size,wrong_hash])
    gate=RecordedArtifactGate(executor,private)
    parent=HarnessSpec('recorded_incumbent',('geometry','decode','content_hash'))
    manager=VersionManager(run.path/'harness_versions',parent);decisions=[]
    def proposal(candidate,change):return {'parent_version':manager.incumbent,'candidate':candidate.to_dict(),
      'change':change,'supporting_evidence':{'diagnoses':diagnoses,'implementation_fact':'geometry calls Image.load before returning'},
      'expected_effect':'remove duplicate decode work while retaining geometry/content checks',
      'affected_modules':['workflow.checks'],'regression_tests':['recorded positives','wrong-size metadata','wrong-digest metadata'],
      'category':'evidence_acquisition_and_validation_orchestration'}
    for candidate,change in [(HarnessSpec('ineffective_reordering',('decode','geometry','content_hash')),'reorder same checks'),
      (HarnessSpec('remove_redundant_decode',('geometry','content_hash')),'remove duplicate decode'),
      (HarnessSpec('regression_drops_identity',('geometry',)),'drop content identity check')]:
        decision=manager.evaluate(proposal(candidate,change),gate);decisions.append(decision)
        for key in ['parent','candidate']:resources.add(Resources(**decision['actual_evaluation_resources'][key]))
    assert not decisions[0]['accepted'] and decisions[1]['accepted'] and not decisions[2]['accepted']
    manager.rollback('H0000','explicit rollback regression check');assert manager.incumbent=='H0000'
    manager.rollback('H0002','restore independently accepted workflow');assert manager.incumbent=='H0002'
    return {'investigated_real_train_episodes':8,'new_diagnostic_queries':32,'gate_unique_train_sources':24,
      'investigation_gate_source_overlap':False,'recorded_positive_cases':24,'injected_metadata_controls':48,
      'diagnoses':diagnoses,'decisions':decisions,'ineffective_rejected':True,'regression_rejected':True,'rollback_verified':True,
      'accepted_workflow':'H0002','selected_result':decisions[1]['measured_validation'],
      'total_search_resources':resources.__dict__,'new_model_calls':0,'new_optimizer_updates':0,
      'natural_image_semantic_improvement':None,'scope':'real saved workflow and file-integrity efficiency only; semantic causes unresolved'}
