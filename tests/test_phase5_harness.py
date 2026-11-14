import copy
import json
import unittest
from src.utils.scratch import scratch_directory
from src.rsi.harness.harness_spec import HarnessSpec
from src.rsi.harness.harness_executor import HarnessExecutor
from src.rsi.harness.trace_collector import TraceCollector,Resources
from src.rsi.harness.failure_investigator import FailureInvestigator,FAILURES
from src.rsi.harness.revision_proposer import propose
from src.rsi.harness.revision_validator import validate_revision
from src.rsi.harness.cost_aware_gate import IndependentGate,GatePolicy
from src.rsi.harness.version_manager import VersionManager
from src.rsi.harness.engineering_fixture import create_cases

class Phase5HarnessTests(unittest.TestCase):
    def test_real_investigation_reject_accept_regress_and_rollback(self):
        with scratch_directory() as root:
            train,train_oracle=create_cases(root,'train',4);cases,oracle=create_cases(root,'gate',32)
            executor=HarnessExecutor(root);initial=HarnessSpec('initial_geometry')
            failed=executor.run(initial,train[0]);self.assertNotEqual(failed['predicted_valid'],train_oracle[train[0]['case_id']])
            trace=TraceCollector(root/'training_trace.jsonl');trace.append('failed_execution',failed)
            investigator=FailureInvestigator(executor,trace,train)
            diagnosis=investigator.investigate_integrity_failure(train[0])
            self.assertEqual(diagnosis['supported'],['dataset_corruption_or_duplication'])
            self.assertIn('geometry_checker_reported_wrong_size',diagnosis['contradicted'])
            self.assertIn('image_quality_measurement_error',diagnosis['unresolved'])
            self.assertEqual(len(diagnosis['supported']+diagnosis['unresolved']),len(FAILURES))
            self.assertEqual(investigator.resources.diagnostic_operations,2)
            gate=IndependentGate(executor,cases,oracle);manager=VersionManager(root/'versions',initial)
            proposal=propose(initial,diagnosis,manager.incumbent)
            ineffective=copy.deepcopy(proposal);ineffective['candidate']=HarnessSpec('decode_only',('decode',)).to_dict()
            rejected=manager.evaluate(ineffective,gate)
            self.assertFalse(rejected['accepted']);self.assertEqual(manager.incumbent,'H0000')
            accepted=manager.evaluate(proposal,gate)
            self.assertTrue(accepted['accepted']);self.assertEqual(manager.incumbent,'H0002')
            result=accepted['measured_validation'];self.assertEqual(result['n_sources'],32)
            self.assertEqual(result['parent_accuracy'],.5);self.assertEqual(result['candidate_accuracy'],1.)
            self.assertNotIn('case_id',json.dumps(result));self.assertNotIn('paired_rows',result)
            self.assertGreater(result['candidate_resources']['bytes_read'],0)
            regression=copy.deepcopy(proposal);regression['parent_version']=manager.incumbent
            regression['candidate']=initial.to_dict()
            self.assertFalse(manager.evaluate(regression,gate)['accepted'])
            self.assertEqual(manager.incumbent,'H0002')
            manager.rollback('H0000','operator regression demonstration')
            self.assertEqual(manager.program().checks,('geometry',))
            self.assertEqual(len(list((root/'versions/history').glob('revision_*.json'))),3)

    def test_editor_and_investigator_cannot_query_gate(self):
        with scratch_directory() as root:
            cases,_=create_cases(root,'gate',32)
            train,_=create_cases(root,'train',2)
            trace=TraceCollector(root/'trace.jsonl');investigator=FailureInvestigator(HarnessExecutor(root),trace,train)
            with self.assertRaises(PermissionError):investigator.observe(cases[0],'content_hash')
            forged={**cases[0],'split':'train'}
            with self.assertRaises(PermissionError):investigator.observe(forged,'content_hash')
            forged={**train[0],'image_path':cases[0]['image_path']}
            with self.assertRaises(PermissionError):investigator.observe(forged,'content_hash')
            with self.assertRaises(ValueError):trace.append('leak',{'answers':['x']},split='test')
            with self.assertRaises(PermissionError):HarnessSpec('curriculum').training_order(cases)

    def test_insufficient_diagnosis_and_natural_validity_never_accepted(self):
        with scratch_directory() as root:
            cases,_=create_cases(root,'train',2)
            trace=TraceCollector(root/'trace.jsonl');i=FailureInvestigator(HarnessExecutor(root),trace,cases)
            diagnosis=i.diagnose(cases[0],[])
            self.assertEqual(diagnosis['diagnosis_status'],'UNRESOLVED')
            self.assertIsNone(propose(HarnessSpec('initial'),diagnosis,'H0000'))
            with self.assertRaises(ValueError):IndependentGate(HarnessExecutor(root),[],{},kind='natural_image_semantics')

    def test_restricted_program_cannot_modify_metric_or_labels(self):
        spec=HarnessSpec('initial');v=spec.to_dict();v['ground_truth']=['malicious']
        with self.assertRaises(ValueError):HarnessSpec.from_dict(v)
        for checks in [('shell',),('geometry','geometry')]:
            with self.assertRaises(ValueError):HarnessSpec('bad',checks)
        with self.assertRaises(ValueError):GatePolicy(min_sources=10)
        with self.assertRaises(ValueError):GatePolicy(minimum_accuracy_gain=-.1)

    def test_resource_memory_and_curriculum_revisions_execute(self):
        with scratch_directory() as root:
            cases,_=create_cases(root,'train',4);executor=HarnessExecutor(root)
            with self.assertRaises(RuntimeError):executor.run(HarnessSpec('budget',('geometry','decode'),max_diagnostics=1),cases[0])
            result=executor.run(HarnessSpec('summary',memory_mode='summary'),cases[0])
            self.assertNotIn('evidence',result['observations'][0])
            order=HarnessSpec('curriculum',curriculum='uncertain_first').training_order(cases)
            self.assertEqual([x['uncertainty'] for x in order],[1.,1.,0.,0.])
            with self.assertRaises(ValueError):Resources(model_calls=-1).validate()

    def test_gate_rejects_duplicate_sources_and_unpaired_coverage(self):
        with scratch_directory() as root:
            cases,oracle=create_cases(root,'gate',32);cases[1]['expected_sha256']=cases[0]['expected_sha256']
            with self.assertRaises(ValueError):IndependentGate(HarnessExecutor(root),cases,oracle)
            with self.assertRaises(ValueError):IndependentGate(HarnessExecutor(root),cases,{'unmatched':True})

    def test_gate_rejects_a_new_error_even_with_large_aggregate_gain(self):
        with scratch_directory() as root:
            cases,oracle=create_cases(root,'gate',32)
            for case in cases:case['independent_review']={'status':'VERIFIED','valid':True,'source':'constructed_oracle_fixture'}
            cases[1]['independent_review']['valid']=False
            gate=IndependentGate(HarnessExecutor(root),cases,oracle)
            result=gate.compare(HarnessSpec('geometry'),HarnessSpec('harm_one_valid_source',('content_hash','independent_review')))
            self.assertGreater(result['accuracy_delta'],.05)
            self.assertEqual(result['new_errors_on_previously_correct_sources'],1)
            self.assertFalse(result['accepted'])

if __name__=='__main__':unittest.main()
