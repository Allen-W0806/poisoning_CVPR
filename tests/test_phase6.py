import json
import unittest
from pathlib import Path
from src.utils.scratch import scratch_directory
from src.progression.controller import run_controller,policy
from src.progression.real_harness import RecordedArtifactGate
from src.rsi.harness.harness_spec import HarnessSpec
from src.rsi.harness.harness_executor import HarnessExecutor
from src.data.dataset import sha256
from src.progression.completion import bounded_repair,validate_completion,process_completions
from src.progression.monitor import invalidate_dependencies
from src.monitoring.integrity_v6 import image_observations

class Phase6Tests(unittest.TestCase):
    def test_monitor_does_not_interpret_audit_counts_as_image_records(self):
        self.assertEqual(image_observations({'observations':500}),[])
        self.assertEqual(image_observations({'observations':[{'count':500}]}),[])
        row={'source_path':'a.png','question':'q','answers':['a']}
        self.assertEqual(image_observations({'observations':[row]}),[row])
    def root(self,path):
        (path/'configs').mkdir();(path/'configs/phase6.json').write_bytes((Path(__file__).parents[1]/'configs/phase6.json').read_bytes())
        return path

    def test_blocked_human_gate_does_not_block_independent_cpu_progress(self):
        with scratch_directory() as p:
            root=self.root(p);visited=[]
            def good(r,s):visited.append(s);return {'state':'PASSED','evidence':'fixture engineering only'}
            callbacks={k:good for k in ['validate_data','review_package','reward_audit','real_harness','rl_checks','comparison']}
            callbacks['human_gate']=lambda r,s:{'state':'NEEDS_HUMAN_REVIEW','reason':'no labels'}
            result=run_controller(root,callbacks)
            self.assertEqual(result['stages']['S1_SEMANTIC_REVIEW']['state'],'NEEDS_HUMAN_REVIEW')
            self.assertIn('S2_REAL_HARNESS',visited);self.assertIn('S4_STATIC_COMPARISON',visited)
            self.assertEqual(result['stages']['S5_PILOT']['state'],'BLOCKED')
            count=len(visited);run_controller(root,callbacks);self.assertEqual(len(visited),count)

    def test_zero_gpu_policy_cannot_be_overridden_by_config(self):
        with scratch_directory() as p:
            root=self.root(p);config=json.loads((root/'configs/phase6.json').read_text());config['scientific_gpu_budget_seconds']=1
            (root/'configs/phase6.json').write_text(json.dumps(config))
            with self.assertRaises(ValueError):policy(root)

    def test_bounded_repair_never_rewrites_finalized_run_and_stops_at_two(self):
        with scratch_directory() as root:
            p=root/'experiments/EXP_fixture';p.mkdir(parents=True);(p/'status.json').write_text('{"status":"RUNNING"}')
            attempts={}
            for i in range(2):self.assertEqual(bounded_repair(root,p,'missing_images',attempts)['action'],'bounded_engineering_repair')
            self.assertEqual(bounded_repair(root,p,'missing_images',attempts)['action'],'BLOCKED')
            (p/'status.json').write_text('{"status":"PASS"}');(p/'metrics.json').write_text('{"negative_result":-1}')
            before=(p/'metrics.json').read_bytes()
            self.assertEqual(bounded_repair(root,p,'missing_run_log',{})['action'],'BLOCKED')
            self.assertEqual((p/'metrics.json').read_bytes(),before)

    def test_empty_or_missing_outputs_cannot_be_completed_success(self):
        with scratch_directory() as p:
            (p/'manifest.json').write_text(json.dumps({'experiment_id':p.name,'slurm_job_id':'123'}))
            (p/'status.json').write_text('{"status":"PASS"}');(p/'metrics.json').write_text('{"x":NaN}')
            result=validate_completion(p,{'123':{'State':'COMPLETED','ExitCode':'0:0'}})
            self.assertEqual(result['decision'],'BLOCKED');self.assertFalse(result['scientific_success'])

    def test_terminal_node_failure_reconciles_unfinalized_cpu_run(self):
        with scratch_directory() as root:
            p=root/'experiments/EXP_fixture_phase6_s1_data';p.mkdir(parents=True)
            (p/'manifest.json').write_text(json.dumps({'experiment_id':p.name,'slurm_job_id':'123'}))
            (p/'status.json').write_text('{"status":"RUNNING"}')
            records=process_completions(root,{'123':{'State':'NODE_FAIL','ExitCode':'1:0'}})
            self.assertEqual(len(records),1);self.assertEqual(records[0]['scheduler_state'],'NODE_FAIL')
            self.assertEqual(json.loads((p/'status.json').read_text())['status'],'FAIL')

    def test_integrity_failure_invalidates_all_dependencies_same_cycle(self):
        state={'stages':{'data':{'state':'FAILED','required_inputs':[]},
          'child':{'state':'PASSED','required_inputs':['data']},'grandchild':{'state':'PASSED','required_inputs':['child']}}}
        invalidate_dependencies(state)
        self.assertEqual(state['stages']['child']['state'],'BLOCKED')
        self.assertEqual(state['stages']['grandchild']['state'],'BLOCKED')

    def test_failure_keeps_current_descendant_instead_of_previous_success(self):
        with scratch_directory() as p:
            root=self.root(p)
            def fail(r,s):
                e=OSError(5,'fixture transient');e.phase6_experiment_id='EXP_current_failed';raise e
            result=run_controller(root,{'validate_data':fail})
            self.assertEqual(result['stages']['S1_DATA']['experiment_id'],'EXP_current_failed')

    def test_invalid_task_success_does_not_clear_scientific_gate(self):
        with scratch_directory() as p:
            root=self.root(p)
            result=run_controller(root,{'validate_data':lambda r,s:{'state':'RUNNING'}})
            self.assertEqual(result['stages']['S1_DATA']['state'],'FAILED')
            self.assertEqual(result['scientific_decision'],'NO-GO')

    def test_transient_stage_retry_is_bounded_and_unrelated_tasks_continue(self):
        with scratch_directory() as p:
            root=self.root(p);calls=[]
            def transient(r,s):calls.append(s);raise OSError(5,'verified fixture transient IO')
            callbacks={'validate_data':transient,'reward_audit':lambda r,s:{'state':'PASSED'}}
            for _ in range(5):result=run_controller(root,callbacks)
            self.assertEqual(len(calls),3) # original plus at most two retries
            self.assertEqual(result['stages']['S1_DATA']['state'],'FAILED')
            self.assertEqual(result['stages']['S1_REWARD_AUDIT']['state'],'PASSED')

    def test_real_file_gate_requires_measurable_benefit_and_rejects_identity_loss(self):
        from PIL import Image
        with scratch_directory() as p:
            cases=[]
            for i in range(24):
                image=p/f'{i}.png';Image.new('RGB',(8,8),(i*7,17,31)).save(image)
                c={'case_id':str(i),'source_group':str(i),'image_path':str(image),'expected_size':[8,8],
                   'expected_sha256':sha256(image),'population':'recorded_train_artifact','expected_valid':True}
                cases.append(c);cases.append({**c,'case_id':str(i)+'bad','expected_sha256':'0'*64,
                    'population':'injected_metadata_control','expected_valid':False})
            gate=RecordedArtifactGate(HarnessExecutor(p),cases)
            parent=HarnessSpec('parent',('geometry','decode','content_hash'))
            self.assertFalse(gate.compare(parent,HarnessSpec('reorder',('decode','geometry','content_hash')))['accepted'])
            good=HarnessSpec('good',('geometry','content_hash'))
            self.assertTrue(gate.compare(parent,good)['accepted'])
            self.assertFalse(gate.compare(good,HarnessSpec('regress',('geometry',)))['accepted'])
