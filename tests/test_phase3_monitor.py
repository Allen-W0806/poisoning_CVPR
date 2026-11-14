import unittest
from src.monitoring.phase3 import phase3_checks
from src.utils.scratch import scratch_directory
from pathlib import Path
from unittest.mock import patch
import json
from src.monitoring.recovery import Recovery
from src.monitoring.core import load_config
from src.monitoring.core import run_monitor
from src.monitoring.diagnostics import inspect_run
from tests.test_monitoring import fixture

class Phase3MonitorTests(unittest.TestCase):
    def test_terminal_scientific_failure_does_not_keep_idle_retry_queue(self):
        with scratch_directory(Path(__file__).resolve().parents[1]) as root:
            path=fixture(root,name='EXP_fixture_phase3_generator_rl_smoke',status='FAIL',job='11')
            (path/'slurm-11.err').write_text("AssertionError: {'adapter_max_difference': 2.8e-6}")
            base=root/'reports/monitoring';base.mkdir()
            (base/'state.json').write_text(json.dumps({'initialized':True,'last_check':'2026-10-27T00:00:00+00:00',
                'pending_failure_jobs':{'11':{'reason':'legacy nontransient'}},'observations':{path.name:{'status':'FAIL'}}}))
            snapshot={'queue':{'exit_code':0,'stdout':''},'accounting':{'exit_code':0,'stdout':''},
                'jobs':{'11':{'State':'FAILED','ExitCode':'1:0','ElapsedRaw':'10'}}}
            result=run_monitor(root,snapshot,load_config(Path(__file__).resolve().parents[1]),scientific_checks=False)
            state=json.loads((base/'state.json').read_text())
            self.assertFalse(state['pending_failure_jobs']);self.assertTrue(result['idle_no_research_submission'])
            self.assertEqual(result['automatic_fixes'],0);self.assertEqual(result['blocked_actions'],1)
    def test_pre_metric_probability_condition_and_gradient_failures_are_blocked(self):
        for message,category in [('Nonfinite transition likelihood','diffusion_probability'),
                ('Conditioning association mismatch','conditioning_mismatch'),
                ('Zero/nonfinite generator gradient','invalid_generator_gradient')]:
            with scratch_directory(Path(__file__).resolve().parents[1]) as root:
                path=fixture(root,name='EXP_fixture_phase3_generator_rl_one',status='FAIL',metrics={})
                (path/'slurm-11.err').write_text(message)
                run=inspect_run(path);finding=next(f for f in run['findings'] if f['code']==category)
                repair=Recovery(root,load_config(Path(__file__).resolve().parents[1]),{}).apply(finding,run,True)
                self.assertEqual(repair['action'],'BLOCKED')
    def test_known_phase3_guard_refusal_is_not_unknown_submission(self):
        for error in ['Previous Phase3 allocation active/unconfirmed','Another user GPU allocation is active/pending','Connection dropped after sbatch']:
            with scratch_directory(Path(__file__).resolve().parents[1]) as root:
                path=fixture(root,name='EXP_fixture_phase3_generator_rl_smoke',status='FAIL',job='11')
                (root/'reports/phase3_jobs.jsonl').write_text(json.dumps({'job_id':'11','task':'smoke','reserved_seconds':600})+'\n')
                state={};recovery=Recovery(root,load_config(Path(__file__).resolve().parents[1]),state,
                    submit=lambda *args,**kwargs:{'exit_code':1,'stdout':'','stderr':error})
                def git(args,**kwargs):return {'exit_code':0,'stdout':'fixture' if args[1]=='rev-parse' else '', 'stderr':''}
                with patch('src.monitoring.recovery.command',side_effect=git):recovery.retry(inspect_run(path),{'code':'preempted'},True)
                self.assertEqual(bool(state.get('submission_uncertain')),error=='Connection dropped after sbatch')
    def test_transient_phase3_uses_own_budget_guard_and_two_retry_limit(self):
        with scratch_directory(Path(__file__).resolve().parents[1]) as root:
            path=fixture(root,name='EXP_fixture_phase3_generator_rl_smoke',status='FAIL',job='11')
            (root/'reports/phase3_jobs.jsonl').write_text(json.dumps({'job_id':'11','task':'smoke','reserved_seconds':600})+'\n')
            run=inspect_run(path);calls=[]
            def submit(args,**kwargs):
                calls.append(args);return {'exit_code':0,'stdout':json.dumps({'job_id':'12','parent_run':'EXP_retry'}),'stderr':''}
            recovery=Recovery(root,load_config(Path(__file__).resolve().parents[1]),{},submit=submit)
            def git(args,**kwargs):return {'exit_code':0,'stdout':'fixture' if args[1]=='rev-parse' else '', 'stderr':''}
            with patch('src.monitoring.recovery.command',side_effect=git):
                for _ in range(3):recovery.retry(run,{'code':'preempted'},True)
            self.assertEqual(len(calls),2)
            self.assertTrue(all('submit_phase3.py' in args[1] for args in calls))

    def test_bad_grad_likelihood_and_conditioning(self):
        with scratch_directory(Path(__file__).resolve().parents[1]) as root:
            r={"id":"EXP_x_phase3_generator_rl_one","status":"RUNNING","metrics":{"updates":[{"step":1,"gradient_norm":0,"finite_gradients":False,"probabilities_finite":False}]}}
            result=phase3_checks(root,[r],{}, {},protect=False)
            self.assertTrue(any(x["code"]=="phase3_optimizer_invalid" for x in result["findings"]))
            self.assertFalse(result["generator_rl_implementation_ready"])

    def test_no_false_pass_from_status_only(self):
        with scratch_directory(Path(__file__).resolve().parents[1]) as root:
            r={"id":"EXP_x_phase3_generator_rl_smoke","status":"PASS","metrics":{}}
            result=phase3_checks(root,[r],{}, {},protect=False)
            self.assertFalse(result["generator_rl_implementation_ready"])
            self.assertTrue(any(x["code"]=="phase3_incomplete_restart_witness" for x in result["findings"]))

    def test_phase3_protected_baseline_preserved(self):
        with scratch_directory(Path(__file__).resolve().parents[1]) as root:
            (root/"configs").mkdir();path=root/"configs/phase3_generator_rl.yaml";path.write_text("original")
            state={"protected_hashes":{"old":"unchanged"}}
            phase3_checks(root,[],state,{})
            expected=dict(state["phase3_protected_hashes_v1"])
            path.write_text("mutation")
            result=phase3_checks(root,[],state,{})
            self.assertEqual(state["phase3_protected_hashes_v1"],expected)
            self.assertEqual(state["protected_hashes"],{"old":"unchanged"})
            self.assertTrue(result["findings"])
