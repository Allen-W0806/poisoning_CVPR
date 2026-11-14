import unittest
from src.operations.night_budget import accounting,can_submit,safe_retry,finite_update


class NightBudgetTests(unittest.TestCase):
    def record(self,**kw):return {'stage':'seed42','attempt':0,'job_id':'10','reserved_seconds':5400,**kw}
    def test_terminal_failure_cost_is_retained(self):
        result=accounting([self.record()],{'10':{'state':'FAILED','elapsed_seconds':71}})
        self.assertEqual(result['actual_GPU_seconds'],71);self.assertEqual(result['committed_GPU_seconds'],71)
    def test_active_job_keeps_full_reservation_and_blocks_concurrent(self):
        states={'10':{'state':'RUNNING','elapsed_seconds':71}}
        self.assertEqual(accounting([self.record()],states)['committed_GPU_seconds'],5400)
        with self.assertRaises(RuntimeError):can_submit([self.record()],states,'seed123',0,5400)
    def test_pending_job_also_blocks(self):
        with self.assertRaises(RuntimeError):can_submit([self.record()],{'10':{'state':'PENDING','elapsed_seconds':0}},'seed123',0,5400)
    def test_duplicate_attempt_and_overcommit_rejected(self):
        states={'10':{'state':'COMPLETED','elapsed_seconds':5000}}
        with self.assertRaises(RuntimeError):can_submit([self.record()],states,'seed42',0,5400)
        with self.assertRaises(RuntimeError):can_submit([self.record()],states,'seed123',0,32000)
    def test_unknown_accounting_and_submission_fail_closed(self):
        for records,states in [([self.record()],{}),([self.record(job_id=None)],{})]:
            with self.assertRaises(RuntimeError):accounting(records,states)
    def test_preemption_exact_resume_one_time(self):
        self.assertTrue(safe_retry(self.record(),'PREEMPTED','',True,64)[0])
        self.assertFalse(safe_retry(self.record(attempt=1),'PREEMPTED','',True,64)[0])
        self.assertFalse(safe_retry(self.record(),'PREEMPTED','',False,64)[0])
    def test_never_retry_oom_nan_assertions_save_failure(self):
        for state,log in [('OUT_OF_MEMORY',''),('FAILED','NaN loss'),('NODE_FAIL','nonfinite'),('FAILED','AssertionError'),('FAILED','PytorchStreamWriter'),('FAILED','No space left on device')]:
            self.assertFalse(safe_retry(self.record(),state,log,True,64)[0])
    def test_only_predeclared_transients(self):
        self.assertTrue(safe_retry(self.record(),'FAILED','Stale file handle',False,0)[0])
        self.assertFalse(safe_retry(self.record(),'FAILED','unexpected coding error',True,64)[0])
    def test_finite_real_update(self):
        self.assertTrue(finite_update({'loss':1.,'raw_gradient_norm':.1,'max_parameter_change':1e-5}))
        self.assertFalse(finite_update({'loss':float('nan'),'raw_gradient_norm':.1,'max_parameter_change':1e-5}))
    def test_retry_reservation_still_obeys_cap(self):
        records=[self.record()];states={'10':{'state':'PREEMPTED','elapsed_seconds':35900}}
        with self.assertRaises(RuntimeError):can_submit(records,states,'seed42',1,5400)
