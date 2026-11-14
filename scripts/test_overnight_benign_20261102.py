"""New CPU capture/controller contract tests, not previous model smoke tests."""
import copy,sys,unittest,random
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from scripts.gradient_boundary_capture_20261102 import snapshot,CaptureClip,norms
from scripts.overnight_benign_controller_20261102 import ledger,can_admit
from scripts.run_ip2pix_checkpoint_diagnostic import same

def state():return dict(run_id='fixture',initialized_UTC='fixture',deadline_UTC='fixture',deadline_epoch=10000,GPU_jobs=[],numerical_budget_seconds=1200,admissions_closed=False)
def job(st,elapsed=0,cap=360):return dict(stage='numerical',scheduler_state=st,elapsed_seconds=elapsed,reservation_seconds=cap,allocated_GPUs=1,job_id='fixture')

class CaptureTests(unittest.TestCase):
 def test_snapshot_no_mutation_or_alias(self):
  p=torch.nn.Parameter(torch.tensor([3.,4.]));p.grad=torch.tensor([6.,8.]);o=torch.optim.AdamW([p],lr=.01);o.step();before=copy.deepcopy(dict(parameter=p.detach(),grad=p.grad,optimizer=o.state_dict(),rng=torch.get_rng_state(),rng_python=random.getstate(),rng_numpy=np.random.get_state()))
  got=snapshot([('p',p)]);got[0][1].zero_();after=dict(parameter=p.detach(),grad=p.grad,optimizer=o.state_dict(),rng=torch.get_rng_state(),rng_python=random.getstate(),rng_numpy=np.random.get_state());self.assertTrue(same(before,after))
 def test_wrapped_clip_exactly_matches_original(self):
  a=torch.nn.Parameter(torch.tensor([1.,2.]));b=torch.nn.Parameter(a.detach().clone());a.grad=torch.tensor([3.,4.]);b.grad=a.grad.clone();r=torch.nn.utils.clip_grad_norm_([a],1.)
  cap=CaptureClip([('p',b)],torch.nn.utils.clip_grad_norm_);z=cap([b],1.);self.assertTrue(torch.equal(r,z));self.assertTrue(torch.equal(a.grad,b.grad));self.assertTrue(torch.equal(cap.records[0]['raw'][0][1],torch.tensor([3.,4.])))
 def test_FP64_norm_independent(self):self.assertEqual(norms([('x',torch.tensor([3.,4.])),('y',torch.tensor([12.]))])['total_L2_FP64'],13.)
 def test_order_is_enforced(self):
  a=torch.nn.Parameter(torch.ones(1));b=torch.nn.Parameter(torch.ones(1));a.grad=torch.ones(1);b.grad=torch.ones(1)
  with self.assertRaises(AssertionError):CaptureClip([('a',a),('b',b)],torch.nn.utils.clip_grad_norm_)([b,a],1.)
 def test_missing_grad_is_observed(self):self.assertTrue(norms(snapshot([('p',torch.nn.Parameter(torch.ones(1)))]))['per_tensor'][0]['missing'])
class ControllerTests(unittest.TestCase):
 def test_failed_cancelled_success_all_charged(self):
  s=state();s['GPU_jobs']=[job('FAILED',11),job('CANCELLED',7),job('COMPLETED',20)];b=ledger(s);self.assertEqual(b['consumed'],38);self.assertEqual(b['reserved'],0)
 def test_pending_reservation_precedes_cost(self):
  s=state();s['GPU_jobs']=[job('PENDING')];self.assertEqual(ledger(s)['reserved'],360);self.assertFalse(can_admit(s,360,'numerical',0)[0])
 def test_live_elapsed_not_double_counted(self):
  s=state();s['GPU_jobs']=[job('RUNNING',100)];b=ledger(s);self.assertEqual(b['consumed']+b['reserved'],360)
 def test_unknown_submission_is_fail_closed(self):
  s=state();s['GPU_jobs']=[job('UNKNOWN')];self.assertFalse(can_admit(s,360,'numerical',0)[0])
 def test_budget_blocks_even_successful_large_allocation(self):
  s=state();s['GPU_jobs']=[job('COMPLETED',35900)];self.assertFalse(can_admit(s,360,'numerical',0)[0])
 def test_deadline_requires_full_projected_job(self):self.assertFalse(can_admit(state(),360,'numerical',9600)[0])
 def test_numerical_subbudget_is_separate(self):
  s=state();s['GPU_jobs']=[job('FAILED',900)];self.assertFalse(can_admit(s,360,'numerical',0)[0]);self.assertEqual(ledger(s)['remaining'],35100)
 def test_repeated_decision_cannot_admit_with_pending(self):
  s=state();self.assertTrue(can_admit(s,360,'numerical',0)[0]);s['GPU_jobs'].append(job('PENDING'));self.assertFalse(can_admit(s,360,'numerical',0)[0])
 def test_closed_admission_never_reopens_on_callback(self):
  s=state();s['admissions_closed']=True;self.assertFalse(can_admit(s,360,'numerical',0)[0])
 def test_one_node_retry(self):
  import re
  self.assertIsNotNone(re.fullmatch(r'epyc\d+','epyc15'));self.assertIsNone(re.fullmatch(r'epyc\d+','epyc15,other'))
class ActualCallbackDryRun(unittest.TestCase):
 def setUp(self):
  import tempfile
  import scripts.overnight_benign_controller_20261102 as c
  from unittest.mock import patch
  self.c=c;(ROOT/'tmp').mkdir(exist_ok=True);self.path=Path(tempfile.mkdtemp(prefix='overnight_callback_cpu_',dir=ROOT/'tmp')).resolve();self.patch=patch.multiple(c,BASE=self.path,OUT=self.path);self.patch.start()
 def tearDown(self):
  import shutil
  self.patch.stop();assert self.path.is_relative_to((ROOT/'tmp').resolve()) and self.path.name.startswith('overnight_callback_cpu_');shutil.rmtree(self.path)
 def test_persisted_duplicate_callback_is_noop_twice(self):
  from unittest.mock import patch
  s=state();s['GPU_jobs']=[{**job('COMPLETED',12), 'processed':True,'job_id':'mock123'}];self.c.write(self.path/'state.json',s)
  with patch.object(self.c,'registered_submit') as sub:
   self.c.controller('event','mock123');self.c.controller('event','mock123');sub.assert_not_called()
  restored=self.c.read(self.path/'state.json');self.assertEqual(restored['duplicate_callback_noops'],2);self.assertEqual(len(restored['GPU_jobs']),1)
 def dry_run(self,passing):
  from unittest.mock import patch
  s=state();s.update(completed_jobs=[],passed_stages=[],failure_history=[],GPU_jobs=[{**job('COMPLETED',12),'job_id':'mock123','processed':False,'exit_code':'0:0','run_directory':'results/CPU_FIXTURE','stage':'numerical'}])
  with patch.object(self.c,'collect'),patch.object(self.c,'render'),patch.object(self.c,'submit_gpu') as submit,patch('scripts.overnight_benign_reporting_20261102.analyze_numerical',return_value={'exact_pass':passing,'tolerance_pass':passing,'first_boundary':None if passing else {'boundary':'raw'}}):
   self.c.advance(s)
   if passing:submit.assert_called_once_with(s,'ordinary_resume')
   else:submit.assert_not_called();self.assertTrue(s['admissions_closed']);self.assertEqual(s['status'],'STOPPED_NUMERICAL_OR_EXECUTION_FAILURE')
 def test_completed_numerical_failure_blocks_learning(self):self.dry_run(False)
 def test_completed_numerical_pass_admits_only_resume_check(self):self.dry_run(True)
 def test_closed_deadline_preserves_numerical_stop(self):
  from unittest.mock import patch
  s=state();s.update(admissions_closed=True,status='STOPPED_NUMERICAL_OR_EXECUTION_FAILURE',GPU_jobs=[{**job('COMPLETED',12),'processed':True}],CPU_jobs=[]);self.c.write(self.path/'state.json',s)
  with patch.object(self.c,'advance') as advance:self.c.controller('deadline');advance.assert_not_called()
  self.assertEqual(self.c.read(self.path/'state.json')['status'],s['status'])
 def test_closed_admission_cannot_launch_resume_even_after_PASS(self):
  from unittest.mock import patch
  s=state();s.update(admissions_closed=True,completed_jobs=[],passed_stages=[],failure_history=[],GPU_jobs=[{**job('COMPLETED',12),'job_id':'mock123','processed':False,'exit_code':'0:0','run_directory':'results/CPU_FIXTURE'}])
  with patch.object(self.c,'collect'),patch.object(self.c,'render'),patch.object(self.c,'submit_gpu') as submit,patch('scripts.overnight_benign_reporting_20261102.analyze_numerical',return_value={'exact_pass':True,'tolerance_pass':True}):self.c.advance(s);submit.assert_not_called()
 def test_accounting_recovery_is_one_shot(self):
  import os
  from unittest.mock import patch
  s=state();s['CPU_jobs']=[];calls=[]
  def submit(s,kind,token,args,record):record['token']=token;s['CPU_jobs'].append(record);calls.append(args);return '99998'
  with patch.dict(os.environ,{'SLURM_JOB_ID':'99999'}),patch.object(self.c,'registered_submit',side_effect=submit),patch.object(self.c,'command'):
   self.c.register_accounting_recovery(s,'12345');self.c.register_accounting_recovery(s,'12345')
  self.assertEqual(len(calls),1);self.assertIn('--dependency=afterany:99999',calls[0]);self.assertIn('--begin=now+1minute',calls[0])
 def test_NODE_FAIL_retry_excludes_only_identified_node(self):
  import time
  from unittest.mock import patch
  s=state();s.update(deadline_epoch=time.time()+10000,completed_jobs=[],passed_stages=[],failure_history=[],GPU_jobs=[{**job('NODE_FAIL',12),'job_id':'mock123','processed':False,'exit_code':'0:9','run_directory':'results/CPU_FIXTURE','attempt':0,'node':'epyc15'}])
  with patch.object(self.c,'collect'),patch.object(self.c,'render'),patch.object(self.c,'submit_gpu') as submit:self.c.advance(s);submit.assert_called_once_with(s,'numerical',1,'epyc15')
 def test_second_NODE_FAIL_cannot_retry(self):
  import time
  from unittest.mock import patch
  s=state();s.update(deadline_epoch=time.time()+10000,completed_jobs=[],passed_stages=[],failure_history=[],GPU_jobs=[{**job('NODE_FAIL',12),'job_id':'mock123','processed':False,'exit_code':'0:9','run_directory':'results/CPU_FIXTURE','attempt':1,'node':'epyc15'}])
  with patch.object(self.c,'collect'),patch.object(self.c,'render'),patch.object(self.c,'submit_gpu') as submit:self.c.advance(s);submit.assert_not_called()
 def resume_fixture(self,empty=False):
  for name in ['ordinary0','ordinary1','resume']:
   d=self.path/name;d.mkdir()
   for phase in ['before','after1','after2']:(d/phase).mkdir();torch.save(torch.ones(1),d/phase/'training_state.pt')
   for file in ['next_trajectory.pt','postclip.pt']:torch.save(torch.ones(1),d/file)
   updates=[dict(gradient_norm=1.,loss=.1)]*(1 if name=='resume' else 2)
   if name=='resume':updates=[dict(gradient_norm=2.,loss=.1)]
   self.c.write(d/'receipt.json',dict(updates=[] if empty else updates,ownership_exact=True,frozen_base_unchanged=True,unmodified_canonical_update=True,original_guard_unchanged=True))
 def test_resume_analysis_requires_recorded_preclip_norm(self):
  from scripts.overnight_benign_reporting_20261102 import analyze_resume
  self.resume_fixture()
  result=analyze_resume(self.path);self.assertFalse(result['exact_pass']);self.assertFalse(result['tolerance_pass'])
 def test_empty_update_receipts_cannot_PASS(self):
  from scripts.overnight_benign_reporting_20261102 import analyze_resume
  self.resume_fixture(empty=True)
  with self.assertRaisesRegex(ValueError,'Missing executed update receipts'):analyze_resume(self.path)
 def test_invalid_artifact_stops_and_records_failed_validation(self):
  from unittest.mock import patch
  s=state();s.update(completed_jobs=[],passed_stages=[],failure_history=[],GPU_jobs=[{**job('COMPLETED',12),'job_id':'mock123','processed':False,'exit_code':'0:0','run_directory':str(self.path)}])
  with patch.object(self.c,'collect'),patch.object(self.c,'render'),patch.object(self.c,'submit_gpu') as submit,patch('scripts.overnight_benign_reporting_20261102.analyze_numerical',side_effect=ValueError('missing fixture artifact')):
   self.c.advance(s);submit.assert_not_called()
  self.assertTrue(s['admissions_closed']);self.assertEqual(s['GPU_jobs'][0]['classification'],'ARTIFACT_VALIDATION_FAILURE');self.assertTrue((self.path/'artifact_validation_failure.json').exists())
if __name__=='__main__':unittest.main()
