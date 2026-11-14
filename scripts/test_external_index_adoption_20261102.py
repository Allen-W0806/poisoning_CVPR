"""Three isolated CPU tests of legacy publisher adoption, no scheduler calls."""
import json,shutil,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import scripts.overnight_supervisor_20261101 as legacy

class Adoption(unittest.TestCase):
 def setUp(self):
  (ROOT/'tmp').mkdir(exist_ok=True);self.root=Path(tempfile.mkdtemp(prefix='verified_index_adoption_',dir=ROOT/'tmp')).resolve();self.base=self.root/'reports/legacy';self.base.mkdir(parents=True);self.override=patch.multiple(legacy,ROOT=self.root,BASE=self.base);self.override.start()
  self.state=dict(tasks=[dict(task_key='SYNTHETIC_TEST',job_id='CPU_FIXTURE',scheduler_state='COMPLETED',classification='SYNTHETIC_TEST_ONLY',run_id='SYNTHETIC_TEST')],blocked_dependencies={},previously_attempted_repairs=[],failure_fingerprints={},last_successful_Google_Drive_synchronization=None)
  self.receipt=dict(checked_UTC='SYNTHETIC_TEST',supervisor_job_id='CPU_FIXTURE',completed_since_previous=[],budget=dict(cap=3600,consumed=0,reserved=0,remaining=3600),stage='SYNTHETIC_TEST',next_action=dict(task='INSPECT_ONLY',reason='synthetic fixture'),cloud=dict(status='SYNTHETIC_TEST'),scheduler={},guard_intact=True)
 def tearDown(self):
  self.override.stop();assert self.root.is_relative_to((ROOT/'tmp').resolve()) and self.root.name.startswith('verified_index_adoption_');shutil.rmtree(self.root)
 def install_index(self):
  p=self.root/'reports/LATEST_REPORT_INDEX.md';p.write_text('# RSI-POISONING\nSYNTHETIC_TEST externally verified index\n');proof=self.root/'results/overnight_benign_20261102_v2/cloud_index_verification.json';proof.parent.mkdir(parents=True);proof.write_text(json.dumps(dict(PROJECT_ID='RSI-POISONING',status='CLOUD_VERIFIED',index_sha256=legacy.sha(p))));return p
 def test_adoption_creates_new_version_and_preserves_old(self):
  old=legacy.reports(self.state,self.receipt);before=(old/'LATEST_REPORT_INDEX.md').read_bytes();p=self.install_index();new=legacy.reports(self.state,self.receipt);self.assertNotEqual(old,new);self.assertEqual((old/'LATEST_REPORT_INDEX.md').read_bytes(),before);self.assertEqual((new/'LATEST_REPORT_INDEX.md').read_bytes(),p.read_bytes())
 def test_repeat_keeps_same_verified_index(self):
  p=self.install_index();one=legacy.reports(self.state,self.receipt);two=legacy.reports(self.state,self.receipt);self.assertEqual(one,two);self.assertEqual((two/'LATEST_REPORT_INDEX.md').read_bytes(),p.read_bytes())
 def test_checksum_mismatch_fails_closed(self):
  p=self.install_index();p.write_text('RSI-POISONING tampered')
  with self.assertRaisesRegex(RuntimeError,'missing/changed'):legacy.reports(self.state,self.receipt)
if __name__=='__main__':unittest.main()
