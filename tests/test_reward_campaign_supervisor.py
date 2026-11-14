"""No victim training before full poison quality/iteration evidence exists."""
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.supervise_reward_campaign import advance


class AdmissionTest(unittest.TestCase):
    def setUp(self):
        self.base=Path(os.environ["PILOT_TEST_TMP"]).resolve()
        self.base.mkdir(parents=True,exist_ok=True)
        self.c=Path(tempfile.mkdtemp(prefix="supervisor_",dir=self.base)).resolve()
        (self.c/"shadowcast").mkdir()
        (self.c/"shadowcast/poison_completed.json").write_text('{}')

    def tearDown(self):
        if not self.c.is_relative_to(self.base) or not self.c.name.startswith('supervisor_'):
            raise RuntimeError("Unsafe test cleanup")
        shutil.rmtree(self.c)

    def test_missing_quality_cannot_train_victim(self):
        with patch('scripts.supervise_reward_campaign.quality_cpu'),patch('scripts.supervise_reward_campaign.gpu') as submit:
            advance(self.c)
        submit.assert_not_called()

    def test_completed_quality_admits_only_fixed_poison50_stage(self):
        (self.c/"shadowcast/poison_quality.json").write_text(json.dumps({
            'PGD_trace_audit':{'all_batches_complete':True},'maximum_8bit_difference':8,'selected_poison_images':50}))
        with patch('scripts.supervise_reward_campaign.ROOT',self.base),patch('scripts.supervise_reward_campaign.quality_cpu'),patch('scripts.supervise_reward_campaign.gpu',return_value=True) as submit:
            advance(self.c)
        self.assertEqual(submit.call_count,1)
        self.assertEqual(submit.call_args.args[1:3],('shadowcast_poisoned',180))
        self.assertEqual(submit.call_args.args[3][-2:],['--poison-count','50'])


if __name__=="__main__":unittest.main()
