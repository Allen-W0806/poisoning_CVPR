import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('night_analysis',Path(__file__).resolve().parents[1]/'scripts/summarize_night_vqa.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class NightAnalysisTests(unittest.TestCase):
    def test_missing_seeds_never_becomes_result(self):
        value=module.summarize('.',{'jobs':[]})
        self.assertEqual(value['status'],'NOT_COMPLETE');self.assertNotIn('splits',value)

    def test_source_bootstrap_identical_arms_zero_delta(self):
        self.assertEqual(module.source_bootstrap([0.]*512,resamples=200),[0.,0.])

    def test_source_not_seed_is_resampling_unit(self):
        # Known +/-0.5 source-level deltas: exact mean=0, SD=0.5.
        values=[-.5,.5]*256
        first=module.source_bootstrap(values,resamples=1000)
        self.assertEqual(first,module.source_bootstrap(values,resamples=1000))
        self.assertLess(first[0],-.03);self.assertGreater(first[1],.03)


if __name__=='__main__':unittest.main()
