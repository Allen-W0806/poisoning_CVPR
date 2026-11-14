"""CPU-only gate bound to exact P0 sources, tests, and fixed protocol."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
MODULES=['tests.test_p0_foundations','tests.test_benign_ddpo_audit','tests.test_diffusion_rl',
         'tests.test_runtime_checkpoint_metadata','tests.test_fresh_batch_remediation',
         'tests.test_bf16_acceptance_remediation','tests.test_mini_pilot','tests.test_lighting_reward_v3',
         'tests.test_vqa_official','tests.test_generator_contract','tests.test_rsi_logging']


def source_hashes():
    paths=list((ROOT/'src').rglob('*.py'))+[ROOT/(n.replace('.','/')+'.py') for n in MODULES]
    paths += [ROOT/'scripts'/name for name in ['test_p0_foundations.py','run_p0_foundations.py','prepare_p0_manifest.py']]
    paths += [ROOT/'configs/p0_foundations_20261111.json']
    return {p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--assets-root');parser.add_argument('--run',required=True);a=parser.parse_args()
    run=Path(a.run).resolve();scratch=run/'tmp/cpu_tests';scratch.mkdir(parents=True,exist_ok=True)
    os.environ['PILOT_TEST_TMP']=str(scratch);os.environ['TMPDIR']=str(scratch);tempfile.tempdir=str(scratch)
    started=time.time()
    try:
        result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(MODULES))
        receipt={'passed':result.wasSuccessful() and not result.skipped,'tests_run':result.testsRun,
                 'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),
                 'elapsed_seconds':time.time()-started,'source_SHA256':source_hashes(),'modules':MODULES,
                 'GPU_work':False,'kind':'CPU engineering regressions; not method efficacy'}
        (run/'cpu_tests.json').write_text(json.dumps(receipt,indent=2));print(json.dumps({k:v for k,v in receipt.items() if k!='source_SHA256'}))
    finally:
        target=scratch.resolve()
        if target.parent!=run/'tmp' or target.name!='cpu_tests':raise ValueError('Unsafe scratch cleanup')
        shutil.rmtree(target)
    sys.exit(0 if receipt['passed'] else 1)


if __name__=='__main__':main()
