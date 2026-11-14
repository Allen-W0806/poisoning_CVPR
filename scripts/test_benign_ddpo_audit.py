"""Reproducible CPU gate for the isolated benign source snapshot."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
TEST_MODULES = ['tests.test_benign_ddpo_audit', 'tests.test_diffusion_rl',
                'tests.test_fresh_batch_remediation', 'tests.test_runtime_checkpoint_metadata',
                'tests.test_bf16_acceptance_remediation', 'tests.test_mini_pilot',
                'tests.test_lighting_reward_v3']

def source_hashes():
    files = list((root/'src').rglob('*.py')) + [root/'scripts/run_benign_ddpo_minimal.py',
            root/'scripts/test_benign_ddpo_audit.py',root/'configs/benign_ddpo_minimal.json']
    files += [root/(name.replace('.','/')+'.py') for name in TEST_MODULES]
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(files)}

if __name__ == '__main__':
    output = Path(sys.argv[1]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ['PILOT_TEST_TMP'] = str(output/'scratch')
    os.environ['TMPDIR'] = str(output/'scratch')
    Path(os.environ['TMPDIR']).mkdir(parents=True, exist_ok=True)
    import tempfile
    tempfile.tempdir = os.environ['TMPDIR']
    modules = TEST_MODULES
    started = time.time()
    suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {'passed': result.wasSuccessful(), 'tests_run': result.testsRun,
               'failures': len(result.failures), 'errors': len(result.errors), 'skips': len(result.skipped),
               'elapsed_seconds': time.time()-started, 'modules': modules, 'source_sha256': source_hashes(),
               'GPU_work_performed': False}
    (output/'cpu_tests.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    import shutil
    target = (output/'scratch').resolve()
    if target.parent != output or target.name != 'scratch': raise ValueError('Unsafe scratch cleanup')
    shutil.rmtree(target)
    print(json.dumps({key:value for key,value in receipt.items() if key!='source_sha256'}))
    sys.exit(0 if result.wasSuccessful() else 1)
