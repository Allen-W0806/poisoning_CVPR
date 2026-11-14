import argparse
import json
import _bootstrap
from src.monitoring.common import safe_path, slurm_snapshot
from src.monitoring.diagnostics import inspect_run

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('experiment_id')
    a=p.parse_args()
    path=safe_path(_bootstrap.ROOT,'experiments/'+a.experiment_id)
    run=inspect_run(path)
    snapshot=slurm_snapshot([run['job_id']] if run['job_id'] else [])
    print(json.dumps(inspect_run(path,snapshot['jobs']),indent=2,allow_nan=True))
