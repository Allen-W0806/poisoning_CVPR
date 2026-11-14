import os
import sys
import _bootstrap
from src.utils.logging import Experiment

if __name__ == "__main__":
    path = os.environ.get("POISONING_JOB_RUN")
    if path:
        try:
            run = Experiment.attach_active(_bootstrap.ROOT, path)
        except ValueError:
            sys.exit(0)
        code = int(sys.argv[1])
        run.finish("PASS" if code == 0 else "FAIL", f"Slurm wrapper exit {code}; see experiment-specific Slurm logs.")
