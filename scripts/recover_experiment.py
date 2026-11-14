"""Run the same recovery gates as the monitor; no independent unguarded submit."""
import json
import _bootstrap
from src.monitoring.core import run_monitor

if __name__ == '__main__':
    print(json.dumps(run_monitor(_bootstrap.ROOT)))
