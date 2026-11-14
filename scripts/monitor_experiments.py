"""Run one bounded CPU monitoring pass. Recurrence is managed by Slurm."""
import json
from pathlib import Path
import argparse
import _bootstrap
from src.monitoring.core_v6 import run_monitor


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--spine', action='store_true')
    args = parser.parse_args()
    if args.spine:
        path = _bootstrap.ROOT / 'tmp/monitor_20261027/F0_inventory.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        registry = _bootstrap.ROOT / 'experiments/registry.jsonl'
        rows = [json.loads(line) for line in registry.read_text().splitlines()]
        path.write_text(json.dumps({'engineering_spine': True, 'registry_entries': len(rows)}))
        print(path)
    else:
        print(json.dumps(run_monitor(_bootstrap.ROOT)))
