"""Compatibility entry used by dashboard/build_dashboard.py and dashboard_sync.py.
Refresh once through the existing source pipeline. Recurrence belongs to Slurm.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.progression.dashboard import refresh
if __name__=='__main__':
    if '--watch' in sys.argv:raise SystemExit('Use the existing two-hour Slurm monitor; no duplicate watch loop.')
    print(json.dumps(refresh(ROOT)))
