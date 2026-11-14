"""Stable live-dashboard entry point; Phase6 minimal renderer retained separately."""
import os
import uuid
from pathlib import Path

def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(text,encoding='utf8');os.replace(tmp,path)

def refresh(root,live=None):
    from .dashboard_live import refresh as render
    result=render(root,live)
    # Reuse the same pages/state when the existing monitor refreshes them.
    if (Path(root)/'reports/SMALL_SCALE_STATUS_20261031.json').exists():
        from scripts.report_small_scale_20261031 import dashboard
        dashboard(Path(root))
    if (Path(root)/'reports/NEXT_VALIDATION_STATUS_20261031.json').exists():
        from scripts.report_next_validation_20261031 import dashboard as next_dashboard
        next_dashboard(Path(root))
    if (Path(root)/'reports/REMEDIATION_STATUS_20261101.json').exists():
        from scripts.report_remediation_20261101 import dashboard as remediation_dashboard
        remediation_dashboard(Path(root))
    if (Path(root)/'reports/METHOD_AUDIT_CLOSURE_20261101.json').exists():
        from scripts.close_method_audit_20261101 import dashboard as method_dashboard
        method_dashboard(Path(root))
    if (Path(root)/'reports/overnight_supervisor/heartbeat.json').exists():
        from scripts.overnight_supervisor_20261101 import dashboard as overnight_dashboard
        overnight_dashboard(Path(root))
    return result
