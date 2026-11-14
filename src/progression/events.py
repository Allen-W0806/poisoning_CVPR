"""Finalized-experiment notification cannot overwrite its immutable outcome."""
from pathlib import Path
from ..monitoring.common import append_json,read_json,now
from ..utils.logging import write_json

def finalized_event(root,experiment_id):
    root=Path(root)
    if not (root/'configs/phase6.json').is_file():return None
    from .dashboard import refresh
    ledger=root/'reports/phase6_followup/dashboard_retry_counts.json';counts=read_json(ledger,{})
    for attempt in range(3):
        try:
            result=refresh(root)
            append_json(root/'reports/phase6_followup/dashboard_events.jsonl',{'timestamp':now(),'event':'experiment_finalized',
               'experiment_id':experiment_id,'snapshot_utc':result['snapshot_utc'],'outcome':'UPDATED'})
            return result
        except Exception as error:
            signature='dashboard_refresh:'+type(error).__name__+':'+str(getattr(error,'errno',None))
            count=counts.get(signature,0)
            append_json(root/'reports/phase6_followup/dashboard_events.jsonl',{'timestamp':now(),'event':'experiment_finalized',
              'experiment_id':experiment_id,'outcome':'REFRESH_FAILED','signature':signature,'error':str(error),'retry_count':count})
            if not isinstance(error,OSError) or getattr(error,'errno',None) not in {5,11,116} or count>=2:return {'status':'FAILED_RECORDED','error':str(error)}
            counts[signature]=count+1;write_json(ledger,counts)
    return {'status':'RETRY_LIMIT_REACHED'}
