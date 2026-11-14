import argparse
import json
import _bootstrap
from src.monitoring.core_v6 import load_config, run_monitor
from src.monitoring.scheduler import Scheduler

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['install','status','pause','resume','remove','tick'])
    p.add_argument('--slot')
    a=p.parse_args()
    config=load_config(_bootstrap.ROOT)
    scheduler=Scheduler(_bootstrap.ROOT,config)
    if a.action in {'install','resume'}:
        result=scheduler.install()
    elif a.action=='status':
        result=scheduler.status()
    elif a.action in {'pause','remove'}:
        result=scheduler.stop(remove=a.action=='remove')
    else:
        if not a.slot:
            p.error('tick requires the scheduled UTC slot')
        def inspect_registered_stages():
            legacy=run_monitor(_bootstrap.ROOT,config=config)
            from rsi_stage_monitor import run_stage_monitor
            legacy['approved_stage_extension']=run_stage_monitor()
            from overnight_supervisor_20261101 import cycle as overnight_cycle
            legacy['overnight_supervisor']=overnight_cycle()
            return legacy
        result=scheduler.tick(a.slot,inspect_registered_stages)
    print(json.dumps(result,indent=2))
