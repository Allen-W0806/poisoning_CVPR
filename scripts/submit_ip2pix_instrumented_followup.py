"""One prespecified observation-only followup of the preserved P2 failure."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from rsi_stage_monitor import BASE,FileLock,read,write,run_stage_monitor,submit
def main():
    report=run_stage_monitor();run=ROOT/'results'/report['run_id'];cfg=read(run/'protocol.json')
    with FileLock(BASE/'monitor.lock') as acquired:
        assert acquired;ledger=read(run/'job_ledger.json');old=[r for r in ledger if r['task']=='P2_IP2Pix'];assert len(old)==1 and old[0]['attempt']==0
        assert all(o['scheduler_state'] in {'COMPLETED','FAILED'} for o in report['observations']);assert report['GPU_charged_seconds_including_prior']+600<=3600
        child=ROOT/'results'/f'{run.name}_ip2pix_a1';child.mkdir(exist_ok=False);protocol=read(ROOT/'results'/old[0]['run_id']/'protocol.json');protocol.update(run_id=child.name,instrumentation='Read-only raw-gradient, preclip norm and layout capture; same source, objective, optimizer and strict criterion');write(child/'protocol.json',protocol)
        task={'task':'P2_IP2Pix','kind':'ip2pix_instrumented','attempt':1,'run_id':child.name,'wall_seconds':600}
        result=submit(task,run,ledger);write(run/'audit/instrumented_followup_authorization.json',{'reason':'Initial earliest difference was measured AFTER clipping. Capture raw gradients to separate backward from clipping/norm effects; no tolerance or science settings changed.','original_failed_job':old[0]['job_id'],'job_id':result['job_id'],'additional_attempt':1,'max_additional_attempts_allowed':2});print(json.dumps(result,indent=2))
if __name__=='__main__':main()
