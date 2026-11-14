"""Execute the registered periodic entry point NOW in an owned CPU allocation."""
import json,subprocess,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.monitoring.scheduler import Scheduler
from src.monitoring.core_v6 import load_config
def main():
    scheduler=Scheduler(ROOT,load_config(ROOT));status=scheduler.status();assert status['state']['enabled'] and status['owned_jobs']
    slot=datetime.now(timezone.utc).isoformat();tag=status['job_tag']+':verify_'+uuid.uuid4().hex;args=['sbatch','--parsable','--partition','epyc-cpu','--account','allusers','--cpus-per-task','1','--mem','2G','--time','00:05:00','--job-name','poisoning-monitor','--comment',tag,'--output',str(ROOT/'reports/rsi_stage_monitor/periodic-entry-%j.out'),'--error',str(ROOT/'reports/rsi_stage_monitor/periodic-entry-%j.err'),str(ROOT/'reports/monitoring/monitor_job.sbatch'),slot]
    p=subprocess.run(args,capture_output=True,text=True);receipt={'command':args,'exit_code':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'job_id':p.stdout.strip().split(';')[0],'slot_UTC':slot,'existing_future_monitor_job':status['state']['next_job_id']};(ROOT/'reports/rsi_stage_monitor/periodic_entry_test_submission.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt,indent=2));assert p.returncode==0
if __name__=='__main__':main()
