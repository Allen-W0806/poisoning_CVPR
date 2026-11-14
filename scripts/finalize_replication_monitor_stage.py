"""Verify final evidence, retain source, export the completed bounded stage."""
import argparse,hashlib,json,re,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;ledger=read(run/'job_ledger.json');checks=[]
    def check(n,b):checks.append({'test':n,'passed':bool(b)});assert b,n
    check('five_real_GPU_submissions',len(ledger)==5 and all(r['job_id'].isdigit() for r in ledger))
    check('bounded_retry_counts',all(sum(x['task']==r['task'] for x in ledger)<=3 for r in ledger))
    for r in ledger:
        out=ROOT/'results'/r['run_id']/('clean_main' if r['kind']=='qwen' else 'diagnostic');inventory=read(out/'artifact_inventory.json')
        check('all_checkpoint_output_hashes_'+r['job_id'],all((out/f['path']).is_file() and sha(out/f['path'])==f['sha256'] for f in inventory['files']))
        data=read(out/'checks.json')
        if r['kind']=='qwen':check('fresh_object_step_'+r['job_id'],data['status']=='PASS' and any(x['test']=='fresh_object_next_step_equal' and x['status']=='PASS' for x in data['checks']))
    check('initial_P2_negative_evidence_preserved',read(ROOT/'results'/f'{run.name}_ip2pix_a0/diagnostic/checks.json')['status']=='FAIL')
    check('instrumented_witness_distinct',read(ROOT/'results'/f'{run.name}_ip2pix_a1/diagnostic/checks.json')['status']=='PASS')
    check('failure_detection_tests',read(ROOT/'reports/rsi_stage_monitor/failure_detection_selftest.json')['status']=='PASS')
    handoff=read(run/'semantic_handoff.json');check('no_fabricated_human_reviews',handoff['human_review_records_created']==0 and not handoff['genuine_submitted_files'])
    schedule=read(ROOT/'reports/monitoring/scheduler.json');check('native_recurring_enabled',schedule['enabled'] and schedule['interval_hours']==2 and schedule['next_job_id']=='86836')
    squeue=subprocess.run(['squeue','-h','-j',schedule['next_job_id'],'-o','%i|%T|%S|%k'],capture_output=True,text=True,check=True).stdout.strip();check('future_successor_actually_registered',squeue.startswith('86836|PENDING|'))
    check('native_entry_execution_verified',schedule['last_tick_job_id']=='86850' and schedule['last_check_outcome']=='COMPLETED_WITH_FINDINGS')
    heartbeat=read(ROOT/'reports/rsi_stage_monitor/heartbeat.json');check('successful_new_coverage_heartbeat',heartbeat['status']=='PASS' and len(heartbeat['observations'])==5)
    resources=read(run/'resource_accounting.json');check('GPU_budget',resources['charged_total_GPU_seconds']==1344<=3600 and resources['new_stage_GPU_seconds']==342 and resources['remaining_GPU_seconds']==2256)
    source=run/'audit/source_final';source.mkdir(exist_ok=True);names=['prepare_replication_stage.py','run_qwen_replication.py','run_ip2pix_checkpoint_diagnostic.py','run_ip2pix_checkpoint_instrumented.py','rsi_stage_monitor.py','install_rsi_stage_monitor.py','verify_registered_rsi_monitor.py','prepare_independent_review_handoff.py','analyze_ip2pix_backward_forensics.py','submit_ip2pix_instrumented_followup.py','report_replication_monitor_stage.py','finalize_replication_monitor_stage.py','manage_monitor.py']
    for name in names:shutil.copy2(ROOT/'scripts'/name,source/name)
    shutil.copy2(ROOT/'slurm/rsi_repro_stage.sbatch',source/'rsi_repro_stage.sbatch')
    for name in ['REPLICATION_AND_MONITORING_STATUS','QWEN_THREE_SEED_REPLICATION','IP2PIX_BF16_DIAGNOSTIC','INDEPENDENT_REVIEW_HANDOFF']:
        p=ROOT/'reports'/f'{name}.md'
        for target in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if not target.startswith(('http:','https:','#')):check('report_link_'+target,(p.parent/target).exists())
    dashboard=ROOT/'dashboard/RSI_Experimental_Progress_Latest.html';text=dashboard.read_text();check('dashboard_new_stage_and_monitor_present','RSI_STAGE_MONITOR_START' in text and 'RSI_REPLICATION_RESULTS_START' in text and 'instrumented witness PASS' in text)
    for target in re.findall(r'(?:src|href)="([^"]+)"',text):
        if not target.startswith(('http:','https:','#')):check('dashboard_link_'+target,(dashboard.parent/target).exists())
    result={'status':'PASS_ENGINEERING_AND_EVIDENCE_INTEGRITY','scientific_IP2Pix_continuation':'NOT VALIDATED; uninstrumented FAIL, instrumented PASS','human_semantics':'BLOCKED','verified_UTC':time.time(),'checks':checks,'registered_monitor':squeue,'heartbeat':heartbeat,'source_sha256':{p.name:sha(p) for p in source.iterdir()},'resource_accounting':resources}
    with (run/'completion.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'checks':len(checks),'future_successor':squeue,'GPU_charged':1344,'GPU_remaining':2256}))
if __name__=='__main__':main()
