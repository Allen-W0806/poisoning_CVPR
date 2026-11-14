"""Final read-only checks and retained evidence hashes; no report regeneration."""
import ast,csv,json,re,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.overnight_benign_controller_20261102 import BASE,OUT,read,write,ledger,now,command
from scripts.numerical_boundaries_20261102 import sha

def main():
 s=read(BASE/'state.json');assert s['admissions_closed'] and len(s['GPU_jobs'])==1 and s['GPU_jobs'][0]['job_id']=='88433';b=ledger(s);assert b['consumed']==172 and b['reserved']==0
 cloud=read(OUT/'cloud_report_verification.json');assert cloud['status']=='CLOUD_VERIFIED';reports={}
 for r in cloud['reports']:
  p=ROOT/'reports'/r['name'];assert sha(p)==r['sha256'];reports[r['name']]=dict(sha256=r['sha256'],cloud_id=r['id'],bytes=p.stat().st_size)
 index=read(OUT/'cloud_index_verification.json');assert sha(ROOT/'reports/LATEST_REPORT_INDEX.md')==index['index_sha256'];assert read(OUT/'cloud_index_after_periodic_copy.json')['byte_exact']
 with (ROOT/'reports/OVERNIGHT_EXPERIMENT_RESULTS.csv').open(newline='') as f:rows=list(csv.DictReader(f))
 assert len(rows)==4 and all(r['ASR']=='NOT TESTED' and r['Actual_allocated_GPU_seconds']=='' for r in rows)
 assert next(r for r in rows if r['Method']=='rsi')['Training_sources']=='10';assert next(r for r in rows if r['Method']=='rsi')['Attempted_training_sources']=='20'
 tex=(ROOT/'reports/OVERNIGHT_EXPERIMENT_RESULTS.tex').read_text();stack=[]
 for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',tex):
  if kind=='begin':stack.append(name)
  else:assert stack and stack.pop()==name
 assert not stack and all(r['Selected_checkpoint_hash'] in tex for r in rows)
 originals=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json');assert all(sha(ROOT/n)==h for n,h in originals.items());protocol=read(OUT/'numerical_a0/protocol.json');assert all(sha(ROOT/n)==h for n,h in protocol['source_hashes'].items())
 assert sha(ROOT/s['sealed_final_sources'])==s['sealed_manifest_sha256']
 sources=['scripts/gradient_boundary_capture_20261102.py','scripts/run_overnight_norm_fixture_20261102.py','scripts/run_overnight_ordinary_resume_20261102.py','scripts/overnight_benign_controller_20261102.py','scripts/overnight_benign_reporting_20261102.py','scripts/overnight_supervisor_20261101.py','scripts/test_overnight_benign_20261102.py','scripts/test_external_index_adoption_20261102.py','sync_rsi_poisoning_reports.py','tests/test_rsi_poisoning_report_sync.py']
 for n in sources:ast.parse((ROOT/n).read_text(encoding='utf-8-sig'))
 queue=command(['squeue','-h','-u','ydk297','-o','%i|%j|%T|%S']).stdout;monitor=read(ROOT/'reports/monitoring/scheduler.json');assert monitor['next_job_id'] in queue
 for prefix in ['overnight_callback_cpu_*','verified_index_adoption_*','overnight_benign_20261102_v2_deploy']:
  assert not list((ROOT/'tmp').glob(prefix)),prefix
 record=dict(PROJECT_ID='RSI-POISONING',finalized_UTC=now(),run_id=s['run_id'],status=s['status'],new_budget=b,monitor_job_id=monitor['next_job_id'],monitor_due_UTC=monitor['next_due_utc'],queue=queue,reports=reports,index_sha256=index['index_sha256'],cloud_status='CLOUD_VERIFIED_AFTER_ACTUAL_PERIODIC_DESKTOP_COPY',source_hashes={n:sha(ROOT/n) for n in sources},protected_sources_unchanged=True,sealed160_manifest_unchanged=True,new_tensor_bytes=sum(p.stat().st_size for p in OUT.rglob('*.pt')),CSV_rows=4,CSV_scope='historical training,new CPU fixed-model quality metrics only',LaTeX_structure_pass=True,LaTeX_rendered='NOT TESTED; no TeX engine in checked PATH',runtime_numeric_NFE='NOT RECORDED; do not replace with inferred count',numerical_result='RAW_BACKWARD_GRADIENT_MISMATCH',ordinary_resume='NOT EXECUTED; prerequisite failed',benign_learning='NOT EXECUTED',production_BF16_fix='NOT ESTABLISHED',unknown_submission_policy='fail closed; no automatic token-to-job reconciliation',review_trace='.aris/traces/experiment-audit/2026-11-02_overnight-bounded',scratch_clean=True)
 write(OUT/'final_delivery_manifest.json',record)
 write(OUT/'artifact_inventory.json',dict(files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']))
 print(json.dumps(dict(finalized_UTC=record['finalized_UTC'],GPU_seconds=b['consumed'],remaining=b['remaining'],reports_verified=len(reports),monitor=monitor['next_job_id'],status=s['status'],cloud=record['cloud_status'])))
if __name__=='__main__':main()
