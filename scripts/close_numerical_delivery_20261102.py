"""Finalize evidence hashes/negative classification; no new scientific job."""
import hashlib,json,shutil,sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.overnight_supervisor_20261101 import read,write,accounting,now,BASE
from scripts.numerical_boundaries_20261102 import sha,put
from src.monitoring.common import FileLock
OUT=ROOT/'results/numerical_root_cause_20261102_v1'

def main():
 with FileLock(BASE/'monitor.lock') as acquired:
  assert acquired
  s=read(BASE/'state.json');task=next(t for t in s['tasks'] if t.get('job_id')=='88422');assert read(OUT/'checks.json')['status']=='FAIL'
  logs=''.join((p.read_text(errors='replace')[-4000:] if p.exists() else '') for p in [OUT/'stderr.log',OUT/'stdout.log']);fp=hashlib.sha256((task['scheduler_state']+task['exit_code']+logs).encode()).hexdigest();task.update(failure_fingerprint=fp,classification='NUMERICAL_MISMATCH_PRESERVED',numerical_retry_allowed=False)
  s['failure_fingerprints'][fp]=dict(job_id='88422',classification='NUMERICAL_MISMATCH_PRESERVED',run_id=OUT.name,numerical_retry_allowed=False);write(BASE/'state.json',s);b=accounting(s)
 for name in ['NUMERICAL_ROOT_CAUSE_LATEST.md','finalize_numerical_evidence_20261102.py']:assert not (ROOT/'tmp'/name).exists()
 for n,h in read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json').items():assert sha(ROOT/n)==h,n
 old=OUT/'report_manifest.json'
 if not (OUT/'report_manifest_before_scope_clarification.json').exists():shutil.copy2(old,OUT/'report_manifest_before_scope_clarification.json')
 put(old,dict(UTC=now(),reports={n:sha(ROOT/'reports'/n) for n in ['NUMERICAL_ROOT_CAUSE_LATEST.md','NEXT_ELIGIBLE_EXPERIMENT.md']},cloud_reports=read(OUT/'cloud_report_verification.json'),cloud_index=read(OUT/'cloud_index_verification.json')))
 queue=subprocess.run(['squeue','-h','-u','ydk297','-o','%i|%j|%T|%S|%R'],capture_output=True,text=True,check=True).stdout
 receipt=dict(UTC=now(),budget=b,queue=queue,tensor_bytes=sum(p.stat().st_size for p in OUT.rglob('*.pt')),all_retained_bytes_before_receipt=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),cloud_index_status=read(OUT/'cloud_index_verification.json')['status'],review='same-family provisional; findings resolved/qualified, numerical FAIL',no_additional_GPU_job=True,staging_files_removed=True,production_sources_unchanged=True)
 put(OUT/'final_delivery_receipt.json',receipt)
 put(OUT/'artifact_inventory.json',dict(files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in OUT.rglob('*') if p.is_file() and p.name!='artifact_inventory.json']))
 print(json.dumps(dict(UTC=receipt['UTC'],budget_remaining=b['remaining'],queue=queue,tensor_bytes=receipt['tensor_bytes'],cloud_index=receipt['cloud_index_status'])))
if __name__=='__main__':main()
