"""CPU final arithmetic audit, schedule verification and delivery manifests."""
import hashlib,json,math,os,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.overnight_benign_controller_20261102 import BASE,OUT,read,write,ledger,collect,command,now,serialized_controller
from scripts.numerical_boundaries_20261102 import load,compare,sha
from scripts.run_ip2pix_checkpoint_diagnostic import same

def main():
 import torch
 torch.set_num_threads(2)
 with serialized_controller(wait=True) as acquired:
  assert acquired;s=read(BASE/'state.json');assert s['admissions_closed'] and len(s['GPU_jobs'])==1;collect(s)
  ids=[t['job_id'] for t in s['CPU_jobs']];acct=command(['sacct','-X','-nP','-j',','.join(ids),'--format=JobID,State,ElapsedRaw,AllocTRES,ExitCode,Start,End']).stdout;by={v[0]:v for line in acct.splitlines() if len(v:=line.split('|'))>=7}
  for t in s['CPU_jobs']:
   if t['job_id'] in by:
    v=by[t['job_id']];t.update(scheduler_state=v[1].split()[0].rstrip('+'),elapsed_seconds=int(v[2]),exit_code=v[4])
  write(BASE/'state.json',s)
  task=ROOT/s['GPU_jobs'][0]['run_directory'];arithmetic=[]
  for mode,index in [('fresh0',0),('fresh1',0),('resident',0),('resident',1)]:
   w=load(task/mode/f'update{index}.pt');c=w['capture'];ref=math.sqrt(math.fsum(float(g.double().square().sum()) for n,g in c['raw']));runtime=float(c['runtime_total_norm']);coef=c['runtime_coefficient_formula'];pred=[(n,g*coef) for n,g in c['raw']];clip=compare(pred,c['postclip']);before=w['before'];after=w['after'];rng=all(same(before[k],after[k]) for k in ['rng_torch','rng_cuda','rng_python','rng_numpy'])
   arithmetic.append(dict(mode=mode,index=index,FP64_total_norm=ref,runtime_total_norm=runtime,runtime_minus_FP64=runtime-ref,runtime_relative_error=abs(runtime-ref)/ref,reconstructed_clipping_coefficient=float(coef),coefficient_provenance='Derived after original clipping from returned runtime norm using pinned2.6 formula; internal intermediate not intercepted',CPU_FP32_predicted_postclip=clip,whole_update_CPU_CUDA_NumPy_Python_RNG_unchanged=rng))
  write(OUT/'norm_and_clipping_arithmetic_audit.json',arithmetic)
  from scripts.overnight_supervisor_20261101 import accounting
  old=accounting(read(ROOT/'reports/overnight_supervisor/state.json'));assert old['consumed']==2899 and old['cap']==3600 and old['separately_authorized_overnight_GPU_jobs']==['88433'] and not old['unreconciled_project_GPU_jobs']
  # Correct historical role/cost attribution; old result receipt stays immutable.
  from scripts.overnight_benign_reporting_20261102 import fixed_model_metrics,render
  original=OUT/'fixed_model_metrics.json'
  if not (OUT/'fixed_model_metrics_initial.json').exists():(OUT/'fixed_model_metrics_initial.json').write_bytes(original.read_bytes())
  fixed_model_metrics(ROOT,OUT);render(s,ROOT,BASE,OUT)
  numerical=read(task/'numerical_results.json');latest=ROOT/'reports/NUMERICAL_ROOT_CAUSE_LATEST.md';body=latest.read_text();body+='\n## CPU norm/clipping arithmetic audit\n\n| Execution | FP64 reference norm | Runtime norm | Absolute norm error | Reconstructed coefficient | CPU predicted postclip exact | All global RNG unchanged |\n|---|---:|---:|---:|---:|---|---|\n'
  for a in arithmetic:body+=f"|{a['mode']}:{a['index']}|{a['FP64_total_norm']:.12g}|{a['runtime_total_norm']:.12g}|{abs(a['runtime_minus_FP64']):.12g}|{a['reconstructed_clipping_coefficient']:.12g}|{a['CPU_FP32_predicted_postclip']['exact']}|{a['whole_update_CPU_CUDA_NumPy_Python_RNG_unchanged']}|\n"
  body+='\nThis audit does not replace the failed raw-gradient gate. The norm coefficient is reconstructed, not an intercepted intermediate. No justified production correction was applied. Conditional uninstrumented and resumed next-update GPU verification was NOT EXECUTED because the prerequisite failed. CPU controller/reporting corrections and tests are separate from a numerical repair.\n';latest.write_text(body,encoding='utf8')
  queue=command(['squeue','-h','-u',os.environ['USER'],'-o','%i|%j|%T|%E|%S']).stdout;monitor=read(ROOT/'reports/monitoring/scheduler.json');dash=read(ROOT/'dashboard/dashboard_data.json');dashboard=[]
  for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']:
   html=(ROOT/'dashboard'/n).read_text();matches=re.findall(r'<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>(.*?)</script>',html,re.S);assert len(matches)==1 and json.loads(matches[0])==dash;assert html.count('id="overnight-benign"')==1;dashboard.append(dict(file=n,embedded_data_exact=True,one_component=True))
  assert (ROOT/'dashboard/state.json').read_bytes()==(ROOT/'dashboard/dashboard_data.json').read_bytes()
  for p in (ROOT/'tmp').glob('overnight_callback_cpu_*'):raise AssertionError('CPU test scratch not cleaned: '+str(p))
  assert not (ROOT/'tmp/overnight_benign_20261102_v2_deploy').exists()
  protected=read(ROOT/'results/method_code_audit_20261101T0510Z_v1/source_manifest.json');assert all(sha(ROOT/n)==h for n,h in protected.items())
  receipts=dict(UTC=now(),run_id=s['run_id'],new_ledger=ledger(s),previous_validation_ledger=dict(cap=old['cap'],consumed=old['consumed'],remaining=old['remaining']),CPU_job_accounting=acct,queue=queue,unique_existing_monitor=monitor,next_fallback_job_id=monitor['next_job_id'],duplicate_callback_noops=s.get('duplicate_callback_noops'),controller_resume_from_persisted_state_verified=True,afterany_callback_id=s['GPU_jobs'][0]['completion_job_id'],afterany_submission_command=next(t['command'] for t in s['CPU_jobs'] if t['purpose']=='afterany completion'),one_shot_deadline_was_registered_and_cancelled_after_early_stop=next(t for t in s['CPU_jobs'] if t['token']=='deadline'),deadline_guard_tests='CPU_preparation_tests_final.log',retained_new_tensor_bytes=sum(p.stat().st_size for p in OUT.rglob('*.pt')),retained_total_bytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()),no_large_C_files=True,dashboard=dashboard,protected_source_hashes_unchanged=True,scratch_clean=True)
  write(OUT/'terminal_receipt.json',receipts)
  summary=ROOT/'reports/OVERNIGHT_FINAL_REPORT.md';f=summary.read_text();f+='\n## Final executed checks\n\nLatest26 targeted CPU tests PASS (capture nonmutation/clip equivalence, real persisted callback dry runs, budget/deadline guards, one-shot accounting recovery, bounded NODE_FAIL policy, conditional preclip-norm acceptance and explicit rejection of empty update receipts, and failed-artifact classification). A real repeated callback for88433 returned IDEMPOTENT_ALREADY_PROCESSED and created no jobs. Install re-entry returned ALREADY_INSTALLED. The original monitor recognizes88433 as a separately authorized allocation and currently accounts2899/3600 in its own ledger; it has not merged the new172 seconds.\n\nNo production BF16 fix. Raw gradients differ before clipping, so ordinary/resume and learned-efficacy stages did not run. Selected RSI historical checkpoint retained10 training sources after its20-source candidate was rejected; attempted cost is preserved. New CPU model quality measurements do not count as new overnight training.\n\nNew tensor evidence retained: '+str(receipts['retained_new_tensor_bytes'])+' bytes on cluster;no activation dumps or large C-drive writes. Original outputs and failed receipts are kept. CPU staging path error was corrected using the absolute canonical interpreter before deployment;GPU cost0. Only own unused deadline finalizer was canceled;the existing two-hour chain remains. Kernel-serialized completion callbacks have one delayed accounting recovery plus the existing two-hour fallback; unresolved scheduler submissions remain fail-closed and are not blindly retried. NODE_FAIL alone permits one identified/excluded-node infrastructure retry; none occurred in this run. Cloud verification is recorded separately after text-only upload.\n';summary.write_text(f,encoding='utf8')
  files=['OVERNIGHT_EXECUTION_STATUS.md','OVERNIGHT_EXPERIMENT_RESULTS.csv','OVERNIGHT_EXPERIMENT_RESULTS.tex','NUMERICAL_ROOT_CAUSE_LATEST.md','NEXT_ELIGIBLE_EXPERIMENT.md','OVERNIGHT_FINAL_REPORT.md'];write(OUT/'delivery_manifest.json',dict(UTC=now(),files={n:sha(ROOT/'reports'/n) for n in files}))
  print(json.dumps(dict(run=s['run_id'],new_GPU_seconds=ledger(s)['consumed'],remaining=ledger(s)['remaining'],initial_raw_mismatch=numerical['first_boundary'],retained_tensor_bytes=receipts['retained_new_tensor_bytes'],next_monitor=monitor['next_job_id'],dashboard_pass=True)))
if __name__=='__main__':main()

