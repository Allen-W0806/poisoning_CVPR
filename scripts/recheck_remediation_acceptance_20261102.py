"""CPU corrections after independent review; never reruns GPU/model work."""
import collections,copy,hashlib,json,os,re,shutil,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False)

def main():
 import torch,numpy as np
 from scripts.run_ip2pix_checkpoint_diagnostic import same
 from scripts.run_bf16_adjoints_20261102 import strict_pair_acceptance
 from src.rl.runtime_checkpoint_metadata import write_runtime_metadata,restore_runtime_metadata,encode
 run=ROOT/'results/execution_remediation_20261102_v1';out=run/'acceptance_recheck';out.mkdir(exist_ok=False);torch.set_num_threads(2);clock=time.perf_counter()
 cmd=[sys.executable,'-B','-m','unittest','tests.test_bf16_acceptance_remediation','tests.test_semantic_review_v3','tests.test_runtime_checkpoint_metadata','-v'];test=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT);(out/'unittest.log').write_text(test.stdout+test.stderr);assert test.returncode==0
 bf=ROOT/'results/execution_remediation_20261102_v1_bf16_adjoints';old=read(bf/'summary.json');states=[torch.load(bf/f'{i}_{m}/before/training_state.pt',map_location='cpu',weights_only=False) for i,m in enumerate(['ordinary','capture','capture','ordinary'])];cross=all(same(states[0],s) for s in states[1:]);accepted=cross and strict_pair_acceptance(old['pairs'],old['capture_raw_error']);assert accepted
 put(out/'BF16_strict_acceptance_recheck.json',dict(status='PASS_BOUNDED_ONLY',original_summary_sha256=sha(bf/'summary.json'),original_script_sha256=read(bf/'protocol.json')['source_hashes']['scripts/run_bf16_adjoints_20261102.py'],all_four_initial_states_exact=cross,raw_exact_required=True,raw_exact=old['capture_raw_error']['exact'],pair_acceptance=accepted,historical_failures_preserved=True,numerical_fix_confirmed=False,new_GPU_seconds=0))
 pilot=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';results=read(pilot/'results.json');observations=read(pilot/'prediction_observations.json');source=read(pilot/'source_manifest.json');prefix=read(pilot/'shared_prefix_receipt.json');qrows=[];runtime=[]
 # Reconstruct the deterministic outer scheduler from its fully recorded inputs.
 param=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([param],lr=1e-5);outer=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=0,threshold=0.,threshold_mode='abs');outer.step(float(np.mean(prefix['initial_validation'])));outer.step(float(np.mean(prefix['prefix_validation'])));outer_state=copy.deepcopy(outer.state_dict())
 for branch in results['branches']:
  arm=branch['arm'];candidate=[x for x in observations if x['phase']==arm+'/candidate_validation'];assessment=[x for x in observations if x['phase']==arm+'/selected_assessment'];assert len(candidate)==len(assessment)==4
  assert {x['sample_id'] for x in candidate}=={x['sample_id'] for x in source['validation']};assert {x['sample_id'] for x in assessment}=={x['sample_id'] for x in source['assessment']}
  assert np.mean([x['reward'] for x in assessment])==branch['assessment_mean'];qrows.append(dict(arm=arm,measured_validation_records=len(candidate),measured_assessment_records=len(assessment),assessment_distinct_source_ids=len({x['source_image_id'] for x in assessment}),selected_checkpoint_sha256=branch['selected_checkpoint_sha256']))
  for phase in ['start','candidate','selected']:
   folder=pilot/arm/phase;state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False);os_state=outer_state if arm=='ordinary_scheduler' and phase in ['candidate','selected'] else None
   if os_state:assert opt.param_groups[0]['lr']==state['optimizer']['param_groups'][0]['lr']
   obj=write_runtime_metadata(folder,state,os_state,reconstructed=True)
   dummy=type('T',(),{})();dummy.optimizer=type('O',(),{'param_groups':state['optimizer']['param_groups']})();dummy.scheduler=type('S',(),{'base_lrs':state['scheduler']['base_lrs']})();dummy.policy=type('P',(),{'config':{'learning_rate':1e-5}})()
   fresh_outer=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=0,threshold=0.,threshold_mode='abs') if os_state else None
   restored=restore_runtime_metadata(dummy,folder,fresh_outer);assert dummy.policy.config['learning_rate']==state['optimizer']['param_groups'][0]['lr']
   runtime.append(dict(arm=arm,phase=phase,active_LR=restored['active_optimizer_lrs'],initial_design_LR=(state['contract'].get('config') or {}).get('learning_rate'),sidecar_sha256=sha(folder/'active_runtime_config.json'),ordinary_scheduler_exact=encode(fresh_outer.state_dict())==encode(outer_state) if fresh_outer else None,reconstructed=True))
 # These sidecars are additional CPU provenance, not original GPU observations.
 for n in ['shared_initial','shared_prefix']:
  folder=pilot/n;state=torch.load(folder/'training_state.pt',map_location='cpu',weights_only=False);write_runtime_metadata(folder,state,reconstructed=True)
 put(out/'pilot_query_budget_recheck.json',dict(status='PASS_MEASURED_FROM_OBSERVATIONS',branches=qrows,observations_sha256=sha(pilot/'prediction_observations.json'),results_sha256=sha(pilot/'results.json'),new_model_calls=0));put(out/'active_runtime_and_scheduler_recheck.json',dict(status='PASS_CPU_STATE_RECONSTRUCTION',rows=runtime,limitation='Ordinary-scheduler state reconstructed from recorded input scores and exact executed source; it was not persisted by the original GPU script. Future source now persists/reloads it.',new_GPU_seconds=0))
 oldkit=run/'semantic_review_v3';newkit=run/'semantic_review_v3_r2';newkit.mkdir();(newkit/'images').mkdir();(newkit/'submitted').mkdir()
 for p in oldkit.iterdir():
  if p.is_file():shutil.copy2(p,newkit/p.name)
 for p in (oldkit/'images').iterdir():os.link(p,newkit/'images'/p.name)
 html=(newkit/'index.html').read_text();needle="['yes','no','ambiguous','not_applicable']";assert needle in html;html=html.replace(needle,"['yes','no','uncertain','not_applicable']");assert needle not in html;(newkit/'index.html').write_text(html)
 for role in ['A','B']:
  with zipfile.ZipFile(run/f'independent_review32_v3_r2_{role}.zip','x',zipfile.ZIP_DEFLATED) as z:
   for p in newkit.rglob('*'):
    if not p.is_file():continue
    b=p.read_bytes()
    if p.name=='index.html':b=b.replace(b'v3-ROLE:',('v3-r2-reviewer-'+role+':').encode())
    z.writestr(str(p.relative_to(newkit)),b)
 put(out/'human_review_enum_recheck.json',dict(status='PASS_ENUM_AND_INGESTION_REGRESSIONS',new_package='semantic_review_v3_r2',legacy_ambiguous_exports_accepted_and_normalized=True,original_exports_never_overwritten=True,human_labels_created=0,genuine_reviewed_pairs=0))
 original=read(run/'data_metric_remediation_summary.json');corrected=copy.deepcopy(original);corrected['results']['prospective_sources']['corrupt_records_covered_by_complete_witnesses']=False;corrected['results']['prospective_sources']['corrupt_record_lineage']='UNKNOWN: complete companion exists, exact job-to-input binding not proved';corrected['scope_correction']='Supersedes only the unsupported historical-corruption coverage claim; measured bytes/IDs/hashes/pHash results retained. Original receipt remains immutable.';corrected['status']='PASS_CONTENT_AND_METRICS_LINEAGE_PENDING';put(out/'data_metric_corrected_summary.json',corrected)
 put(out/'summary.json',dict(status='PASS_SCOPED_CORRECTIONS',job_id=os.getenv('SLURM_JOB_ID'),test_returncode=test.returncode,strict_BF16_recheck=accepted,pilot_query_counts_measured=True,runtime_metadata_sidecars_verified=len(runtime),human_review_enum_repaired=True,corruption_coverage_overclaim_retracted=True,new_GPU_seconds=0,new_model_calls=0,seconds=time.perf_counter()-clock));print(json.dumps(read(out/'summary.json')),flush=True)
 return 0

if __name__=='__main__':raise SystemExit(main())
