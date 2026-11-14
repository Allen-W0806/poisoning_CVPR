"""Execute targeted reward regressions and historical arithmetic compatibility."""
import csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN='execution_remediation_20261102_v1'

def main():
 out=ROOT/'results'/RUN/'reward_regressions';out.mkdir(exist_ok=False);clock=time.perf_counter()
 cmd=[sys.executable,'-B','-m','unittest','tests.test_benign_reward_remediation',
      'tests.test_evaluation_reward.EvaluationRewardTest.test_reward_clipping_and_unknown_semantics',
      'tests.test_phase5_contracts','tests.test_phase5_protocol','-v']
 p=subprocess.run(cmd,capture_output=True,text=True,cwd=ROOT);(out/'unittest.stdout.log').write_text(p.stdout);(out/'unittest.stderr.log').write_text(p.stderr)
 from src.rl.reward import validation_reward,RewardWeights
 from src.rl.reward_legacy_v1 import validation_reward as old
 from src.editing.semantic_validator import SemanticResult
 import yaml
 cfg=yaml.safe_load((ROOT/'configs/phase3_generator_rl.yaml').read_text())['reward'];weights=RewardWeights.from_config(cfg)
 rows=list(csv.DictReader((ROOT/'results/audit_remediation_20261101T0120Z_v1/cpu_audit/historical_feedback_recomputed.csv').open()));assert len(rows)==500
 errors=[];compat=[]
 for x in rows:
  s,a,d,c=[float(x[k]) for k in ['S','A','D','C']];label=SemanticResult(False,s,'historical_proxy','preserved')
  now=validation_reward(label,a,d,c,weights);before=old(label,a,d,c)
  compat.append(now==before);errors.append(abs(now['reward']-float(x['reward'])))
 receipt=dict(status='PASS' if p.returncode==0 and all(compat) and max(errors)<1e-12 else 'FAIL',job_id=os.environ.get('SLURM_JOB_ID'),unittest_returncode=p.returncode,unittest_command=cmd,historical_rows=len(rows),exact_legacy_implementation_agreement=all(compat),max_historical_record_error=max(errors),executed_weights=weights.to_dict(),legacy_source_sha256=hashlib.sha256((ROOT/'src/rl/reward_legacy_v1.py').read_bytes()).hexdigest(),guard_sha256=hashlib.sha256((ROOT/'src/rl/phase5_backend.py').read_bytes()).hexdigest(),scientific_reward_release=False,seconds=time.perf_counter()-clock,new_GPU_seconds=0)
 (out/'summary.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True);return 0 if receipt['status']=='PASS' else 1

if __name__=='__main__':raise SystemExit(main())
