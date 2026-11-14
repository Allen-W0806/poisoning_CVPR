"""Supplement saved tables; no image metrics, model calls, or training are rerun."""
import csv,json,hashlib,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import build_baseline_comparisons_20261101 as core
RUN=ROOT/'results/baseline_comparison_20261101T2255Z_v1'
OUT=RUN/'publication_supplement'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir()
 # Predeclared supplemental stream; original two RSI intervals are preserved.
 protocol=dict(contrast='adaptive_scheduler_minus_fixed',resamples=5000,
               bootstrap_seed=2026110123,unit='crossed training seeds and shared test conditions',
               source='saved 3x32 outcomes only',new_GPU_seconds=0,new_SSIM_calls=0)
 (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
 d=core.read(core.ABL/'threeway/results.json')
 by={(x['seed'],x['arm']):x for x in d['branches']}
 delta=np.array([np.array(by[s,'adaptive_scheduler']['selected_test'])-
                 np.array(by[s,'fixed']['selected_test']) for s in [42,123,3407]])
 rng=np.random.default_rng(protocol['bootstrap_seed'])
 si=rng.integers(3,size=(5000,3)); ci=rng.integers(32,size=(5000,32))
 draws=np.array([delta[s][:,c].mean() for s,c in zip(si,ci)])
 bounds=core.ci(draws)
 extra=dict(row_type='contrast',comparison=protocol['contrast'],mean=float(delta.mean()),
            ci95_low=bounds[0],ci95_high=bounds[1],unit='cosine terminal reward, not accuracy',
            CI_method='5000 crossed seed/shared-condition resamples, supplemental seed2026110123')
 saved=core.read(RUN/'ablation/recomputed.json')
 # Do not recalculate the original two intervals or any training observations.
 contrasts=saved['contrasts']+[extra]
 receipt=dict(protocol=protocol,contrast=extra,
              inputs={str(p.relative_to(ROOT)):sha(p) for p in
                      [core.ABL/'threeway/results.json',RUN/'ablation/recomputed.json',
                       RUN/'real_image/summary.json',RUN/'checks.json']},
              original_RSI_intervals_unchanged=True)
 (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False))
 rows=saved['per_seed']+saved['aggregates']+contrasts+[
     dict(row_type='shared_allocation',job_id=x['job_id'],
          allocated_GPU_seconds=x['allocated_GPU_seconds'],allocated_scope=x['scope'])
     for x in saved['allocations']]
 target=OUT/'RL_RSI_ABLATION_RESULTS.csv';core.csvwrite(target,rows)
 core.emit(RUN,target.name,target.read_text())
 core.publish_ablation(RUN,saved['per_seed'],saved['aggregates'],saved['contrasts'],d)
 md=(ROOT/'reports/RL_RSI_ABLATION_RESULTS.md').read_text()
 md+='\n## Supplemental ordinary-scheduler contrast\n\n'
 md+=f"Scheduler minus Fixed = {extra['mean']:+.12f}, crossed 95% CI [{bounds[0]:.12f}, {bounds[1]:.12f}]. "
 md+='5000 draws, fixed supplemental seed2026110123; original two RSI intervals remain bitwise identical. '
 md+='This is exploratory, and also includes zero. No training or image measurement was repeated.\n'
 core.emit(RUN,'RL_RSI_ABLATION_RESULTS.md',md)
 tex=(ROOT/'reports/RL_RSI_ABLATION_RESULTS.tex').read_text()
 replacement=f"{extra['mean']:+.9f} {core.interval(bounds,digits=9)}"
 tex=tex.replace(f"{extra['mean']:+.9f} (descriptive)",replacement)
 core.emit(RUN,'RL_RSI_ABLATION_RESULTS.tex',tex)
 real=core.read(RUN/'real_image/summary.json');cost=real['resource_accounting']
 assert cost['gpu_seconds']==sum(x['GPU_count']*x['allocated_elapsed_seconds'] for x in cost['jobs'])==584
 md=(ROOT/'reports/BASELINE_REAL_IMAGE_RESULTS.md').read_text()
 md+='\n## Shared historical allocation (not per-condition costs)\n\n'
 md+='The matched-feedback campaign allocated **584 GPU-seconds (0.162222 GPU-hours)**: '
 md+='86793 pilot 129s; 86795 failed main 436s; 86796 repair 19s. CPU86794 consumed 32 allocated seconds and zero GPU. '
 md+='These jobs include pilots, failed execution, repeats, loading and writes. '
 md+='They cannot be partitioned exactly over C0-C5. The 160 C4 primary-generation records total '
 md+=f"{real['conditions'][4]['generation_primary_seconds']:.6f}s; this is included within, not added to, allocation. "
 md+='This recomputation job88317 allocated 20 CPU-job seconds, zero GPUs. '
 md+='Historical failure86795 is retained; no failed run was discarded.\n'
 core.emit(RUN,'BASELINE_REAL_IMAGE_RESULTS.md',md)
 rows=[dict(row_type='condition',**x) for x in real['conditions']]
 rows.append(dict(row_type='shared_allocation',condition='SHARED_ALLOCATION',
                  allocated_GPU_seconds=584,allocated_GPU_scope='86793+86795+86796, all historical matched controls including failures; count once',
                  new_GPU_seconds=0,evaluation_scope='cost only; no score or sample count'))
 target=OUT/'BASELINE_REAL_IMAGE_RESULTS.csv';core.csvwrite(target,rows)
 core.emit(RUN,target.name,target.read_text())
 tex=(ROOT/'reports/BASELINE_REAL_IMAGE_RESULTS.tex').read_text()
 tex=tex.replace('LPIPS unavailable.', 'LPIPS unavailable. Shared historical allocation:584 GPU-seconds including failed main; current CPU recomputation:0 GPU-seconds.')
 core.emit(RUN,'BASELINE_REAL_IMAGE_RESULTS.tex',tex)
 print(json.dumps(dict(status='PASS_SAVED_DATA_SUPPLEMENT',contrast=extra,new_GPU_seconds=0)))
if __name__=='__main__':main()
