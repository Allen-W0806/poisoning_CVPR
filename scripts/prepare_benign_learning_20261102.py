"""CPU deterministic brightness control and immutable next-run cohort lock.

No model inference, training, new annotations, scientific-reward replacement or
selection on the 160-source unopened reservation.
"""
import json,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import put,sha

def main(out):
 import numpy as np
 from PIL import Image
 pilot=ROOT/'results/execution_remediation_20261102_v1_pretrained_pilot';data=json.loads((pilot/'source_manifest.json').read_text())
 for role,rows in data.items():
  for r in rows:
   r['historical_role']=r.get('role');r['role']={'train':'development_training','validation':'development_selection_validation','assessment':'development_assessment'}[role]
 def y(x):return float(np.mean(x@np.array([.2126,.7152,.0722],dtype=np.float64)))
 def reward(source,edited):
  before=y(source);after=y(edited);target=min(1.,1.03*before)
  return abs(before-target)-abs(after-target)
 # Literal independent unit examples: RGB grey .5 -> target .515, delta .015.
 tests=dict(identity_zero=reward(np.full((2,2,3),.5),np.full((2,2,3),.5))==0.,target_reward=np.isclose(reward(np.full((2,2,3),.5),np.full((2,2,3),.515)),.015,atol=1e-14),darkening_negative=np.isclose(reward(np.full((2,2,3),.5),np.full((2,2,3),.485)),-.015,atol=1e-14),black_identity=reward(np.zeros((1,1,3)),np.zeros((1,1,3)))==0.)
 controls=[]
 for role,rows in data.items():
  for r in rows:
   image=ROOT/r['image_path'];assert sha(image)==r['image_sha256']
   with Image.open(image) as im:x=np.asarray(im.convert('RGB'),dtype=np.float64)/255.
   before=y(x);delta=min(1.,1.03*before)-before
   # Quantize to the same RGB uint8 domain as generated PNG evaluation.
   bright=np.round(np.clip(x+delta,0,1)*255)/255.;dark=np.round(np.clip(x-delta,0,1)*255)/255.
   controls.append(dict(role=role,sample_id=r['sample_id'],image_id=r['image_id'],image_sha256=r['image_sha256'],source_Y=before,target_Y=min(1.,1.03*before),identity_reward=reward(x,x),deterministic_bright_reward=reward(x,bright),deterministic_dark_reward=reward(x,dark)))
 tests['all_real_identity_zero']=all(r['identity_reward']==0 for r in controls)
 tests['all_real_brightness_improves_proxy']=all(r['deterministic_bright_reward']>0 for r in controls)
 tests['all_real_darkening_worsens_proxy']=all(r['deterministic_dark_reward']<0 for r in controls)
 ids={k:{str(r['image_id']) for r in v} for k,v in data.items()};tests['cohorts_ID_disjoint']=all(not ids[a]&ids[b] for a,b in [('train','validation'),('train','assessment'),('validation','assessment')])
 candidate=ROOT/'results/execution_remediation_20261102_v1/prospective/candidate_content_manifest.json'
 lock=dict(task='Next isolated benign brightness learning test; not original scientific reward',training=data['train'],selection_validation=data['validation'],development_assessment=data['assessment'],cohort_provenance='ALL historically exposed development data; assessment never becomes a locked final test',source_manifest_sha256=sha(pilot/'source_manifest.json'),candidate_reservation_sha256=sha(candidate),unopened_160_role='Preserved and excluded from this control; not silently consumed as a test',independent_development_evaluation='REQUIRED BEFORE MODEL LEARNING; a new source-disjoint cohort with complete exposure/selection lineage and lock must be admitted, no eligible independent cohort asserted',baseline_checkpoint=str((pilot/'shared_initial/training_state.pt').relative_to(ROOT)),baseline_checkpoint_sha256=sha(pilot/'shared_initial/training_state.pt'),compute='Future single learning arm + matched frozen checkpoint; no four-arm campaign; cap must be separately preregistered within live residual budget',acceptance=dict(numerical='ordinary repeat, full accumulation/raw-gradient and checkpoint continuation exact; tolerant checks reported separately',reward_units='absolute normalized mean-luminance error reduction in [0,1] luminance units, reward bounded [-1,1], not percent accuracy',learning='Preregister one update and one fixed evaluation; trained minus frozen mean reward must be positive with paired source-bootstrap 95% CI lower bound >0 and improve absolute target error; no step/seed/LR search on assessment',semantic='two real independent reviewers + adjudication for changed semantic cases; no invented labels; brightness score is insufficient',budget='Match initial checkpoint, source IDs/seeds, inference settings, evaluation query count; include generation, shared prefix and failed allocations',stop='Failed learning/numerical criteria preserved; no more steps merely to improve proxy'),eligible_now=False)
 put(out/'next_benign_cohort_lock.json',lock);put(out/'deterministic_benign_control.json',dict(tests={k:bool(v) for k,v in tests.items()},rows=controls,new_GPU_seconds=0,new_model_calls=0,learning_performed=False,metric_code='this script; literal RGB unit cases plus source-content-bound CPU controls',semantic_validity='UNREVIEWED',scientific_reward_release=False))
 print(json.dumps(dict(tests={k:bool(v) for k,v in tests.items()},sources=len(controls),learning_eligible=False)))
if __name__=='__main__':main(ROOT/'results/numerical_root_cause_20261102_v1')
