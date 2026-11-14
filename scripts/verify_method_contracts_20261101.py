"""CPU checks of existing guarded diagnostic interfaces; no scientific loss edit."""
import copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from PIL import Image
from src.rl.reward import RewardWeights
from src.rl.reward_v2 import RewardConfigV2,aggregate_diagnostic
from src.evaluation.semantic_validation_v2 import decision_v2
from src.rl.diffusion_rl_trainer_v2 import GuardedDiffusionRLTrainer,validate_unique_batch
def verify(run):
 rows=[];im=Image.new('RGB',(8,8),'white');record=dict(sample_id='SYNTHETIC_PIXEL_IDENTITY_ONLY',question='Synthetic identity fixture');validity=decision_v2(im,im,record,dict(score=1.),{});cfg=RewardConfigV2(**RewardWeights().__dict__);receipt=aggregate_diagnostic(cfg,cfg.to_dict(),validity,1.,0.,1/3)
 rows.append(dict(test='v2_verified_pixel_identity_consistency_credit',passed=receipt['answer_consistency']==1. and abs(receipt['reward']-1.4666666666666666)<1e-15,observed=receipt['reward'],training_eligible=receipt['training_eligible']))
 recorded=cfg.to_dict();recorded['accuracy_weight']=.4
 try:aggregate_diagnostic(cfg,recorded,validity,1.,0.,1/3);rejected=False
 except ValueError:rejected=True
 rows.append(dict(test='v2_executed_metadata_mismatch_rejected',passed=rejected))
 t=dict(trajectory_id='T0',sample_id='S0',image_id='I0',source_image_sha256='0'*64,reward_receipt=dict(training_eligible=True));c=dict(source_image_sha256='0'*64);episode=(t,c,1.)
 trainer=GuardedDiffusionRLTrainer.__new__(GuardedDiffusionRLTrainer)
 try:trainer.update([episode]);rejected=False
 except RuntimeError:rejected=True
 rows.append(dict(test='v2_guard_rejects_even_true_receipt_before_optimizer',passed=rejected))
 try:validate_unique_batch([episode,copy.deepcopy(episode)]);rejected=False
 except ValueError:rejected=True
 rows.append(dict(test='v2_duplicate_logical_source_batch_rejected',passed=rejected))
 result=dict(status='PASS' if all(r['passed'] for r in rows) else 'FAIL',checks=rows,scope='existing diagnostic v2 interface only; not a released training reward; legacy historical caller remains separate',new_human_labels=0,new_model_calls=0,new_GPU_seconds=0)
 with (run/'versioned_contract_checks.json').open('x') as f:json.dump(result,f,indent=2)
 assert result['status']=='PASS';return result
