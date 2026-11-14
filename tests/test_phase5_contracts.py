import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import yaml
from src.utils.scratch import scratch_directory
from src.data.dataset import sha256
from src.data.semantic_contract_v2 import from_original,validate,eligible_for_semantic_study
from src.data.dataset_v2 import VerifiedVQADataset,validate_content_disjoint
from src.evaluation.semantic_validation_v2 import decision_v2,validate_validity
from src.rl.reward_v2 import RewardConfigV2,aggregate_diagnostic,saturation_and_redundancy
from src.rl.diffusion_rl_trainer_v2 import validate_unique_batch,GuardedDiffusionRLTrainer
from src.rl.diffusion_policy_v2 import ContentBoundDiffusionPolicy

ROOT=Path(__file__).resolve().parents[1]

def fixture(root):
    p=root/'COCO_train2014_000000000001.jpg';Image.new('RGB',(8,8),'white').save(p)
    row={'sample_id':'11','image_id':'1','image_path':p.name,'image_sha256':sha256(p),
         'question':'Is the bus red?','answers':['yes']*10,'split':'train','source_question_image_id':'1'}
    row['semantic_contract']=from_original(row)
    manifest=root/'train.json';manifest.write_text(json.dumps([row]),encoding='utf8')
    return row,manifest,p

class Phase5ContractTests(unittest.TestCase):
    def config(self):return RewardConfigV2.from_dict(yaml.safe_load((ROOT/'configs/phase5_reward.yaml').read_text()))
    def evidence(self,status,verified,score,provisional=False):
        return {'status':status,'verified':verified,'score':score,'provisionally_valid':provisional,'source':'unit_exact_oracle'}
    def aggregate(self,v,**kw):
        c=self.config();return aggregate_diagnostic(c,c.to_dict(),v,kw.get('a',1.),kw.get('d',0.),kw.get('cost',1/3))

    def test_explicit_coefficients_are_executed_and_recorded(self):
        c=self.config();row=self.aggregate(self.evidence('VALID',True,1.))
        self.assertAlmostEqual(row['reward'],1.5-.1/3)
        altered={**c.to_dict(),'accuracy_weight':.2};new=RewardConfigV2.from_dict(altered)
        result=aggregate_diagnostic(new,altered,self.evidence('VALID',True,1.),1.,0.,1/3)
        self.assertAlmostEqual(result['reward'],1.2-.1/3)
        with self.assertRaises(ValueError):aggregate_diagnostic(c,altered,self.evidence('VALID',True,1.),1.,0.,1/3)
        for bad in [None,True,float('nan'),-1.]:
            with self.assertRaises((ValueError,TypeError)):RewardConfigV2.from_dict({**c.to_dict(),'accuracy_weight':bad})
        with self.assertRaises(ValueError):RewardConfigV2.from_dict({'accuracy_weight':.5})

    def test_valid_invalid_unknown_disputed_are_separate(self):
        valid=self.aggregate(self.evidence('VALID',True,1.))
        invalid=self.aggregate(self.evidence('INVALID',True,0.))
        unknown=self.aggregate(self.evidence('UNKNOWN',False,0.))
        provisional=self.aggregate(self.evidence('UNKNOWN',False,.5,True))
        disputed=self.aggregate(self.evidence('DISPUTED',False,0.))
        self.assertEqual(valid['invalid_penalty'],0)
        self.assertTrue(valid['consistency_eligible'])
        self.assertEqual(invalid['reward'],-1)
        self.assertFalse(unknown['consistency_eligible']);self.assertTrue(provisional['consistency_eligible'])
        self.assertEqual(unknown['validity_status'],'UNKNOWN');self.assertEqual(disputed['validity_status'],'DISPUTED')
        self.assertFalse(provisional['training_eligible'])
        self.assertEqual(unknown['raw_components']['answer_consistency'],1.)
        self.assertEqual(unknown['answer_consistency'],0.)
        self.assertTrue(unknown['saturated_low'])
        info=saturation_and_redundancy([invalid,unknown,disputed]);self.assertEqual(info['at_lower_bound'],3)

    def test_identity_retains_consistency_credit(self):
        image=Image.new('RGB',(8,8),'white')
        decision=decision_v2(image,image.copy(),{'sample_id':'1','question':'What color?'},{'score':1.},{})
        self.assertEqual(decision['status'],'VALID')
        self.assertTrue(self.aggregate(decision)['consistency_eligible'])
        self.assertAlmostEqual(self.aggregate(decision)['reward'],1.5-.1/3)

    def test_ambiguous_verification_rejected_before_reward(self):
        image=Image.new('RGB',(8,8),'white');edit=Image.new('RGB',(8,8),'gray')
        with self.assertRaises(ValueError):decision_v2(image,edit,{'sample_id':'1','question':'x'},{'score':1.},{'parse_ok':False})
        for key,bad in [('verified',1),('score',float('nan')),('status','VALID')]:
            value=self.evidence('UNKNOWN',False,0.);value[key]=bad
            with self.assertRaises(ValueError):validate_validity(value)
        with self.assertRaises(ValueError):self.aggregate(self.evidence('VALID',True,1.),d=float('nan'))

    def test_semantic_contract_does_not_copy_unknown_facts(self):
        with scratch_directory() as root:
            row,_,_=fixture(root);c=row['semantic_contract']
            self.assertIsNone(c['edited_factual_answer']['value']);self.assertIsNone(c['supplied_training_annotation']['value'])
            self.assertEqual(c['candidate_semantic_categories'],['color','vehicle'])
            self.assertFalse(eligible_for_semantic_study(c))
            bad=copy.deepcopy(c);bad['edited_factual_answer']['value']=['yes']
            with self.assertRaises(ValueError):validate(bad)
            bad=copy.deepcopy(c);bad['image_question_consistency']={'status':'VERIFIED','value':'SI-V','source':None}
            with self.assertRaises(ValueError):validate(bad)
            bad=copy.deepcopy(c);bad['edited_factual_answer']={'status':'DISPUTED','value':['yes'],'source':{'kind':'review','artifact':'dispute.json'}}
            with self.assertRaises(ValueError):validate(bad)
            partial=copy.deepcopy(c);partial['semantic_category']='vehicle'
            partial['image_question_consistency']={'status':'VERIFIED','value':'SC','source':{'kind':'review','artifact':'sc.json'}}
            self.assertFalse(eligible_for_semantic_study(partial))
            complete=copy.deepcopy(partial);complete['edited']={'image_id':'edited_1','sha256':row['image_sha256'],'image_path':row['image_path']}
            source={'kind':'exact_identity_fixture','artifact':row['image_path']}
            complete['edited_factual_answer']={'status':'VERIFIED','value':['yes'],'source':source}
            complete['visual_edit_validity']={'status':'VERIFIED','value':'VALID','source':source}
            complete['verification']={'status':'VERIFIED','value':True,'source':source}
            self.assertTrue(eligible_for_semantic_study(complete))
            for bad_answer in [[None],[''],[' '],[1]]:
                bad=copy.deepcopy(complete);bad['edited_factual_answer']['value']=bad_answer
                with self.assertRaises(ValueError):validate(bad)

    def test_runtime_manifest_and_content_checks(self):
        with scratch_directory() as root:
            row,m,p=fixture(root);d=VerifiedVQADataset(m,root,'train',sha256(m),'editor')
            self.assertEqual(d[0]['image_id'],'1')
            Image.new('RGB',(8,8),'black').save(p)
            with self.assertRaises(ValueError):d[0]
        with scratch_directory() as root:
            row,m,p=fixture(root);d=VerifiedVQADataset(m,root,'train',sha256(m))
            m.write_text('[]')
            with self.assertRaises(ValueError):d[0]

    def test_runtime_join_and_content_split_guards(self):
        with scratch_directory() as root:
            row,m,p=fixture(root);row['source_question_image_id']='2';m.write_text(json.dumps([row]))
            with self.assertRaises(ValueError):VerifiedVQADataset(m,root,'train',sha256(m))
            with self.assertRaises(PermissionError):VerifiedVQADataset(m,root,'test',sha256(m),'editor')
            other={**row,'image_id':'2'}
            with self.assertRaises(ValueError):validate_content_disjoint({'train':[row],'test':[other]})

    def test_batch_duplicates_rejected_before_optimizer(self):
        episodes=[({'trajectory_id':str(i),'sample_id':str(i),'image_id':str(i),'source_image_sha256':f'{i:064x}'},
                   {'source_image_sha256':f'{i:064x}'},0.) for i in range(10)]
        self.assertTrue(validate_unique_batch(episodes))
        for field in ['trajectory_id','sample_id','image_id']:
            bad=copy.deepcopy(episodes);bad[1][0][field]=bad[0][0][field]
            trainer=GuardedDiffusionRLTrainer.__new__(GuardedDiffusionRLTrainer)
            with patch('src.rl.diffusion_rl_trainer.DiffusionRLTrainer.update') as backend:
                with self.assertRaises(ValueError):trainer.update(bad)
                backend.assert_not_called()
        bad=copy.deepcopy(episodes);bad[1][0]['source_image_sha256']=bad[0][0]['source_image_sha256'];bad[1][1]['source_image_sha256']=bad[0][1]['source_image_sha256']
        with self.assertRaises(ValueError):validate_unique_batch(bad)
        trainer=GuardedDiffusionRLTrainer.__new__(GuardedDiffusionRLTrainer)
        with patch('src.rl.diffusion_rl_trainer.DiffusionRLTrainer.update') as backend:
            with self.assertRaises(RuntimeError):trainer.update(episodes)
            backend.assert_not_called()

    def test_policy_provenance_rejects_missing_source_hash_before_model_use(self):
        policy=ContentBoundDiffusionPolicy.__new__(ContentBoundDiffusionPolicy)
        with patch('src.rl.diffusion_policy.DiffusionPolicy.condition') as original:
            with self.assertRaises(ValueError):policy.condition({'sample_id':'x'})
            original.assert_not_called()
            original.return_value={'sample_id':'x'}
            condition=policy.condition({'sample_id':'x','image_sha256':'a'*64})
            self.assertEqual(condition['source_image_sha256'],'a'*64)

if __name__=='__main__':unittest.main()
