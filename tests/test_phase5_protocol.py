import copy
import json
import unittest
from pathlib import Path
from src.utils.scratch import scratch_directory
from src.data.prepare_v2 import prepare_v2
from src.data.dataset_v2 import validate_content_disjoint
from src.evaluation.phase5_comparability import require_launch_ready,compare_checkpoints
from src.evaluation.security_readonly_v2 import validate_lineage,aggregate_fixed_observations,QWEN_REV
from src.rsi.harness.harness_spec import HarnessSpec
from src.rsi.harness.revision_validator import validate_revision

class Phase5ProtocolTests(unittest.TestCase):
    def test_v2_groups_duplicate_content_and_never_recycles_old_test(self):
        old={s:[] for s in ['train','validation','test']}
        for i,split in enumerate(['train','train','validation','validation','validation','test']):
            old[split].append({'sample_id':str(i),'image_id':str(i),'image_sha256':f'{i:064x}',
                'question':'Is the bus red?','answers':['yes']*10,'split':split,'image_path':f'COCO_train2014_{i:012d}.jpg'})
        old['test'][0]['image_sha256']=old['train'][0]['image_sha256']
        new,p=prepare_v2(old,{'2'})
        self.assertTrue(validate_content_disjoint(new));self.assertEqual(sum(map(len,new.values())),6)
        self.assertIn('0',{r['sample_id'] for r in new['test']});self.assertIn('5',{r['sample_id'] for r in new['test']})
        self.assertIn('2',{r['sample_id'] for r in new['development']})
        self.assertEqual(p['category_counts']['train']['independently_eligible'],0)

    def test_preintervention_check_rejects_existing_branch_divergence(self):
        import torch
        a={'adapter':{'x':torch.tensor([0.])},'optimizer':{},'scheduler':{},'step':4,'version':4,'contract':{'algorithm':'DDPO_SF_v1'},
            'consumed':{'a'},'rng_torch':torch.tensor([1]),'rng_cuda':[],'rng_numpy':('x',),'rng_python':()}
        with scratch_directory() as root:
            x=root/'a.pt';y=root/'b.pt';torch.save(a,x);torch.save(a,y)
            self.assertTrue(compare_checkpoints(x,y)['all_required_fields_exact'])
            self.assertFalse(compare_checkpoints(x,y)['causal_intervention_comparison_ready'])
            empty=copy.deepcopy(a);empty['adapter']={};empty['contract']={};torch.save(empty,y);torch.save(empty,x)
            self.assertFalse(compare_checkpoints(x,y)['all_required_fields_exact'])
            torch.save(a,x)
            b=copy.deepcopy(a);b['adapter']['x']+=1e-5;torch.save(b,y)
            result=compare_checkpoints(x,y);self.assertFalse(result['causal_intervention_comparison_ready'])
            self.assertGreater(result['adapter_max_abs_difference'],0)

    def test_launch_guard_requires_all_independent_evidence(self):
        readiness={k:True for k in ['independent_editing_validity','validated_training_reward','content_disjoint_splits',
           'active_harness_tests','conventional_scheduler_baseline','pre_intervention_comparability','private_gate_isolation','frozen_resource_protocol']}
        readiness['independent_editing_validity']=False
        with self.assertRaises(RuntimeError):require_launch_ready(readiness)
        with self.assertRaises(ValueError):require_launch_ready({'automatic_proxy_agrees':True})

    def test_security_fixture_does_not_become_posttraining_result(self):
        node={'artifact_id':'pretrained_qwen','stage':'pretrained','model_id':'Qwen/Qwen3-VL-2B-Instruct','revision':QWEN_REV,
          'parent_artifact_id':None,'training_data_sha256':None,'authorization_id':None,'artifact_manifest_sha256':'a'*64}
        protocol={'definition_id':'fixed_external_target_observation_v1','evaluation_artifact_id':'synthetic_fixture',
          'source':'https://arxiv.org/html/2506.07214','independent_evaluator':'external_fixture','fixture_kind':'synthetic_engineering'}
        rows=[{'sample_id':'1','source_image_id':'1','condition':'SI-V','model_artifact_id':node['artifact_id'],
            'definition_id':protocol['definition_id'],'predefined_target_observed':True,'phase':'inference_only'}]
        result=aggregate_fixed_observations(node,protocol,rows)
        self.assertEqual(result['by_phase']['inference_only']['supplied_conditioned_behavior_rate'],1.)
        self.assertIsNone(result['by_phase']['post_training']['supplied_conditioned_behavior_rate'])
        self.assertIsNone(result['post_training_ASR']);self.assertFalse(result['real_security_result'])
        with self.assertRaises(ValueError):validate_lineage({**node,'model_id':'Gemini'})
        bad=copy.deepcopy(rows);bad[0]['factual_reference_score']=1.
        with self.assertRaises(ValueError):aggregate_fixed_observations(node,protocol,bad)

    def test_revision_category_cannot_hide_undeclared_changes(self):
        parent=HarnessSpec('initial');candidate=HarnessSpec('candidate',('content_hash',),memory_mode='summary')
        p={'parent_version':'H0000','candidate':candidate.to_dict(),'change':'test','supporting_evidence':{'observed':'x'},
            'expected_effect':'x','affected_modules':['workflow.memory_mode'],'regression_tests':['x'],'category':'experience_memory'}
        with self.assertRaises(ValueError):validate_revision(p,'H0000',parent)

    def test_inference_and_posttraining_populations_never_pool(self):
        node={'artifact_id':'clean_fixture','stage':'clean_finetuned','model_id':'Qwen/Qwen3-VL-2B-Instruct','revision':'fixed_clean_revision',
          'parent_artifact_id':'pretrained','training_data_sha256':'a'*64,'authorization_id':'synthetic_unit_fixture','artifact_manifest_sha256':'b'*64}
        protocol={'definition_id':'fixed','evaluation_artifact_id':'synthetic_fixture','source':'https://arxiv.org/html/2506.07214',
          'independent_evaluator':'external_fixture','fixture_kind':'synthetic_engineering'}
        rows=[{'sample_id':str(i),'source_image_id':str(i),'condition':'SI-V','model_artifact_id':'clean_fixture',
            'definition_id':'fixed','predefined_target_observed':i==0,'phase':phase}
            for i,phase in enumerate(['inference_only','post_training'])]
        result=aggregate_fixed_observations(node,protocol,rows)
        self.assertEqual(result['by_phase']['inference_only']['supplied_conditioned_behavior_rate'],1.)
        self.assertEqual(result['by_phase']['post_training']['supplied_conditioned_behavior_rate'],0.)
        self.assertNotIn('supplied_conditioned_behavior_rate',result)
        self.assertIsNone(result['post_training_ASR'])
        sc=[{**rows[0],'sample_id':str(i),'condition':'SC','source_image_id':'shared_image',
            'factual_reference_score':1.,'factual_reference_verified':True} for i in [5,6]]
        coverage=aggregate_fixed_observations(node,protocol,sc)['by_phase']['inference_only']['semantic_consistency_accuracy']
        self.assertEqual(coverage['denominator'],2);self.assertEqual(coverage['scored_unique_source_images'],1)
        sc[1]['source_image_id']='other_image'
        self.assertEqual(aggregate_fixed_observations(node,protocol,sc)['by_phase']['inference_only']['semantic_consistency_accuracy']['scored_unique_source_images'],2)

if __name__=='__main__':unittest.main()
