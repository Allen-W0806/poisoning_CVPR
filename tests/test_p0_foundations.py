"""P0 gates: source leakage, response masking, controller restore, and budgets."""
import copy
import json
from pathlib import Path
import unittest

import numpy as np
import torch

from src.data.p0_protocol import validate_splits,metric_summary
from src.training.clean_vlm_pipeline import response_labels
from src.mini_pilot.controller_v3 import LearnedController,controller_state
from src.mini_pilot.engine import compare_trees


class SplitAndStatsTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'sample_id':str(i),'image_id':str(i),'image_sha256':str(i)*64,
                    'answers':['yes']*10,'training_answer':'yes'} for i in range(1,5)]

    def test_source_content_question_disjointness(self):
        data={name:[row] for name,row in zip(('train','dev','test','transfer'),self.rows)}
        self.assertEqual(validate_splits(data,{key:1 for key in data})['source_IDs'],4)
        for field in ('sample_id','image_id','image_sha256'):
            bad=copy.deepcopy(data);bad['test'][0][field]=bad['train'][0][field]
            with self.assertRaises(ValueError):validate_splits(bad,{key:1 for key in data})

    def test_require_real_qa_answers(self):
        row=copy.deepcopy(self.rows[0]);row['answers']=[]
        with self.assertRaises(ValueError):validate_splits({'train':[row]},{'train':1})

    def test_source_bootstrap_is_seeded_and_has_correct_units(self):
        result=metric_summary([0.,1.,.5,.5])
        self.assertEqual(result['mean'],.5)
        self.assertEqual(result,metric_summary([0.,1.,.5,.5]))
        self.assertEqual(result['training_seed_uncertainty'],'NOT_ESTIMATED')
        with self.assertRaises(ValueError):metric_summary([float('nan')])


class CleanResponseLabelsTests(unittest.TestCase):
    def test_only_assistant_response_is_supervised(self):
        prompt=torch.tensor([[10,20,30]]);full=torch.tensor([[10,20,30,40,50]])
        labels=response_labels(prompt,full)
        self.assertEqual(labels.tolist(),[[-100,-100,-100,40,50]])
        self.assertTrue(torch.equal(full,torch.tensor([[10,20,30,40,50]])))

    def test_wrong_prefix_empty_response_and_multi_image_rejected(self):
        for prompt,full in [(torch.tensor([[1,2]]),torch.tensor([[1,3,4]])),
                            (torch.tensor([[1,2]]),torch.tensor([[1,2]])),
                            (torch.ones(2,2,dtype=torch.long),torch.ones(2,3,dtype=torch.long))]:
            with self.assertRaises(ValueError):response_labels(prompt,full)


class LearnedOuterRestoreTests(unittest.TestCase):
    def test_real_parameter_update_and_next_action_update_restore(self):
        torch.manual_seed(42);first=LearnedController(123)
        state=controller_state(.1,.2,.3,1e-5,.9);config={'lr':1e-5,'weight_sem':2.}
        first.choose(state,config);receipt=first.learn(.02,{'scope':'synthetic_benign_CPU_fixture'})
        self.assertGreater(receipt['controller_max_parameter_change'],0)
        saved=copy.deepcopy(first.state_dict())
        second=LearnedController(1);second.load_state_dict(saved)
        self.assertTrue(compare_trees(saved,second.state_dict())['exact'])
        self.assertEqual(first.choose(state,config),second.choose(state,config))
        first.learn(-.01,{'scope':'fixture'});second.learn(-.01,{'scope':'fixture'})
        self.assertTrue(compare_trees(first.state_dict(),second.state_dict())['exact'])

    def test_incomplete_or_unsettled_restore_rejected(self):
        outer=LearnedController(42);saved=copy.deepcopy(outer.state_dict());saved.pop('memory')
        with self.assertRaises(ValueError):outer.load_state_dict(saved)
        saved=copy.deepcopy(outer.state_dict());outer.choose(controller_state(0,0,0,1e-5,1),{'lr':1e-5,'weight_sem':2.})
        with self.assertRaises(RuntimeError):outer.load_state_dict(saved)


class FixedProtocolTests(unittest.TestCase):
    def test_quality_steps_and_learning_gate_are_explicit(self):
        root=Path(__file__).resolve().parents[1]
        protocol=json.loads((root/'configs/p0_foundations_20261111.json').read_text())
        self.assertGreaterEqual(protocol['editor_resolution'],512)
        self.assertTrue(all(row['steps']>=20 for row in protocol['editor_configs']))
        self.assertFalse(protocol['P1_authorized_by_this_protocol'])
        self.assertLessEqual(protocol['allocation_cap_seconds'],1200)


if __name__=='__main__':unittest.main()
