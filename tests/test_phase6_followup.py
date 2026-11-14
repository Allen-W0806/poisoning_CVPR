"""Only new functionality; fixtures are SYNTHETIC and never review evidence."""
import copy
import json
import unittest
from unittest.mock import patch
from pathlib import Path
from src.progression.review_workflow import DIMENSIONS,validate_submission,analyze_submissions
from src.utils.scratch import scratch_directory

def fixture():
    pair={'pair_id':'SYNTHETIC_0','original_image_sha256':'a'*64,'edited_image_sha256':'b'*64,
       'automatic_metrics':{'ssim':.8,'historical_scalar_reward':-.2},'automatic_edited_answer':'yes'}
    manifest={'pairs':[pair]}
    row={k:pair[k] for k in ['pair_id','original_image_sha256','edited_image_sha256']}
    row.update(dimensions={k:v[1] for k,v in DIMENSIONS.items()},edited_factual_answer={'status':'REVIEWED','value':['yes']},reason='SYNTHETIC fixture, not human evidence')
    sub={'schema_version':'independent_edit_review_v1','provenance':{'reviewer_id':'SYNTHETIC_A','review_session_id':'fixture_A',
      'reviewed_at':'2026-10-28T10:00:00Z','reviewer_type':'human','independence_attestation':True,'model_assisted':False},'reviews':[row]}
    return manifest,sub

class FollowupReviewTests(unittest.TestCase):
    def test_empty_submissions_never_create_labels_or_agreement(self):
        manifest,_=fixture();result=analyze_submissions(manifest,[])
        self.assertEqual(result['independently_reviewed_consensus_pairs'],0)
        self.assertIsNone(result['proxy_answer_match']['rate']);self.assertFalse(result['training_reward_released'])

    def test_image_mismatch_and_unknown_nonnull_answers_are_rejected(self):
        manifest,sub=fixture();bad=copy.deepcopy(sub);bad['reviews'][0]['edited_image_sha256']='c'*64
        with self.assertRaises(ValueError):validate_submission(bad,manifest)
        bad=copy.deepcopy(sub);bad['reviews'][0]['edited_factual_answer']['status']='UNKNOWN'
        with self.assertRaises(ValueError):validate_submission(bad,manifest)

    def test_self_declaration_does_not_equal_operator_verified_truth(self):
        manifest,a=fixture();b=copy.deepcopy(a);b['provenance'].update(reviewer_id='SYNTHETIC_B',review_session_id='fixture_B')
        result=analyze_submissions(manifest,[('a',a),('b',b)])
        self.assertEqual(result['provenance_consistent_submissions'],2)
        self.assertEqual(result['independently_reviewed_consensus_pairs'],0)
        self.assertEqual(result['reviewer_pair_agreement']['instruction_completion'][0]['agreement'],1)

    def test_operator_attested_disagreement_retained_and_reward_stays_blocked(self):
        manifest,a=fixture();b=copy.deepcopy(a);b['provenance'].update(reviewer_id='SYNTHETIC_B',review_session_id='fixture_B')
        att={'verified_by':'SYNTHETIC_OPERATOR','evidence_reference':'SYNTHETIC_ONLY','verified_human_reviewer_ids':['SYNTHETIC_A','SYNTHETIC_B']}
        result=analyze_submissions(manifest,[('a',a),('b',b)],att)
        self.assertEqual(result['independently_reviewed_consensus_pairs'],1);self.assertFalse(result['scientific_pilot_ready'])
        b['reviews'][0]['dimensions']['instruction_completion']='NOT_FULFILLED'
        result=analyze_submissions(manifest,[('a',a),('b',b)],att)
        self.assertEqual(result['independently_reviewed_consensus_pairs'],0);self.assertEqual(len(result['disagreements']),1)

    def test_malformed_submission_is_recorded_without_crashing_monitor(self):
        manifest,_=fixture();result=analyze_submissions(manifest,[('bad.json',None)])
        self.assertEqual(len(result['validation_errors']),1)

    def test_full_dashboard_rebuilds_status_and_measurements_without_releasing_science(self):
        from src.progression.dashboard_full import normalize
        root=Path(__file__).resolve().parents[1]
        state={'updated_utc':'SYNTHETIC_STAMP','source_commit':'fixture','review_analysis':{'independently_reviewed_consensus_pairs':0},
            'followup':{'experiment_id':'SYNTHETIC_FINALIZED','results':{'reward_separation':{'n':500,'lower_clipped':230}}},
            'jobs':[],'live':{},'experiments':[{'experiment_id':'SYNTHETIC_FINALIZED','created_at':'fixture','ended_at':'fixture','status':'COMPLETED','metrics_present':True,'scope':'engineering'}],
            'stage_status':{'stages':{'SYNTHETIC_STAGE':{'state':'BLOCKED','reason':'fixture'}}},
            'stage_receipts':{},'next_eligible_task':'Independent review','blockers':['independent truth'],
            'actual':{'new_gpu_hours':0},'authorized_budget':{'scientific_gpu_seconds':0},'current_phase':'fixture','package_status':{}}
        data=normalize(root,state)
        self.assertEqual(data['phase6_stages'][0]['status'],'BLOCKED')
        self.assertEqual(data['followup']['experiment_id'],'SYNTHETIC_FINALIZED')
        self.assertEqual(data['headline_metrics']['verified_semantic_samples'],0)
        self.assertEqual(data['scientific_decision'],'NO-GO')
        self.assertEqual(data['current']['engineering_gate'],'NEEDS_VERIFICATION')
        self.assertEqual(data['live']['verified_results'],[])

    def test_finalization_notification_refreshes_actual_snapshot_and_bounds_retries(self):
        from src.progression.events import finalized_event
        with scratch_directory() as root:
            (root/'configs').mkdir();(root/'configs/phase6.json').write_text('{}')
            with patch('src.progression.dashboard.refresh',return_value={'snapshot_utc':'SYNTHETIC_TIMESTAMP'}) as refresh:
                finalized_event(root,'SYNTHETIC_RUN');self.assertEqual(refresh.call_count,1)
            with patch('src.progression.dashboard.refresh',side_effect=OSError(5,'SYNTHETIC transient')) as refresh:
                finalized_event(root,'SYNTHETIC_RUN');self.assertEqual(refresh.call_count,3)
                finalized_event(root,'SYNTHETIC_RUN');self.assertEqual(refresh.call_count,4)
