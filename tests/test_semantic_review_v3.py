import unittest
from src.evaluation.independent_review_v3 import ATTRS,consensus,validate_row

class ReviewV3(unittest.TestCase):
 def setUp(self):
  self.case=dict(source_image_id='SYNTHETIC_ONLY',question_id='SYNTHETIC_Q',original_sha256='a'*64,edited_sha256='b'*64)
  self.row={**self.case,'reviewer_id':'SYNTHETIC_R1','independence_attestation':'yes','reviewed_utc':'2026-11-02T00:00:00Z',**{k:'yes' for k in ATTRS},'semantic_label':'valid','confidence':'high','failure_reason':'','notes':'UNIT FIXTURE ONLY'}
 def test_missing_text_label_rejected(self):
  x=dict(self.row);del x['text_preserved']
  with self.assertRaises(ValueError):validate_row(x,self.case)
 def test_real_UI_ambiguous_export_is_accepted_and_normalized(self):
  raw={**self.row,'text_preserved':'ambiguous','semantic_label':'ambiguous','failure_reason':'text uncertain'}
  self.assertEqual(validate_row(raw,self.case)['text_preserved'],'uncertain')
  self.assertEqual(raw['text_preserved'],'ambiguous')
 def test_one_human_is_pending(self):self.assertEqual(consensus([self.row],self.case)['status'],'PENDING_TWO_HUMANS')
 def test_two_distinct_agree(self):self.assertEqual(consensus([self.row,{**self.row,'reviewer_id':'SYNTHETIC_R2'}],self.case)['status'],'HUMAN_REVIEWED')
 def test_duplicate_reviewer_rejected(self):
  with self.assertRaises(ValueError):consensus([self.row,self.row],self.case)
 def test_text_conflict_blocks_valid_label(self):
  with self.assertRaises(ValueError):validate_row({**self.row,'text_preserved':'no'},self.case)
 def test_disagreement_needs_third_human(self):
  other={**self.row,'reviewer_id':'SYNTHETIC_R2','semantic_label':'invalid','text_preserved':'no','failure_reason':'text changed'}
  self.assertEqual(consensus([self.row,other],self.case)['status'],'NEEDS_HUMAN_ADJUDICATION')
  with self.assertRaises(ValueError):consensus([self.row,other],self.case,other)
  self.assertEqual(consensus([self.row,other],self.case,{**other,'reviewer_id':'SYNTHETIC_R3'})['semantic_label'],'invalid')
 def test_synthetic_identifier_cannot_certify_real_case(self):
  case={**self.case,'source_image_id':'123'}
  with self.assertRaises(ValueError):validate_row({**self.row,**case},case)

if __name__=='__main__':unittest.main()
