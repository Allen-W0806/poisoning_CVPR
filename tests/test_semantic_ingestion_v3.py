import csv,hashlib,json,unittest
from src.utils.scratch import scratch_directory
from src.evaluation.independent_review_v3 import ATTRS
from scripts.ingest_semantic_review_v3 import ingest
class Ingestion(unittest.TestCase):
 def fixture(self,p):
  kit=p/'kit';kit.mkdir();(kit/'a.png').write_bytes(b'fixture-original');(kit/'b.png').write_bytes(b'fixture-edited')
  c=dict(source_image_id='SYNTHETIC_ONLY',question_id='SYNTHETIC_Q',original='a.png',edited='b.png',original_sha256=hashlib.sha256(b'fixture-original').hexdigest(),edited_sha256=hashlib.sha256(b'fixture-edited').hexdigest());(kit/'blinded_manifest.json').write_text(json.dumps({'cases':[c]}));return kit,c
 def test_empty_input_never_certifies(self):
  with scratch_directory() as p:
   kit,c=self.fixture(p);r=ingest(kit,[],p/'out');self.assertEqual(r['accepted_final_pairs'],0);self.assertEqual(r['human_judgments_generated_by_assistant'],0)
 def test_UI_ambiguous_CSV_roundtrip_without_rewriting(self):
  with scratch_directory() as p:
   kit,c=self.fixture(p);paths=[]
   for i in [1,2]:
    row={k:c[k] for k in ['source_image_id','question_id','original_sha256','edited_sha256']};row.update(reviewer_id=f'SYNTHETIC_R{i}',independence_attestation='yes',reviewed_utc='2026-11-02T00:00:00Z',semantic_label='ambiguous',confidence='low',failure_reason='uncertain text',notes='UNIT FIXTURE ONLY',**{k:'ambiguous' for k in ATTRS});path=p/f'{i}.csv'
    with path.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    paths.append(path)
   before=[path.read_bytes() for path in paths];r=ingest(kit,paths,p/'out');self.assertEqual(r['accepted_final_pairs'],1);self.assertEqual(r['case_results']['SYNTHETIC_ONLY']['semantic_label'],'ambiguous');self.assertEqual([path.read_bytes() for path in paths],before)
if __name__=='__main__':unittest.main()
