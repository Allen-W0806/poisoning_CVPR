"""Verify the saved blinded kit and prepare a handoff; create no human labels."""
import argparse,csv,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;kit=ROOT/'results/semantic_clean_20261029_v1/semantic_review';cases=json.loads((kit/'blinded_manifest.json').read_text())['cases'];assert len(cases)==160
    for c in cases:
        assert hashlib.sha256((kit/c['original']).read_bytes()).hexdigest()==c['original_sha256'];assert hashlib.sha256((kit/c['edited']).read_bytes()).hexdigest()==c['edited_sha256']
    assignment=run/'independent_review_assignments.csv'
    with assignment.open('x',newline='') as f:
        fields=['source_image_id','question_id','original_sha256','edited_sha256','reviewer_1_id','reviewer_2_id','adjudicator_id'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for c in cases:w.writerow({k:c.get(k,'') for k in fields})
    result={'interface_engineering':'PASS','independent_semantic_review':'BLOCKED','verified_cases':160,'verified_image_hashes':320,'complete_contact_sheets':len(list((kit/'contact_sheets').glob('*.jpg'))),'human_review_records_created':0,'genuine_submitted_files':[str(p.relative_to(ROOT)) for p in (kit/'submitted').glob('*') if p.is_file()],'review_interface':str((kit/'index.html').relative_to(ROOT)),'assignment_template':str(assignment.relative_to(ROOT))}
    (run/'semantic_handoff.json').write_text(json.dumps(result,indent=2))
    (ROOT/'reports/INDEPENDENT_REVIEW_HANDOFF.md').write_text('''# Independent dual-review handoff

Interface engineering PASS; human review BLOCKED. No human judgments are invented. Open `results/semantic_clean_20261029_v1/semantic_review/index.html` from the copied offline kit. All160 cases and320 original/edit hashes were verified. The current GPU experiments continue independently.

Assign two distinct human reviewers using `results/'''+a.run_id+'''/independent_review_assignments.csv`. Each reviews every image alone, using a separate browser profile/computer, without target predictions, old answer-bearing contact sheets, risk flags or the other reviewer's labels. Keep reviewer1 and reviewer2 exports separate until both are locked. Each reviewer truthfully attests independence and exports the provided CSV. Then a third person adjudicates disagreements. Do not mark empty assignment cells as completed reviews.

Use the existing INSTRUCTIONS.md, reviewer_1_template.csv, reviewer_2_template.csv and adjudication_template.csv in that kit. Submit genuine exports to the existing analysis tool using a new exclusive analysis ID. Agreement and Cohen kappa are computed on paired original labels before adjudication. Independent preservation rate and valid-only accuracy effects remain unknown until real reviews exist. No reviewers have been contacted by this automation.
''')
    print(json.dumps(result))
if __name__=='__main__':main()
