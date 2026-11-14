"""Synthetic unit fixtures only; never create human reviews for research images."""
import csv,json,shutil,tempfile
from pathlib import Path
from analyze_semantic_clean_stage import validated_reviews,ATTRS
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/'tmp/semantic_review_contracts';base.mkdir(parents=True,exist_ok=False);checks=[]
    try:
        c={'source_image_id':'SYNTHETIC_ONLY','question_id':'SYNTHETIC_Q','original_sha256':'synthetic_original','edited_sha256':'synthetic_edit'}
        row={'reviewer_id':'SYNTHETIC_ONLY_R1','independence_attestation':'yes','reviewed_utc':'2026-10-29T00:00:00Z','source_image_id':c['source_image_id'],'question_id':c['question_id'],'original_sha256':c['original_sha256'],'edited_sha256':c['edited_sha256'],**{k:'yes' for k in ATTRS},'semantic_label':'valid','confidence':'high','failure_reason':'','notes':'Synthetic unit test, not a human review'}
        def write(name,rows):
            p=base/name
            with p.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows(rows)
            return p
        r1=write('one.csv',[row]);result,_=validated_reviews([r1],[c]);checks.append(('single_review_does_not_certify',not result['final_labels']))
        r2=write('two.csv',[{**row,'reviewer_id':'SYNTHETIC_ONLY_R2'}]);result,_=validated_reviews([r1,r2],[c]);checks.append(('dual_complete_agreement_path',result['final_labels']=={c['source_image_id']:'valid'}))
        bad=write('bad.csv',[{**row,'reference_answer_valid':'no'}]);result,_=validated_reviews([bad],[c]);checks.append(('contradictory_valid_rejected',len(result['rejected_records'])==1 and not result['final_labels']))
        result,_=validated_reviews([r1,r1],[c]);checks.append(('duplicate_reviewer_not_independent',len(result['rejected_records'])==1 and not result['final_labels']))
        missing=write('missing.csv',[{**row,'independence_attestation':'no'}]);result,_=validated_reviews([missing],[c]);checks.append(('unattested_review_rejected',not result['accepted_records']))
        mismatch=write('hash.csv',[{**row,'edited_sha256':'wrong'}]);result,_=validated_reviews([mismatch],[c]);checks.append(('wrong_image_hash_rejected',not result['accepted_records']))
        disagreement=write('disagree.csv',[{**row,'reviewer_id':'SYNTHETIC_ONLY_R2','semantic_label':'invalid','failure_reason':'scene_change','scene_preserved':'no','reference_answer_valid':'no'}]);result,_=validated_reviews([r1,disagreement],[c]);checks.append(('disagreement_requires_adjudication',not result['final_labels'] and result['agreement']==0))
        receipt={'status':'PASS' if all(x[1] for x in checks) else 'FAIL','tests':[{'test':n,'passed':bool(ok)} for n,ok in checks],'human_research_reviews_created':0}
        (ROOT/'results/semantic_clean_20261029_v1/audit/review_contract_checks.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt));assert receipt['status']=='PASS'
    finally:
        target=base.resolve();assert target.parent== (ROOT/'tmp').resolve() and target.name=='semantic_review_contracts';shutil.rmtree(target)
if __name__=='__main__':main()
