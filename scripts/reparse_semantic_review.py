"""CPU-only derived audit of already saved VLM responses. No new inference."""
import json
from collections import Counter
from pathlib import Path
import _bootstrap
from _phase3 import phase3_run,mark_latest
from src.evaluation.semantic_inventory_schema import parse_inventory_v2
from src.evaluation.semantic_review import compare_inventories
from src.utils.logging import write_json

if __name__=='__main__':
    root=_bootstrap.ROOT
    source=Path(json.loads((root/'reports/phase3_latest_semantics.json').read_text())['path'])
    rows=json.loads((source/'semantic_validity_table.json').read_text())
    with phase3_run('semantic_schema_audit',{'seed':42,'source_run':source.name,'parser':'v2_integer_count_representation',
            'model_calls':0,'ground_truth_labels_created':False},('validation',),source_parent=source.name) as run:
        results=[]
        for row in rows:
            original=parse_inventory_v2(row['source_inventory']['raw_output'])
            edited=parse_inventory_v2(row['edited_inventory']['raw_output'])
            checks=compare_inventories(original,edited,row['question'])
            differences=[k for k,v in checks['checks'].items() if v['status']=='DESCRIPTION_DIFFERENCE_REVIEW']
            category='INVENTORY_PARSE_FAILURE' if not original['parse_ok'] or not edited['parse_ok'] else 'AUTOMATIC_DESCRIPTION_DISAGREEMENT' if differences else 'AWAITING_INDEPENDENT_REVIEW'
            result={**row,'original_schema_uncertainty_category':row['uncertainty_category'],
                'source_inventory':original,'edited_inventory':edited,'automatic_checks':checks,
                'automatic_disagreement_fields':differences,'uncertainty_category':category,
                'cross_validator_status':'prior_proxy_agreement_but_inventory_review_flag' if row['prior_proxy_provisionally_valid'] and differences else 'unresolved_no_ground_truth'}
            assert result['human_status']=='UNVERIFIED' and result['independent_validity_judgment']==''
            results.append(result)
        write_json(run.path/'semantic_validity_table_v2.json',results)
        run.metrics({'pairs':len(results),'source_run':source.name,'new_model_calls':0,'human_verified_changed_edits':0,
            'automatic_human_agreement':None,'model_independence':False,
            'parsed_pairs':sum(r['source_inventory']['parse_ok'] and r['edited_inventory']['parse_ok'] for r in results),
            'description_disagreement_pairs':sum(bool(r['automatic_disagreement_fields']) for r in results),
            'uncertainty_categories':dict(Counter(r['uncertainty_category'] for r in results)),
            'original_review_form':str(source/'review_form.csv'),'original_contact_sheet':str(source/'images/review_contact_sheet.png'),
            'semantic_truth_ready':False})
        run.finish(summary='Representation-only reparse of saved numeric count outputs. Prior schema failure retained; all human validity labels remain blank.')
        mark_latest('semantic_schema_audit',run);print(run.path)
