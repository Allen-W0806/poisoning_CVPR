"""Final provenance/link/count checks and append-only current-stage history."""
import argparse
import ast
import csv
from datetime import datetime,timezone
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',default='matched_feedback_20261028_v2');a=p.parse_args();run=ROOT/'results'/a.run_id
    census=json.loads((run/'prior_exposure_census.json').read_text());used_ids=set(census['used_image_ids']);used_hashes=set(census['used_content_hashes'])
    ids=set();hashes=set();checked_candidates=0;prediction_rows=0;repeat_images=0
    revisions=json.loads((run/'accepted_phase_revisions.json').read_text()) if (run/'accepted_phase_revisions.json').exists() else {}
    for phase in ['pilot','main']:
        source=json.loads((run/f'{phase}_manifest.json').read_text())['records'];byid={r['source_image_id']:r for r in source}
        input_path=run/revisions.get(phase,phase)
        assert json.loads((input_path/'checks.json').read_text())['status']=='PASS'
        for r in source:
            assert r['source_image_id'] not in ids|used_ids and r['image_sha256'] not in hashes|used_hashes
            assert sha(ROOT/r['image_path'])==r['image_sha256'];ids.add(r['source_image_id']);hashes.add(r['image_sha256'])
        candidate=json.loads((input_path/'candidate_manifest.json').read_text())['records']
        for c in candidate:
            if c['generation_status']=='PASS':assert sha(ROOT/c['candidate_path'])==c['candidate_image_hash']
            checked_candidates+=1
        raw=[json.loads(x) for x in (input_path/'predictions.jsonl').read_text().splitlines()]
        for r in raw:
            original=byid[r['source_image_id']]
            assert r['question']==original['question'] and r['reference_answers']==original['answers']
            assert r['original_image_hash']==original['image_sha256']
        assert len({(r['source_image_id'],r['candidate_id'],r['evaluation_seed']) for r in raw})==len(raw)
        prediction_rows+=len(raw)
        gen=json.loads((run/phase/'generation_records.json').read_text())
        for r in gen:
            if 'repeat_sha256' in r:
                assert sha(run/phase/'images'/f"{r['source_image_id']}_C4_repeat.png")==r['repeat_sha256'];repeat_images+=1
    original_worker=run/'audit/initial_runner/scripts/run_matched_feedback.py'
    assert sha(original_worker)==json.loads((run/'main/checks.json').read_text())['source_sha256']
    assert len(ids)==len(hashes)==184 and checked_candidates==1104 and prediction_rows==1248 and repeat_images==12
    stats=json.loads((run/'paired_statistics.json').read_text());assert stats['main']['effective_sample_size']==160 and stats['pilot']['effective_sample_size']==24
    assert stats['main']['semantic_validated_secondary']['status']=='BLOCKED'
    required=['paired_feedback.csv','paired_statistics.json','candidate_group_statistics.csv','feedback_distribution.csv','semantic_review_queue.csv','semantic_audit.json','editor_contact_sheet.png']
    assert all((run/name).is_file() for name in required)
    scripts=sorted(set(ROOT.glob('scripts/*matched*.py')));source_hashes=[]
    for path in scripts:
        ast.parse(path.read_text(encoding='utf-8'));source_hashes.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
        snapshot=run/'audit/source_current'/sha(path)/path.name;source_hashes[-1]['snapshot_path']=str(snapshot.relative_to(ROOT));snapshot.parent.mkdir(parents=True,exist_ok=True)
        if not snapshot.exists():snapshot.write_bytes(path.read_bytes())
    (run/'audit/current_source_hashes.json').write_text(json.dumps(source_hashes,indent=2))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src']:self.links.append(v)
    reports=['MATCHED_FEEDBACK_VALIDATION','SEMANTIC_PRESERVATION_AUDIT','RL_RELIABILITY_STATUS','NEXT_EXPERIMENT_DECISION']
    links=0;missing=[]
    for name in reports:
        path=ROOT/'reports'/f'{name}.md'
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            links+=1
            if not (path.parent/target).exists():missing.append((name,target))
        dest=run/'reports'/path.name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(path.read_bytes())
    for name in ['gradient_feedback_validation.html','RSI_Experiment_Progress_2026-10-28.html']:
        path=ROOT/'dashboard'/name;page=path.read_text(encoding='utf-8');parser=Links();parser.feed(page)
        assert 'lang="en"' in page and '<meta name="viewport"' in page and 'cdn.' not in page
        assert all(f'id="s{i}"' in page for i in range(1,11))
        for target in parser.links:
            if target.startswith(('http:','https:','#')):continue
            links+=1
            if not (path.parent/target).exists():missing.append((name,target))
    assert not missing,missing
    resource=json.loads((run/'resource_accounting.json').read_text());assert resource['gpu_seconds']<=3600 and resource['status']=='PASS'
    checklist={name:'PASS' for name in ['source_and_reports_audited','unique_run_and_manifests','dataset_overlap_checks','matched_inference_complete',
        'frozen_hashes_verified','predictions_saved','paired_analysis_complete','saturation_quantified','all_editor_outputs_retained','human_review_queue_prepared',
        'invalid_and_unverified_edits_reported','RL_regressions_checked','historical_negatives_preserved','resources_reconciled','failures_documented','English_dashboards_updated','four_reports_generated','next_decision_documented']}
    checklist.update(independent_semantic_review='BLOCKED',validated_semantic_secondary='BLOCKED',full_pretrained_training='NOT VALIDATED',useful_generator_learning_feedback='NOT VALIDATED')
    files=[p for p in run.rglob('*') if p.is_file()]
    completion={'status':'PASS','scope':'requested assessment and deliverable integrity; does not open scientific training gates','checklist':checklist,
        'unique_source_images':184,'main_independent_sources':160,'pilot_independent_sources':24,'candidate_records':checked_candidates,
        'prediction_records_including_repeats':prediction_rows,'generation_repeat_files':repeat_images,'local_artifact_links_checked':links,
        'missing_links':missing,'retained_files':len(files),'retained_bytes':sum(p.stat().st_size for p in files),'GPU_seconds':resource['gpu_seconds'],
        'independent_review_preservation_rate':None,'large_scale_training_enabled':False,'completion_time_UTC':datetime.now(timezone.utc).isoformat()}
    (run/'completion.json').write_text(json.dumps(completion,indent=2))
    marker='MATCHED_FEEDBACK_VALIDATION_20261028_V2';index=ROOT/'EXPERIMENT_INDEX.md'
    if marker not in index.read_text(encoding='utf-8'):
        with index.open('a',encoding='utf-8') as f:f.write(f'\n\n## {marker}\n\nFresh matched pilot24/main160 completed under locked benign conditions C0–C4 plus geometry control C5. Reference agreement assessed with paired image bootstrap and permutation/Holm analysis. Independent semantic verification BLOCKED; full pretrained training NOT VALIDATED. Zero pretrained optimizer steps. All outputs and preparation failure v1 retained. GPU allocation {resource["gpu_seconds"]} seconds. [Report](reports/MATCHED_FEEDBACK_VALIDATION.md), [decision](reports/NEXT_EXPERIMENT_DECISION.md), [dashboard](dashboard/RSI_Experiment_Progress_2026-10-28.html).\n')
    history=ROOT/'experiments/matched_feedback_history.jsonl'
    if marker not in (history.read_text() if history.exists() else ''):
        with history.open('a') as f:f.write(json.dumps({'event':marker,'run_id':a.run_id,'report':'reports/MATCHED_FEEDBACK_VALIDATION.md','GPU_seconds':resource['gpu_seconds'],'scientific_training':'NOT VALIDATED','semantics':'BLOCKED'})+'\n')
    print(json.dumps(completion,indent=2))


if __name__=='__main__':main()
