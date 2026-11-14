"""Validate deliverable links, record retained sizes, append experiment history."""
import ast
import csv
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import shutil
import time
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/gradient_feedback_summary_20261028'


def main():
    scripts=[ROOT/'scripts'/p for p in ['validate_policy_gradients.py','validate_benign_models.py','validate_ddpo_wiring.py',
        'validate_group_reference.py','validate_reviewed_edits.py','report_gradient_feedback.py','collect_gradient_feedback_inventory.py','finalize_gradient_feedback.py']]
    for p in scripts:ast.parse(p.read_text(encoding='utf-8'))
    class Links(HTMLParser):
        def __init__(self):super().__init__();self.links=[]
        def handle_starttag(self,tag,attrs):
            for k,v in attrs:
                if k in ['href','src']:self.links.append(v)
    parser=Links();dashboard=ROOT/'dashboard/gradient_feedback_validation.html';parser.feed(dashboard.read_text(encoding='utf-8'))
    missing=[];nlinks=0
    for p in [ROOT/'reports/GRADIENT_AND_FEEDBACK_AUDIT.md',ROOT/'reports/EXPERIMENT_RESULTS.md',dashboard]:
        links=parser.links if p.suffix=='.html' else re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8'))
        for value in links:
            if value.startswith(('http:','https:','#')):continue
            nlinks+=1
            if not (p.parent/value).exists():missing.append({'document':str(p.relative_to(ROOT)),'target':value})
    assert not missing,missing
    required=['gradient_validation.csv','batch_feedback.csv','benign_vlm_evaluation.csv','parameter_update_audit.csv']
    csvrows={}
    for name in required:
        with (ROOT/'results'/name).open() as f:csvrows[name]=len(list(csv.DictReader(f)))
        assert csvrows[name]>0
    inventory=json.loads((OUT/'inventory_local.json').read_text())
    cluster=json.loads((OUT/'inventory_cluster.json').read_text())
    left={r['path']:r['normalized_lf_sha256'] for r in inventory['source_inventory']}
    right={r['path']:r['normalized_lf_sha256'] for r in cluster['source_inventory']}
    matched=left==right
    for r in inventory['source_inventory']:
        path=ROOT/r['snapshot_path'];assert path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
    accepted=['gradient_feedback_20261028_remote_v2/synthetic_checks.json','ddpo_wiring_20261028_v3/wiring_checks.json',
              'benign_editor_20261028_v2/model_checks.json','group_reference_20261028_v3/checks.json']
    assert all(json.loads((ROOT/'results'/p).read_text())['passed'] for p in accepted)
    checks=[]
    for directory in ROOT.glob('results/*20261028*'):
        if not directory.is_dir():continue
        for name in ['synthetic_checks.json','wiring_checks.json','model_checks.json','checks.json']:
            p=directory/name
            if not p.exists():continue
            value=json.loads(p.read_text())
            if isinstance(value.get('checks'),list):
                for check in value['checks']:checks.append({'run_id':directory.name,**check})
    keys=list(dict.fromkeys(k for r in checks for k in r))
    with (OUT/'all_check_outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(checks)
    retained=[]
    for directory in ROOT.glob('results/*20261028*'):
        if not directory.is_dir():continue
        files=[p for p in directory.rglob('*') if p.is_file()]
        retained.append({'path':str(directory.relative_to(ROOT)),'files':len(files),'bytes':sum(p.stat().st_size for p in files)})
    disk={name:dict(zip(['total','used','free'],shutil.disk_usage(name+':/'))) for name in ['C','H']}
    record={'recorded_unix':time.time(),'syntax_checked_scripts':len(scripts),'local_links_checked':nlinks,'missing_links':missing,
        'requested_csv_data_rows':csvrows,'source_sync_normalized_equal':matched,'source_count':len(left),
        'plot_visual_QA':'six PNG plots inspected together; axes/legends/CI/error trends legible',
        'dashboard_QA':'English HTML parsed and all local href/src targets checked; no browser-rendered certification',
        'retained_run_directories':retained,'retained_run_bytes':sum(r['bytes'] for r in retained),'disk_after':disk,
        'new_multi_GB_C_artifacts':False,'intermediate_scratch_cleanup':'run-specific tmp/gradient_feedback_transfer verified and removed at closure',
        'scientific_decision':'NO_GO_large_scale_finetuning; feedback candidate comparability and editor semantics unresolved',
        'failed_assertions_in_check_receipts':sum(not r['passed'] for r in checks),
        'other_failures':'initial editor JSON serialization and rejected Slurm submission recorded separately in EXPERIMENT_RESULTS.md; not disguised as successful assertions'}
    (OUT/'completion.json').write_text(json.dumps(record,indent=2))
    marker='GRADIENT_FEEDBACK_VALIDATION_20261028'
    text='''

## GRADIENT_FEEDBACK_VALIDATION_20261028

Bounded benign validation completed. Synthetic categorical/DDPO gradients and tiny-UNet wiring PASS; frozen Qwen 160 images × 3 seeds is exactly repeatable (official VQA consensus 80.9375%). Ten-batch candidate feedback NOT VALIDATED (composition p=.07179; 0/45 Holm-significant pairs). Editor file/seed checks PASS after logging repair, but semantic preservation FAILS. Zero pretrained-model optimizer steps; no large-scale fine-tuning. All negative runs retained. Allocated GPU time including failures: 198 seconds; CPU-only Slurm allocation: 28 seconds.

Reports: [Code audit](reports/GRADIENT_AND_FEEDBACK_AUDIT.md), [Results](reports/EXPERIMENT_RESULTS.md), [English dashboard](dashboard/gradient_feedback_validation.html). Source data and completion receipt: `results/gradient_feedback_summary_20261028/`.
'''
    index=ROOT/'EXPERIMENT_INDEX.md'
    if marker not in index.read_text(encoding='utf-8'):
        with index.open('a',encoding='utf-8') as f:f.write(text)
    history=ROOT/'experiments/gradient_feedback_history.jsonl'
    previous=history.read_text() if history.exists() else ''
    if marker not in previous:
        with history.open('a',encoding='utf-8') as f:f.write(json.dumps({'event':marker,'created_unix':time.time(),'report':'reports/EXPERIMENT_RESULTS.md','raw_results':'results/gradient_feedback_summary_20261028','status':'BOUNDED_GRADIENT_PASS_FEEDBACK_AND_EDITOR_GATES_OPEN','gpu_seconds':198})+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['retained_run_directories','disk_after']},indent=2))


if __name__=='__main__':main()
