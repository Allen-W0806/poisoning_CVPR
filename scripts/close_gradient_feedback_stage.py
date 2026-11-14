"""Canonical append-only history and removal of this run's transfer files only."""
import json
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parents[1]


def main():
    marker='GRADIENT_FEEDBACK_VALIDATION_20261028'
    index=ROOT/'EXPERIMENT_INDEX.md'
    if marker not in index.read_text(encoding='utf-8'):
        with index.open('a',encoding='utf-8') as f:f.write('''

## GRADIENT_FEEDBACK_VALIDATION_20261028

Bounded benign validation complete: synthetic gradients/tiny DDPO wiring PASS; frozen Qwen 160 unique images × 3 seeds exactly repeatable, consensus 80.9375%. Ten-batch candidate feedback NOT VALIDATED; editor strict semantics FAIL. Pretrained-model optimizer steps: zero. Negative runs retained. GPU allocation including failures: 198s; CPU-only allocation: 28s.

[Audit](reports/GRADIENT_AND_FEEDBACK_AUDIT.md), [Results](reports/EXPERIMENT_RESULTS.md), [English dashboard](dashboard/gradient_feedback_validation.html), evidence: `results/gradient_feedback_summary_20261028/`.
''')
    history=ROOT/'experiments/gradient_feedback_history.jsonl'
    if marker not in (history.read_text(encoding='utf-8') if history.exists() else ''):
        with history.open('a',encoding='utf-8') as f:f.write(json.dumps({'event':marker,'created_unix':time.time(),'report':'reports/EXPERIMENT_RESULTS.md','status':'BOUNDED_GRADIENT_PASS_FEEDBACK_AND_EDITOR_GATES_OPEN','gpu_seconds':198})+'\n')
    intended=(ROOT/'tmp').resolve();removed=[]
    names=['gradient_feedback_validation_source.tar','gradient_feedback_20261028_evidence.tar','gradient_feedback_20261028_final_models.tar',
        'gradient_feedback_20261028_complete.tar','gradient_feedback_inventory.tar','gradient_feedback_inventory_final.tar',
        'gradient_feedback_inventory_final2.tar','gradient_feedback_delivery_20261028.tar','validate_policy_gradients.py','gradient_feedback_validation_v2.json']
    for name in names:
        target=(intended/name).resolve()
        if target.parent!=intended or target.name!=name:raise RuntimeError('Unsafe cleanup target')
        if target.exists():
            if not target.is_file():raise RuntimeError('Unexpected cleanup directory')
            target.unlink();removed.append(str(target.relative_to(ROOT)))
    receipt={'removed_transfer_files':removed,'existing_models_checkpoints_logs_preserved':True,'recursive_deletion':False}
    (ROOT/'results/gradient_feedback_summary_20261028/canonical_cleanup.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt))


if __name__=='__main__':main()
