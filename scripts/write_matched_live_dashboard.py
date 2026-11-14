"""Offline interim status; main-stage acceptance stays NOT VALIDATED while running."""
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'results/matched_feedback_20261028_v2'
stats=json.loads((run/'pilot/paired_statistics.json').read_text());old=ROOT/'dashboard/gradient_feedback_validation.html'
backup=ROOT/'dashboard/gradient_feedback_validation_previous_20261028.html'
if not backup.exists():shutil.copyfile(old,backup)
items=[('Project Overview','Matched benign feedback validation. Pilot engineering PASS; main execution is in progress and acceptance is NOT VALIDATED. No pretrained-model optimization.'),
 ('Experiment Timeline','Prior n160 development study retained separately. New pilot24 and main160 are disjoint from historical project source IDs and content hashes.'),
 ('Gradient Validation','Fresh finite exact-vs-MC regression and tiny checkpoint replay PASS in their tested synthetic scope. Full pretrained training remains NOT VALIDATED.'),
 ('Frozen VLM Evaluation','Prior original-image development accuracy80.9375% (160 unique images, 480 dependent repeated calls). New pilot n24 is separate; main n160 pending.'),
 ('Matched Candidate Feedback','Pilot all-five tied-source fraction: '+str(stats['group_statistics']['all_candidates_tied_fraction'])+'. Conditions and analysis are frozen; pilot/main pools stay separate.'),
 ('Semantic Preservation','Independent human-review status BLOCKED. Automated similarity and target-answer agreement do not certify semantics. Every editor output retained.'),
 ('RL Algorithm Audit','Categorical REINFORCE; production DDPO-SF; isolated group-relative reference. Production GRPO not implemented. Reward-release guard remains active.'),
 ('Resource Accounting','Pilot Slurm86793 completed in129 GPU allocation seconds; CPU regression86794 completed in32 seconds. Main86795 active with1200s wall limit. Stage budget3600 GPU seconds.'),
 ('Failure and Repair Log','Preparation v1 stopped at historical CSV parser capacity; versioned v2 fixes parser capacity and explicitly records empty receipts. Science protocol unchanged; prior failures retained.'),
 ('Next Experimental Gates','Complete main matched inference and paired analysis; independent semantic review; separate full pretrained BF16 continuation validation. No training gate is opened by infrastructure tests.')]
text='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RSI Experiment Progress</title><style>body{font:16px/1.6 system-ui;background:#f4f5f7;color:#243142;max-width:1100px;margin:auto;padding:24px}section{background:white;border:1px solid #ddd;padding:20px;margin:16px 0}h2{font-size:22px}</style><h1>RSI Experiment Progress — 2026-10-28</h1><p>Updated '+datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')+'</p>'+''.join(f'<section><h2>{i}. {html.escape(title)}</h2><p>{html.escape(content)}</p></section>' for i,(title,content) in enumerate(items,1))+'</html>'
for name in ['gradient_feedback_validation.html','RSI_Experiment_Progress_2026-10-28.html']:(ROOT/'dashboard'/name).write_text(text,encoding='utf-8')
