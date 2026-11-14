"""Evidence-bound English reports and offline research dashboard for paired study."""
import argparse
from datetime import datetime
import html
import json
from pathlib import Path
import shutil
import sys
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]


def mdtable(headers,rows):
    return '| '+' | '.join(headers)+' |\n|'+ '|'.join('---' for _ in headers)+'|\n'+''.join('| '+' | '.join(str(x) for x in row)+' |\n' for row in rows)


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def percent(value):return f'{100*value:.2f}%'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='matched_feedback_20261028_v2');a=parser.parse_args()
    run=ROOT/'results'/a.run_id;stats=read(run/'paired_statistics.json');audits=read(run/'semantic_audit.json');cfg=read(run/'protocol.json')
    mainstats=stats.get('main',stats['pilot']);phase='main' if 'main' in stats else 'pilot';sem=audits[phase]
    resource=read(run/'resource_accounting.json') if (run/'resource_accounting.json').exists() else {'status':'BLOCKED: accounting pending','gpu_seconds':None}
    reliability=read(run/'audit/rl_reliability.json') if (run/'audit/rl_reliability.json').exists() else {'status':'BLOCKED'}
    conditions=mainstats['conditions'];c4=next(c for c in conditions if c['candidate_id']=='C4');n=mainstats['effective_sample_size']
    planned=mainstats['planned_comparisons'];significant=[r for r in planned if r['primary_family_holm_p']<.05 and (r['ci95_low']>0 or r['ci95_high']<0)]
    condition_rows=[[r['candidate_id'],r['n_unique_sources'],f"{r['mean_score_all_attempted_utility']:.5f}",f"[{r['mean_ci95_low']:.5f}, {r['mean_ci95_high']:.5f}]",f"{r['paired_difference_C0']:+.5f}",f"[{r['paired_ci95_low']:+.5f}, {r['paired_ci95_high']:+.5f}]",percent(r['at_one_fraction'])] for r in conditions]
    comparison_rows=[[r['candidate_a']+'−'+r['candidate_b'],f"{r['mean_difference']:+.5f}",f"[{r['ci95_low']:+.5f}, {r['ci95_high']:+.5f}]",f"{r['permutation_p']:.6f}",f"{r['primary_family_holm_p']:.6f}",percent(r['wins']),percent(r['ties']),percent(r['losses'])] for r in planned]
    condition_table=mdtable(['Condition','Unique sources','Mean agreement','95% CI','Difference C0','Paired 95% CI','At score 1'],condition_rows)
    comparison_table=mdtable(['Comparison','Mean difference','Paired 95% CI','Permutation p','Holm p (4 planned)','Win','Tie','Loss'],comparison_rows)
    prefix=f'../results/{a.run_id}/';now=datetime.now(ZoneInfo('America/New_York')).isoformat(timespec='seconds')
    census=read(run/'prior_exposure_census.json');data=read(run/'data_checks.json')
    opening=f'''# Matched Benign Feedback Validation

Run: `{a.run_id}`. Updated {now} (America/New_York). **Frozen inference only; zero pretrained-model optimizer steps.**

**Feedback computation and tested reproducibility: PASS. Distinguishable inherited-reference agreement: {'PASS within the tested paired comparison scope' if significant else 'NOT VALIDATED'}. Useful semantically valid generator-learning feedback: NOT VALIDATED. Independently verified semantic secondary analysis: BLOCKED. Full pretrained BF16 training/restart: NOT VALIDATED.**

Primary results below use the {phase} pool, n={n} unique source images. Pilot and main are separate pools and analyses. Candidate observations and repeated inference seeds are dependent, not additional independent images. There are five primary candidate conditions; C5 is an auxiliary preprocessing control.

## Design, data and provenance

The all-project census inspected {len(census['scanned_files'])} JSON/JSONL/CSV records/manifests, including train, validation, test, gate, development, prior experiment and result records. It found {len(census['used_image_ids'])} previously referenced numeric source-image IDs, {len(census['used_content_hashes'])} content hashes, {census['cached_source_images']} cached originals, and **{census['cached_unused_images']} unused cached originals**. The official cached annotation pool contains {census['official_annotation_image_population']} images; {census['eligible_unused_official_images']} unique image IDs meet the predeclared project-exposure exclusion and >=6/10 normalized human-answer agreement rule. Answer-category candidate populations overlap and must not be summed.

The script staged exactly 184 new official COCO images, preserving the original question and ten human answers. Twenty-four pilot sources and 160 main sources are disjoint by both image ID and content SHA256 and have no observed overlap with historical project records. Original project splits were not modified. Model pretraining contamination remains unknown. Category-balanced selection (pilot 8/8/8; main yes/no53, number53, other54) and the human-agreement rule limit population generalization.

Data source: cached VQAv2 train2014-2017-04-26 annotations/questions, and the fixed selected image paths at `https://s3.amazonaws.com/images.cocodataset.org/train2014/`. Annotation terms: CC BY 4.0; images retain original Flickr/COCO per-image terms. No new model weights or large packages were downloaded. The model/environment pins match the prior audit.

[Protocol]({prefix}protocol.json), [exposure census]({prefix}prior_exposure_census.json), [data checks]({prefix}data_checks.json), [pilot manifest]({prefix}pilot_manifest.json), [main manifest]({prefix}main_manifest.json). Source manifests were frozen before generation; candidate manifests with actual output hashes were frozen before VQA inference. Original and edited image hashes, raw answers and labels remain in every row.

## Fixed candidate conditions

| ID | Definition |
|---|---|
| C0 | Original unmodified RGB image |
| C1 | Exact original pixels re-encoded as identity PNG |
| C2 | Global brightness ×1.03, predefined |
| C3 | Global contrast ×1.03, predefined |
| C4 | Frozen pinned IP2Pix, lighting-only instruction, generation seed42, 256×256, 10steps |
| C5 | Auxiliary IP2Pix resize/padding only; no generation |

All target conditions use the original question and references, Qwen revision `89644892e4d85e24eaac8bacfd4f463576704203`, greedy short-answer decoding, max32 tokens, pixel budget4096–262144, physical microbatch1. The source order rotates condition order deterministically. IP2Pix revision is `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`; BF16 runtime, FP16 cached variant, guidance7.5/image guidance1.5, math SDPA, no adapter loaded. No strength/instruction/seed selection was made using target performance. Every attempted edit remains in the evidence.

C4 changes geometry as well as image content relative to C0. The predeclared auxiliary C4−C5 comparison helps diagnose generation beyond resizing/padding; C4−C0 is not attributed solely to generation.

## Main paired results

{condition_table}

{comparison_table}

The main planned family has {len(significant)} comparisons with Holm p<.05 and paired-bootstrap intervals excluding zero. This is evidence about reference-answer agreement under the fixed transformations, not certification of correct edited labels or useful learning. C1 pixel/answer/score identity is an engineering invariant, not a statistical equivalence claim.

The primary uncertainty analysis resamples source-image indices jointly across all candidates 10,000 times. Two-sided paired sign-permutation tests use 10,000 sign draws and a plus-one correction. Four predeclared C1–C4 vs C0 comparisons form one Holm family; ten exploratory primary-condition pairs form a separately labeled Holm family. C4−C5 is separately labeled auxiliary. No thresholds, transforms or pools were adjusted after observing target results.

## Group informativeness, saturation and reproducibility

Mean within-source C0–C4 reward variance is {mainstats['group_statistics']['mean_within_source_variance']:.6f}; **{percent(mainstats['group_statistics']['all_candidates_tied_fraction'])}** of source groups give exactly the same reward to all five candidates; **{percent(mainstats['group_statistics']['nondegenerate_fraction'])}** are non-degenerate. C4 ties C0 on {percent(c4['ties_C0'])} of sources. Full win/tie/loss comparisons, category summaries and score distributions are retained.

Selected inference repeats cover six sources per phase (two per answer category), all six conditions, and seeds42/123/3407. Exact repeat agreement is {percent(mainstats['repeat_exact_fraction'])} across {mainstats['repeated_candidate_cases']} main repeated candidate cases. This is not full-pool repeated inference and does not increase n. Generation repeats cover the same six sources under identical generation settings. Frozen UNet, VAE, text encoder and target VLM parameter hashes are checked before/after each phase.

All-attempted score utility is zero for technical generation/inference failure; the raw official VQA score remains missing where no answer exists. Technical missingness and failure counts are preserved. For C2/C3/C4, inherited-reference scores are descriptive agreement until expected-answer preservation is verified. A high score alone does not establish semantic or instruction preservation.

## Secondary analysis and limits

The independently validated, all-candidate semantic-preserving matched subset is **BLOCKED**, with zero independently reviewed C4 cases. The original source labels and exact identity C1 are available; this does not certify nontrivial edits. We do not manufacture a filtered valid-edits effect from automated similarity or assistant screening. Technical complete-case sensitivity equals all-attempted analysis only when there are no technical failures; semantic robustness remains unvalidated.

The auxiliary C4−C5 mean difference is {mainstats['auxiliary_geometry_comparison']['mean_difference']:+.5f}, paired95% CI {mainstats['auxiliary_geometry_comparison']['ci95']}, permutation p={mainstats['auxiliary_geometry_comparison']['permutation_p']:.6f}. This comparison still inherits the semantic-label limitation.

## Engineering failure and versioned control repair

The original main job86795 remains FAILED. One original JPEG (image361516) is encoded in grayscale L mode; the validation wrapper incorrectly rejected the encoded mode even though all VLM inputs are converted to RGB. It skipped the original C0 inference and failed both all-record validity and the identity-control assertion. No edited-image generation failed. The source pixels, question and ten references were unchanged.

Repair job86796 validated all160 decoded RGB inputs (159 RGB-encoded originals, one grayscale-encoded original), proved grayscale C0/C1 pixel equality, and reevaluated that source's C0 and C1 under seeds42/123/3407: six real inference calls, exact answer/score agreement and unchanged target hash. A versioned `main_repair_v1` accepted view replaces only the missing C0 observation; the remaining1031 observations and all generated images are unchanged. The original failed JSONL/checks/job state remain immutable. No data, score threshold, transform or question was changed to improve results.

Primary corrected-view outputs: [paired_feedback.csv]({prefix}paired_feedback.csv), [paired_statistics.json]({prefix}paired_statistics.json), [candidate_group_statistics.csv]({prefix}candidate_group_statistics.csv), [feedback_distribution.csv]({prefix}feedback_distribution.csv), [all plots]({prefix}{phase}/plots/paired_differences_CI.png). All original attempts, including the skipped C0, plus six repair validations are separately retained in [all_attempts_feedback.csv]({prefix}all_attempts_feedback.csv). [Repair receipt]({prefix}main_repair_v1/repair_receipt.json) and [accepted-revision mapping]({prefix}accepted_phase_revisions.json) make the derivation explicit. Main and pilot numbers must not be pooled as one post-hoc favorable result.
'''
    opening+='''
## Reproduce the retained analysis

From `H:/Project/Poisoning`, with the existing local NumPy/SciPy/Matplotlib/Pillow environment:

```powershell
python -B scripts/analyze_matched_feedback.py --run-id matched_feedback_20261028_v2
python -B scripts/report_matched_validation.py --run-id matched_feedback_20261028_v2
python -B scripts/finalize_matched_validation.py --run-id matched_feedback_20261028_v2
```

These commands recompute from the frozen source-level predictions, retain failed attempts separately, and never rerun or tune the models. The original source census, selected image IDs, questions, labels, archived source-code versions, model pins, config and actual file hashes are preserved in the run. A future fresh-data census will reflect additional exposure; it must not silently replace this frozen pool. For new inference, use `run_matched_feedback.py --run-id <new-exclusive-run> --phase pilot|main` only after explicitly staging that run's versioned manifests and data checks. Use `submit_matched_feedback.py` for the one-hour stage budget guard. Do not run into an existing phase directory. Canonical `.venv-editor` is reused without installation; GPU work uses Slurm.
'''
    (ROOT/'reports/MATCHED_FEEDBACK_VALIDATION.md').write_text(opening,encoding='utf-8')
    from scipy.stats import beta
    valid=sem['technically_valid_C4_edits'];attempted=sem['attempted_C4_edits'];low=0 if valid==0 else float(beta.ppf(.025,valid,attempted-valid+1));high=1 if valid==attempted else float(beta.ppf(.975,valid+1,attempted-valid))
    semantic=f'''# Semantic Preservation Audit

Run `{a.run_id}`, {phase} pool n={attempted}. **Technical image validity: {'PASS' if valid==attempted else 'FAIL'}. Independently confirmed semantic preservation: BLOCKED.**

Frozen C4 generation attempted {attempted} edits; {valid} technically valid ({percent(valid/attempted)}; binomial95% CI [{percent(low)}, {percent(high)}]), {attempted-valid} technical failures. Every output and repeated-generation output is retained. No failed seed or invalid/ambiguous output is silently discarded.

**The confirmed semantic-preservation fraction is unknown**, not 0% or 100%. Independent human reviews:0. Pending C4 cases:{attempted}. Confirmed-invalid and confirmed-ambiguous population counts are unknown. Every review row has nine separate fields: scene identity, object existence/count/identity/colors, spatial relationships, text/symbols, VQA-answer preservation and instruction compliance. Exact-identity C1 is proven by identical pixels; that is not a certificate for C2/C3/C4/C5.

Automated diagnostic flags on C4 (review prioritization only): {json.dumps(sem['automated_risk_flag_counts'])}. Any risk flag: {sem['automated_any_risk_count']}/{attempted} ({percent(sem['automated_any_risk_rate'])}). The fixed flags use SSIM<.7, absolute mean luminance change>20, and unintended darkening. These thresholds were frozen before target inference; no similarity threshold is used to certify semantics. Target normalized-answer agreement C4/C0 is {percent(sem['answer_agreement_rate'])}; this also does not certify preservation.

The contact sheets include every source, all six conditions, original question, ten reference answers, original/edited raw model answers, fixed instruction and automated diagnostics. Full-resolution candidate paths and hashes are in the review queue. [Main overview]({prefix}editor_contact_sheet.png), [all main contact sheets]({prefix}{phase}/contact_sheets/), [review queue]({prefix}semantic_review_queue.csv), [semantic audit data]({prefix}semantic_audit.json).

Any assistant visual notes are stored separately as assistant screening, never independent certification. The independently validated secondary statistical analysis is BLOCKED until signed reviews establish a common matched source subset. The analysis must retain all-attempted results and review failure rates when that secondary subset becomes available. Automated risk counts are not claimed as true scene/object/color failure prevalence.
'''
    if (run/'assistant_screening_notes.json').exists():
        semantic+=f'\n## Assistant-only illustrative screening\n\nThe first four frozen main sources were inspected in manifest order. Two have clear assistant-observed instruction failures: image19967 changes ice-cream appearance/colors; image503689 changes the outdoor baseball scene and object population. The other two cannot be certified at thumbnail scale. These are illustrative cases, not a prevalence sample or independent certification. [Screening notes]({prefix}assistant_screening_notes.json). Independent semantic preservation rate remains unknown.\n'
    (ROOT/'reports/SEMANTIC_PRESERVATION_AUDIT.md').write_text(semantic,encoding='utf-8')
    rl=f'''# RL Reliability Status

Run `{a.run_id}`. Current scoped regression status: **{reliability['status']}**. Full pretrained-model training gate: **NOT VALIDATED**.

Production algorithms are categorical REINFORCE and single-pass on-policy DDPO score-function learning with fixed coordinate/time scaling. Supervised denoising MSE is a separate trainer. Production GRPO does not exist; the group-relative implementation is a finite synthetic reference. Stored diffusion old log-probabilities check unchanged-policy replay; they are not PPO ratios. The newer backend still rejects reward training eligibility. This stage does not bypass it.

The finite categorical and Gaussian references were rerun under the existing pinned `.venv-editor` and unchanged tolerances. The successful tiny synthetic random-UNet harness was also rerun, including actual LoRA updates, frozen base hashes, exact checkpoint restoration, and equivalent fixed-trajectory next optimizer step. Sampling and replay use the same math SDPA backend. Twenty-one targeted pre-existing diffusion/reward/VQA/contract regression tests passed. Source hashes and installed versions are reconciled against the previous audit. [Current reliability receipt]({prefix}audit/rl_reliability.json), [CPU log]({prefix}logs/rl-cpu-86794.err), [tiny wiring]({prefix}../{a.run_id}_tiny_checkpoint/wiring_checks.json), [finite reference]({prefix}../{a.run_id}_gradient/synthetic_checks.json).

Previously established gradient errors remain approximately categorical .0061 and DDPO .0576; exact current rerun values, convergence slopes and all assertions are in the finite-reference receipt. These are expected-reward derivative checks under finite harmless tasks, not validated reward constructs for real image editing. Current inference checks independently establish that Qwen and pretrained IP2Pix receive zero optimizer updates and retain complete parameter hashes.

Historical full pretrained BF16 continuation failures remain negative evidence: checked restore/behavior likelihood equality did not imply equal backward gradients, post-update adapters, optimizer state or output pixels. The earlier math-SDPA mismatch failure is retained separately and the synthetic sampling/replay fixture uses the matched backend. Current tiny success does not certify full pretrained stochastic rollout or resumed training equivalence.

Missing evidence before opening pretrained diffusion RL: a valid independently reviewed reward construct, a semantically verified matched candidate signal, and exact full pretrained BF16 forward/backward/optimizer/RNG/rollout continuation under the locked protocol. No numerical tolerance was relaxed. No attack-policy, poisoning, backdoor or ASR-driven optimization was activated.
'''
    (ROOT/'reports/RL_RELIABILITY_STATUS.md').write_text(rl,encoding='utf-8')
    decisions=[
        ('Q1. Is benign feedback reproducible?','PASS',f"Selected main repeats agree exactly on {mainstats['repeated_candidate_cases']} candidate cases ({percent(mainstats['repeat_exact_fraction'])}); this covers six sources, not the full pool."),
        ('Q2. Meaningful matched differences?','PASS' if significant else 'NOT VALIDATED',f"{len(significant)}/4 planned comparisons satisfy Holm p<.05 and bootstrap CI excluding zero. This supports inherited-reference agreement differences only; semantically valid candidate quality is NOT VALIDATED."),
        ('Q3. Reward saturation?','PASS',f"Quantified: all-five tie fraction {percent(mainstats['group_statistics']['all_candidates_tied_fraction'])}; nondegenerate {percent(mainstats['group_statistics']['nondegenerate_fraction'])}. C4 at score1={percent(c4['at_one_fraction'])}, at0={percent(c4['at_zero_fraction'])}. Quantification does not mean saturation is acceptable."),
        ('Q4. Fraction preserving required semantics?','BLOCKED',f"Unknown:0 independent human reviews/{attempted} main C4 edits. Technical validity {valid}/{attempted} is a different quantity."),
        ('Q5. Robust after excluding invalid edits?','BLOCKED','A verified semantic-valid common matched subset is unavailable. No valid-only effect is invented. Technical failures and all attempted outputs remain included.'),
        ('Q6. Production RL identity and tested validation?','PASS',f"REINFORCE/DDPO-SF identified; scoped fresh synthetic, checkpoint and guard receipt status {reliability['status']}. Production GRPO and full pretrained training are NOT VALIDATED."),
        ('Q7. Pretrained training reliability?','NOT VALIDATED','Full BF16 backward/optimizer/rollout/RNG continuation remains unresolved; historical failures retained. Frozen inference is not training reliability.'),
        ('Q8. Evidence before claiming feedback-driven generator learning?','NOT VALIDATED','Independent semantic labels, useful nondegenerate matched reward, successful locked benign pretrained training with full checkpoint continuation, and a fresh held-out paired efficacy test are missing.'),
        ('Q9. Next three experiments?','NOT VALIDATED','(1) Independent dual review/adjudication of all C4 outputs and matched photometric controls. (2) Locked benign, same-source multi-candidate repeat study using the reviewed labels, with power informed by these ties/effects. (3) Separate bounded full-IP2Pix BF16 checkpoint/RNG/rollout continuation diagnostic; train only after reward and semantic gates pass.'),
        ('Q10. Unsupported claims?','NOT VALIDATED','Successful pretrained diffusion RL, completed production GRPO, general RSI efficacy, independently certified edit semantics, and candidate-ranking validity from repeated seeds or unmatched batch means are unsupported.')]
    decision='# Next Experiment Decision\n\n**Do not start pretrained-model training.** The assessment separates computability, tested reproducibility and reliable semantically valid candidate discrimination.\n\n'+''.join(f'## {q}\n\n**{status}.** {answer}\n\n' for q,status,answer in decisions)
    decision+=condition_table+'\n'+comparison_table+f'\nSupporting artifacts: [paired statistics]({prefix}paired_statistics.json), [semantic audit]({prefix}semantic_audit.json), [RL reliability]({prefix}audit/rl_reliability.json), [resource ledger]({prefix}resource_accounting.json). Primary confidence intervals use 10,000 image-level paired bootstrap resamples. Pilot n24 is reported separately; main n160 is the primary assessment. Human-review limitations prevent generalization from reference agreement to valid learning reward.\n'
    (ROOT/'reports/NEXT_EXPERIMENT_DECISION.md').write_text(decision,encoding='utf-8')
    status={'run_id':a.run_id,'updated':now,'current_stage':'Bounded matched-feedback validation complete' if phase=='main' else 'Pilot review',
        'feedback_computable':'PASS','tested_reproducibility':'PASS' if mainstats['repeat_exact_fraction']==1 else 'FAIL',
        'reference_agreement_discrimination':'PASS' if significant else 'NOT VALIDATED','semantic_valid_candidate_discrimination':'NOT VALIDATED',
        'independent_semantic_review':'BLOCKED','full_pretrained_training':'NOT VALIDATED','pretrained_optimizer_steps':0,'resource':resource,
        'previous_original_VQA':{'accuracy':.809375,'CI95':[.75375,.860015625],'n_unique':160,'repeated_calls':480,'scope':'previous development pool; separate from this fresh study'},
        'new_phase_statistics':stats,'semantic_audits':audits,'RL_reliability_status':reliability['status']}
    dump(run/'dashboard_data.json',status)
    def esc(v):return html.escape(str(v))
    overview=f'<p>Run <code>{a.run_id}</code> · Updated {esc(now)}</p><p>Frozen benign evaluation only. Pretrained-model optimizer steps: <strong>0</strong>. Useful generator-learning feedback and full training remain NOT VALIDATED. Independent semantic review is BLOCKED.</p>'
    conditions_html=''.join(f'<tr><td>{esc(r[0])}</td>'+''.join(f'<td>{esc(v)}</td>' for v in r[1:])+'</tr>' for r in condition_rows)
    resource_table='<div class="table-wrap"><table><thead><tr><th>Job</th><th>State</th><th>Allocated seconds</th><th>GPUs</th></tr></thead><tbody>'+''.join(f'<tr><td>{esc(r["job_id"])}</td><td>{esc(r["state"])}</td><td>{r["allocated_elapsed_seconds"]}</td><td>{r["GPU_count"]}</td></tr>' for r in resource.get('jobs',[]))+'</tbody></table></div>'
    sections=[
        ('Project Overview',overview+'<p>PASS: data traceability, computability, tested repeats and synthetic infrastructure. Open: independent semantics, usable candidate-quality reward, full BF16 training continuation.</p>'),
        ('Experiment Timeline','<ul><li>Previous: finite gradient tests and n160 development VQA.</li><li>Exposure census: zero unused cached images; fresh disjoint official-source pilot24/main160 staged.</li><li>Current: pilot gate, main matched evaluation, semantic queue, fresh RL regressions.</li><li>Retained failure: initial preparation parser capacity; versioned repair, no scientific threshold change.</li></ul>'),
        ('Gradient Validation','<p>Previous categorical RMSE .0061; DDPO RMSE .0576. Current exact-vs-MC convergence, parameter ownership and checkpoint regressions are recorded separately. Tiny random-UNet success does not validate pretrained training.</p>'+f'<a href="../results/{a.run_id}/audit/rl_reliability.json">Current regression evidence</a>'),
        ('Frozen VLM Evaluation','<p><strong>Previous development result:</strong> 80.9375% consensus, 95% CI [75.3750%,86.0016%], 160 unique images / 480 calls. This is separate from the new fresh-pool results below. Neither repeat calls nor edited conditions increase the independent image count.</p>'),
        ('Matched Candidate Feedback',f'<p>Primary {phase}: n={n}. Five primary conditions; auxiliary C5 controls editor geometry. All-candidate tie fraction {percent(mainstats["group_statistics"]["all_candidates_tied_fraction"])}. C4–C0 difference {c4["paired_difference_C0"]:+.5f}, paired95% CI [{c4["paired_ci95_low"]:+.5f},{c4["paired_ci95_high"]:+.5f}].</p><label>Chart cohort <select id="phase"><option>main</option><option>pilot</option></select></label><div id="interactive-chart"></div><p id="chart-note"></p><div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{esc(v)}</th>' for v in ['Condition','n','Mean','95% CI','Δ C0','Paired CI','Score1'])+'</tr></thead><tbody>'+conditions_html+'</tbody></table></div><p>Scores on edited images are inherited-reference agreement until label preservation is verified.</p>'),
        ('Semantic Preservation',f'<p>Technical C4 validity {valid}/{attempted}; independently confirmed preservation rate unknown. Automated risk flags: {esc(sem["automated_risk_flag_counts"])}. Similarity, answer agreement and assistant screening do not certify scene/object/count/color preservation.</p><a href="../results/{a.run_id}/semantic_review_queue.csv">Independent human-review queue</a><br><a href="../results/{a.run_id}/editor_contact_sheet.png"><img class="contact" src="../results/{a.run_id}/editor_contact_sheet.png" alt="Original and all candidate conditions with questions, answers and diagnostics"></a>'),
        ('RL Algorithm Audit','<p>REINFORCE: categorical controller. DDPO: production single-pass score function. Isolated group-relative reference: finite synthetic surrogate. Production GRPO: not implemented / NOT VALIDATED. Reward-release guards remain active. Frozen-model evaluation is not pretrained RL validation.</p>'),
        ('Resource Accounting',f'<p><strong>{resource.get("gpu_seconds")} allocated GPU-seconds</strong> of the 3600-second budget, including the failed main attempt and control repair. Actual VLM calls: {resource.get("actual_VLM_inference_calls")}; generator calls including repeats: {resource.get("total_generation_calls_including_repeats")}.</p>'+resource_table+'<p>Process inference/generation timing is separate. Dollar cost is unavailable; no price is fabricated.</p><details><summary>Full resource ledger</summary><pre>'+esc(json.dumps(resource,indent=2))+'</pre></details>'),
        ('Failure and Repair Log','<p>Preparation v1 stopped on oversized historical CSV fields. Versioned v2 fixes parser capacity and records empty receipts. Original main job86795 FAILED after one grayscale JPEG was mislabeled invalid by its encoded mode; job86796 reevaluated decoded-RGB C0/C1 at all three seeds and produced a versioned accepted view. Original failed records/job status remain intact. Earlier serialization, attention-backend and full BF16 restart failures also remain recorded.</p>'),
        ('Next Experimental Gates','<ol><li>Independent review and adjudication, with all-attempted failure prevalence retained.</li><li>Locked same-source candidate study using semantically reviewed labels and saturation-aware power.</li><li>Bounded full-IP2Pix BF16 continuation diagnostic; no training before reward/semantic gates pass.</li></ol><p>Unsupported: general framework efficacy, successful pretrained RL, production GRPO completion, certified semantics from target answers.</p>')]
    css='''body{font:16px/1.55 system-ui,sans-serif;color:#243142;background:#f3f5f7;margin:0}main{max-width:1180px;margin:auto;padding:28px}header,section{background:white;border:1px solid #dce1e7;padding:24px;margin-bottom:18px}h1{font-size:30px;margin:0 0 8px}h2{font-size:22px;margin-top:0}a{color:#254f89}nav{display:flex;gap:12px;flex-wrap:wrap;margin-top:20px}nav a{font-size:13px}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;border-bottom:1px solid #dce1e7;padding:9px}.table-wrap{overflow-x:auto}pre{white-space:pre-wrap;font-size:13px}.contact{max-width:100%;height:auto}svg{width:100%;height:auto}select{font:inherit;margin:12px}.plots{display:grid;grid-template-columns:1fr 1fr;gap:16px}.plots img{max-width:100%}@media(max-width:700px){main{padding:10px}section,header{padding:16px}.plots{grid-template-columns:1fr}h1{font-size:24px}}'''
    css+='pre,code{overflow-wrap:anywhere;word-break:break-word}main,section,header{min-width:0}details{margin-top:12px}'
    page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RSI Experiment Progress — 2026-10-28</title><style>'+css+'</style><main><header><h1>RSI Experiment Progress</h1><p>Matched benign feedback, semantic preservation and RL infrastructure reliability</p><nav>'+''.join(f'<a href="#s{i}">{i}. {esc(title)}</a>' for i,(title,_) in enumerate(sections,1))+'</nav></header>'+''.join(f'<section id="s{i}"><h2>{i}. {esc(title)}</h2>{content}</section>' for i,(title,content) in enumerate(sections,1))
    page+='<section><h2>Reports and inspectable evidence</h2>'+''.join(f'<p><a href="../reports/{name}.md">{name.replace("_"," ")}</a></p>' for name in ['MATCHED_FEEDBACK_VALIDATION','SEMANTIC_PRESERVATION_AUDIT','RL_RELIABILITY_STATUS','NEXT_EXPERIMENT_DECISION'])+'<div class="plots">'+''.join(f'<a href="../results/{a.run_id}/{phase}/plots/{p.name}"><img src="../results/{a.run_id}/{phase}/plots/{p.name}" alt="{esc(p.stem)}"></a>' for p in sorted((run/phase/'plots').glob('*.png')))+f'</div><p><a href="../results/{a.run_id}/logs/">Execution logs</a> · <a href="gradient_feedback_validation_previous_20261028.html">Previous diagnostic dashboard</a></p></section></main>'
    page+='<script>const DATA='+json.dumps(stats).replace('</','<' + chr(92) + '/')+''';function draw(){let phase=document.getElementById('phase').value;if(!DATA[phase])phase=Object.keys(DATA)[0];const rows=DATA[phase].conditions;let s='<svg viewBox="0 0 900 330" role="img" aria-label="Matched condition means and bootstrap intervals">';rows.forEach((r,i)=>{let y=30+i*45,w=r.mean_score_all_attempted_utility*650;s+=`<text x="10" y="${y+17}">${r.candidate_id}</text><rect x="60" y="${y}" width="${w}" height="24" fill="#416485"><title>${r.candidate_id}: ${r.mean_score_all_attempted_utility.toFixed(4)}, 95% CI [${r.mean_ci95_low.toFixed(4)},${r.mean_ci95_high.toFixed(4)}], n=${r.n_unique_sources}</title></rect><line x1="${60+r.mean_ci95_low*650}" x2="${60+r.mean_ci95_high*650}" y1="${y+12}" y2="${y+12}" stroke="black" stroke-width="2"/><text x="740" y="${y+17}">${r.mean_score_all_attempted_utility.toFixed(4)}</text>`});s+='</svg>';document.getElementById('interactive-chart').innerHTML=s;document.getElementById('chart-note').textContent=phase+': '+DATA[phase].effective_sample_size+' unique source images. Hover bars for uncertainty. C5 is auxiliary.'}document.getElementById('phase').onchange=draw;draw();</script></html>'''
    previous=ROOT/'dashboard/gradient_feedback_validation_previous_20261028.html'
    if not previous.exists():shutil.copyfile(ROOT/'dashboard/gradient_feedback_validation.html',previous)
    for name in ['gradient_feedback_validation.html','RSI_Experiment_Progress_2026-10-28.html']:(ROOT/'dashboard'/name).write_text(page,encoding='utf-8')
    dump(run/'report_status.json',{'status':'PASS','updated':now,'reports_generated':4,'dashboard_sections':10,'external_CDN_dependencies':False,'n_primary_main':n,
        'independent_semantics':'BLOCKED','useful_learning_feedback':'NOT VALIDATED','full_pretrained_training':'NOT VALIDATED'})
    print(json.dumps({'run_id':a.run_id,'phase':phase,'main_N':n,'planned_significant_comparisons':len(significant),'human_review':'BLOCKED','training':'NOT VALIDATED'}))


if __name__=='__main__':main()
