# Phase 2 smoke tests

PASS is engineering acceptance; semantic truth and generator-RL readiness have separate blockers.

| Test | Status | Exact script command | Job | Experiment | Actual result | Logs / diagnosis / fix |
|---|---|---|---|---|---|---|
| 1 Pretrained editor | PASS | `.venv-editor/bin/python scripts/smoke_pretrained_editor.py` | 86292 | `EXP_20261027T045901_48eded21ef7d_phase2_editor_inference` | 5 actual images; grid and resources saved | metrics.json, train.log; parent slurm-JOB.out/.err. PASS; green-shift fidelity limitation retained |
| 2 Actual generator adapter | PASS | `.venv-editor/bin/python scripts/smoke_generator_training.py` | 86306 | `EXP_20261027T052543_a20882fb60de_phase2_generator_lora_smoke` | 1594368 adapter parameters; public PNG difference 0; exact resume PASS | metrics.json, train.log; parent slurm-JOB.out/.err. Earlier offline naming, alpha, precision and SDPA-context failures fixed; all attempts retained in PHASE2_FAILURES.md |
| 3 Reward-trained generator | BLOCKED | `.venv/bin/python scripts/check_generator_rl_contract.py` | None | `EXP_20261027T051832_ba75b1635cca_phase2_generator_rl_contract` | Raises explicit unsupported error; supervised updates do not satisfy this test | metrics.json, train.log; parent slurm-JOB.out/.err. Missing validated native trajectory likelihood and gradient mechanism; no substitute RL claim |
| 4 Semantic decision interface | PASS | `.venv/bin/python scripts/validate_edited_images.py` | 86307 | `EXP_20261027T052713_2e70f825d4bd_phase2_semantic_validation` | 20 pairs: 5 verified identities, 5 constructed invalid; 10 changed pairs await independent review | metrics.json, train.log; parent slurm-JOB.out/.err. Interface/controls PASS; independent natural-image oracle still BLOCKED |
| 5a VQA development 200 | PASS | `.venv/bin/python scripts/run_vqa_baseline_full.py --split validation` | 86300 | `EXP_20261027T050746_4556a0cd5c4e_phase2_qwen_baseline_validation` | 200 examples; 81.15% official score | metrics.json, train.log; parent slurm-JOB.out/.err. Official scoring parity repaired; no post-result protocol changes |
| 5b Internal held-out 300 once | PASS | `.venv/bin/python scripts/run_vqa_baseline_full.py --split test` | 86305 | `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test` | 300 examples once; 77.63% internal held-out official score | metrics.json, train.log; parent slurm-JOB.out/.err. Same frozen protocol; once-only guard retained |
| 6 Real-editor controller RL update smoke | PASS | `.venv-editor/bin/python scripts/compare_rl_rsi.py` | 86308 | `EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation` | 6 fixed updates; 3 base + 3 candidate updates; revision accepted=False | metrics.json, train.log; parent slurm-JOB.out/.err. Real proxy feedback; single seed and nominal operation match only; no superiority conclusion |
| 7 Real-observation RSI | PASS | `.venv-editor/bin/python scripts/compare_rl_rsi.py` | 86308 | `EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation` | 6 fixed updates; 3 base + 3 candidate updates; revision accepted=False | metrics.json, train.log; parent slurm-JOB.out/.err. Real proxy feedback; single seed and nominal operation match only; no superiority conclusion |
| 8 Registry/versioning | PASS | `.venv/bin/python scripts/collect_phase2_evidence.py` | CPU | reports/phase2_evidence_inventory.json | uniqueness, all files, source commits verified | Failures remain separately indexed; no finalized runs overwritten |

Commands above are executed from the canonical project root. Each experiment ID resolves to `experiments/ID/`. Its manifest records the parent ID and executing commit; parent directories contain exact job.sbatch and Slurm stdout/stderr. CPU contract checks have no Slurm ID. Full raw metrics, images and checkpoints accompany the focused reports.

Actual failure attempts are preserved and itemized in PHASE2_FAILURES.md. Parent Slurm output/error files identify the command and failure. No failed full generator attempt is rewritten as PASS.

Unit suite: 14 tests passed in the Qwen and editor environments; separate fresh-agent documented editor invocation also passed 14 without edits (PHASE2_DOC_INVOCATION_REVIEW.md). Two-round same-family source review is provisional; runtime follow-through is documented in implement-stage/PHASE2_REVIEW_FOLLOWUP.md.

Remaining scientific blockers: valid reward-based generator trajectory optimizer; independent natural-image human verification; meaningful repeated-seed compute-matched advantage evaluation.
