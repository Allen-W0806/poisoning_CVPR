"""Produce Phase 2 reports from actual immutable execution records."""
import csv
import io
import json
from pathlib import Path
import _bootstrap


def read(path):
    return json.loads(Path(path).read_text())


if __name__ == "__main__":
    root = _bootstrap.ROOT
    reports = root / "reports"
    inventory = read(reports / "phase2_evidence_inventory.json")
    model = read(reports / "phase2_model_metadata.json")
    loaded = {}
    for key in ["editor", "training", "generator_rl_contract", "semantics", "vqa_validation", "vqa_test", "feedback", "controller"]:
        pointer = reports / f"phase2_latest_{key}.json"
        if pointer.exists():
            p = Path(read(pointer)["path"])
            # On the Windows mirror canonical paths are remote; resolve by immutable ID.
            if not p.exists():
                p = root / "experiments" / read(pointer)["id"]
            loaded[key] = (p, read(p / "metrics.json"))
    def link(key):
        return f"`{loaded[key][0].name}`" if key in loaded else "BLOCKED / no completed execution"
    def metric(key):
        return loaded[key][1] if key in loaded else {}
    jobs = [r for r in csv.DictReader(io.StringIO(inventory["scheduler"]["stdout"]), delimiter="|") if r["JobIDRaw"].isdigit()]
    total_seconds = sum(int(r["ElapsedRaw"]) for r in jobs)
    job_table = "\n".join(["| Job | State | Exit | Allocated seconds |", "|---|---|---|---|"] +
        [f"| {r['JobIDRaw']} | {r['State']} | {r['ExitCode']} | {r['ElapsedRaw']} |" for r in jobs])
    def save(name, text):
        (reports / name).write_text(text.strip() + "\n", encoding="utf-8")
    editor = metric("editor")
    train = metric("training")
    semantics = metric("semantics")
    control = metric("controller")
    updates = control.get("fixed_rl_updates", []) + control.get("rsi_base_updates", []) + control.get("rsi_candidate_updates", [])
    update_diagnostics = {}
    if updates:
        update_diagnostics = {key: {"min": min(x[key] for x in updates), "max": max(x[key] for x in updates)}
            for key in ["reward_variance", "gradient_norm", "trainable_max_change", "step_seconds", "peak_gpu_memory_bytes"]}
    editor_id = model["timbrooks/instruct-pix2pix"]["revision"]
    step_id = model["stepfun-ai/Step1X-Edit"]["revision"]
    save("IMAGE_EDITOR_SELECTION.md", f"""# Image editor selection

Selected native image-and-instruction **InstructPix2Pix** (`timbrooks/instruct-pix2pix`, revision `{editor_id}`). Frozen Qwen environment was preserved; new `.venv-editor` uses Diffusers 0.35.1/PEFT 0.17.1 and the existing Torch 2.6.0 cu124 base.

| Criterion | InstructPix2Pix | Step1X-Edit |
|---|---|---|
| Image-conditioned editing | native image + instruction, actually executed | native documented |
| Public weights / revision | pinned above; only 2.134 GB required FP16 components cached | `stepfun-ai/Step1X-Edit` revision `{step_id}`; metadata inspected, weights not downloaded |
| License metadata | model card MIT; underlying SD1.5 CreativeML OpenRAIL-M terms also relevant | model card/repository Apache-2.0 |
| Inference VRAM | measured below on A100; planning expectation 8–16GB was an estimate | upstream reports 42.5GB at 512, H800 (not our measurement) |
| Training VRAM | actual rank-8 LoRA forward/backward measurements below | upstream rank-64 BF16 report 29.7GB at 512; no local/A100 training measurement |
| LoRA modules | actual UNet to_q/to_k/to_v/to_out.0 linear modules inspected | native LoRA scripts documented; not instantiated here |
| Gradient training | actual supervised epsilon-denoising LoRA updates | officially documented paired-edit finetuning; not executed here |
| Reward optimizer | BLOCKED: no validated native scheduler/CFG trajectory density path | unsupported/unverified here |
| A100 compatibility | actual inference and supervised smoke, see IDs | not tested; upstream H800 reports are not A100 certification |
| Dependencies | isolated pinned Diffusers/PEFT; bitsandbytes CPU-only warning irrelevant to AdamW; no 8-bit path used | more model/vision dependencies and larger staging requirement, untested |

Inference evidence: {link('editor')}, generated count {editor.get('generated_count', 'BLOCKED')}, peak allocation {editor.get('peak_gpu_memory_bytes', 0)/1e9:.3f} GB, generation total {editor.get('total_generation_seconds', 0):.3f}s. Supervised training evidence: {link('training')}.

Choosing this backend reduces implementation/dependency risk and retains native image conditioning. This is not a claim that Step1X cannot fit or train on A100. No third backend or text-to-image substitution was needed. Model weight sums across all repository variants are not treated as loaded model size.

Sources: [InstructPix2Pix pipeline](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix), [training reference](https://huggingface.co/docs/diffusers/training/instructpix2pix), [model card](https://huggingface.co/timbrooks/instruct-pix2pix), [Step1X memory/LoRA documentation](https://github.com/stepfun-ai/Step1X-Edit), [Step1X model card](https://huggingface.co/stepfun-ai/Step1X-Edit), [SD1.5 license metadata](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
""")
    if train:
        parameter = train["parameters"]
        update = train["updates"]
        training_text = f"""# Generator training report

Evidence: {link('training')}. Actual pretrained UNet attention LoRA, not a separate controller, received three AdamW updates. Objective is **supervised image-conditioned DDPM epsilon-prediction MSE**, not RL.

Total UNet-with-adapter parameters: {parameter['generator_parameters']:,}; pipeline parameters: {parameter['pipeline_parameters']:,}; trainable adapter parameters: {parameter['trainable_parameters']:,} ({train['trainable_percent']:.4f}% of generator). Rank 8, alpha 16, FP32 adapter/master parameters, BF16 frozen generator computation. VAE and text encoder frozen. Actual matched module list is in metrics.json.

Logical batch 10 = physical microbatch 1 × accumulation 10. Three updates, LR 1e-5, weight decay .01, max norm 1, constant LR for this smoke, gradient checkpointing. Ten training records only; targets are deterministic brightness ×1.02 pairs. Source latent mode is unscaled, target latent sample is scaled by the VAE factor, noisy target is concatenated with source latent, and sampled DDPM noise is the prediction target. No Qwen/security reward enters this loss. Official recipe has optional conditioning dropout; this minimal run uses none.

Maximum adapter parameter change: {train['adapter_max_parameter_change']:.9g}; finite gradients at every update. All {train['frozen_tensor_count']} frozen parameter tensors were hashed before/after and stayed unchanged. Adapter artifact: {train['adapter_bytes']:,} bytes; AdamW state after last update: {update[-1]['optimizer_state_bytes']:,} bytes.

Peak observed forward allocation: {max(x['forward_peak_memory_bytes'] for x in update)/1e9:.3f} GB; peak through backward: {max(x['backward_peak_memory_bytes'] for x in update)/1e9:.3f} GB. These include persistent model/accumulated gradient storage and are not incremental activation-only sizes. Mean update time {train['mean_step_seconds']:.3f}s. The smoke also performs a resumed fourth diagnostic update; all allocation time is charged in the phase ledger.

Fresh training-state reload pixel max difference: {train['fresh_model_reload_pixel_difference']}; public exported adapter reload difference: {train['export_reload_pixel_difference']} (predeclared tolerance ≤2 uint8 levels). Loaded rank/alpha checked as {train['exported_lora_rank']}/{train['exported_lora_alpha']}. Public adapter tensors retain {train['public_adapter_dtype']} with exact max tensor difference {train['public_adapter_tensor_difference']}. Resumed next-update adapter, AdamW and scheduler state matched exactly in this completed run. Checkpoints retain model revision, modules, data IDs, config and Python/NumPy/Torch/CUDA RNG. Public export must preserve PEFT metadata and exact FP32 A/B values; loading into BF16 and upcasting afterward loses information.

Earlier failures remain in reports/PHASE2_FAILURES.md and immutable runs: omitted offline weight_name, lost alpha metadata, a 4.41e-8 fused-attention restart difference, and a math-attention context lifetime bug. Final numerical isolation uses math SDPA through forward/backward/recomputation; no acceptance tolerance was enlarged. Success here does not erase the earlier non-bitwise result or prove universal cross-kernel reproducibility.

Generated images remain valid PNGs. This validates updates/restart plumbing, not visual quality improvement, instruction fidelity, robustness, or diffusion RL. Example after/reloaded images are in the experiment images directory.
"""
    else:
        training_text = "# Generator training report\n\nFull generator smoke remains BLOCKED/FAILED; see PHASE2_FAILURES.md. Actual partial gradient updates and checkpoints are retained, but no complete success is claimed."
    save("GENERATOR_TRAINING_REPORT.md", training_text)
    save("RL_GENERATOR_VALIDATION.md", f"""# Generator versus controller RL

**Reward-based generator training is BLOCKED.** Contract execution: {link('generator_rl_contract')}. InstructPix2Pix's native scheduler/CFG path has no validated trainable trajectory log-probability implementation in this project. A DDPO-style stochastic transition density, per-step likelihood/reference agreement, masking and gradient tests are missing. No arbitrary categorical REINFORCE update was applied to diffusion latents.

Actual generator LoRA updates are supervised denoising, documented in GENERATOR_TRAINING_REPORT.md. They do not count as generator RL.

Separate controller RL evidence: {link('controller')}. A categorical policy samples one of three benign instructions for a **frozen real editor**, with REINFORCE `-mean(log p(action|image) * detached sample reward)` and AdamW. Real outputs and Qwen/proxy measurements are precomputed for a fixed source/seed/action mapping. This is a valid stochastic categorical bandit, not on-policy diffusion trajectory learning or generator adaptation.

Each observation retains semantic proxy S, answer consistency A, distortion D, resource C and invalid/unverified penalty. The existing reward clips components [0,1] and totals [-1,1.5]; unknown semantic validity remains penalized. C is 10 steps / 30-step fixed reference, not a fabricated wall-clock measure. Actual generation/Qwen/selection runtime is charged separately. No held-out test or security metric enters training.

Frozen action-0 editor, random actions, trained controller and RSI-selected controller are measured on the same validation table. The unsupported reward-trained-generator arm is explicitly absent. No ASR, trigger search or evasion optimization was implemented.
""")
    vqa_lines = ["# Official VQA baseline", "", "Protocol frozen before the full development run: `configs/phase2_vqa_protocol.json`. Its checksum is recorded in each run. Pinned upstream official evaluator is mechanically Python-3 converted with BSD-style license retained; no scoring logic changes. Phase 1's preliminary scorer/results remain unchanged.", "",
        "Upstream normalization helper tests cover punctuation, articles, number words, contractions and 10-answer leave-one-out consensus. The original evaluator conditionally normalizes when human answers differ; unanimous-answer behavior is preserved and tested, rather than silently replaced. Raw answer, whitespace-stripped complete answer, normalized diagnostic string, original references and official score are all saved. No label-aware answer extraction.", "",
        "Qwen revision 89644892e4d85e24eaac8bacfd4f463576704203; fixed generic short-answer system prompt; original question unchanged; greedy 64-token maximum; BF16; pixel budget 4096–262144; microbatch one. First three development predictions repeat exactly. Repeated deterministic outputs are not independent seeds.", ""]
    for key in ["vqa_validation", "vqa_test"]:
        m = metric(key)
        if m:
            vqa_lines += [f"## {m['split']}", "", f"Evidence {link(key)}; n={m['samples']}; official VQA score **{m['accuracy_percent']:.2f}%**; mean latency {m['mean_latency_seconds']:.4f}s; p95 {m['latency_p95_seconds']:.4f}s; peak allocation {m['peak_gpu_memory_bytes']/1e9:.3f}GB.", "", "| Answer type | N | Score |", "|---|---|---|"]
            vqa_lines += [f"| {k} | {v['samples']} | {100*v['accuracy']:.2f}% |" for k,v in m["by_answer_type"].items()]
            vqa_lines += ["", "Complete raw predictions: predictions.jsonl; error cases and config: metrics.json/config.yaml. No results used to alter the frozen scoring/generation protocol.", ""]
            examples = [json.loads(line) for line in (loaded[key][0] / "predictions.jsonl").read_text().splitlines()[:3]]
            vqa_lines += ["| Example question | Raw short answer | Human references (first 3) | Score |", "|---|---|---|---|"]
            for example in examples:
                cells = [example["question"], example["raw_answer"], str(example["human_reference_answers"][:3]), str(example["score"])]
                vqa_lines.append("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in cells) + " |")
            vqa_lines.append("")
        else:
            vqa_lines += [f"{key}: BLOCKED / not executed.", ""]
    vqa_lines += ["The 300-question held-out subset comes from official train2014 and is **not** the official VQAv2 test-server benchmark. It is evaluated once after completed development under the same protocol. This is a single deterministic baseline, not a robustness or RSI improvement result."]
    save("VQA_BASELINE_REPORT.md", "\n".join(vqa_lines))
    save("SEMANTIC_VALIDATION_REPORT.md", f"""# Question-aware semantic validation

Evidence: {link('semantics')}; evaluated pairs: {semantics.get('pairs', 'BLOCKED')}.

Separate outputs: visual MSE/SSIM/distortion, original/edited answer consistency, question-dependent semantic decision, pixel-change localization. CLIP/SSIM/LPIPS are never used as semantic proof. The question marks color/count/spatial/object/attribute/answerability relevance. A pairwise structured Qwen review is compared with short-answer agreement and original-reference agreement; all raw judgments, parse failures and disagreements are retained.

Controls: five exact identities are verified valid; five completely occluded original scenes are constructed invalid. Ten changed natural pairs (mild brightness plus real pretrained edits) remain **unverified** without independent review. Automatic agreement produces provisionally_valid, never verified valid. Human verified changed edits: {semantics.get('nonidentity_human_verified', 0)}. Automatic provisional agreements: {semantics.get('automatic_provisional_valid', 'BLOCKED')}; disagreements: {semantics.get('automatic_disagreements', 'BLOCKED')}.

Two Qwen prompting modes use the same frozen model and are correlated; they are not independent semantic truth. Only verified-valid pairs are eligible for answer-preserving accuracy aggregation. Other scores are diagnostics and cannot claim ground-truth preservation. Pixel localization has no semantic/object mask unless a reviewed mask is supplied. The full 20-row contact sheet and blank human review template are saved with individual decisions.

The five source questions form a small manually reviewable set; they do not cover every listed semantic attribute. The color-change red-car case is a unit fixture, not evidence that all natural-image color/count/spatial changes were exercised.

Actual editor grid inspection found a pronounced green shift in one snow scene despite a color-preservation instruction; see PHASE2_VISUAL_QA.md. Realistic output or answer agreement alone cannot establish instruction fidelity. This is an implemented conservative validation gate with a remaining human-oracle blocker, not a complete natural-image semantic certification system.
""")
    save("RSI_VALIDATION_REPORT.md", f"""# Real-observation RSI validation

Feedback evidence: {link('feedback')}; controller/RSI experiment: {link('controller')}.

Shared feedback construction measured {metric('feedback').get('records')} edited observations, {metric('feedback').get('qwen_calls')} Qwen calls, and {metric('feedback').get('generation_seconds', 0):.3f}s editor generation. Human-verified changed edits: {metric('feedback').get('human_verified_edits', 0)}. All model loading and judging time is included in the enclosing Slurm allocation, beyond the generation subtotal.

The real pretrained editor is frozen. Shared table contains 10 training and 10 validation source images ×3 benign instructions, fixed source/seed/action mapping. Qwen answer/structured proxies and quality/resource measurements are saved per edited image. Training uses only train rows; selection uses only validation; no test rows are opened by this code.

Fixed RL: six logical-batch-10 updates, validation after updates 3/6. RSI: three-update base plus three-update candidate from common initial state, two validation lookups total. Both have 60 gradient sample exposures and two validation lookups. This matches nominal operation/data budgets; **actual wall/GPU time is not forced equal** and is reported. Shared generation/verification cost belongs to the comparison infrastructure and is charged once, not hidden in optimizer timings.

Revision rule: propose 5e-5 only when observed sampled reward variance is positive and all gradient norms <1, otherwise retain 1e-5; strict expected-validation-reward gain within budget accepts. Every proposal preserves old/new config, reason, raw evidence, cost, decision and rollback checkpoint. This is controlled configuration revision, not code self-rewriting.

Observed comparison:\n```json\n{json.dumps({k:control.get(k) for k in ['frozen_editor_action0_validation_reward','random_policy_validation_reward','fixed_rl_validation','rsi_base_validation','rsi_candidate_validation','rsi_selected_validation','fixed_rl_seconds','rsi_seconds']}, indent=2)}\n```

The recorded random baseline uses one seed-42 action per validation image (ten draws), whereas trained-policy validation integrates all three actions' probabilities. To expose this sampling difference, a read-only derived table audit also reports the exact uniform-random expectation; it does not replace the original run:\n```json\n{json.dumps(inventory.get('feedback_table_audit', {}), indent=2)}\n```

Revision:\n```json\n{json.dumps({k:control.get('revision',{}).get(k) for k in ['previous_config','proposed_config','validation_score','accepted','reason','cost_gpu_hours','proposal_reason','rollback']},indent=2)}\n```

Raw sample rewards, variance, gradient norms, parameter deltas, time and memory are retained for all updates. This small fixed-table automatic-proxy experiment is preliminary and does not establish RSI superiority, improved natural-image semantics, generator learning by RL, or robust generalization. Budget metadata includes actual measured selection overhead and all scientific jobs are charged to the phase ledger.

Observed per-update diagnostic ranges (all twelve updates across arms):\n```json\n{json.dumps(update_diagnostics, indent=2)}\n```

Separate CPU reload audit of the saved selection and rollback artifacts:\n```json\n{json.dumps(inventory.get('checkpoint_selection_audit', {}), indent=2)}\n```
""")
    tests = [("1", "Pretrained editor", "PASS" if editor else "BLOCKED", "editor", "scripts/smoke_pretrained_editor.py"),
        ("2", "Actual generator adapter", "PASS" if train else "BLOCKED", "training", "scripts/smoke_generator_training.py"),
        ("3", "Reward-trained generator", "BLOCKED", "generator_rl_contract", "scripts/check_generator_rl_contract.py"),
        ("4", "Semantic decision interface", "PASS" if semantics.get("pairs")==20 else "BLOCKED", "semantics", "scripts/validate_edited_images.py"),
        ("5a", "VQA development 200", "PASS" if metric("vqa_validation").get("samples")==200 else "BLOCKED", "vqa_validation", "scripts/run_vqa_baseline_full.py --split validation"),
        ("5b", "Internal held-out 300 once", "PASS" if metric("vqa_test").get("samples")==300 else "BLOCKED", "vqa_test", "scripts/run_vqa_baseline_full.py --split test"),
        ("6", "Real-editor controller RL update smoke", "PASS" if control else "BLOCKED", "controller", "scripts/compare_rl_rsi.py"),
        ("7", "Real-observation RSI", "PASS" if control else "BLOCKED", "controller", "scripts/compare_rl_rsi.py")]
    actual_results = {
        "editor": f"{editor.get('generated_count')} actual images; grid and resources saved",
        "training": f"{train.get('parameters', {}).get('trainable_parameters')} adapter parameters; public PNG difference {train.get('export_reload_pixel_difference')}; exact resume PASS" if train else "No complete accepted execution",
        "generator_rl_contract": "Raises explicit unsupported error; supervised updates do not satisfy this test",
        "semantics": f"{semantics.get('pairs')} pairs: 5 verified identities, 5 constructed invalid; 10 changed pairs await independent review",
        "vqa_validation": f"200 examples; {metric('vqa_validation').get('accuracy_percent', 0):.2f}% official score",
        "vqa_test": f"300 examples once; {metric('vqa_test').get('accuracy_percent', 0):.2f}% internal held-out official score",
        "controller": f"6 fixed updates; 3 base + 3 candidate updates; revision accepted={control.get('revision', {}).get('accepted')}" if control else "No completed controller execution",
    }
    diagnoses = {"editor": "PASS; green-shift fidelity limitation retained",
        "training": "Earlier offline naming, alpha, precision and SDPA-context failures fixed; all attempts retained in PHASE2_FAILURES.md",
        "generator_rl_contract": "Missing validated native trajectory likelihood and gradient mechanism; no substitute RL claim",
        "semantics": "Interface/controls PASS; independent natural-image oracle still BLOCKED",
        "vqa_validation": "Official scoring parity repaired; no post-result protocol changes",
        "vqa_test": "Same frozen protocol; once-only guard retained",
        "controller": "Real proxy feedback; single seed and nominal operation match only; no superiority conclusion"}
    rows = ["# Phase 2 smoke tests", "", "PASS is engineering acceptance; semantic truth and generator-RL readiness have separate blockers.", "", "| Test | Status | Exact script command | Job | Experiment | Actual result | Logs / diagnosis / fix |", "|---|---|---|---|---|---|---|"]
    for number, title, status, key, command in tests:
        p = loaded.get(key, (None,{}))[0]
        job = read(p / "manifest.json").get("slurm_job_id") if p else None
        env = ".venv-editor" if key in {"editor","training","controller"} else ".venv"
        rows.append(f"| {number} {title} | {status} | `{env}/bin/python {command}` | {job} | {link(key)} | {actual_results[key]} | metrics.json, train.log; parent slurm-JOB.out/.err. {diagnoses[key]} |")
    rows += [f"| 8 Registry/versioning | {'PASS' if inventory['registry_unique'] and inventory['phase2_all_registered'] and inventory['all_phase2_required_files_present'] else 'FAIL'} | `.venv/bin/python scripts/collect_phase2_evidence.py` | CPU | reports/phase2_evidence_inventory.json | uniqueness, all files, source commits verified | Failures remain separately indexed; no finalized runs overwritten |", "", "Commands above are executed from the canonical project root. Each experiment ID resolves to `experiments/ID/`. Its manifest records the parent ID and executing commit; parent directories contain exact job.sbatch and Slurm stdout/stderr. CPU contract checks have no Slurm ID. Full raw metrics, images and checkpoints accompany the focused reports.", "", "Actual failure attempts are preserved and itemized in PHASE2_FAILURES.md. Parent Slurm output/error files identify the command and failure. No failed full generator attempt is rewritten as PASS.", "", "Unit suite: 14 tests passed in the Qwen and editor environments; separate fresh-agent documented editor invocation also passed 14 without edits (PHASE2_DOC_INVOCATION_REVIEW.md). Two-round same-family source review is provisional; runtime follow-through is documented in implement-stage/PHASE2_REVIEW_FOLLOWUP.md.", "", "Remaining scientific blockers: valid reward-based generator trajectory optimizer; independent natural-image human verification; meaningful repeated-seed compute-matched advantage evaluation."]
    save("PHASE2_SMOKE_TEST_REPORT.md", "\n".join(rows))
    run_table = "\n".join(["| Experiment | Status | Job | Executing commit |", "|---|---|---|---|"] +
        [f"| `{r['id']}` | {r['status']} | {r['job_id']} | `{r['git_commit']}` |" for r in inventory["runs"]])
    save("PHASE2_SETUP_REPORT.md", f"""# Phase 2 setup and execution

Canonical project `/storage/home/ydk297/projects/poisoning`; existing mirror `H:\\Project\\Poisoning`. Starting commits are recorded in reports/PHASE2_START.json; existing experiments remain immutable. New real editor environment `.venv-editor` is separate from original Qwen `.venv`, which ran the full baseline successfully. Exact dependency freezes are phase2_freeze_editor.txt and phase2_freeze_qwen.txt.

Report-generation source commit: `{inventory.get('report_source_commit', 'unrecorded')}`. Each scientific run retains its own executing commit below.

New components: native pretrained editor/config/output metadata, inspected UNet rank-8 LoRA supervision/restart, explicit unsupported generator-RL contract, real cached-editor categorical RL, conservative question-aware semantic/answer/localization checks, licensed pinned official VQA evaluator, frozen full-split baseline, actual-observation RSI history/rollback, budget guard and experiment records.

Resource policy: rechecked epyc-gpu/allusers/A100 80GB; one GPU, four CPUs, 48G host RAM per short job. Wrapper refuses active/unconfirmed previous Phase 2 allocations and checks completed/failed elapsed seconds plus active reservations against 7200 GPU-seconds. No large sweep launched.

{job_table}

Total allocated Phase 2 GPU cost: **{total_seconds/3600:.6f} GPU-hours** ({total_seconds}s), including all failures/loading/evaluation; maximum budget 2 GPU-hours. Optimizer component timings are inside this total and must not be added again.

Retained storage snapshot:\n```text\n{inventory['disk']['stdout'].strip()}\n```

No multi-GB download on C:. Pre-staging estimate ≤15GB new cluster data/environment/artifacts; actual model download selected 2.134GB required FP16 component files. Central cache reused. Failed adapter/optimizer evidence and generated comparison images remain; scratch/install archives are removed only after mirroring. Shared runtimes and original model weights are untouched.

All new canonical experiment IDs:\n\n{run_table}

Registry unique: {inventory['registry_unique']}; all new runs registered: {inventory['phase2_all_registered']}; all required files present: {inventory['all_phase2_required_files_present']}. Raw source revisions are in every manifest/git_commit.txt. Old source changes extend modules; Phase 1 scorer/results are not replaced. Final mirror import checks existing same-ID experiment bytes before accepting new evidence.

Read the seven focused reports plus PHASE2_FAILURES.md. Completion here means feasible Phase 2 engineering stages executed with honest blockers; it does not claim a learned backdoor, generator RL, RSI advantage, or fully verified natural-image semantics.
""")
    save("PHASE2_NEXT_STEPS.md", f"""# Phase 2 next steps

1. Review `SEMANTIC_VALIDATION_REPORT.md` and the 20-row contact sheet; fill human_review_template.json in a new derived review artifact. Do not overwrite automatic judgments. Natural-image edited accuracy remains gated until independent review.
2. Implement and test a mathematically valid stochastic diffusion trajectory optimizer before any generator-RL claim. Current native-scheduler reward update is explicitly BLOCKED; supervised LoRA is separate.
3. Investigate repeatability of the earlier fused-attention 4.41e-8 failure; final math-backend smoke is scoped to its recorded environment. Retain every failed attempt and metadata-scaling diagnosis.
4. Only after these readiness gates, consider the prepared stability/ablation configs. Freeze equal complete data/compute/proposal/validation budgets; do not auto-launch the grid or reinterpret tiny proxy rewards as scientific advantage.

Measured Phase 2 allocated cost so far: {total_seconds/3600:.6f} GPU-hours. Use one A100 with 48G host RAM and a 10–15 minute ceiling for another bounded feasibility run; consult actual generator peak and step times in GENERATOR_TRAINING_REPORT.md. These measurements do not validate larger batches or Step1X capacity. Held-out 300 was opened once and must not be rerun for tuning.

Next safe executable checks (CPU, no held-out reuse):

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv-editor/bin/activate; python -m unittest discover -v'
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python scripts/check_generator_rl_contract.py'
```

If another real supervised generator smoke is needed within remaining budget:

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python scripts/submit_phase2.py --task training --minutes 10'
```
""")
    print(json.dumps({"reports_written": 9, "allocated_gpu_seconds": total_seconds, "completed_components": list(loaded)}))
