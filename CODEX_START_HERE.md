# Codex task: audit, minimal reconstruction, and BENIGN experimental validation

## Read this first

You are working in an existing research repository. The attached `ARCHITECTURE_DECISION.md` is the author's current *proposal*, NOT proof that the implementation works. Check all claims against the actual repo. This task concerns generic image-editing RL/RSI methodology only. **Do not implement, improve, tune, or automate a VLM backdoor/poisoning generation loop; do not use ASR, stealth, target-answer redirection, or poisoning ratios as train-time rewards or as a selection signal.** If code has these capabilities, separate and disable them in the benign experiment entrypoints. Any security-oriented metrics remain offline, read-only robustness-reporting activities; no feedback from them should enter generator or controller training.

## Read and inspect (in order)

1. `ARCHITECTURE_DECISION.md` and the latest main paper `.tex` (if present).
2. Experiment scripts, configs, GPU launch scripts, data handling, model loading, training loops, sampling/schedulers, metrics, and saved logs.
3. The existing brightness pilot, including state of the UNet LoRA, stochastic sampler, log-prob calculations, optimizer, AMP/BF16, gradient scaling, and reproducibility.
4. Existing reports; compare reported runs with checkpoint and log evidence. Explicitly mark unverified assertions.

If an input is missing, report the missing path; do not invent a file, checkpoint, result, or dataset.

## IMPORTANT corrections to the attached architecture file

A. DDPO **does not require distinct per-step rewards or learned per-step credit**. The original DDPO paper formalizes reverse diffusion as an MDP, with the terminal image reward providing the trajectory return, and uses the score-function/importance-sampling policy-gradient estimators over denoising transitions. The actual sampled stochastic transitions and the log-probabilities used by training must be consistent. Verify this in the original paper and correct `ARCHITECTURE_DECISION.md` §1.4–1.5 and the method draft where needed.

B. A one-off score-function update is not automatically a verified DDPO implementation. Do not call it DDPO merely because it updates the LoRA from a scalar reward. Validate the estimator, stochastic sampling semantics, per-step likelihoods, gradient flow, repeated collection/update cycles, and independent test behavior.

C. The Outer Controller in the proposal is *not proven* to be a trained sequential RL policy. UCB/Thompson bandit selection is acceptable to study, but document it precisely. If all the outer controller does is hyperparameter search, call it HPO/adaptive search instead of claiming RSI. RSI claims require held-out transfer evidence that the **improvement procedure** has improved, not just one model configuration.

D. Avoid declaring DRaFT "worse" or "better" without a controlled benchmark. It is a differentiable-reward optimization alternative, not interchangeable with a score-function/DDPO update.

Primary references (verify rather than merely citing):
- DDPO: https://proceedings.iclr.cc/paper_files/paper/2024/hash/14f75513f0f1ca01de1e826b52e6b840-Abstract-Conference.html
- DDPO project / official implementation: https://rl-diffusion.github.io/
- DRaFT: https://deepmind.google/research/publications/51081/
- InstructPix2Pix: https://www.timothybrooks.com/instruct-pix2pix
- RRSI: https://github.com/google-research/rrsi
- AIDE2: https://arxiv.org/abs/2609.26457

## Execution workflow: DO work, not only create plans

### P0: Non-destructive repo and environment audit
- Log `git status`, current commit, Python/PyTorch/Diffusers/Accelerate versions, CUDA devices and free memory, dependency lock/state, and available model/data assets. Do not hardcode user hardware or assume unused GPUs. Do not erase outputs or change any production job.
- Inventory experiment entrypoints and their real vs expected behavior. Build or fix a simple test command using existing dependencies where possible.
- Run unit/smoke tests first. For any heavy job, check available GPUs and resource conflicts before use.

### P1: Repair the benign training plumbing
- First reproduce the smallest existing frozen IP2P image-editing inference for known-safe edit prompts.
- Confirm the reported BF16 / optimizer issue and fix it without hiding NaNs or silently skipping optimizer steps; log LoRA parameter dtype, gradient norm/finite status, actual optimizer steps, output hashes, and checkpoint IDs.
- Establish paired-seed baseline for frozen vs benign fine-tuned candidate. Keep train/dev/test splits separate by source image or subject where appropriate.
- Confirm a genuine stochastic sampling process and log-prob consistency *before* labeling the result DDPO. If incompatible with the current scheduler, log a blocker and use an appropriately supported sampler for benign validation, documenting the change.

### P2: Minimal benign Inner RL experiment
- Use **only harmless editing tasks** (e.g., day-to-night, change weather/lighting, artistic style) with fixed text instructions and source images. Use reward/evaluation solely for instruction adherence, identity/content preservation, and visual quality. Keep train-time metrics and held-out test metrics distinct.
- Compare frozen model with the smallest correct RL-trained LoRA under matched seeds and compute. Add a non-RL differentiable reward tuning comparator if readily available, but do not hold up the first functioning RL test just to build all baselines.
- Report reward trajectory, held-out measures, paired examples, seed count, variance/uncertainty, compute and any failures. Never claim a positive gain from a null pilot.
- Begin with a short, bounded smoke test, then one bounded pilot only if all assertions/gradients/checkpoint tests pass. Use only available hardware without disrupting existing experiments. Avoid uncontrolled long GPU jobs.

### P3: Minimal benign Outer/RSI experiment
- Run only after the Inner control path works. Implement a simple, bounded candidate proposal + held-out validation + accept/reject harness over **benign learning recipe / improvement procedure** configurations. Retain full history with reason for rejection and cost.
- Compare fixed recipe, fixed-rule proposal, and bandit-guided proposal under matched total compute/evaluation budgets. Do not claim PPO is trained without enough independent outer trajectories.
- Explicitly test whether held-out improvement transfers to a separate benign editing benchmark. If the only effect is tuning model hyperparameters on the same dev split, call it adaptive search/HPO instead of RSI.

### P4: Align the paper AFTER verifying actual behavior
- Update only the relevant method sections in a separate patch / branch if the repo supports it. Do not overwrite the author's original `.tex` without a diff or backup.
- Provide an editable, clean framework diagram, Algorithm 1, and clear separation of IMPLEMENTED / PROPOSED / VALIDATED claims. Leave unmeasured values as TBD, not invented numbers.
- Preserve the VLM backdoor discussion only as a clearly separated offline robustness evaluation context, without train-time ASR/stealth optimization instructions.

## Mandatory outputs

1. `reports/IMPLEMENTATION_AUDIT.md`: observed file paths, identified bugs, concrete fixes, verified vs unverified claims.
2. `reports/MINIMAL_EXPERIMENT_STATUS.md`: job command, launched/not launched, process identifier if available, exact data split, seed, GPU, start/end timestamps, checkpoint, metrics, error trace.
3. `reports/ARCHITECTURE_CORRECTIONS.md`: what was changed in architecture and why; especially DDPO terminal reward vs per-step likelihood, and bandit vs PPO distinction.
4. `reports/NEXT_EXPERIMENT_DECISION.md`: pass/fail criteria for expanding experiments; honest experimental blockers.
5. Reproducible benign configuration(s), smoke tests, and small held-out evaluation output.
6. A minimal LaTeX diff / method outline with no fabricated results.

## Acceptance / stopping conditions

- Do **not** expand experiments if basic GPU/memory/dependency checks, reward sanity checks, log-prob invariants, gradient-finite checks, or save/reload checks fail.
- Do **not** interpret a single seed, four source images, or an unchanged frozen baseline as evidence of learning.
- Do **not** mix security-study metrics into generator optimization and do **not** build any closed-loop attack-success optimization.
- If execution is unavailable, still produce code audit, small local tests, exact runnable benign commands, and a precise unlaunched status; do not claim experiments were run.

## Final report language

At the end, report in **Chinese** with a compact status table: implemented/verified/running/blocked, changed file paths, exact commands, known results (with sample sizes), and the next concrete blocker. Favor reproducible evidence over claims or polished prose.
