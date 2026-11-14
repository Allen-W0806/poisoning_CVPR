# Method and scope

> Historical infrastructure-stage description retained below. For the current diffusion DDPO, clean Qwen, evaluation and audit boundaries, use [CURRENT_METHOD_20261101.md](CURRENT_METHOD_20261101.md). This dated entry distinguishes implemented components from unvalidated scientific claims.

## Architecture

1. Immutable data records preserve image, image ID, question and all ten annotator answers. Split allocation occurs by image ID before sample selection. Answer-type proportions are approximately maintained; exact counts and checksums are saved in dataset provenance.
2. `QwenVQA` uses Transformers `AutoModelForImageTextToText` and `AutoProcessor`, frozen parameters, greedy generation (64 new tokens), left padding, explicit pixel budgets and batch-aware prompt trimming. BF16 is enabled only on supporting CUDA devices; CPU uses FP32. Questions are passed unchanged. The implementation follows the [official Qwen3-VL examples](https://github.com/QwenLM/Qwen3-VL).
3. `BenignEditor(images, conditions, seed)` currently runs PIL brightness adjustments bounded at 3%. It preserves image geometry. The API can receive a different editing backend through `load_editor`; no diffusion backend is presented as already implemented. Saving/quality/semantic validation are separate from generation.
4. `EditPolicy` has a frozen feature projection and a trainable categorical head selecting factors 1.0/0.97/1.03. REINFORCE uses detached per-sample benign validation rewards. The mean reward is a logging statistic, not a shared replacement for sample-level returns. Mini-batches, accumulation, AdamW, warmup, clipping, BF16 and full RNG/optimizer/scheduler checkpoints are supported. This is an infrastructure test, not adaptation of a pretrained diffusion generator.
5. `RSIController` is separate from RL. It stores current policy/config version, immutable proposal records, validation scores/identity, failures and cumulative cost. Revisions can change batch size, curriculum or LR. Test data, labels and metrics are outside its writable interface. Toy smoke scores verify acceptance/rejection persistence only; no empirical RSI benefit follows.
6. `security_metrics` aggregates independent existing observations. Conditional behavior, clean false activation, detection, inference-time answer change and post-training behavior change use separate names and denominators. Missing measurements are null. It neither edits inputs nor updates any policy or VLM.

## Evidence boundaries

The [BadSem paper](https://arxiv.org/abs/2506.07214) motivates robustness questions; its attack is not reimplemented. This setup excludes automatic poisoning, target-output objectives, textual triggers and ASR/evasion rewards.

Synthetic red-circle images have an exact count oracle suited only to these separated shapes. Natural-image semantic validity is explicitly unverified without an external verifier. Brightness bounds, MSE and SSIM do not prove natural-image semantics. Clean unmodified evaluation has preservation by identity.

Clean VQA consensus averages the ten leave-one-reference-out agreement scores, each capped at 1 after dividing by 3. `clean_exact_any` is reported separately. Current answer normalization is labelled `preliminary_v1`; full official-toolkit string normalization parity has not been established. Comparisons intended for publication must use/parity-check the [official VQA evaluator](https://github.com/GT-Vision-Lab/VQA/tree/master/PythonEvaluationTools). A ten-example inference check is not a benchmark-quality estimate.

Same-device seeded replay is tested exactly. Cross-device/kernel/precision equality is not asserted. Component wall-time GPU hours and scheduler allocation GPU hours are distinct. Total resource comparisons must include model loading, proposals, failed/rejected trials, training and validation.

Limited clean evaluations select a seeded shuffled prefix of indices (seed 42), rather than the initial answer-type block. Previously executed prefix-only measurements remain in their original runs and are labelled in the setup report. Checkpoint resume checks the config plus reward/editor/evaluator/device/data/batch contract before loading state; the caller must reconstruct that same contract. Security observation inputs are copied into each evaluation run, and optional change rates report their own observed denominators. Cached model availability is recorded as actual model use only in Qwen runs.

The fixed short-answer system prompt is `Answer the question using only a short word or phrase. Do not include an explanation.` It is identical across examples and independent of labels; the original question is unchanged. Earlier no-system-prompt runs returned verbose explanations and scored zero under whole-answer string matching. Those results are preserved. Short-answer calibration is a different inference protocol, not evidence of learned improvement. `--answer-style raw` reproduces the original no-system-prompt mode.
