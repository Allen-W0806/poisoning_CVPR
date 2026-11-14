# Versioned method proposal: separate diffusion-policy learning and harness evolution

Research direction remains image-only semantic manipulation in controlled training-time VLM security research. The broader efficacy/efficiency hypothesis is not established by the engineering results below. No target labels, poisoned victim training or security-reward optimizer is supplied.

The four subsystems are (A) versioned original/edited factual records and generation environment; (B) an image-conditioned diffusion policy whose UNet attention LoRA receives score-function gradients; (C) a separate restricted executable harness editor; and (D) a fixed independent evaluator. A categorical edit selector is a historical baseline, not the primary trainable policy.

The policy conditions on the source image and externally supplied editing instruction. Its actions are stochastic latent transitions, not categorical editing commands. Original UNet, VAE and text encoder weights are frozen; LoRA A/B matrices are the learned parameters. The original DDPO-SF update and fixed coordinate/time normalization remain unchanged. A versioned policy adds source-content provenance, and its guarded trainer rejects duplicate source content and unreleased training rewards.

During any future candidate comparison, the generator checkpoint and all model services must be frozen and identical for candidate/incumbent executions. Harness changes affect the procedure for acquiring/retaining evidence, invoking validation checks, limiting diagnostic resources or ordering training-side cases. They cannot revise factual labels, evaluator definitions, datasets, security objectives or generator parameters. Different generator checkpoints cannot masquerade as a harness intervention.

Active investigation retrieves a registered training episode and acquires an observation absent from its old trace. The investigator records competing explanations, supporting/contradicting observations and unresolved alternatives. This implementation uses deterministic, scoped diagnostic/proposal rules and a restricted declarative workflow interpreter. It is an adaptation of the Video-RSI program-level idea, not a reproduction of its full language-model code editor.

The independent engineering Gate compares the same32 distinct constructed integrity cases, using a frozen exact oracle and explicit resource thresholds. It returns aggregates only. A candidate must cause no newly incorrect source outcome; a meaningful task gain may use at most twice the diagnostic operations, or equal accuracy must reduce operations by20%. Wall/CPU time and total search cost are measured separately. These engineering thresholds are not a validated natural-image selection protocol.

Natural-image task/semantic evaluation and the new editing training reward remain unreleased: no independent labels exist. The repaired compatibility reward records explicit coefficients, raw inputs, validity eligibility and saturation/redundancy, but cannot reach the Phase5 optimizer. Scientific experiments remain PREPARED_NOT_RUN.

Training-time security effects require a documented downstream model-training lineage and an independently supplied fixed evaluation artifact. Frozen-Qwen answer changes measure inference feedback only. Security observations remain read-only and never become generator rewards or Editor Gate feedback.

Prior-work classification: DDPO transition score learning is directly supported; IP2Pix LoRA conditioning, restricted workflow evolution and the engineering Gate are adaptations; jointly learning an editor and evolving its research workflow is a new research hypothesis; any backdoor-effectiveness or real end-to-end efficiency benefit is currently unverified.

Original local HTML draft is retained. No LaTeX draft is present in this checkout. `docs/phase5_method_patch.tex` is a proposed replacement snippet, not an applied manuscript edit or a compiled paper.
