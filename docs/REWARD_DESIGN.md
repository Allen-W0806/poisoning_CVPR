# Benign validation reward

`R = clip(w_v V + w_a A - w_d D - w_c C - invalid_penalty, -1, 1.5)`.

Default weights: semantic 1.0, answer consistency 0.5, distortion 0.2, cost 0.1; invalid/unverified semantics penalty 1.0. Each input component must be finite and is clipped to [0,1]. The total is clipped after applying the penalty. There is no per-batch centering or reward standardization in this smoke implementation. Individual components, unclipped/clipped totals, actions and sample IDs are logged.

| Component | Current measurement | Limit |
|---|---|---|
| V | exact count preservation on labelled synthetic separated red circles | natural images return unverified and receive the penalty until an independent verifier is supplied |
| A | agreement between original and edited synthetic oracle answers | consistency is not correctness; real VQA ground-truth accuracy is a separate metric |
| D | `(MAE + (1 - SSIM)/2)/2`, clamped [0,1]; uint8 images scaled to [0,1] | visual proxy only; not proof of semantic integrity |
| C | deterministic edit-operation count divided by a 10-operation reference budget, capped at 1 | this smoke uses C=0.1; measured time is logged separately, not substituted into a deterministic reward |

The cluster uses windowed `skimage.metrics.structural_similarity` with `data_range=1`. A CPU environment lacking scikit-image may use an explicitly labelled `global_ssim_fallback`; it must not be pooled silently with the windowed metric.

The invalid penalty applies unless semantics are verified and V=1. A natural-image verifier may provide graded values, but a partial score still triggers the penalty in this initial contract. Pixel similarity alone cannot set V to 1.

Loss is `-mean(log pi(action|image) * detached_sample_reward)` over each microbatch, divided by accumulation count before backward. Batch reward is mean of all samples across accumulated microbatches. No entropy bonus, learned baseline, PPO clipping or security metric is used. VLM weights and the policy feature projection remain frozen.

Engineering defaults: seed 42, microbatch 10, accumulation 1, AdamW LR 1e-5, weight decay .01, max gradient norm 1, 5% linear warmup then constant LR, BF16 on a capable GPU, otherwise FP32. The short smoke has 3 updates; ceil(.05*3)=1 warmup step, so this run does not characterize warmup behavior. FP32 master parameters receive optimizer updates. These hyperparameters have not been optimized or scientifically validated.

Optional batch/LR/seed/accumulation/reward-ablation/fixed-policy grids are listed in `configs/rl.yaml` and are not launched. The CLI/API can consume user-supplied lightweight training settings; this is not evidence that diffusion adaptation is feasible at the same cost. Neither ASR nor detection/evasion is available to the reward function.
