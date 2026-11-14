# Lighting-v2 comparison

Status: TRAINING

Only seed42. Source-image bootstrap intervals do not estimate training-seed uncertainty. Frozen/Fixed share the same observation by user choice. Development mini-assessment was previously exposed. No poisoned-victim training, ASR, clean accuracy or independent semantic labels.

Reward v2 = 2*(CLIP_edit-CLIP_source) + brightness_progress - color_cost - LPIPS. The new task is explicit +3% linear-light exposure. Lighting/color use original content boxes; CLIP/LPIPS/SSIM retain full canvases. Old and new raw reward scales are not comparable. DDIM50 eta1, guidance5, image guidance1.5, rank16/alpha32, LR5e-6, clipping0.5. Full FP32 computation reuses upcast cached FP16-variant weights; BF16 repair is not asserted.

| Method | status | optimizer_steps | common_reward | LPIPS | SSIM | content_encoded_luma_delta | content_chromaticity_MAE | ASR |
|---|---|---|---|---|---|---|---|---|
| Frozen | GENERATOR_READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Fixed Editing (equivalent control) | GENERATOR_READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| DDPO-only | RUNNING | 40 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| Rule-based RSI | READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
| RL-RSI | READY | 0 | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED | NOT TESTED |
