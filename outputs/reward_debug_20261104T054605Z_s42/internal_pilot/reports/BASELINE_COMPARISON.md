# Lighting-v3 comparison

Status: COMPLETED

Only seed42. Source-image bootstrap intervals do not estimate training-seed uncertainty. Frozen/Fixed share the same observation by user choice. Development mini-assessment was previously exposed. No poisoned-victim training, ASR, clean accuracy or independent semantic labels.

Reward v3 = 2*(CLIP_edit-CLIP_source + mean_brightness_progress) - color_cost - LPIPS. The new task is explicit +3% linear-light exposure. Lighting/color use original content boxes; CLIP/LPIPS/SSIM retain full canvases. Old and new raw reward scales are not comparable. DDIM50 eta1, guidance5, image guidance1.5, rank16/alpha32, LR5e-6, clipping0.5. Full FP32 computation reuses upcast cached FP16-variant weights; BF16 repair is not asserted.

| Method | status | optimizer_steps | common_reward | LPIPS | SSIM | content_encoded_luma_delta | content_chromaticity_MAE | ASR |
|---|---|---|---|---|---|---|---|---|
| Frozen | MEASURED | 0 | -2.483343 | 0.085933 | 0.752509 | 0.005918 | 0.041251 | NOT TESTED |
| Fixed Editing (equivalent control) | MEASURED_EQUIVALENT_CONTROL | 0 | -2.483343 | 0.085933 | 0.752509 | 0.005918 | 0.041251 | NOT TESTED |
| DDPO-only | MEASURED | 100 | -2.344747 | 0.084585 | 0.753064 | 0.003982 | 0.041282 | NOT TESTED |
| Rule-based RSI | MEASURED | 100 | -2.364122 | 0.084915 | 0.752834 | 0.003836 | 0.041271 | NOT TESTED |
| RL-RSI | MEASURED | 100 | -2.345174 | 0.084783 | 0.752923 | 0.004061 | 0.041347 | NOT TESTED |
