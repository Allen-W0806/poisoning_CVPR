# Baseline comparison

UTC: 2026-11-04T04:27:09.691624+00:00

Status: COMPLETED

All numbers below come from saved observations. This is a benign lighting-editing pilot. CLIP cosine is semantic text alignment, not ASR, instruction correctness, or VQA accuracy. The 16 mini-test images come from a historically exposed training pool and are development assessment. Frozen edits images with unchanged pretrained weights; it is not a pixel identity baseline. Fixed shares the identical prompt, seed, weights, sampler and observations by explicit user choice; it is an equivalence control, not independent evidence.

| Method | status | optimizer_steps | common_reward | CLIP_cosine | LPIPS | SSIM | delta_common_vs_Frozen | ASR |
|---|---|---|---|---|---|---|---|---|
| Frozen | MEASURED | 0 | -0.051463 | 0.152862 | 0.357187 | 0.595992 | 0.000000 | NOT TESTED |
| Fixed Editing (equivalent control) | MEASURED_EQUIVALENT_CONTROL | 0 | -0.051463 | 0.152862 | 0.357187 | 0.595992 | 0.000000 | NOT TESTED |
| DDPO-only | MEASURED | 100 | -0.020859 | 0.153833 | 0.328526 | 0.619803 | 0.030603 | NOT TESTED |
| Rule-based RSI | MEASURED | 100 | -0.020859 | 0.153833 | 0.328526 | 0.619803 | 0.030603 | NOT TESTED |
| RL-RSI | MEASURED | 100 | -0.014295 | 0.153746 | 0.321787 | 0.625512 | 0.037168 | NOT TESTED |

Common evaluator: 2 × CLIP cosine − calibrated AlexNet LPIPS. All arms use this same evaluator even when RL-RSI changes its training semantic weight. 95% intervals in the CSV resample paired source images (10,000 draws); one training seed cannot estimate between-seed uncertainty.
