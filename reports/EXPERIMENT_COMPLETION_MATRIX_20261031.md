# Experiment completion matrix

Updated 2026-10-31T19:41:25.706707+00:00.

|Stage|Executed|Run ID|Slurm job|Result|Numerical checks|Scientific conclusion|Blockers|
|---|---|---|---|---|---|---|---|
|CPU source/provenance|True|small_scale_validation_20261031T1928Z_v1|87920|80 unique source/content; 32/16/32|PASS|Source-disjoint engineering data|Untouched-test status BLOCKED|
|Image controls|True|small_scale_validation_20261031T1928Z_v1|87920|400 valid traced outputs|PASS: decode/SSIM/pixel|Transformation engineering only|LPIPS missing; edited-label validity unreviewed|
|Frozen IP2Pix|True|small_scale_validation_20261031T1928Z_v1|87920|160 existing outputs/hashes audited|PASS: existing traceability|Descriptive reuse only|Independent human review0/160; BF16 continuation unresolved|
|Clean Qwen LoRA|True|small_scale_validation_20261031T1928Z_v1|87921|3 seeds complete|PASS|No inferred generalization benefit|historical development assessment; not an untouched scientific test|
|Benign RL exact/MC|True|small_scale_validation_20261031T1928Z_v1|87920|finite categorical/group reference|PASS|Synthetic mathematical consistency only||
|Production DDPO optimizer|True|small_scale_validation_20261031T1928Z_v1|87921|Tiny random UNet; pretrained ownership ZERO STEPS|PASS|No production GRPO or pretrained reliability claim|Full pretrained BF16 gate remains blocked|
|Synthetic RSI|True|small_scale_validation_20261031T1928Z_v1|87920|2 reproducible accept/reject/regression/rollback paths|PASS|Integrity oracle improvement only; not autonomous scientific RSI||
|Benign augmentation training|False|small_scale_validation_20261031T1928Z_v1|87920|NOT RUN|BLOCKED|No claim|Label validity not independently established|
|Attack reproduction / ASR|False|small_scale_validation_20261031T1928Z_v1|87920|NOT PERFORMED|N/A|No efficacy claim|Outside this benign pilot|

## Completed saved-data comparison 2026-11-01T23:10:07.416627+00:00

|Stage|Run ID|Slurm|Engineering result|Scientific scope|Remaining blocker|
|---|---|---|---|---|---|
|Matched160 controls and aligned SSIM|baseline_comparison_20261101T2255Z_v1|88317|PASS:960 primary records;640 new CPU SSIM|Frozen original-reference agreement only|0 independent labels;development exposure;LPIPS null|
|Tiny3-seed/3-arm statistics|baseline_comparison_20261101T2255Z_v1|88317 + bounded CPU supplement|PASS:saved-data CIs reproduce;12 disk-prefix comparisons reused|All3 contrasts include0;RSI superiority NOT ESTABLISHED|3 seeds;synthetic task|
|Publication tables/Results|baseline_comparison_20261101T2255Z_v1|CPU only|PASS:8 deliverables;current CVPR style compile|No external baseline reproduced|Full-method scientific gate NO-GO|
