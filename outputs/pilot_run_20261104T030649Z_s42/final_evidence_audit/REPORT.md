# Final read-only proxy and numerical audit

All values below were recomputed on CPU from saved validation images. The same calibrated reward evaluates the unedited source and final edits. This audit did not alter training, controller feedback or checkpoint selection.

| Reference / method | Common reward | CLIP cosine | LPIPS | SSIM |
|---|---:|---:|---:|---:|
| Unedited source (diagnostic) | 0.273499 | 0.136750 | 0.000000 | 1.000000 |
| frozen | -0.095900 | 0.146963 | 0.389826 | 0.601948 |
| ddpo_only | -0.027125 | 0.148044 | 0.323213 | 0.646556 |
| rule_rsi | -0.027125 | 0.148044 | 0.323213 | 0.646556 |
| rl_rsi | -0.015453 | 0.148959 | 0.313372 | 0.652327 |

The identity score is a proxy-validity diagnostic, not a sixth trained arm or instruction-success metric.

Post-hoc signed RGB luma changes: {"frozen": -0.11243483232629072, "ddpo_only": -0.10622581312759255, "rule_rsi": -0.10622581312759255, "rl_rsi": -0.10938630244030674}. Mean encoded RGB weighted 0.2126/0.7152/0.0722, [0,1]; no gamma linearization; post-hoc direction diagnostic, not instruction-success accuracy.

CPU/GPU metric differences and the untriggered-rule state comparison are fully retained in results.json. Exact and tolerance outcomes are separate; a rule that never activated is not evidence about an active heuristic.
