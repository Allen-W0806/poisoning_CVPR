# Phase 5 versioned reward corrections

Confirmed implementation defects are corrected in new entry points, preserving all old scores and protected modules. `RewardConfigV2` requires every named coefficient and exact equality of recorded/executed configuration. Compatibility defaults remain semantic1.0, accuracy0.5, distortion0.2, cost0.1, invalid penalty1.0. Changing these numbers to manufacture gains was not performed.

Validity has strict VALID/INVALID/UNKNOWN/DISPUTED states with boolean/numeric consistency. Malformed verifier fields fail rather than receiving favorable defaults. Verified exact identity is valid and answer-consistency eligible; an unsupported edit remains unknown. Receipts expose raw components, eligibility, unclipped and clipped scores, lower/upper saturation, constant components and component correlations. Duplicate source content is rejected before optimizer activity, even across distinct IDs.

These repairs fix wiring and accounting; they do not fix a semantic construct by assertion. Legacy U=1, S=0.5A and clipping motivate independent validation; no renamed scalar or coefficient adjustment establishes an informative editing reward. Compatibility aggregation is explicitly diagnostic_only and every receipt has training_eligible=false. The guarded trainer calls the release guard before its preserved DDPO optimizer; raw scalar episodes cannot bypass it.

No new benign training reward is released. Release requires an independently labeled pilot with verified identity, valid SC, valid SI-V, invalid edits, UNKNOWN and DISPUTED cases; report class/source counts, missingness, inter-reviewer disagreement, construct/component redundancy and saturation. Lock a release version and its validation criteria before training. Until then all Phase 5 scientific optimizer updates remain blocked. This is an unresolved scientific prerequisite rather than a completed reward-validation result.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [configs/phase5_reward.yaml](<H:/Project/Poisoning/configs/phase5_reward.yaml>), [src/rl/reward_v2.py](<H:/Project/Poisoning/src/rl/reward_v2.py>), [src/evaluation/semantic_validation_v2.py](<H:/Project/Poisoning/src/evaluation/semantic_validation_v2.py>), [src/rl/diffusion_rl_trainer_v2.py](<H:/Project/Poisoning/src/rl/diffusion_rl_trainer_v2.py>), [src/rl/phase5_backend.py](<H:/Project/Poisoning/src/rl/phase5_backend.py>).

Tests: [tests/test_phase5_contracts.py](<H:/Project/Poisoning/tests/test_phase5_contracts.py>).

Evidence: [reports/phase5_new_tests_final.log](<H:/Project/Poisoning/reports/phase5_new_tests_final.log>), [reports/phase5_canonical_cpu_tests_final.log](<H:/Project/Poisoning/reports/phase5_canonical_cpu_tests_final.log>), [reports/phase45_recompute.json](<H:/Project/Poisoning/reports/phase45_recompute.json>).
