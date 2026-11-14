# Scientific gate matrix

**Overall: NO-GO.** Benign engineering checks may pass; semantic validity, independent real-image evaluation, pretrained numerical stability and scientific efficacy remain unestablished.

| Historical ID | Current scoped status | Actual verification | Remaining scientific risk |
|---|---|---|---|
| F01 Claim/estimand mismatch | PASS | Protocol names cosine engineering task and no VLM/security reward. | No BadSem reproduction or security efficacy has been established. |
| F02 Same-model semantic verification | BLOCKED | 0 genuine reviewed pairs; no human/model error rate available. | Independent reviewer agreement and semantic validity remain unknown. |
| F03 Reward redundancy and floor censoring | FAIL | 500/500 S=.5A;230/500 floor; centered component rank2. | Feedback construct is not validated for scientific learning. |
| F04 Instruction success absent | BLOCKED | C0/C1 scores identical for160; identity exceeds all historical changed rewards. | No human-confirmed lighting success rate; no new objective introduced. |
| F05 RSI versus ordinary scheduling | BLOCKED | New comparison PASS; modest LR/rollback mechanism explicitly described. | Providing a scheduler baseline does not prove novel RSI or autonomous superiority. |
| F06 Preadaptation causal confounding | PASS | Disk initial/prefix comparisons: 12; all equal=True. | Only the new FP32 tiny-model comparison is repaired; historical BF16 causal claims remain invalid. |
| F07 Reused data and weak replication | BLOCKED | Newest32 assessment sources:4 prior inference,10 in old opened-test manifest. | No real-image generalization claim; three seeds cannot substitute independent data. |
| F08 Exact-content split leakage | PASS | Zero ID/hash overlaps within latest32/16/32; all80 file hashes valid. | PASS for new internal split only, not historical v1 or exposure independence. |
| F09 Pretrained stochastic continuation | FAIL | Both new paths PASS; earlier ordinary and instrumented failures preserved. | Intermittent path-dependent history is unresolved; batch10 continuation is unaccepted. |
| F10 Question versus global preservation | BLOCKED | Forms prepared; zero human labels. | Rubric availability is not semantic remediation. |
| F11 Step proxy versus measured compute | PASS | Both GPU attempts charged; per-branch actual GPU-held seconds retained. | No claim that old C is measured cost or has become a validated cost objective. |
| F12 Identity loses consistency credit | FAIL | Caller gives A0 and reward.9666667 versus1.4666667 with declared credit. | Bug still present in legacy caller; no historical reward overwritten. |
| F13 Exposure versus realized compute | PASS | Each branch4 updates,320 forward NFEs,48 validation+32 test queries. | Measured times differ; sequential same-node timings are not a latency benchmark. |
| F14 Suspected editing/padding failures | BLOCKED | Geometry reconstruction160/160; aligned SSIM mean.7340322949; LPIPS absent. | No human-confirmed edit/instruction failure rates. |
| F15 Active-contract reconstruction | BLOCKED | New config/optimizer and rollback checks PASS. | General production pretrained resume across a historical revision remains unvalidated. |
| F16 Obsolete method entry | PASS | Original body byte text preserved after removing banner; code paths checked. | Documentation does not certify scientific benefit. |
| F17 Backward primitive not localized | BLOCKED | Previous instrumented raw error2.834317e-5; new pair exact; no stable path effect. | No specific CUDA primitive or unique causal mechanism identified. |
| F18 Independent validity unknown | BLOCKED | 0 reviewed pairs; disagreement and validity rates remain unknown. | Human-dependent task cannot be replaced by model flags. |
| F19 Reward config/default mismatch | FAIL | Values match historical campaign; wiring gap persists. | Editing YAML alone would not change coefficients; future version needs explicit contract. |
| F20 Runtime data guards weaker | FAIL | All80 latest image hashes PASS; historical official join receipt retained. | Audit success does not make runtime loader a corruption detector. |
| F21 Near-duplicates/pretraining exposure | BLOCKED | All80 prior-exposure flags true; latest split only source-disjoint. | Foundation-model contamination and unrecorded prior exposure unknown. |
| F22 Real post-training scientific/security evidence absent | BLOCKED | Absent denominators remain missing; scientific gate NO-GO. | Natural-image efficacy and training-induced security behavior untested. |
| F23 Within-batch duplicate freshness gap | PASS | Prepatch duplicate accepted;5 regression tests PASS including real trainer early rejection. | Distinct trajectories for the same condition remain explicitly permitted; no historical duplicate was inferred. |

## October31 versus current evidence

October31: real clean LoRA updates/checkpoint recovery in3 seeds; finite-gradient reference checks; aligned160-image metrics;32-pair blinded interface; tiny two-arm DDPO–RSI memory/rollback. These did not validate natural-image semantics, instruction success, unused assessment data, general pretrained continuation, or RSI superiority.

New passing remediation: F23 trajectory-ID guard plus5 regressions; F16 current documentation; F06 strict preadaptation equivalence in the bounded FP32 experiment; F08 latest internal content/ID separation; F11/F13 measured resource reporting. F01 is a claim-scope correction only. F05 now has an ordinary scheduler comparator but its novelty/superiority question remains BLOCKED. Historical adverse findings are not retrospectively changed to PASS.

## Feedback evidence

{
  "total": 500,
  "all_saved_reward_arithmetic_exact": true,
  "reward_definition_unchanged": true,
  "S_equals_half_A": 500,
  "constant_C": [
    0.3333333333333333
  ],
  "constant_U": [
    1.0
  ],
  "all_negative": true,
  "floor_count": 230,
  "upper_count": 0,
  "raw_below_floor": 230,
  "centered_component_rank": 2,
  "distribution": {
    "n": 500,
    "mean": -0.4955117319948971,
    "variance": 0.2173682721302256,
    "minimum": -1.0,
    "maximum": -0.03733959666763742,
    "unique_at_12dp": 260,
    "duplicate_pairs": 26346,
    "pair_tie_fraction": 0.21119038076152305
  },
  "by_role": {
    "train": {
      "n": 320,
      "mean": -0.5475048743720011,
      "variance": 0.21877436412192677,
      "minimum": -1.0,
      "maximum": -0.03733959666763742,
      "unique_at_12dp": 145,
      "duplicate_pairs": 13541,
      "pair_tie_fraction": 0.265301724137931,
      "floor_count": 165
    },
    "development": {
      "n": 180,
      "mean": -0.4030794788800455,
      "variance": 0.2026526479880975,
      "minimum": -1.0,
      "maximum": -0.044186480094989156,
      "unique_at_12dp": 116,
      "duplicate_pairs": 2080,
      "pair_tie_fraction": 0.12911235257603973,
      "floor_count": 65
    }
  },
  "by_arm": {
    "M0": {
      "n": 20,
      "mean": -0.34616713044544056,
      "variance": 0.19299358030362432,
      "minimum": -1.0,
      "maximum": -0.04423587073882418,
      "unique_at_12dp": 15,
      "duplicate_pairs": 15,
      "pair_tie_fraction": 0.07894736842105263
    },
    "M1_42": {
      "n": 120,
      "mean": -0.49282959208021765,
      "variance": 0.21960078550042872,
      "minimum": -1.0,
      "maximum": -0.037342060388376286,
      "unique_at_12dp": 66,
      "duplicate_pairs": 1485,
      "pair_tie_fraction": 0.20798319327731093
    },
    "M1_123": {
      "n": 120,
      "mean": -0.5182866218986196,
      "variance": 0.21903448884125565,
      "minimum": -1.0,
      "maximum": -0.04489515883227191,
      "unique_at_12dp": 63,
      "duplicate_pairs": 1653,
      "pair_tie_fraction": 0.23151260504201682
    },
    "M2_42": {
      "n": 120,
      "mean": -0.4929522057868437,
      "variance": 0.219497869006242,
      "minimum": -1.0,
      "maximum": -0.03733959666763742,
      "unique_at_12dp": 66,
      "duplicate_pairs": 1485,
      "pair_tie_fraction": 0.20798319327731093
    },
    "M2_123": {
      "n": 120,
      "mean": -0.5028692751388169,
      "variance": 0.2181981564416002,
      "minimum": -1.0,
      "maximum": -0.04472130884726844,
      "unique_at_12dp": 65,
      "duplicate_pairs": 1540,
      "pair_tie_fraction": 0.21568627450980393
    }
  },
  "interpretation": "constant U and C cannot rank observations; S/A duplicate the same gate; floor censoring hides D/C differences; no independently verified semantics"
}
{
  "n_sources": 160,
  "raw_records": 1032,
  "primary_records": 960,
  "conditions": [
    {
      "condition": "C0",
      "n": 160,
      "mean": 0.8487500000000001,
      "variance": 0.10842610062893082,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 8292,
      "pair_tie_fraction": 0.6518867924528302,
      "at_zero": 0.1,
      "at_one": 0.8,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 160
    },
    {
      "condition": "C1",
      "n": 160,
      "mean": 0.8487500000000001,
      "variance": 0.10842610062893082,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 8292,
      "pair_tie_fraction": 0.6518867924528302,
      "at_zero": 0.1,
      "at_one": 0.8,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 160
    },
    {
      "condition": "C2",
      "n": 160,
      "mean": 0.8506250000000002,
      "variance": 0.10578577044025159,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 8285,
      "pair_tie_fraction": 0.6513364779874213,
      "at_zero": 0.09375,
      "at_one": 0.8,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 159
    },
    {
      "condition": "C3",
      "n": 160,
      "mean": 0.8506250000000002,
      "variance": 0.10578577044025156,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 8285,
      "pair_tie_fraction": 0.6513364779874213,
      "at_zero": 0.09375,
      "at_one": 0.8,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 157
    },
    {
      "condition": "C4",
      "n": 160,
      "mean": 0.649375,
      "variance": 0.20364740566037734,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 5475,
      "pair_tie_fraction": 0.43042452830188677,
      "at_zero": 0.2875,
      "at_one": 0.5875,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 116
    },
    {
      "condition": "C5",
      "n": 160,
      "mean": 0.764375,
      "variance": 0.1561442610062893,
      "minimum": 0.0,
      "maximum": 1.0,
      "unique_at_12dp": 5,
      "duplicate_pairs": 6758,
      "pair_tie_fraction": 0.5312893081761006,
      "at_zero": 0.16875,
      "at_one": 0.70625,
      "invalid": 0,
      "missing_official": 0,
      "ties_original": 137
    }
  ],
  "all_candidates_tied_fraction": 0.70625,
  "mean_within_source_variance": 0.04513125,
  "C0_C1_utility_equal": true,
  "metric_mismatches": [],
  "between_batch_mean_variance": {
    "C0": 0.005267361111111111,
    "C1": 0.005267361111111111,
    "C2": 0.005047309027777777,
    "C3": 0.006088975694444443,
    "C4": 0.010542100694444442,
    "C5": 0.009687934027777779
  },
  "repeats_exact": true,
  "scores_for_C2_C3_C4_C5": "inherited-reference agreement, not certified label-valid accuracy"
}

Between-batch variation combines sample composition/difficulty and system response. Comparisons across C0–C5 are paired within identical sources. Ten diagnostic source blocks are not ten comparable GRPO candidates. Only29.375% of five-candidate source groups have nonzero score dispersion. Technical invalid/missing counts are separate from true zero scores. Same-model consistency is not independent semantic proof.

## Real-data independence

[
  {
    "split": "train",
    "n": 32,
    "known_prior_exposure_flags": 32,
    "prior_executed_inference_overlap": 3,
    "prior_training_or_training_manifest_overlap": 4,
    "prior_dataset_manifest_overlap": {
      "data/processed/vqav2_seed42_v1/train.json": 23,
      "data/processed/vqav2_seed42_v1/validation.json": 4,
      "data/processed/vqav2_seed42_v1/test.json": 5
    }
  },
  {
    "split": "development",
    "n": 16,
    "known_prior_exposure_flags": 16,
    "prior_executed_inference_overlap": 1,
    "prior_training_or_training_manifest_overlap": 1,
    "prior_dataset_manifest_overlap": {
      "data/processed/vqav2_seed42_v1/train.json": 12,
      "data/processed/vqav2_seed42_v1/validation.json": 1,
      "data/processed/vqav2_seed42_v1/test.json": 3
    }
  },
  {
    "split": "independent_evaluation",
    "n": 32,
    "known_prior_exposure_flags": 32,
    "prior_executed_inference_overlap": 4,
    "prior_training_or_training_manifest_overlap": 4,
    "prior_dataset_manifest_overlap": {
      "data/processed/vqav2_seed42_v1/train.json": 17,
      "data/processed/vqav2_seed42_v1/validation.json": 5,
      "data/processed/vqav2_seed42_v1/test.json": 10
    }
  }
]

All80 flags were previously exposed. The latest32 assessment includes4 confirmed earlier inference sources and10 old opened-test manifest sources;4 overlap prior training or training manifests. A manifest exposure is not proof of an optimizer update. Cross-split IDs/SHA are disjoint inside the latest split, so source-disjoint is valid; genuinely independent is not. Historical v1 exact duplicate remains preserved.

## Clean Qwen conclusion

{
  "existing_recheck": "PASS",
  "stats": [
    {
      "seed": null,
      "n": 32,
      "accuracy": 0.7843749999999999,
      "paired_difference": 0.0,
      "paired_ci95": [
        0.0,
        0.0
      ],
      "changed_answers": 0,
      "fresh_object_repeat_exact": true,
      "mean_latency": 0.26470065202738624,
      "scope": "historical development only"
    },
    {
      "seed": 42,
      "n": 32,
      "accuracy": 0.7843749999999999,
      "paired_difference": 0.0,
      "paired_ci95": [
        0.0,
        0.0
      ],
      "changed_answers": 0,
      "fresh_object_repeat_exact": true,
      "mean_latency": 0.31916784221539274,
      "scope": "historical development only"
    },
    {
      "seed": 123,
      "n": 32,
      "accuracy": 0.8062499999999999,
      "paired_difference": 0.021875,
      "paired_ci95": [
        0.0,
        0.06562499999999999
      ],
      "changed_answers": 1,
      "fresh_object_repeat_exact": true,
      "mean_latency": 0.3188282387036452,
      "scope": "historical development only"
    },
    {
      "seed": 3407,
      "n": 32,
      "accuracy": 0.8062499999999999,
      "paired_difference": 0.021875,
      "paired_ci95": [
        0.0,
        0.06562499999999999
      ],
      "changed_answers": 1,
      "fresh_object_repeat_exact": true,
      "mean_latency": 0.3191470800265961,
      "scope": "historical development only"
    }
  ],
  "paired_sign_diagnostics": [
    {
      "seed": 42,
      "changed_sample_ids": [],
      "nonzero_score_pairs": 0,
      "exact_two_sided_paired_sign_flip_p": 1.0,
      "interpretation": "no significant clean-task gain; discrete consensus delta on at most one source, not extra independent replications"
    },
    {
      "seed": 123,
      "changed_sample_ids": [
        "362793004"
      ],
      "nonzero_score_pairs": 1,
      "exact_two_sided_paired_sign_flip_p": 1.0,
      "interpretation": "no significant clean-task gain; discrete consensus delta on at most one source, not extra independent replications"
    },
    {
      "seed": 3407,
      "changed_sample_ids": [
        "362793004"
      ],
      "nonzero_score_pairs": 1,
      "exact_two_sided_paired_sign_flip_p": 1.0,
      "interpretation": "no significant clean-task gain; discrete consensus delta on at most one source, not extra independent replications"
    }
  ],
  "same_changed_source_across_123_and_3407": true,
  "conclusion": "engineering reproducibility PASS; clean-task scientific improvement unsupported; no equivalence claim; development-only",
  "new_inference_calls": 0,
  "new_optimizer_steps": 0
}

Single nonzero paired outcome per improving seed gives exact two-sided sign-flip p1.0; both seeds change the same source. Bootstrap intervals include0 and no equivalence follows from a zero empirical interval. No clean-task scientific gain established.

## Image and human gate

{
  "SSIM_available": true,
  "SSIM_reference": "decoded C5 == ImageOps.pad(original RGB,256,white,BICUBIC=3); C4 and C5 same 256 geometry; full-frame including padding",
  "n_reconstruction_exact": 160,
  "SSIM_mean": 0.7340322949377474,
  "LPIPS": null,
  "modules": {
    "lpips": false,
    "piq": false,
    "torchmetrics": false
  },
  "calibrated_LPIPS_weights": "not verified; cached AlexNet classification weights alone are not LPIPS",
  "no_substitution": true
}

C5 was reconstructed exactly as RGB ImageOps.pad256, white, BICUBIC; SSIM compares C4 with C5 at identical dimensions. Full-frame padding remains part of the measurement. Native originals remain distinct. Missing LPIPS is null; cached classifier weights are not silently treated as calibrated LPIPS. The existing predeclared32 kit is reused, not resampled. Two independent reviewer exports and separate adjudication remain BLOCKED.
