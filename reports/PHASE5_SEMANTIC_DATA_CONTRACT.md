# Phase 5 semantic data contract

Version `vqav2_content_v2` preserves all 1,500 records. Every source image byte hash and official question/image/ten-answer join was reverified on the canonical cluster. No edited factual labels were manufactured.

| Split | Records | Image IDs | Exact content groups | Independently eligible |
|---|---:|---:|---:|---:|
| train | 999 | 988 | 988 | 0 |
| development | 104 | 103 | 103 | 0 |
| gate | 96 | 94 | 94 | 0 |
| test | 301 | 299 | 298 | 0 |

Original image ID/path/SHA, question and dataset answers are separate from edited-image identity, independently supplied edited factual answers, supplied training annotation, category, SC/SI-V relation, validity and verification provenance. VERIFIED edited answers require nonempty strings; UNKNOWN/DISPUTED canonical answers remain None. Metadata validity is not proof of authentic human review. Eligibility requires complete independent edited facts, visual validity, relation and category; all current records have UNKNOWN semantics, 0 verified SC and 0 verified SI-V. Keyword color/animal/food/vehicle matches only nominate candidates. Counts by category are in the receipt.

Exact content grouping moved train sample397819015 to test alongside old test388662037 (same SHA `fcc1fd100ed50e30ea87b9d84597cb40876c76d955716fc9dd6e936787faf181`). All old test records remain test. Same-ID inconsistencies are rejected. Runtime loader checks manifest SHA, image bytes, source joins and semantic-record correspondence at initialization and access; Editor role only accepts train. Original v1 files remain unchanged.

Development/Gate derive from historically exposed v1 validation. They are engineering partitions, not fresh scientific evidence. Known Phase 4 development images remain development. Old test was already opened historically and cannot acquire new freshness by renaming. Final-test selection must be operator-only after freezing the harness; future confirmatory evaluation needs an independently justified unseen cohort.

A separate 64-bit dHash (9x8 grayscale; Hamming <=4) candidate audit found 0 candidate pairs. This does not prove absence of near duplicates and causes no automatic deletion.

Frozen manifest SHA256:

- train: `f6fc1dd7b3b3fd1a9e1fd1bba6ab791c3df87acedecc013305fb8de9c58f09b4`
- development: `b82b9a3cac57241d151a294f067312807288dfee353565a43c8e0e694ac86131`
- gate: `ce7ffba3d124b4a5373560f6eaa644d3bf5ad4c664dc1112f10b8135aee90c00`
- test: `98c3b47853e07343a848ea6361713b03f64487ad0c06598e893049c2e06a59ce`

Portable mirrors are under `data/manifests/cluster/vqav2_content_v2`; canonical manifests remain `data/processed/vqav2_content_v2`. Raw images were not downloaded anew.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [src/data/semantic_contract_v2.py](<H:/Project/Poisoning/src/data/semantic_contract_v2.py>), [src/data/dataset_v2.py](<H:/Project/Poisoning/src/data/dataset_v2.py>), [src/data/prepare_v2.py](<H:/Project/Poisoning/src/data/prepare_v2.py>), [scripts/prepare_phase5_data.py](<H:/Project/Poisoning/scripts/prepare_phase5_data.py>).

Tests: [tests/test_phase5_contracts.py](<H:/Project/Poisoning/tests/test_phase5_contracts.py>), [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_dataset_v2.json](<H:/Project/Poisoning/reports/phase5_dataset_v2.json>), [data/manifests/cluster/vqav2_content_v2/provenance.json](<H:/Project/Poisoning/data/manifests/cluster/vqav2_content_v2/provenance.json>), [data/manifests/cluster/vqav2_content_v2/near_duplicate_candidates.json](<H:/Project/Poisoning/data/manifests/cluster/vqav2_content_v2/near_duplicate_candidates.json>).
