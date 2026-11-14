# Phase 5 fixed-artifact security evaluation

Implemented a read-only aggregation interface, not a security-result generator. It accepts independently supplied observations and explicit pretrained/clean_finetuned/authorized_fixed_condition lineage. Qwen revision is pinned to `89644892e4d85e24eaac8bacfd4f463576704203`. Every trained node needs a documented parent, data, authorization and artifact manifest. A real artifact must be a local JSON bundle with listed checkpoint files confined under its directory; the interface reads and hashes them. External/remote paths alone are unsupported. Content hashes attest identity, not scientific authorization or authentic annotation.

Inference_only and post_training populations are never pooled. Report clean accuracy and SC accuracy with scored/all row counts, missing scores, available/scored/unscored unique sources. False activation on clean controls and SC-conditioned false activation are distinct. SI-V-conditioned target-observed rate reports the explicit supplied boolean denominator and unique sources. Missing scores are not silently converted into incorrect predictions or complete coverage.

Published BadSem ASR concerns a predefined target word under the applicable trained-artifact/evaluation setting. No target is generated here. A genuine post-training ASR requires real authorized artifact lineage plus post-training rows; missing or synthetic artifacts return None and denominator0. Pretrained observations cannot be labeled post_training. Frozen-Qwen answer changes are inference feedback, not demonstrated backdoor implantation.

Only synthetic fixtures exercised this interface. There is no new clean fine-tuning result, no supplied real security artifact, and no real ASR. Report numerator, denominator, source-cluster coverage, uncertainty, checkpoint revision, dataset lineage and evaluation resources when valid artifacts become available; small/absent denominators remain explicit. Future independent evaluator costs are unmeasured. Security feedback is excluded from generator reward, harness selection and active Editor observations. No poisoned-target training, target-label generator, evasion optimizer or automated ASR maximization is implemented.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [src/evaluation/security_readonly_v2.py](<H:/Project/Poisoning/src/evaluation/security_readonly_v2.py>), [src/data/semantic_contract_v2.py](<H:/Project/Poisoning/src/data/semantic_contract_v2.py>).

Tests: [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_canonical_cpu_tests_final.log](<H:/Project/Poisoning/reports/phase5_canonical_cpu_tests_final.log>), [implement-stage/phase5/ASSUMPTIONS.md](<H:/Project/Poisoning/implement-stage/phase5/ASSUMPTIONS.md>).
