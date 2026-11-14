# Official VQA baseline

Protocol frozen before the full development run: `configs/phase2_vqa_protocol.json`. Its checksum is recorded in each run. Pinned upstream official evaluator is mechanically Python-3 converted with BSD-style license retained; no scoring logic changes. Phase 1's preliminary scorer/results remain unchanged.

Upstream normalization helper tests cover punctuation, articles, number words, contractions and 10-answer leave-one-out consensus. The original evaluator conditionally normalizes when human answers differ; unanimous-answer behavior is preserved and tested, rather than silently replaced. Raw answer, whitespace-stripped complete answer, normalized diagnostic string, original references and official score are all saved. No label-aware answer extraction.

Qwen revision 89644892e4d85e24eaac8bacfd4f463576704203; fixed generic short-answer system prompt; original question unchanged; greedy 64-token maximum; BF16; pixel budget 4096–262144; microbatch one. First three development predictions repeat exactly. Repeated deterministic outputs are not independent seeds.

## validation

Evidence `EXP_20261027T050746_4556a0cd5c4e_phase2_qwen_baseline_validation`; n=200; official VQA score **81.15%**; mean latency 0.2091s; p95 0.3056s; peak allocation 4.353GB.

| Answer type | N | Score |
|---|---|---|
| number | 25 | 64.40% |
| other | 100 | 74.60% |
| yes/no | 75 | 95.47% |

Complete raw predictions: predictions.jsonl; error cases and config: metrics.json/config.yaml. No results used to alter the frozen scoring/generation protocol.

| Example question | Raw short answer | Human references (first 3) | Score |
|---|---|---|---|
| What time is it? | 7:25 | ['6:55', '6:52', 'uncertain'] | 0.0 |
| How many people are shown? | 7 | ['9', '7', '8'] | 0.9 |
| How many windows are in this picture? | 12 | ['16', '16', '23'] | 0.0 |

## test

Evidence `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test`; n=300; official VQA score **77.63%**; mean latency 0.2011s; p95 0.3007s; peak allocation 4.352GB.

| Answer type | N | Score |
|---|---|---|
| number | 38 | 68.42% |
| other | 150 | 67.27% |
| yes/no | 112 | 94.64% |

Complete raw predictions: predictions.jsonl; error cases and config: metrics.json/config.yaml. No results used to alter the frozen scoring/generation protocol.

| Example question | Raw short answer | Human references (first 3) | Score |
|---|---|---|---|
| How many planes? | 1 | ['1', '1', '1'] | 1.0 |
| How many beds are in this hotel room? | 2 | ['2', '2', '2'] | 1.0 |
| How many human hands are shown? | 2 | ['3', '3', '2'] | 1.0 |

The 300-question held-out subset comes from official train2014 and is **not** the official VQAv2 test-server benchmark. It is evaluated once after completed development under the same protocol. This is a single deterministic baseline, not a robustness or RSI improvement result.
