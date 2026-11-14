# RSI-Poisoning: next validation decision

PROJECT_ID = RSI-POISONING

Evidence as of 2026-11-01T00:41:48.281335+00:00. **NO-GO remains.** Actual bounded CPU and one-GPU diagnostic work completed; this publication contains deidentified prose and aggregates only.

| Check | Status |
|---|---|
| Source separation within the 80-source pilot | PASS |
| Semantic validity | BLOCKED |
| Image quality metrics | BLOCKED |
| Pretrained generator numerical reliability | FAIL |
| Clean target fine-tuning and existing checkpoint recheck | PASS |
| Benign actual generator RL learning | PASS |
| Integrated RSI revision, memory and rollback | PASS |
| Independent evaluation readiness | BLOCKED |
| Training-induced behavioral efficacy | NOT TESTED |

A random32 of160 historical image pairs are prepared for independent dual human review. Genuine reviews received0; no validity/changed/ambiguous frequencies can yet be estimated. Image SSIM is geometry-aligned; mean0.7340322949377474. LPIPS is missing because the calibrated implementation is unavailable, not substituted. Original, geometry-only and generated conditions remain distinct.

The current paired BF16 diagnostic ordinary path PASS and instrumented path FAIL reverse the previous ordinary FAIL/instrumented PASS pattern. In the new failed path, the earliest mismatch is raw backward gradients, maxabs2.834317274391651e-5 and relativeL2 0.0001648017040489566; forward, restoration and RNG comparisons passed. Different allocations/nodes and instrumentation are confounded. One fixed repetition per path does not establish a fix or a unique cause. All negative evidence is retained.

Existing clean Qwen language-attention LoRA checkpoints were evaluated without further optimization, using fixed greedy decoding and original references. Each checkpoint used two fresh model loads. Evaluation remains historical development-only because unused-source provenance cannot be certified. The aggregate CSV records paired bootstrap intervals. Changes on one common image across two seeds are not independent replication of an efficacy gain.

The integrated benign test uses a tiny random UNet with the actual production DDPO score-function trainer, not GRPO. Each fixed/RSI arm received4 logical updates plus2 extra checkpoint-reference/replay updates,272 forward NFEs and120 backward transition calls. Actual LoRA updates, finite gradients, stochastic recovery, persistent revision memory, strict fixed acceptance/rejection and disk rollback were exercised. The accepted learning-rate proposal has gain3.591412678360939e-06 with95% interval[-9.631318971514701e-06, 1.6934884479269377e-05]; the interval crosses zero. This is not evidence of autonomous improvement, pretrained generator reliability or backdoor behavior.

Cumulative validation GPU budget: 1955/3600 A100 seconds; remaining1645. This is a validation subtotal, not lifetime project costs. Existing two-hour monitoring remains a single Slurm successor chain; no duplicate monitor was installed.

Next: obtain real independent semantic review and adjudication, certify a private evaluation split, and localize ordinary/instrumented numerical instability before pretrained generator learning. Generator reward-release guards remain active. No attack reward, poisoning optimization or ASR-driven learning was enabled.
