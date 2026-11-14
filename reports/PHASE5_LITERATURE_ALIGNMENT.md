# Phase 5 literature alignment

Engineering implementation is complete within the restricted Phase 5 scope; efficacy remains unverified. These classifications apply to mechanisms, not to transferable performance claims.

| Component | Classification | Evidence and transfer boundary |
|---|---|---|
| Diffusion transition score-function learning | DIRECTLY SUPPORTED BY PRIOR WORK | DDPO; existing SF backend retained, not relabeled PPO/GRPO |
| Image-conditioned IP2Pix UNet attention LoRA | ADAPTED FROM PRIOR WORK | DDPO learning applied to the existing conditional editor; no equivalence to every DDPO implementation claimed |
| Original versus edited facts; SC/SI-V distinction | ADAPTED FROM PRIOR WORK | BadSem supplies semantic distinctions; strict provenance/UNKNOWN contract is our implementation |
| Active observations, reusable program edits, external Gate | ADAPTED FROM PRIOR WORK | Video-RSI; restricted deterministic workflow editor rather than full LLM code editing |
| Training-side integrity curriculum and diagnostic memory edits | NEW RESEARCH HYPOTHESIS | Allowed interfaces exist; effectiveness beyond integrity fixtures is unverified |
| Combined LoRA RL and harness evolution | NEW RESEARCH HYPOTHESIS | No cited paper establishes this combination's efficacy |
| Backdoor effectiveness, improved semantic editing, end-to-end efficiency | CURRENTLY UNVERIFIED | No new scientific training, human labels or post-training security result |

## Primary sources actually inspected

[BadSem, method and evaluation sections 3–5](https://arxiv.org/html/2506.07214): image-only semantic manipulation distinguishes semantic consistency (SC) and visual semantic inconsistency (SI-V). Its predefined-target-word ASR requires the relevant downstream training/evaluation context; a frozen model changing its answer is not this ASR. Category keywords are candidate filters, not independently verified labels.

[Video-RSI, method sections 3.1–3.4 and Algorithm 1](https://arxiv.org/html/2609.37950): an Editor investigates additional observations and revises executable solver procedures. Candidate/incumbent evaluation uses the same private, video-disjoint Gate with accuracy/frame costs and aggregate feedback. Its frame accounting is not total offline search cost; ours reports both selected execution and all attempted revisions. Its 72-video Gate and thresholds are not transplanted as validated natural-image thresholds.

[DDPO, ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/14f75513f0f1ca01de1e826b52e6b840-Paper-Conference.pdf): diffusion denoising transitions form a policy with reward at the sampled image. On-policy score-function updates differ from importance-sampled/PPO multi-pass variants. This project preserves its existing single-pass SF path and fixed normalization; no claim of implementing another estimator is made.

[MetaPoison, method and algorithm](https://proceedings.neurips.cc/paper_files/paper/2020/file/8ce6fc704072e351679ac97d4a985574-Paper.pdf): downstream training is an essential part of evaluating training-time effects. This motivates separating frozen inference observations from post-training outcomes. Its poisoning optimization is not implemented or adapted here.

## Official initial/final code inspection

[Official Video-RSI source](https://github.com/bingjunluo/Video-RSI/tree/5ad014499a5e009b75e3ef05ac83595b78adda09), commit `5ad014499a5e009b75e3ef05ac83595b78adda09`, was read without execution. README's default initial workspace is `solver_full_workspace`; the final candidate is `artifacts/final_harness`. Compared agent/main/tools/skill_policy and the initial evolve/investigation/gate_feedback engine. The final tools expand from 8,615 to 39,355 bytes and add graph scope, event occurrence/order handling, observed-window memory and routing. These are changes to reusable execution, not a learning-rate schedule. MIT reference files and license are retained locally. Our interpreter supports bounded check ordering, evidence acquisition, verification, resource limits, memory representation and training-only curriculum. It does not reproduce the complete Video-RSI solver or its results.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [src/rl/diffusion_policy_v2.py](<H:/Project/Poisoning/src/rl/diffusion_policy_v2.py>), [src/rsi/harness/harness_spec.py](<H:/Project/Poisoning/src/rsi/harness/harness_spec.py>), [src/rsi/harness/failure_investigator.py](<H:/Project/Poisoning/src/rsi/harness/failure_investigator.py>).

Tests: [tests/test_phase5_contracts.py](<H:/Project/Poisoning/tests/test_phase5_contracts.py>), [tests/test_phase5_harness.py](<H:/Project/Poisoning/tests/test_phase5_harness.py>), [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_literature/VideoRSI_SOURCE.json](<H:/Project/Poisoning/reports/phase5_literature/VideoRSI_SOURCE.json>), [reports/phase5_literature/VideoRSI/LICENSE](<H:/Project/Poisoning/reports/phase5_literature/VideoRSI/LICENSE>), [reports/phase5_harness_smoke.json](<H:/Project/Poisoning/reports/phase5_harness_smoke.json>).
