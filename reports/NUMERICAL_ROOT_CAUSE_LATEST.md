# Numerical root cause — latest

PROJECT_ID: RSI-POISONING. Updated UTC: 2026-11-02T16:12:42.049377+00:00. Canonical root `/storage/home/ydk297/projects/poisoning`.

**ROOT_CAUSE_PARTIALLY_LOCALIZED. No production numerical repair is verified. The launch-blocking candidate FAILED ordinary verification.**

## Earliest observed boundary

Existing88433 CPU analysis confirms the same32 affected LoRA parameters in all four fresh/resident contrasts, within `down_blocks.0.attentions.0/1`, self/cross attention projections. All exact names,module paths,shapes,dtypes,element counts,max absolute and relative-L2 errors are in `results/numerical_fix_20261102_v1/affected_gradients32.csv` (128 pair/parameter rows);the32-name table follows below. All256 gradients remain in strict acceptance; no failing tensor was excluded.

Job88497 used identical node/parameter hooks in two fresh processes,matched checkpoint/optimizer/RNG/conditioning,B10/T2,constant coefficient1. For `down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight`, first source/transition0/CFG call0 has exact forward input/output but unequal incoming `grad_output`:32,768 differing elements,max abs2.2879531602626457e-11,relativeL2 .005070520188901492. Local `MmBackward0` weight VJP differs afterward (max3.972786544181872e-10),then parameter contribution and accumulated gradients. The responsible upstream CUDA operation is **NOT ESTABLISHED**. This is the earliest fully observed boundary in the selected path, not a full-network temporal trace.

Three-branch local VJP sums match a valid explicit FP32 order in every1280 audited case. The selected parameter's matching order is the same across paired workers. This does not identify leaf accumulation ordering as the first origin. It does not exclude upstream graph scheduling. Local LoRA-A already receives an unequal adjoint,so its local VJP or clipping cannot be blamed as the primary origin from this evidence.

Job88499 additionally captured Q/K/V,SDPA output/incoming gradient/dQ/dK/dV and the LoRA-B bridge for the first source's two transitions. Those6 covered SDPA calls and the covered LoRA-B bridge were exact. Under this larger intervention,the first selected upstream mismatch moved to transition12/call38 (208 differing values,max3.8009179137432625e-13). Full forward tensors at that later transition were not retained. Instrumentation changes reproduction; covered SDPA equality is not an ordinary-path release or proof about all SDPA calls.

## Controlled hypotheses and candidate

| Hypothesis | Actual evidence | Conclusion |
|---|---|---|
| H1 backward ordering | CPU1280 sum-order cases valid;88502 ABBA default/single-thread with no layer hooks | Turning off autograd multithreading does NOT fix exact/tolerance; upstream ordering remains unresolved |
| H2 BF16 attention/backward | Existing MATH/deterministic/checkpoint-off settings retained;88499 six observed SDPA bridges exact | No demonstrated offending SDPA primitive; no backend setting relabeled as a new fix |
| H3 CUDA/process execution |88503 only changed CUDA_LAUNCH_BLOCKING0/1 before process CUDA init | Blocked captured raw pair exact,async pair FAIL;candidate support only |
| H4 local VJP versus upstream |88497 selected input/output exact but incoming adjoint unequal;local VJP follows | First observed inequality precedes selected local LoRA VJP |
| H5 mixed precision/casting | Raw LoRA FP32;lossless packing requires exact roundtrip;independent FP64 local references retained | Reference rounding quantified,not a proof of expected nondeterminism or its cause |

No whole-model replacement,scientific-loss/reward coefficient change,SDPA switch,activation-checkpointing change,hyperparameter search or threshold relaxation occurred. The task is a non-adversarial constant-coefficient numerical fixture,not scientific reward evaluation.

An opt-in launch configuration candidate was implemented in `src/rl/cuda_execution_guard.py:9-14` and `src/models/pretrained_editor_loader.py:13-18` (current working-tree line numbers;strict_backward defaultsFalse). It checks CUDA initialization and sets CUDA_LAUNCH_BLOCKING1;late initialization without it fails closed. Original loader and exact executed failed-candidate versions are preserved in `candidate_original/` and `candidate_failed_source/`;the current guard has additional failure-warning documentation. The candidate passed captured raw checks but FAILED uninstrumented ordinary checks. It is not enabled by any production training caller. Guard docs explicitly mark it experimental. No justified mathematical/backward-operator source correction is established.

The88503 subprocess setting was established before interpreter CUDA use;88504 exercised a bootstrap that imports Torch then invokes the guard before Torch-reported CUDA initialization,then loader opt-in. The placement difference and observation styles are unresolved confounds,not an established explanation of failure. Do not generalize the captured pair's PASS to a complete BF16 fix.

## Actual execution and resource accounting

| Job | Stage | Slurm / exit | A100 seconds | Actual result |
|---|---|---|---:|---|
|88495|localization|FAILED / 1:0|47|Capture size guard stopped worker;no paired numerical result|
|88497|localization_bounded|COMPLETED / 0:0|74|Identical selected forward,unequal incoming adjoint before local VJP|
|88499|attention_bridge|COMPLETED / 0:0|87|Captured first-source SDPA bridge exact;later transition12 upstream mismatch remains|
|88502|h1_order|COMPLETED / 0:0|139|Single-thread autograd fails exact and tolerance;not a fix|
|88503|h3_launch|COMPLETED / 0:0|165|Launch-blocked raw pair exact;candidate only,not ordinary verification|
|88504|verification|FAILED / 124:0|340|Raw pair exact;ordinary pair FAIL;resumed optimizer update TIMEOUT|


Total new GPU allocation: **852/960 A100 seconds**,including47-second capture-size failure and340-second bounded verification failure,reserved0,remaining108. This separate urgent-debug ledger uses the still-valid authorization window ending2026-11-02T16:17:57.636501+00:00;the finalized overnight controller remains closed. Previous overnight spend172 plus this852 is1024 under the shared36000 envelope;the older3600 validation ledger is separate. No jobs from RSI-Video were charged,changed or canceled. No further GPU task is admitted.

Each job's exact protocol SHA256,declared source revisions,paths and results are in `job_revisions_and_results.json`;Git base `06bd889d9392c4d53621c3e4af15ad718e84f6c5` plus explicit working-tree hashes identify code,not a fictitious new commit. Failed source versions/logs remain.88504 returned124 because its own predeclared process timeout fired,not because Slurm reported a successful recovery.

## Required verification outcome

Raw captured pair and postclip/Adam/adapter/RNG exact and tolerance PASS. Ordinary first-update adapter:744,464 differing elements,max9.978311936720274e-7,exact/tolerance FAIL. Adam first moment:1,593,772 differing elements,max5.123263690620661e-6,exact/tolerance FAIL. Adam second moment:1,589,926 unequal,max1.657298298596288e-9,exact FAIL,tolerance PASS. Do not call its tolerance result bitwise equality.

Ordinary second-update adapter:1,387,812 differing elements,max6.400543043127982e-6. Ordinary second postclip:1,594,366 differing elements,max.0009832559153437614. These fail exact/tolerance. All ordinary initial checkpoints match. Raw0 after-state equals ordinary0 first-update common state;the ordinary pair still fails,so this one cross-mode match does not release the candidate.

The actual resume process restored the checkpoint and produced an exact92-tensor next trajectory. It did not complete/save the next optimizer update before timeout:after2,postclip,update receipt and final guard receipt are missing. Recovery-next-update is **NOT VERIFIED**,not PASS and not an invented numerical failure. Completed ordinary workers verified frozen weights,256-parameter optimizer ownership and unchanged reward guard. Final CPU verifier explicitly lists missing evidence.

Six new guard-unit tests and seven bounded-capture/API tests passed. Independent same-family provisional review cross-checked20 saved-tensor verification comparisons with zero discrepancies;it rejected a fix claim. Unit tests validate configuration/capture mechanics,not BF16 reliability. See `reports/BF16_FIX_REVIEW_20261102.md`.

# Existing88433 affected-gradient map

All32 affected parameters recur in all four comparisons: True.

| Exact parameter | Shape | Dtype | Fresh/fresh differing elements | Max abs | Relative L2 |
|---|---|---|---:|---:|---:|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight`|8x320|torch.float32|2538|2.77111666946e-12|2.62228538578e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight`|320x8|torch.float32|2538|1.90921127796e-08|5.72793294967e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight`|8x320|torch.float32|2497|2.16004991671e-12|1.23257538894e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight`|320x8|torch.float32|2541|1.43481884152e-08|3.45242177405e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight`|8x320|torch.float32|2450|2.54658516496e-11|7.34790974448e-07|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight`|320x8|torch.float32|2528|2.1792948246e-07|5.88425996227e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight`|8x320|torch.float32|2414|1.11413100967e-11|7.20970069337e-07|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight`|320x8|torch.float32|2549|1.60420313478e-07|8.49733675683e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight`|8x320|torch.float32|2500|5.00222085975e-12|1.30272566477e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight`|320x8|torch.float32|2432|1.30385160446e-08|1.45799127958e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight`|8x768|torch.float32|5576|4.2064129957e-12|4.87829155835e-07|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight`|320x8|torch.float32|2447|1.7462298274e-08|2.084136226e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight`|8x768|torch.float32|5609|2.91038304567e-11|2.21549807309e-07|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight`|320x8|torch.float32|2388|2.19675712287e-07|3.48737452712e-06|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight`|8x320|torch.float32|1734|1.25055521494e-11|3.95780582031e-07|
|`down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight`|320x8|torch.float32|2537|6.50761649013e-08|6.5508495537e-06|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight`|8x320|torch.float32|2303|2.84217094304e-13|3.35118167833e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight`|320x8|torch.float32|2426|1.92085281014e-09|6.96474811921e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight`|8x320|torch.float32|2288|2.55795384874e-13|3.00503390121e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight`|320x8|torch.float32|2441|3.60887497663e-09|6.5546471155e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight`|8x320|torch.float32|1910|1.81898940355e-12|1.05254671616e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight`|320x8|torch.float32|2360|1.16415321827e-08|5.65893199339e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight`|8x320|torch.float32|1438|9.09494701773e-13|6.17882565159e-08|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight`|320x8|torch.float32|2373|1.28638930619e-08|6.47211460691e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight`|8x320|torch.float32|2060|4.54747350886e-13|9.9928882096e-08|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight`|320x8|torch.float32|1257|4.88944351673e-09|3.69916038956e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight`|8x768|torch.float32|2915|4.54747350886e-13|9.65108318105e-08|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight`|320x8|torch.float32|1123|1.18307070807e-08|8.01360359251e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight`|8x768|torch.float32|2726|7.27595761418e-12|8.11736941658e-08|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight`|320x8|torch.float32|923|8.19563865662e-08|7.64223081648e-07|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight`|8x320|torch.float32|490|1.81898940355e-12|3.81059943565e-08|
|`down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight`|320x8|torch.float32|1984|2.32830643654e-09|2.09347861119e-07|

