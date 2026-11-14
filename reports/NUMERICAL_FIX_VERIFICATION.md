# Numerical fix verification

PROJECT_ID: RSI-POISONING. UTC 2026-11-02T16:12:42.049377+00:00.

**ROOT_CAUSE_PARTIALLY_LOCALIZED — candidate FAILED ordinary verification. No FIX_VERIFIED claim.**

| Required test | Actual result |
|---|---|
| Matched checkpoint/optimizer/data/RNG | PASS for compared starts |
| Captured exact raw gradients | PASS,raw0/raw1,all256 tensors/1,594,368 FP32 elements |
| Separate tolerance raw checks | PASS for captured pair only |
| Raw pair postclip/Adam/adapter | PASS exact/tolerance |
| Ordinary uninstrumented paired execution | FAIL from first optimizer update |
| Ordinary postclip and Adam comparison | FAIL;some second-moment tolerance checks pass but exact fails |
| Final ordinary adapter | FAIL exact/tolerance |
| Checkpoint save/load and next trajectory | Restore actually executed;next trajectory exact |
| Next optimizer update after checkpoint resume | NOT VERIFIED/TIMEOUT;no saved completed update |
| Frozen weights/trainable ownership | PASS in completed raw/ordinary workers;resume completion receipt missing |

Source change attempted: optional launch guard in `src/rl/cuda_execution_guard.py`,loader flag in `src/models/pretrained_editor_loader.py`. No production caller opt-in was enabled. Original versions and failed candidate source remain. Model/loss/coefficients/strict criteria are unchanged. The helper controls process configuration only and is explicitly not a reproducibility certificate. New CPU regressions6 guard +7 capture/API PASS;full ordinary acceptance FAIL. Per-tensor exact/tolerance,max absolute,relativeL2 and affected counts are stored in `verification/mandatory_verification.json`. Missing fields cannot become a vacuous PASS.

| Job | Stage | Slurm / exit | A100 seconds | Actual result |
|---|---|---|---:|---|
|88495|localization|FAILED / 1:0|47|Capture size guard stopped worker;no paired numerical result|
|88497|localization_bounded|COMPLETED / 0:0|74|Identical selected forward,unequal incoming adjoint before local VJP|
|88499|attention_bridge|COMPLETED / 0:0|87|Captured first-source SDPA bridge exact;later transition12 upstream mismatch remains|
|88502|h1_order|COMPLETED / 0:0|139|Single-thread autograd fails exact and tolerance;not a fix|
|88503|h3_launch|COMPLETED / 0:0|165|Launch-blocked raw pair exact;candidate only,not ordinary verification|
|88504|verification|FAILED / 124:0|340|Raw pair exact;ordinary pair FAIL;resumed optimizer update TIMEOUT|


Actual budget852/960,reserved0,remaining108;old controller not reopened;deadline2026-11-02T16:17:57.636501+00:00. No retry of the failed candidate and no benign training job. Lowest CUDA primitive and software-defect versus numerical-property distinction remain unresolved. No broad audit or completed model smoke test was recreated.
