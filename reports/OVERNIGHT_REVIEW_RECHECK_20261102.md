# Overnight controller and report recheck, 2 November 2026

**WARN, same-family provisional.** The revised reports accurately represent the completed run. Job 88433 was the only overnight GPU allocation. It used one A100 for 172 seconds. Admissions closed after the raw-gradient mismatch, no ordinary/resume or efficacy stage ran, and no GPU time remains reserved. Slurm records the unused CPU deadline job 88431 as cancelled with zero elapsed seconds.

This post-execution review covers the revised controller, comparator, fixed-model table and CPU tests. The initial review is preserved in `OVERNIGHT_REVIEW_20261102.md` and its companion JSON. No GPU jobs or model calls were made. No image, gradient tensor or checkpoint files were read for this recheck.

| Initial finding | Verification of revision |
|---|---|
| R1, event delivery and registration | Event, bootstrap and deadline handlers now wait on the kernel lock. An event that sees stale active accounting can register one delayed, token-deduplicated CPU reconciliation. GPU state, budget reservation and intent are written before submission. The existing two-hour monitor remains a fallback. A prolonged accounting delay or unknown job ID after an interrupted submission still requires later reconciliation. |
| R2, deadline handling | A deadline event preserves a processed terminal scientific status. Otherwise it first closes admissions, checks the registration token and pending state before requesting cancellation, and reconciles terminal results. The real terminal receipt records one duplicate callback no-op and one GPU allocation. Install has an existing-state no-op. No regression was observed in the actual closed campaign. |
| R3, ordinary scalar comparison | Two ordinary update sequences and the resumed second update now compare recorded preclip norm and loss in addition to tensors. The CPU regression catches a changed norm with identical tensor evidence. The first revised comparator still accepted completely empty update sequences as documented below. No conditional GPU verification was executed. |
| R4, training exposure and cost | RSI now lists 10 retained and 20 attempted training sources and 1 retained versus 2 attempted optimizer steps. Per-arm NFE is explicitly incremental, the shared 168 is counted once, and historical total NFE is 720. The four branches retain null allocated GPU costs. The 16 per-image records and all previously published numeric metrics are identical between the initial and final small JSON files. |
| R5, coefficient provenance | The user-facing numerical report now identifies the observed runtime norm and explicitly labels the clipping coefficient as reconstructed afterward. The dated immutable protocol retains its original wording. |
| R6, NODE_FAIL handling | Two new tests call the actual decision method. They verify a single excluded-node retry and the stop after a second NODE_FAIL. The configured exception can repeat a partially started payload. A single diagnostic actually ran here. |

## Residual comparator defect at revision c39c0814

The recorded scalar check compared empty lists successfully for both pairs when all three receipts had `updates=[]`. In a read-only, in-memory CPU fixture with identical mock tensor files and valid boolean guards, `analyze_resume` returned both `exact_pass=True` and `tolerance_pass=True`. The result is reproducible from reporting lines 24–35. A validator must require exactly two ordinary updates and one resumed update for these comparisons to support a scientific decision. This defect did not affect job 88433.

## Preservation and accounting

Five archived runtime and reporting scripts match the original reviewed hashes, and the initial fixed-model JSON matches its original hash. The archive inspected at 06:36 UTC did not contain the original 18-test module. Its hash and executed log are retained. The revised CPU log reports 24 tests passing. The terminal receipt and current state agree with the scheduler and the separate 36,000-second budget. The older 3,600-second ledger later reported 2,899 consumed and 701 remaining, with no entry for 88433. The two ledgers remain separate.

The accompanying JSON preserves the compared revisions and residual limitations. The initial numerical verdict and the lack of an efficacy comparison stand.
