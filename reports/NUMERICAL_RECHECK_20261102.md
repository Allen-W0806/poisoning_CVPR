# Targeted numerical evidence recheck — 2026-11-02

Review independence: **same-family**. Acceptance status: **provisional**. Initial review and its findings remain preserved in `NUMERICAL_REVIEW_20261102.md/.json`. This recheck reads only the revised helper, final regression receipt and six test definitions, corrected cohort lock, current comparisons, preserved-version inventory, and the two requested reports. No model/GPU work or large-witness rehash was performed.

**Scoped recheck: PASS for closure of all six documentation/comparator findings, with the experimental limitations accurately qualified. Numerical acceptance remains FAIL; root cause, repair, and learning eligibility remain unestablished.** This scoped PASS is not a numerical or scientific acceptance.

| Initial finding | Recheck |
|---|---|
| Missing ordinary raw/forward boundaries | Accurately qualified. Main report lines 18–22 explicitly says these are unobserved. Comparator lines 65–80 records cross-mode missing keys and A2/B1 equality. The saved preclip norm scalar now supplies an earlier measurement than postclip, without inventing a raw-gradient witness. |
| Overstated capture scope | Resolved in main report lines 18–19: 32 selected LoRA module inputs/outputs/adjoints at transition 0; 32-parameter contributions over 20 transitions. Original protocol is preserved unchanged. |
| Checkpoint round-trip versus continuation | Resolved in main report line 49 and next-experiment lines 11/35: four round-trips passed; resumed next optimizer update was not tested and remains required. |
| Misleading inherited independent-evaluation roles | Resolved. Corrected/latest locks have identical SHA256 `b2f16a196fb8d351c92cfc1416149db9df1f588d5777419eca10cf821ca5cf95`. All effective roles are development-only; `historical_role` preserves the original labels; `eligible_now` is false. |
| Constant-coefficient versus scientific objective | Resolved by the final clarification at main-report line 7: executed job 88422 used constant coefficient 1 instead of original scientific rewards; the production loss implementation remained unchanged; original scientific-reward efficacy was not evaluated. Verified in both canonical and local report bytes. |
| Tolerance reference and undefined relative-L2 aggregation | Resolved in helper lines 16–29: absolute tolerance scales the first tensor consistently; undefined count and null maximum are retained; aggregate L2 is separate. Six meaningful regression definitions inspected; canonical final receipt reports six tests OK. No additional rerun needed. |

The current JSON and report agree on first measured A1/A2 difference: preclip norm `1.0337369441986084` versus `1.033760666847229`, absolute delta `2.372264862060547e-05`. This is a scalar returned by the original clipping call. Ordinary raw gradients remain missing, so backward-gradient divergence versus norm reduction remains unresolved. A2/B1 equality is disclosed and no instrumentation-causality claim is made. Family-specific postclip, adapter, and Adam metrics match the current comparison JSON.

Preservation check: original protocol SHA256 remains `a77287fb2c51955451d35b4b5206dfdafaee10d9c918fd5f80407857adbf31e5`; original execution summary remains `82dc00b9802d3e55e937910a7640846f7a0775cbe1fd3a50cd95682ddca1eb8c`. Initial/v2 comparisons and initial helpers exist under the declared preserved paths. Recheck local/canonical helper and report hashes matched at read time.

Reviewed helper SHA256: `e708e5da9fcb488e3998d245af5355abc6897eb2402446023225781622d53b6c`. Final main report SHA256, matching canonical/local copies after the requested wording clarification: `49e64376f76d90a089bb8818294a1014e59be8e90642a21761e66ac821327df5`. Next-experiment report: `ad8eb73f3e25fee939c1059e5b85efd8996c5e3c98317cc05ce06b0b24e17064`.
