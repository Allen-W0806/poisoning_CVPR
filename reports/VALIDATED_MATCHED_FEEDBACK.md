# Validated Matched Feedback

Run `semantic_clean_20261029_v1`. **PASS** for saved-score computation, traceability and tested reproducibility. Semantic interpretation and candidate-learning suitability **NOT VALIDATED**; human-valid secondary effects **BLOCKED**. No editing parameters, target predictions or old results were modified. Zero new matched inference or generation calls.

Analysis A retains all 160 source images and all C4 generation attempts; every C4 generated successfully. Accepted raw input is the parent `main_repair_v1` revision, 1,032 records including dependent repeats. The original failed main attempt, its skipped C0 record and six repair controls remain in the parent all-attempt ledger. Scores were recomputed from raw answers and all ten references, and all associations reconciled. Repeated seeds are excluded from bootstrap sample size. Human-valid Analysis B and invalid/ambiguous Analysis C each have n=0 and **BLOCKED**, not an estimated zero effect.

| Condition | Mean original-reference agreement | 95% paired-bootstrap CI | At zero | At one |
|---|---:|---|---:|---:|
| C0 | 84.88% | [79.62%, 89.69%] | 10.00% | 80.00% |
| C1 | 84.88% | [79.62%, 89.69%] | 10.00% | 80.00% |
| C2 | 85.06% | [79.88%, 89.88%] | 9.38% | 80.00% |
| C3 | 85.06% | [80.00%, 89.81%] | 9.38% | 80.00% |
| C4 | 64.94% | [58.00%, 71.69%] | 28.75% | 58.75% |
| C5 | 76.44% | [70.37%, 82.44%] | 16.88% | 70.62% |


| Analysis | n | Comparison | Difference | 95% CI | Win / tie / loss | Holm p |
|---|---:|---|---:|---|---|---:|
| A_all_attempted | 160 | C4-C0 | -19.94 pp | [-26.56 pp, -13.62 pp] | 1.88% / 72.50% / 25.62% | 0.000200 |
| A_all_attempted | 160 | C4-C5 | -11.50 pp | [-17.19 pp, -6.19 pp] | 2.50% / 80.62% / 16.88% | 0.000200 |
| auto_flagged_descriptive | 102 | C4-C0 | -24.02 pp | [-32.65 pp, -15.59 pp] | 1.96% / 67.65% / 30.39% | 0.000200 |
| auto_flagged_descriptive | 102 | C4-C5 | -18.14 pp | [-26.57 pp, -10.20 pp] | 1.96% / 73.53% / 24.51% | 0.000200 |
| auto_unflagged_descriptive | 58 | C4-C0 | -12.76 pp | [-22.59 pp, -3.62 pp] | 1.72% / 81.03% / 17.24% | 0.023998 |
| auto_unflagged_descriptive | 58 | C4-C5 | +0.17 pp | [-2.41 pp, +3.45 pp] | 3.45% / 93.10% / 3.45% | 1.000000 |


Uncertainty uses 10,000 paired image bootstrap resamples. New two-comparison overall Holm families and six-comparison category families are explicitly exploratory, separate from the unchanged parent's four planned comparisons (C4-C0 Holm p=.000400). Small endpoint differences from parent intervals are Monte Carlo bootstrap seed variation, not changed raw values or thresholds. Automated strata are post-edit diagnostics, not semantic-valid subsets. Their selection can bias effects. Category-stratified results are retained in [category_statistics.csv](../results/semantic_clean_20261029_v1/analysis_initial/category_statistics.csv).

Within-image C0-C4 mean sample reward variance is 0.04513125. Exactly 113/160 groups tie (70.625%); 47/160 are nondegenerate (29.375%). C4-C0 wins/ties/losses are 3/116/41. C4-C5 are 4/129/27. A significant population paired difference does not make each group informative. The 36 repeated candidate cases (six sources, three seeds) match exactly; this is not a full 160-source repeat experiment.

The 8.4375-pp C5-C0 loss is associated with geometry preprocessing. C4 adds an average 11.50-pp inherited-reference loss beyond C5 under matched geometry. This separates generation from resizing at the pipeline level; it does not distinguish valid appearance editing from scene/object/label damage. In the 58 unflagged cases C4-C5 is +0.1724 pp, with interval spanning zero; flags cannot establish the true semantic-valid effect.

| Feedback property | Status | Evidence / boundary |
|---|---|---|
| Computable | PASS | Complete raw-answer/reference/sample traceability; official scorer recomputation |
| Tested reproducibility | PASS | 36 selected candidate cases repeat exactly |
| Statistically distinguishable | PASS | Aggregate paired original-reference differences; two overall comparisons survive Holm |
| Semantically interpretable | NOT VALIDATED | Original references may be invalid after edits; 0 independent reviews |
| Suitable for benign original VQA | PASS | Frozen original-image agreement is a defined descriptive task metric |
| Suitable for semantically valid editor ranking or RL | NOT VALIDATED | 70.625% ties, saturation, unknown label preservation; no declared power/sufficiency threshold |

Answer changes are not semantic certification. VQA degradation is not attack success. [Raw per-image feedback](../results/semantic_clean_20261029_v1/analysis_initial/per_image_feedback.csv), [full analysis](../results/semantic_clean_20261029_v1/analysis_initial/analysis.json), [effect plot](../results/semantic_clean_20261029_v1/plots/matched_effects.png).
