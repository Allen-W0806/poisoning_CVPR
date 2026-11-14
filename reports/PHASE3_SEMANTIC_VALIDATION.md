# Semantic validation

Review package: `EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review`.

```json
{
  "pairs": 10,
  "source_images": 5,
  "human_verified_changed_edits": 0,
  "automatic_human_agreement": null,
  "inventory_model_calls": 15,
  "model_independence": false,
  "independent_inference_outputs": true,
  "description_disagreement_pairs": 0,
  "unverified": 10,
  "independent_semantic_truth_ready": false
}
```
`review_form.csv` and `review_form.jsonl` include exact original/edited images, question, all reference answers, editing instruction, blank independent validity judgment, reason and reviewer fields. `semantic_validity_table.json` retains per-sample checks, raw separate single-image inventories, uncertainty categories and disagreements with prior proxy feedback. `images/review_contact_sheet.png` is a 10-row original/edited sheet.

All ten changed Phase2 natural pairs remain **UNVERIFIED**, human verified count0. Object presence/identity, color, count, spatial, attributes, plausibility and artifact descriptors are obtained in separate single-image calls, not reused training pair judgments. Exact description matches/differences and independent numerical RGB/saturation/roughness checks are diagnostics. Wording differences are not proven semantic changes; unsupported/uncertain fields remain UNKNOWN. These calls use the same frozen Qwen family, so they are separate inference outputs but **not independent model-family or human truth**. Human–automatic agreement is N/A because labels do not exist. No accuracy-on-semantically-preserved-edits claim. The previously observed green-shift snow scene remains an instruction-fidelity warning, not a new human label.

The original v1 parser required all strings; actual model output used integer counts and all10 pairs were marked INVENTORY_PARSE_FAILURE. This engineering failure is retained. A separate CPU-derived audit normalizes only explicit nonnegative integer count representation without inference or human labels. Use the v2 table for automatic review flags, and the original immutable package for images/blank forms.

```json
{
  "derived_run": "EXP_20261027T150420_0c1cc77f77e2_phase3_semantic_schema_audit",
  "metrics": {
    "pairs": 10,
    "source_run": "EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review",
    "new_model_calls": 0,
    "human_verified_changed_edits": 0,
    "automatic_human_agreement": null,
    "model_independence": false,
    "parsed_pairs": 10,
    "description_disagreement_pairs": 9,
    "uncertainty_categories": {
      "AUTOMATIC_DESCRIPTION_DISAGREEMENT": 9,
      "AWAITING_INDEPENDENT_REVIEW": 1
    },
    "original_review_form": "/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review/review_form.csv",
    "original_contact_sheet": "/storage/home/ydk297/projects/poisoning/experiments/EXP_20261027T145346_155d3b0a29c0_phase3_semantic_review/images/review_contact_sheet.png",
    "semantic_truth_ready": false
  }
}
```
