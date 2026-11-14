# Phase6 data and independent evaluation readiness

Dataset engineering checks PASSED; independent natural-image semantic evaluation NEEDS_HUMAN_REVIEW. No independent edited factual answers or SC/SI-V labels are certified.

Official question/annotation/image joins and bytes were reverified for1500 records. Exact-content splits remain disjoint. Source-content sampling is deterministic,10 distinct contents per logical batch. Old test membership is preserved. All v1/v2 files are unchanged, so no corrected manifest was needed. Duplicate records within a split are reported rather than silently deleted:

```json
{
  "train": {
    "records": 999,
    "unique_image_ids": 988,
    "content_groups": 988,
    "repeated_content_records": 11,
    "repeated_id_records": 11,
    "independently_eligible": 0
  },
  "development": {
    "records": 104,
    "unique_image_ids": 103,
    "content_groups": 103,
    "repeated_content_records": 1,
    "repeated_id_records": 1,
    "independently_eligible": 0
  },
  "gate": {
    "records": 96,
    "unique_image_ids": 94,
    "content_groups": 94,
    "repeated_content_records": 2,
    "repeated_id_records": 2,
    "independently_eligible": 0
  },
  "test": {
    "records": 301,
    "unique_image_ids": 299,
    "content_groups": 298,
    "repeated_content_records": 3,
    "repeated_id_records": 2,
    "independently_eligible": 0
  }
}
```

Original dataset answers are source facts, not edited-image answers. Edited facts remain UNKNOWN without independently supplied evidence. Runtime v2 contracts, exact-content batch guards and identity/unknown/disputed tests remain active. Development/Gate and old test were historically exposed; they are not new confirmatory evaluation.

Created24 pair review queue: [experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/index.html](<H:/Project/Poisoning/experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/index.html>) and [experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/review_queue.json](<H:/Project/Poisoning/experiments/EXP_20261028T044803_6cec2c9966a5_phase6_s1_review_package/review_queue.json>). Includes original/edited images, question,10 original answers, fixed instruction, automatic proxies, priority reasons and blank reviewer fields. First-occurrence representatives per source are selected in fixed historical parent/update order, then ranked by proxy disagreement, lower clipping and distortion. Later edits cannot compete. This is diagnostic triage, not a representative or all-pair worst-case estimate.

Seven independent dimensions: instruction completion, realism, content preservation, factual QA validity, image-question relationship, visible artifacts and uncertainty. Edited factual answer, reviewer identity, second reviewer, disagreement, reason and adjudication are separate fields. Reviewers work independently; automatic proxies are collapsed to reduce anchoring. JSON export is UNADJUDICATED and never promotes labels to truth. No reviewer was fabricated, contacted or assumed. A supplied form still requires identity/provenance and independent adjudication before any reward release. Metadata checks cannot establish human truth.

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/data/dataset_v2.py](<H:/Project/Poisoning/src/data/dataset_v2.py>), [src/data/semantic_contract_v2.py](<H:/Project/Poisoning/src/data/semantic_contract_v2.py>), [src/progression/tasks.py](<H:/Project/Poisoning/src/progression/tasks.py>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

24对/48图及原始来源核验通过，工作台和来源核实/一致性/代理分析接口已完成；独立提交0，事实与语义仍UNKNOWN。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
