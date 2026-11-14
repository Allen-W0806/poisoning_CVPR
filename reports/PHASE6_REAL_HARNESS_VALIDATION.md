# Phase6 recorded-artifact harness validation

Eight distinct TRAIN-source historical episodes were investigated using32 new CPU queries (headers, content digest, pair quality and saved trajectory). Context includes source/edited images, question, source answers, instruction, generation settings, quality/semantic proxies, reward components and execution costs. Source identities and finite saved states were verified. Same-implementation quality arithmetic is reproducible; instruction following, edited factual answers, independent semantic validity and strict continuation remain unresolved.

Fixed source order was written before observations. Eight investigation sources and24 other validation sources do not overlap; all come from historical training, not scientific Gate/test. They are original-source PNGs; Gate does not score the edited images or diagnosis accuracy. Trusted structural oracle uses recorded file bytes and dimensions. Two explicitly injected metadata controls per source create wrong-size and wrong-digest cases. No original/edited image or historical metric was modified.

The accuracy denominator is **72 equally weighted rows**,24 recorded-positive PNGs and48 controls (1:2 mixture), clustered in24 source groups. Incumbent checks geometry+decode+hash; geometry already fully decodes. Proposed reusable revision removes redundant decode. An ineffective reorder was rejected. The revision was accepted, a content-check-dropping regression rejected, and rollback/restore verified. Version programs, proposal evidence, private paired results and all decisions are immutable.

Accepted result: accuracy1.0→1.0, diagnostic operations216→144 (33.3% reduction); zero new row/source errors. This passes the predeclared equal-accuracy/zero-regression/>=20%-operation-saving rule. One-run wall/CPU measurements are in the receipt and may reflect ordering/cache effects; no general speedup or semantic-editing gain is claimed.

Total investigation plus all accepted/rejected Gate attempts:

```json
{
  "wall_seconds": 5.820236200466752,
  "cpu_seconds": 5.723810066000006,
  "diagnostic_operations": 1040,
  "bytes_read": 332716553,
  "model_calls": 0,
  "gpu_seconds": 0.0
}
```

No model calls, no optimizer update, no security evaluation. Independent acceptance here means a separate exact structural oracle with aggregate Editor feedback, not human semantic truth or same-UID filesystem secrecy. Scope is recorded-artifact workflow efficiency only.

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/progression/real_harness.py](<H:/Project/Poisoning/src/progression/real_harness.py>), [src/rsi/harness/harness_executor.py](<H:/Project/Poisoning/src/rsi/harness/harness_executor.py>), [src/rsi/harness/version_manager.py](<H:/Project/Poisoning/src/rsi/harness/version_manager.py>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).

## Phase6 follow-up 20261028

两项新修订在不变Gate下均拒绝：144→144操作，准确率1→1。576次新诊断操作与所有既有调查成本分开保留；没有新增语义改善证据。

完整新增证据与范围：[PHASE6_FOLLOWUP.md](PHASE6_FOLLOWUP.md)。旧报告内容及不可变实验输出保留。
