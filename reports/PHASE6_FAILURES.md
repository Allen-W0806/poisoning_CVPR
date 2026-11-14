# Phase6 failures, corrections and claim limits

All original Phase1–5 finalized evidence and all first Phase6 audit runs remain. Corrected audits are new descendants with explicit source provenance; no overwritten metric or selected favorable seed.

Observed engineering issues and bounded fixes:

```json
{
  "observed_failures": [
    {
      "kind": "monitor_schema_collision",
      "original_error": "TypeError: int object is not iterable at check_associations; metrics.observations=500",
      "correction": "versioned v6 monitor handles image-record lists only",
      "regression": "test_monitor_does_not_interpret_audit_counts_as_image_records",
      "original_runs_preserved": true
    },
    {
      "kind": "PowerShell_runtime_compatibility",
      "errors": [
        "File.Move3 overload unavailable",
        "File.Replace null backup marshaled as illegal path"
      ],
      "correction": "explicit same-volume scratch backup path; atomic Replace/Move; two bounded retries then successful real sync"
    },
    {
      "kind": "dashboard_overflow",
      "observed": "headless Chrome detected horizontal overflow",
      "correction": "box sizing, fixed table layout, min-width and wrapping; desktop/mobile browser QA passed"
    },
    {
      "kind": "browser_tool_unavailable",
      "original_error": "cua failed to write kernel assets",
      "fallback": "installed headless Chrome via bundled Playwright; local file rendering verified"
    },
    {
      "kind": "preview_network",
      "observed": "localhost browser access denied/timeout",
      "correction": "direct local HTML browser QA; no server-hosting claim"
    },
    {
      "kind": "source_provenance_and_completion_review",
      "observed": "two-round gaps retained; first audits used context versions as source labels",
      "correction": "new immutable descendant audits86602; source manifests separated from audit context"
    },
    {
      "kind": "last_review_recovery_branches",
      "observed": "failed-attempt identity, same-cycle dependent invalidation and interrupted Slurm recovery gaps",
      "correction": "current-attempt pointer, transitive invalidation, terminal accounting reconciliation; executor tests only after final review"
    }
  ],
  "scientific_blockers": [
    "independent labels absent",
    "reward validity unresolved",
    "fresh private evaluator absent",
    "strict continuation unresolved",
    "scientific GPU budget0"
  ],
  "no_failed_or_negative_results_deleted": true
}
```

First review found4 undeclared semantic assumptions and2 stale rows: Gate denominator/control weighting, first-occurrence source selection, question-strata versus edit-category coverage, generating versus audit provenance, retry coverage, and provisional versus final completion. These were declared/repaired in code and corrected descendant runs. Raw two-round same-family reviews remain provisional; no cross-family acquittal. Later executor-only fixes are identified in BUILD_NOTE.

Automatic repair is limited to known active output directories/logs and retryable transient IO, at most2 retries. Active-run repair cannot rewrite a finalized run. Failed/transient task retry creates a new descendant; no arbitrary source patching or reward change. Unknown scientific failures remain blocked. Checkpoint continuation and poor reward/semantic evidence are scientific blockers, not ordinary retry targets.

## Source and evidence

Report build source: `181d306b631afd7298faf01fbc4de1233cc31b64`; executed-source commits remain in each immutable run manifest. Final local/canonical commits: [reports/phase6_completion.json](<H:/Project/Poisoning/reports/phase6_completion.json>).

Implementation: [src/progression/completion.py](<H:/Project/Poisoning/src/progression/completion.py>), [src/progression/controller.py](<H:/Project/Poisoning/src/progression/controller.py>), [src/monitoring/integrity_v6.py](<H:/Project/Poisoning/src/monitoring/integrity_v6.py>), [scripts/sync_phase6.ps1](<H:/Project/Poisoning/scripts/sync_phase6.ps1>).

Regression: [tests/test_phase6.py](<H:/Project/Poisoning/tests/test_phase6.py>), [reports/phase6_final_tests.log](<H:/Project/Poisoning/reports/phase6_final_tests.log>). Stage evidence: [reports/PHASE6_STAGE_STATUS.json](<H:/Project/Poisoning/reports/PHASE6_STAGE_STATUS.json>), [reports/phase6/live.json](<H:/Project/Poisoning/reports/phase6/live.json>).
