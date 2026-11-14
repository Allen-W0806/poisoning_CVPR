# Phase 5 tests and evidence preservation

**Scoped engineering GO; scientific comparison NO-GO.**

| Verification | Actual outcome |
|---|---|
| Local new Phase 5 tests |22/22 PASS |
| Canonical full suite, final reviewed source |81/81 PASS,3.500sec |
| Earlier local full suite |75pass +3errors: local Diffusers missing; retained failure log, not reported as full pass |
| Real32-source reusable-harness integration |PASS; actual diagnosis, ineffective rejection, accepted general correction, regression rejection and rollback |
| Saved real training-episode investigation |Four new CPU observations; semantic causes unresolved; no unsupported revision |
| Official data joins/bytes |1500/1500 reverified; exact-content split isolation; independent eligible0 |
| Historical preservation |Local3418 files; canonical3344 files unchanged;32 protected Phase 4 source hashes unchanged on both |
| New scientific execution |0 model calls,0 GPUh,0 optimizer updates; M0–M5 prepared only |

Commands: local `python -B -m unittest tests.test_phase5_contracts tests.test_phase5_harness tests.test_phase5_protocol -v`; canonical `.venv-editor/bin/python -B -m unittest discover -v`. Existing canonical runtime used; no shared runtime installation. Tests include explicit mismatched config, identity/unknown/disputed/malformed verifier, invalid factual answers, runtime image/hash/join mismatch, duplicate-content-before-update, immutable train capability, scope-changing revision rejection, both-direction hard regression, rollback, source coverage, separate inference/post-training and real-artifact requirements. They do not establish human semantic validity or GPU continuation.

Exactly two fresh same-family GPT-6-Astra xhigh assumption-review rounds were performed. Round1 findings led to eligibility, content-batch, security-population, causal-readiness, regression and training-guard fixes. Round2 flagged a bundle-layout interface assumption and stale factual-answer/source-coverage claims. A26 was recorded; string validation and unique-source coverage were corrected and executor-tested afterward. Raw reviews retain their provisional/gaps verdict. No third review, independent acquittal or clean-review claim is made.

Monitor snapshot: enabled=True, installed=True; exactly1 owned pending CPU successor, job86589; next eligibility2026-10-28T04:32:35+00:00; last success2026-10-28T02:32:49.970766+00:00; failures0. It remains the existing two-hour Slurm monitor; no duplicate scheduler was created. Actual starts depend on Slurm. This is a timestamped snapshot, not a guarantee of future state.

All legacy failed/negative experiments, old manifests and original draft remain. Protected historical reward, DDPO backend, drivers and monitor source hashes were not modified. New source, versioned manifests/configs, reference code and audit evidence are retained. Run-specific patch scratch is removed at closure; no checkpoints, model weights or historical evidence are cleanup targets.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [scripts/phase5_preservation.py](<H:/Project/Poisoning/scripts/phase5_preservation.py>), [scripts/phase5_smoke.py](<H:/Project/Poisoning/scripts/phase5_smoke.py>), [scripts/investigate_phase5_episode.py](<H:/Project/Poisoning/scripts/investigate_phase5_episode.py>).

Tests: [tests/test_phase5_contracts.py](<H:/Project/Poisoning/tests/test_phase5_contracts.py>), [tests/test_phase5_harness.py](<H:/Project/Poisoning/tests/test_phase5_harness.py>), [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_new_tests_final.log](<H:/Project/Poisoning/reports/phase5_new_tests_final.log>), [reports/phase5_canonical_cpu_tests_final.log](<H:/Project/Poisoning/reports/phase5_canonical_cpu_tests_final.log>), [reports/phase5_local_cpu_tests.log](<H:/Project/Poisoning/reports/phase5_local_cpu_tests.log>), [reports/phase5_preservation_verified.json](<H:/Project/Poisoning/reports/phase5_preservation_verified.json>), [reports/phase5_canonical_preservation_verified.json](<H:/Project/Poisoning/reports/phase5_canonical_preservation_verified.json>), [reports/phase5_monitor_snapshot.json](<H:/Project/Poisoning/reports/phase5_monitor_snapshot.json>), [implement-stage/phase5/SWEEP_ROUND1.json](<H:/Project/Poisoning/implement-stage/phase5/SWEEP_ROUND1.json>), [implement-stage/phase5/SWEEP_ROUND2.json](<H:/Project/Poisoning/implement-stage/phase5/SWEEP_ROUND2.json>).
