# Phase 5 reusable harness implementation

The CPU integration is a real execution, not a mocked acceptance. Run `EXP_20261028T025518_c497cfd041ee_phase5_harness_evolution` used 32 distinct constructed PNG sources and an exact integrity oracle. A geometry-only workflow accepted altered file content. New header and content-hash observations distinguished a digest discrepancy from a specific wrong-size explanation. The broad quality-measurement hypothesis remained unresolved.

An ineffective decode-only revision was rejected. Reusable geometry+hash verification H0002 was accepted: accuracy 0.5 -> 1.0, false-valid outcomes 16 -> 0, with no new errors on previously correct sources. A regression was rejected; rollback to H0000 and restoration of H0002 were verified. Every attempted program and decision is retained with parent, evidence, affected module, expectation, regression checks, measured cost and accept/reject status.

Selected parent/candidate costs were 32/64 diagnostic operations; candidate execution wall time was 0.00617149 seconds. Total search including rejected attempts cost 259 operations, 25,389 bytes, 0.09014638 wall seconds and 0.07946718 CPU seconds; 0 model calls and 0 GPU seconds. This is not a wall-time speedup claim. Synthetic oracle success establishes only the engineering mechanism.

Run `EXP_20261028T025521_e24fe78ca96a_phase5_active_saved_episode` additionally investigated saved real Phase 4 training case 225164001 using four newly executed CPU diagnostics: headers, content hash, pair quality and trajectory summary. Source hash matched; stored arithmetic reproduced; ten transition states were finite. Cost: 0.32622429 wall seconds, 3.01851716 CPU seconds, 3,325,648 bytes, no model call or update. Recomputing the same metric does not verify its implementation or construct validity. Instruction following, factual correctness and verifier disagreement remain unresolved, so no correction was proposed.

The Editor receives exact registered training-case capabilities; caller split tags alone are insufficient. Gate returns aggregates only and stores private per-item evidence outside the Editor response. A strict declarative interpreter bounds tools, files, diagnostics and revision scope. Memory/resource/curriculum interfaces exist, but the actual demonstrated accepted change is content verification. No arbitrary shell/code generation or independent LLM Editor was implemented. Natural-image Gate refuses absent independent labels.

## Implementation and reproducibility

Implementation source commits: local `82a3febe7049639f2b43ec9a68ed60ff60113619`; canonical `afd9c1a4159d88b00388b0db7ea23e96d8c2e689`. The final documentation commit is recorded separately in [reports/phase5_completion.json](<H:/Project/Poisoning/reports/phase5_completion.json>).

Code: [src/rsi/harness/trace_collector.py](<H:/Project/Poisoning/src/rsi/harness/trace_collector.py>), [src/rsi/harness/failure_investigator.py](<H:/Project/Poisoning/src/rsi/harness/failure_investigator.py>), [src/rsi/harness/revision_proposer.py](<H:/Project/Poisoning/src/rsi/harness/revision_proposer.py>), [src/rsi/harness/revision_validator.py](<H:/Project/Poisoning/src/rsi/harness/revision_validator.py>), [src/rsi/harness/cost_aware_gate.py](<H:/Project/Poisoning/src/rsi/harness/cost_aware_gate.py>), [src/rsi/harness/version_manager.py](<H:/Project/Poisoning/src/rsi/harness/version_manager.py>), [scripts/phase5_smoke.py](<H:/Project/Poisoning/scripts/phase5_smoke.py>), [scripts/investigate_phase5_episode.py](<H:/Project/Poisoning/scripts/investigate_phase5_episode.py>).

Tests: [tests/test_phase5_harness.py](<H:/Project/Poisoning/tests/test_phase5_harness.py>), [tests/test_phase5_protocol.py](<H:/Project/Poisoning/tests/test_phase5_protocol.py>).

Evidence: [reports/phase5_harness_smoke.json](<H:/Project/Poisoning/reports/phase5_harness_smoke.json>), [reports/phase5_saved_episode_investigation.json](<H:/Project/Poisoning/reports/phase5_saved_episode_investigation.json>), [experiments/EXP_20261028T025518_c497cfd041ee_phase5_harness_evolution](<H:/Project/Poisoning/experiments/EXP_20261028T025518_c497cfd041ee_phase5_harness_evolution>).
