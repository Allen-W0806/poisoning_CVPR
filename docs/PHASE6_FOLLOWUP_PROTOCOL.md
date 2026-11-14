# Phase6 follow-up scope and decisions

Reuse completed Phase6 results; do not run its 92-test suite or its stage controller again. Zero scientific GPU budget. Only new review-ingestion, packet/provenance, dashboard-refresh and candidate-revision checks may execute.

Review workflow: original24 prioritized TRAIN pairs remain immutable. New per-pair manifest resolves exact saved edited bytes and episode metadata. Seven categorical dimensions default UNKNOWN; original answers never become edited truth. Review submissions declare human/source independence, image identity and time. Metadata consistency is not proof of human authorship. `reviews/reviewer_provenance.json` is a separately supplied operator attestation; absent attestation means no independently verified human labels. Different human IDs and their independent statements remain distinct; disagreements are retained, never silently majority-voted. Even complete agreement cannot automatically release a training reward or scientific pilot.

Agreement: report overlapping non-UNKNOWN pair counts, coverage of24 sources, exact categorical agreement and Cohen kappa (undefined for constant marginal distributions). Proxy comparisons use only operator-attested submissions and explicit per-dimension group summaries; do not convert the original prioritized queue into representative performance. All synthetic review fixtures remain tests only, never real evidence.

Harness: test new check-order/memory candidates against the same24-source/72-row structural Gate and same >=20% operation-saving/no-regression criterion. Retain rejection when measured operations do not improve; serialized bytes alone do not alter the frozen criterion. No new semantic improvement claim.

Dashboard: reuse the existing live monitor/state/completion infrastructure. Preserve the minimal-page snapshot. Integrate the supplied complete package if its actual source becomes available; do not claim to have deployed an inaccessible package. Chinese/English views share one authoritative JSON snapshot; finalized-experiment events and verified stage changes refresh it. HTTP preview polls JSON with no-store; file views require explicit sync/reload, with an opt-in timer clearly labeled as rereading the local copy.

Label availability is checked in each existing two-hour monitor cycle. Individual malformed submissions yield recorded validation errors, not a monitor crash or fake labels. Review stages remain scientifically NO-GO until independent validity, accepted reward, fresh private evaluation and explicit budget are separately satisfied.
