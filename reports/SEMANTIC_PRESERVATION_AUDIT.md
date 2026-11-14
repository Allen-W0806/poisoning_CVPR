<!-- semantic_clean_20261029_v1 -->
> Latest stage: bounded clean-target LoRA training now passes; independent edited-image review remains BLOCKED. Historical results below retain their original scope. See [current decision](NEXT_EXPERIMENT_DECISION.md) and [clean baseline](CLEAN_TARGET_TRAINING_BASELINE.md).

# Semantic Preservation Audit

Run `matched_feedback_20261028_v2`, main pool n=160. **Technical image validity: PASS. Independently confirmed semantic preservation: BLOCKED.**

Frozen C4 generation attempted 160 edits; 160 technically valid (100.00%; binomial95% CI [97.72%, 100.00%]), 0 technical failures. Every output and repeated-generation output is retained. No failed seed or invalid/ambiguous output is silently discarded.

**The confirmed semantic-preservation fraction is unknown**, not 0% or 100%. Independent human reviews:0. Pending C4 cases:160. Confirmed-invalid and confirmed-ambiguous population counts are unknown. Every review row has nine separate fields: scene identity, object existence/count/identity/colors, spatial relationships, text/symbols, VQA-answer preservation and instruction compliance. Exact-identity C1 is proven by identical pixels; that is not a certificate for C2/C3/C4/C5.

Automated diagnostic flags on C4 (review prioritization only): {"risk_darkening": 81, "risk_low_SSIM": 52, "risk_large_luminance_change": 16}. Any risk flag: 102/160 (63.75%). The fixed flags use SSIM<.7, absolute mean luminance change>20, and unintended darkening. These thresholds were frozen before target inference; no similarity threshold is used to certify semantics. Target normalized-answer agreement C4/C0 is 68.12%; this also does not certify preservation.

The contact sheets include every source, all six conditions, original question, ten reference answers, original/edited raw model answers, fixed instruction and automated diagnostics. Full-resolution candidate paths and hashes are in the review queue. [Main overview](../results/matched_feedback_20261028_v2/editor_contact_sheet.png), [all main contact sheets](../results/matched_feedback_20261028_v2/main/contact_sheets/), [review queue](../results/matched_feedback_20261028_v2/semantic_review_queue.csv), [semantic audit data](../results/matched_feedback_20261028_v2/semantic_audit.json).

Any assistant visual notes are stored separately as assistant screening, never independent certification. The independently validated secondary statistical analysis is BLOCKED until signed reviews establish a common matched source subset. The analysis must retain all-attempted results and review failure rates when that secondary subset becomes available. Automated risk counts are not claimed as true scene/object/color failure prevalence.

## Assistant-only illustrative screening

The first four frozen main sources were inspected in manifest order. Two have clear assistant-observed instruction failures: image19967 changes ice-cream appearance/colors; image503689 changes the outdoor baseball scene and object population. The other two cannot be certified at thumbnail scale. These are illustrative cases, not a prevalence sample or independent certification. [Screening notes](../results/matched_feedback_20261028_v2/assistant_screening_notes.json). Independent semantic preservation rate remains unknown.
