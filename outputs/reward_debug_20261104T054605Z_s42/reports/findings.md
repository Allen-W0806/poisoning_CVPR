# Research findings — seed42 debug campaign

Status: completed bounded diagnostic/reference experiment; broad attack/self-improvement claim unsupported. All planned GPU stages are terminal at44166 GPU-seconds (12.2683333h), within20h. No automatic experiment follows this negative result.

The tested hypothesis was that an explicitly task-aligned reward and a learned hyperparameter controller would yield better benign edits than DDPO, as a precursor to semantic-backdoor work. The revised objective passed its analytical controls and avoided the rejected v2 saturation; FP32 ordinary/reload witnesses and actual updates were consistent. This is infrastructure and objective-calibration evidence.

The learned controller did not demonstrate an advantage over DDPO on the16 development sources. Target error remains worse than the pixel-identity reference on average, and visible color failures remain. The proposed method's poisoned-victim ASR was not evaluated. A different, released-code Shadowcast experiment achieved94.5% keyword ASR while showing a0.34375-percentage-point VizWiz decline under26/27 updates. These endpoints must stay separate.

Possible explanations to investigate later, not demonstrated root causes: the frozen editor already changes colors and grayscale appearance; the composite reward does not impose a hard non-target-attribute constraint;100 updates with a three-action controller may be insufficient to alter that behavior. This run did not isolate those explanations, so rank, learning rate, or controller design cannot be assigned causal blame.

Constraints for future work: preserve the rejected v2 and negative v3 evidence; do not equate reward calibration or finite gradients with semantic success; do not tune against these assessed16 sources and call them untouched test data; do not treat source bootstrap as seed uncertainty; do not convert actions to accepted revisions; do not use Shadowcast's result as the proposed method's ASR; do not launch new trials merely to fill tables or obtain a positive headline. BF16 remains an unresolved, separate numerical path.

Recommended next scientific step is to define and independently check target-edit and non-target-preservation criteria before a newly specified efficacy study. A later broader study would need untouched evaluation sources, repeated training seeds, matched source/update/compute controls, a clean-utility margin, poisoned-victim endpoints, and independently validated semantics. These are recommendations, not additional execution in this campaign.

Claim gate: broad claim_supported=no; scoped measurements_supported=yes; confidence=high in the unsupported-broad-claim judgment. Same-family review is provisional; formal integrity audit unavailable. See CLAIMS_FROM_RESULTS.md and the preserved review trace.
