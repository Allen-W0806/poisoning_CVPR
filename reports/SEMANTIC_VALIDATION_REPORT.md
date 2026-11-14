# Question-aware semantic validation

Evidence: `EXP_20261027T052713_2e70f825d4bd_phase2_semantic_validation`; evaluated pairs: 20.

Separate outputs: visual MSE/SSIM/distortion, original/edited answer consistency, question-dependent semantic decision, pixel-change localization. CLIP/SSIM/LPIPS are never used as semantic proof. The question marks color/count/spatial/object/attribute/answerability relevance. A pairwise structured Qwen review is compared with short-answer agreement and original-reference agreement; all raw judgments, parse failures and disagreements are retained.

Controls: five exact identities are verified valid; five completely occluded original scenes are constructed invalid. Ten changed natural pairs (mild brightness plus real pretrained edits) remain **unverified** without independent review. Automatic agreement produces provisionally_valid, never verified valid. Human verified changed edits: 0. Automatic provisional agreements: 7; disagreements: 3.

Two Qwen prompting modes use the same frozen model and are correlated; they are not independent semantic truth. Only verified-valid pairs are eligible for answer-preserving accuracy aggregation. Other scores are diagnostics and cannot claim ground-truth preservation. Pixel localization has no semantic/object mask unless a reviewed mask is supplied. The full 20-row contact sheet and blank human review template are saved with individual decisions.

The five source questions form a small manually reviewable set; they do not cover every listed semantic attribute. The color-change red-car case is a unit fixture, not evidence that all natural-image color/count/spatial changes were exercised.

Actual editor grid inspection found a pronounced green shift in one snow scene despite a color-preservation instruction; see PHASE2_VISUAL_QA.md. Realistic output or answer agreement alone cannot establish instruction fidelity. This is an implemented conservative validation gate with a remaining human-oracle blocker, not a complete natural-image semantic certification system.
