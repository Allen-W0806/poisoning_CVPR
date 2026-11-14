# Actual image inspection

Inspected the saved five-row original/generated comparison grid from the actual InstructPix2Pix inference run, not a rendered mock-up.

- The first snowboarding image has a pronounced green color shift despite the instruction to preserve colors. This is a concrete instruction-fidelity limitation; it must not be described as guaranteed mild brightness preservation.
- The remaining rows retain visibly recognizable scene content at this contact-sheet scale, but small object/count/attribute changes cannot be ruled out by visual similarity. The sky/cloud and missing-pizza questions illustrate why question-specific checks matter.
- This inspection is by the assistant, not an independent human-labeling exercise. Human-verification fields remain unset. Full pair images and a blank review form are retained for genuine review.

The grid is `reports/phase2_editor_grid.png` in the Windows mirror and also in the immutable editor experiment images directory. Semantic scores and quality metrics are separate.

The completed semantic experiment's native 512×6200 contact sheet was also inspected: `EXP_20261027T052713_2e70f825d4bd_phase2_semantic_validation/images/human_review_contact_sheet.png`. It contains all twenty original/edited rows, including five identity controls, five mild brightness pairs, five fully occluded diagnostic outputs, and five real pretrained edits. The snow-scene green shift is visible again. This confirms the sheet and negative controls were produced; it does not supply independent human labels or resolve small-object/count/attribute uncertainty.
