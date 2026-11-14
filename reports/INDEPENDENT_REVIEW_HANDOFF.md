# Independent dual-review handoff

Interface engineering PASS; human review BLOCKED. No human judgments are invented. Open `results/semantic_clean_20261029_v1/semantic_review/index.html` from the copied offline kit. All160 cases and320 original/edit hashes were verified. The current GPU experiments continue independently.

Assign two distinct human reviewers using `results/replication_monitor_20261029_v1/independent_review_assignments.csv`. Each reviews every image alone, using a separate browser profile/computer, without target predictions, old answer-bearing contact sheets, risk flags or the other reviewer's labels. Keep reviewer1 and reviewer2 exports separate until both are locked. Each reviewer truthfully attests independence and exports the provided CSV. Then a third person adjudicates disagreements. Do not mark empty assignment cells as completed reviews.

Use the existing INSTRUCTIONS.md, reviewer_1_template.csv, reviewer_2_template.csv and adjudication_template.csv in that kit. Submit genuine exports to the existing analysis tool using a new exclusive analysis ID. Agreement and Cohen kappa are computed on paired original labels before adjudication. Independent preservation rate and valid-only accuracy effects remain unknown until real reviews exist. No reviewers have been contacted by this automation.
