# Clean Qwen three-seed replication

**PASS for bounded training engineering; no accuracy improvement observed.** The three independent adapter-initialization seeds42/123/3407 each execute16 logical steps plus one fresh-object next-step replay. Train32 cached original examples; fresh held-out32 original images excluded from known project exposure by image ID/content hash. All questions/ten-answer lists originate in cached official VQAv2 archives;6.57MB of new original COCO images were fetched from the official source. Source IDs, original hashes, ten human answers, consensus labels, source URLs and selection decisions were frozen before inference. Freshness is project-level; model-pretraining exposure and perceptual near duplicates remain unresolved. The high-consensus category-balanced diagnostic pool is not population representative.

Pinned Qwen revision89644892e4d85e24eaac8bacfd4f463576704203; existing .venv-editor. BF16 base, FP32 rank8/alpha16 language q/v attention LoRA, AdamW1e-4, constant scheduler, accumulation2 and norm cap1. Exactly112 tensors /1,605,632 elements update in every seed. Base and visual parameters remain frozen. The three final adapter hashes differ, confirming separate adaptation states.

| Seed | Job | Before | After | Paired difference | 95% paired CI | Fresh-object next step |
|---:|---|---:|---:|---:|---|---|
| 42 | 86846 | 90.00% | 90.00% | +0.00pp | [+0.00, +0.00]pp | PASS |
| 123 | 86848 | 90.00% | 90.00% | +0.00pp | [+0.00, +0.00]pp | PASS |
| 3407 | 86851 | 90.00% | 90.00% | +0.00pp | [+0.00, +0.00]pp | PASS |

All three before/after image-bootstrap intervals are [81.56%,96.56%]. Shared evaluation images give effective n32, not96; the three training seeds are separate engineering replications. Zero observed score changes produces a degenerate empirical paired interval; this does not establish population equivalence. Seed-to-seed mean-score standard deviation is0 in this small fixture.

Fresh-object continuation creates a new pinned model and LoRA object at logical step8, restores optimizer/scheduler/all saved RNG states, takes the same next update, and compares gradient tensors, adapters, optimizer, scheduler and RNG bitwise. All pass in all three seeds. Repeated and fresh-final-reloaded predictions also match. Finite nonzero gradients, exact optimizer ownership, saved-checkpoint binding and per-tensor parameter changes pass. No assertion was relaxed.

Per-seed statistics, checkpoint hashes and decoded predictions: [replication_statistics.json](../results/replication_monitor_20261029_v1/replication_statistics.json), [evaluation CSV](../results/replication_monitor_20261029_v1/clean_replication_evaluation.csv). Source manifest: [fresh_source_receipt.json](../results/replication_monitor_20261029_v1/fresh_source_receipt.json). This is a small original-VQA LoRA infrastructure test, not full-model training, generalization gain or backdoor evidence.
