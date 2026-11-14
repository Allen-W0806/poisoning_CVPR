# Fast validation status

- Status: COMPLETED
- Dataset audit: `{"image_count": 64, "content_count": 64, "split_counts": {"train": 32, "validation": 16, "test": 16}, "image_ID_and_SHA256_disjoint": true, "all_images_decoded_and_rehashed": true}`
- Requested: 32/16/16
- Total planned updates per trained arm: 100, divided over 3 rounds.
- Figures: READY
- Numerically highest measured common proxy: RL-RSI (single-seed descriptive ranking only).
- Failure: none recorded

All numbers below come from saved observations. This is a benign lighting-editing pilot. CLIP cosine is semantic text alignment, not ASR, instruction correctness, or VQA accuracy. The 16 mini-test images come from a historically exposed training pool and are development assessment. Frozen edits images with unchanged pretrained weights; it is not a pixel identity baseline. Fixed shares the identical prompt, seed, weights, sampler and observations by explicit user choice; it is an equivalence control, not independent evidence.

## Preserved execution history

- TimeoutError: Preregistered pilot wall-time cap reached; no automatic continuation
- ModuleNotFoundError: No module named 'matplotlib'

The time-cap continuation and CPU-only plotting repair are recorded separately. Completed measurements were preserved; see reporting_repair/receipt.json.
