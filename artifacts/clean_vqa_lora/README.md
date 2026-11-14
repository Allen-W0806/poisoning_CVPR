# Selected clean-VQA LoRA

Three selected adapters only: seeds42/123/3407, each 3,227,648 bytes, FP32 language attention LoRA. Original files are byte-identical; SHA256 and base model revision are in ADAPTER_MANIFEST.json. Base Qwen3-VL-2B-Instruct revision must be 89644892e4d85e24eaac8bacfd4f463576704203. The upstream model card declares apache-2.0; no new license for the overall repository is granted by this export.

These are normal original-image QA SFT adapters, with no ASR reward or poisoning data. They are not generator/Inner/Outer/RSI checkpoints. Full optimizer/RNG states remain at the original checkpoint locations. Selection used dev data only.
