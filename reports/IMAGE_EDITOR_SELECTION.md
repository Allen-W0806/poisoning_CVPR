# Image editor selection

Selected native image-and-instruction **InstructPix2Pix** (`timbrooks/instruct-pix2pix`, revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`). Frozen Qwen environment was preserved; new `.venv-editor` uses Diffusers 0.35.1/PEFT 0.17.1 and the existing Torch 2.6.0 cu124 base.

| Criterion | InstructPix2Pix | Step1X-Edit |
|---|---|---|
| Image-conditioned editing | native image + instruction, actually executed | native documented |
| Public weights / revision | pinned above; only 2.134 GB required FP16 components cached | `stepfun-ai/Step1X-Edit` revision `752c0bc28a43f56249f588432a5964f1878c01ff`; metadata inspected, weights not downloaded |
| License metadata | model card MIT; underlying SD1.5 CreativeML OpenRAIL-M terms also relevant | model card/repository Apache-2.0 |
| Inference VRAM | measured below on A100; planning expectation 8–16GB was an estimate | upstream reports 42.5GB at 512, H800 (not our measurement) |
| Training VRAM | actual rank-8 LoRA forward/backward measurements below | upstream rank-64 BF16 report 29.7GB at 512; no local/A100 training measurement |
| LoRA modules | actual UNet to_q/to_k/to_v/to_out.0 linear modules inspected | native LoRA scripts documented; not instantiated here |
| Gradient training | actual supervised epsilon-denoising LoRA updates | officially documented paired-edit finetuning; not executed here |
| Reward optimizer | BLOCKED: no validated native scheduler/CFG trajectory density path | unsupported/unverified here |
| A100 compatibility | actual inference and supervised smoke, see IDs | not tested; upstream H800 reports are not A100 certification |
| Dependencies | isolated pinned Diffusers/PEFT; bitsandbytes CPU-only warning irrelevant to AdamW; no 8-bit path used | more model/vision dependencies and larger staging requirement, untested |

Inference evidence: `EXP_20261027T045901_48eded21ef7d_phase2_editor_inference`, generated count 5, peak allocation 2.831 GB, generation total 6.992s. Supervised training evidence: `EXP_20261027T052543_a20882fb60de_phase2_generator_lora_smoke`.

Choosing this backend reduces implementation/dependency risk and retains native image conditioning. This is not a claim that Step1X cannot fit or train on A100. No third backend or text-to-image substitution was needed. Model weight sums across all repository variants are not treated as loaded model size.

Sources: [InstructPix2Pix pipeline](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix), [training reference](https://huggingface.co/docs/diffusers/training/instructpix2pix), [model card](https://huggingface.co/timbrooks/instruct-pix2pix), [Step1X memory/LoRA documentation](https://github.com/stepfun-ai/Step1X-Edit), [Step1X model card](https://huggingface.co/stepfun-ai/Step1X-Edit), [SD1.5 license metadata](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5).
