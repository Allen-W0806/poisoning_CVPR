# Pretrained editor survey (no diffusion weights downloaded)

| Candidate checkpoint | Published license metadata | Interface / adaptation prospect | Resource planning |
|---|---|---|---|
| [timbrooks/instruct-pix2pix](https://huggingface.co/timbrooks/instruct-pix2pix) | model card: MIT; underlying components/data terms need review | Diffusers InstructPix2Pix, UNet LoRA is a plausible extension | plan roughly 8–16 GB inference and 16–32 GB LoRA at modest resolution; estimates, unmeasured here |
| [Qwen/Qwen-Image-Edit](https://huggingface.co/Qwen/Qwen-Image-Edit) | Apache-2.0 | Diffusers QwenImageEditPipeline, adapter integration needs separate engineering validation | 20B-class diffusion transformer alone roughly 40 GB BF16 weights, with encoders/activations additional; one 80GB device may require offload; no measured capacity claim |
| [black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) | FLUX.1 dev non-commercial license | Diffusers FluxKontextPipeline; adaptation is a future backend | 12B-class transformer roughly 24 GB BF16 weights before encoders/activations; plan 48–80GB or offload for inference and measure LoRA separately |

Parameter-count memory arithmetic and planning ranges are estimates, not measured VRAM requirements or training guarantees. Licensing metadata is not legal advice or permission to redistribute all components. Only the lightweight PIL editor has been enabled/tested in this setup. Changing the backend must preserve the images/conditions/seed contract and supply independent semantic checks.
