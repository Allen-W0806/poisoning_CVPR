import time
from pathlib import Path
import numpy as np
import torch
from PIL import ImageOps
from .editor_config import PretrainedEditorConfig
from .editor_outputs import EditedOutput
from .transforms import BenignEditor
from ..utils.seed import seed_everything, resolve_device


class PretrainedEditor:
    backend = "instructpix2pix"

    def __init__(self, config: PretrainedEditorConfig, device="cuda", adapter_checkpoint=None):
        from diffusers import StableDiffusionInstructPix2PixPipeline
        self.config = config.validate()
        self.device = resolve_device(device)
        seed_everything(config.seed)
        self.dtype = torch.bfloat16 if config.precision == "bf16" and self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        start = time.perf_counter()
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(config.model_id,
            revision=config.revision, variant=config.variant, torch_dtype=self.dtype,
            use_safetensors=True, local_files_only=True, safety_checker=None, feature_extractor=None,
            requires_safety_checker=False).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        for module in [self.pipe.unet, self.pipe.vae, self.pipe.text_encoder]:
            module.eval().requires_grad_(False)
        if adapter_checkpoint:
            # PEFT places new adapters at the base dtype. Load with FP32 UNet to
            # preserve exact saved FP32 A/B values, then restore only base weights.
            self.pipe.unet.to(dtype=torch.float32)
            self.pipe.load_lora_weights(adapter_checkpoint, adapter_name="default",
                weight_name="pytorch_lora_weights.safetensors", local_files_only=True, low_cpu_mem_usage=False)
            for name, parameter in self.pipe.unet.named_parameters():
                parameter.data = parameter.data.to(torch.float32 if ".lora_" in name else self.dtype)
            self.pipe.unet.eval().requires_grad_(False)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - start
        self.last_outputs = []

    def prepare_image(self, image):
        image = BenignEditor.load(image)
        # Full image retained: padding, never center cropping answer-relevant content.
        return ImageOps.pad(image, (self.config.image_resolution,) * 2, color="white", method=3)

    def edit(self, image, instruction, source_image_id, seed=42, output_path=None, steps=None):
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Natural-language editing instruction is required")
        source = self.prepare_image(image)
        steps = steps or self.config.num_inference_steps
        if not 1 <= steps <= 100:
            raise ValueError("Invalid inference step count")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with torch.inference_mode(), torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            generated = self.pipe(prompt=instruction, image=source, num_inference_steps=steps,
                guidance_scale=self.config.guidance_scale, image_guidance_scale=self.config.image_guidance_scale,
                generator=generator).images[0]
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        a = np.asarray(generated)
        if a.dtype != np.uint8 or generated.size != source.size or not np.isfinite(a).all():
            raise ValueError("Invalid generated image")
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            generated.save(path)
        metadata = {"source_image_id": str(source_image_id), "instruction": instruction,
            "model_name": self.config.model_id, "model_revision": self.config.revision,
            "seed": seed, "settings": {**self.config.to_dict(), "num_inference_steps": steps},
            "generation_seconds": elapsed,
            "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if self.device.type == "cuda" else 0,
            "output_path": str(output_path) if output_path else None,
            "preprocessing": "aspect-preserving resize and white padding to square; no crop",
            "scheduler": type(self.pipe.scheduler).__name__, "trajectory_metadata": None}
        return EditedOutput(generated, metadata)

    def __call__(self, images, conditions, seed=42):
        if len(images) != len(conditions):
            raise ValueError("Image/condition batch length mismatch")
        self.last_outputs = [self.edit(image, c["instruction"], c.get("source_image_id", i), seed + i,
                                      c.get("output_path"), c.get("steps")) for i, (image, c) in enumerate(zip(images, conditions))]
        return [o.image for o in self.last_outputs]
