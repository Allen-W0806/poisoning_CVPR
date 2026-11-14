from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class PretrainedEditorConfig:
    model_id: str = "timbrooks/instruct-pix2pix"
    revision: str = ""
    variant: str | None = None
    image_resolution: int = 512
    precision: str = "bf16"
    inference_batch_size: int = 1
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5
    seed: int = 42
    attention_backend: str = "math"

    def validate(self):
        if not self.revision or self.image_resolution % 8 or self.image_resolution < 64:
            raise ValueError("Pinned revision and resolution divisible by eight are required")
        if self.inference_batch_size != 1 or self.num_inference_steps < 1:
            raise ValueError("Initial physical inference batch is one and steps must be positive")
        if self.precision not in {"bf16", "fp32"}:
            raise ValueError("Supported precision: bf16 or fp32")
        if self.attention_backend != "math":
            raise ValueError("Only the recorded math SDPA backend is supported by this editor interface")
        return self

    def to_dict(self):
        return asdict(self)
