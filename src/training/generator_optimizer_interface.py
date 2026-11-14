from dataclasses import dataclass


@dataclass(frozen=True)
class GeneratorOptimizerContract:
    architecture: str = "InstructPix2Pix UNet with image-conditioned diffusion"
    inference_scheduler: str = "native pretrained EulerAncestralDiscreteScheduler"
    supervised_objective: str = "DDPM epsilon prediction MSE on benign paired targets"
    reward_based_supported: bool = False
    blocker: str = "No validated trainable transition-log-probability implementation for this native editor scheduler/CFG path; a DDPO-style stochastic scheduler and trajectory likelihood/reference tests are required"

    def reward_update(self, *args, **kwargs):
        raise NotImplementedError(self.blocker)
