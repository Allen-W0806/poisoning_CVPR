"""Real UNet LoRA supervised denoising updates. This is explicitly NOT RL."""
import random
import time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageEnhance
from .generator_adapter import add_attention_lora, adapter_state, load_adapter_state


class GeneratorTrainer:
    def __init__(self, editor, records, config):
        from diffusers import DDPMScheduler
        self.editor, self.pipe, self.config = editor, editor.pipe, dict(config)
        if any(r["split"] != "train" for r in records):
            raise ValueError("Generator updates require training-only records")
        if config["micro_batch_size"] != 1 or config["gradient_accumulation_steps"] != len(records):
            raise ValueError("This feasibility trainer uses microbatch=1, one logical batch of all records")
        self.parameters = add_attention_lora(self.pipe, config["lora_rank"], config["lora_alpha"])
        if config["gradient_checkpointing"]:
            self.pipe.unet.enable_gradient_checkpointing()
        self.optimizer = torch.optim.AdamW([p for p in self.pipe.unet.parameters() if p.requires_grad],
            lr=config["learning_rate"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: 1.0)
        self.noise_scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        if self.noise_scheduler.config.prediction_type != "epsilon":
            raise ValueError("Only verified epsilon-prediction training is enabled")
        self.step = 0
        self.cache = []
        self.instruction = "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."
        with torch.no_grad():
            tokens = self.pipe.tokenizer(self.instruction, padding="max_length", truncation=True,
                max_length=self.pipe.tokenizer.model_max_length, return_tensors="pt").input_ids.to(editor.device)
            hidden = self.pipe.text_encoder(tokens)[0]
            for i, r in enumerate(records):
                source = editor.prepare_image(r["image"])
                target = ImageEnhance.Brightness(source).enhance(1.02)
                def tensor(image):
                    return (torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].to(editor.device, editor.dtype) / 127.5) - 1
                source_latent = self.pipe.vae.encode(tensor(source)).latent_dist.mode()
                generator = torch.Generator(device=editor.device).manual_seed(config["seed"] + 1000 + i)
                target_latent = self.pipe.vae.encode(tensor(target)).latent_dist.sample(generator=generator) * self.pipe.vae.config.scaling_factor
                self.cache.append({"source": source_latent, "target": target_latent, "hidden": hidden, "sample_id": r["sample_id"]})

    @torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    def update(self):
        self.pipe.unet.train()
        self.optimizer.zero_grad(set_to_none=True)
        started = time.perf_counter()
        losses, forward_peak, backward_peak = [], 0, 0
        for item in self.cache:
            noise = torch.randn_like(item["target"])
            t = torch.randint(0, self.noise_scheduler.config.num_train_timesteps, (1,), device=self.editor.device).long()
            noisy = self.noise_scheduler.add_noise(item["target"], noise, t)
            # Official InstructPix2Pix conditioning uses unscaled source latent mode.
            latent_input = torch.cat([noisy, item["source"]], dim=1)
            torch.cuda.reset_peak_memory_stats()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = self.pipe.unet(latent_input, t, item["hidden"], return_dict=False)[0]
                loss = F.mse_loss(prediction.float(), noise.float())
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite generator denoising loss")
            forward_peak = max(forward_peak, torch.cuda.max_memory_allocated())
            (loss / len(self.cache)).backward()
            backward_peak = max(backward_peak, torch.cuda.max_memory_allocated())
            losses.append(float(loss.detach()))
        gradients = [p.grad for p in self.pipe.unet.parameters() if p.requires_grad and p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/nonfinite actual UNet adapter gradients")
        norm = torch.nn.utils.clip_grad_norm_([p for p in self.pipe.unet.parameters() if p.requires_grad], self.config["max_grad_norm"])
        self.optimizer.step()
        self.scheduler.step()
        self.step += 1
        torch.cuda.synchronize()
        return {"step": self.step, "loss": float(np.mean(losses)), "sample_losses": losses,
            "grad_norm": float(norm), "finite_gradients": True, "micro_batches": len(self.cache),
            "logical_batch_size": len(self.cache), "step_seconds": time.perf_counter() - started,
            "forward_peak_memory_bytes": forward_peak, "backward_peak_memory_bytes": backward_peak,
            "optimizer_state_bytes": sum(v.numel() * v.element_size() for s in self.optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)),
            "objective": "supervised_epsilon_denoising_MSE_NOT_RL"}

    def save(self, directory, contract):
        from diffusers.utils import convert_state_dict_to_diffusers
        directory.mkdir(parents=True, exist_ok=False)
        state = adapter_state(self.pipe.unet)
        self.pipe.save_lora_weights(str(directory / "diffusers_adapter"), unet_lora_layers=convert_state_dict_to_diffusers(state),
            unet_lora_adapter_metadata=self.pipe.unet.peft_config["default"].to_dict(), safe_serialization=True)
        torch.save({"adapter": state, "optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict(),
            "step": self.step, "config": self.config, "contract": contract,
            "rng_torch": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state_all(),
            "rng_python": random.getstate(), "rng_numpy": np.random.get_state()}, directory / "training_state.pt")

    def restore(self, directory, contract):
        state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
        if state["config"] != self.config or state["contract"] != contract:
            raise ValueError("Generator checkpoint config/model/data contract mismatch")
        load_adapter_state(self.pipe.unet, state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])
        self.step = state["step"]
        torch.set_rng_state(state["rng_torch"])
        torch.cuda.set_rng_state_all(state["rng_cuda"])
        random.setstate(state["rng_python"])
        np.random.set_state(state["rng_numpy"])
