"""Frozen semantic cosine and calibrated perceptual distance, with explicit ranges."""
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from .data import sha256


class ProxyRewardCalculator:
    def __init__(self, root, device="cuda"):
        from huggingface_hub import snapshot_download
        from transformers import CLIPModel, CLIPProcessor
        self.device = torch.device(device)
        self.clip_path = snapshot_download("openai/clip-vit-large-patch14", local_files_only=True)
        self.clip_model = CLIPModel.from_pretrained(
            self.clip_path, local_files_only=True, torch_dtype=torch.float32).to(device).eval().requires_grad_(False)
        self.processor = CLIPProcessor.from_pretrained(self.clip_path, local_files_only=True)
        # Reuse the verified vendored dependency; never pip-install or fetch weights.
        receipt = json.loads((Path(root) / "results/execution_remediation_20261102_v1/"
                              "lpips_dependency_verification.json").read_text())
        module_root = Path(receipt["module_root"])
        calibration = module_root / "lpips/weights/v0.1/alex.pth"
        trunk = Path(receipt["base_file"])
        if sha256(calibration) != receipt["calibration_sha256"] or sha256(trunk) != receipt["base_sha256"]:
            raise ValueError("LPIPS calibration or pretrained AlexNet checksum mismatch")
        # torchvision must resolve precisely the already-verified torch cache.
        os.environ["TORCH_HOME"] = str(trunk.parent.parent.parent)
        if not (Path(os.environ["TORCH_HOME"]) / "hub/checkpoints" / trunk.name).samefile(trunk):
            raise ValueError("AlexNet cache mapping mismatch")
        sys.path.insert(0, str(module_root))
        import lpips
        self.lpips_model = lpips.LPIPS(net="alex", version="0.1", lpips=True,
                                      pretrained=True, pnet_rand=False, verbose=False).to(device).eval().requires_grad_(False)
        self.receipt = {"CLIP_snapshot": str(self.clip_path), "CLIP_revision": Path(self.clip_path).name,
                        "LPIPS_calibration_SHA256": sha256(calibration), "AlexNet_SHA256": sha256(trunk),
                        "LPIPS_input_range": "[-1,1]", "CLIP_input": "PIL RGB via pinned processor",
                        "ASR": "NOT TESTED", "all_reward_parameters_frozen": True}

    @staticmethod
    def _validate_images(images):
        if images.ndim != 4 or images.shape[1] != 3 or not torch.isfinite(images).all():
            raise ValueError("Expected finite Bx3xHxW images")
        if float(images.min()) < -1.00001 or float(images.max()) > 1.00001:
            raise ValueError("Expected images in [-1,1]")

    @torch.no_grad()
    def compute_reward(self, original_imgs, edited_imgs, target_prompt,
                       weight_sem=2.0, weight_stealth=1.0):
        self._validate_images(original_imgs)
        self._validate_images(edited_imgs)
        if original_imgs.shape != edited_imgs.shape or not 0 < weight_sem <= 5 or weight_stealth <= 0:
            raise ValueError("Matched image shapes and both positive reward weights required")
        from PIL import Image
        pixels = ((edited_imgs.detach().cpu().float() + 1) * 127.5).round().clamp(0, 255).byte()
        pil = [Image.fromarray(item.permute(1, 2, 0).numpy()) for item in pixels]
        inputs = self.processor(text=[target_prompt] * len(pil), images=pil,
                                return_tensors="pt", padding=True).to(self.device)
        with torch.autocast(self.device.type, enabled=False):
            outputs = self.clip_model(**inputs)
            similarities = F.cosine_similarity(outputs.image_embeds.float(), outputs.text_embeds.float(), dim=-1)
            # LPIPS's default contract is [-1,1]; preserve a length-B vector even for B=1.
            distances = self.lpips_model(original_imgs.to(self.device).float(),
                                         edited_imgs.to(self.device).float()).flatten(1).mean(1)
        semantic = weight_sem * similarities
        perceptual = -weight_stealth * distances
        if not torch.isfinite(semantic + perceptual).all():
            raise FloatingPointError("Nonfinite CLIP/LPIPS reward")
        return (semantic + perceptual).detach(), semantic.detach(), perceptual.detach()

    @torch.no_grad()
    def measure(self, original, edited, target_prompt, weight_sem=2., weight_stealth=1.):
        import numpy as np
        from skimage.metrics import structural_similarity
        def tensor(image):
            array = np.array(image.convert("RGB"), dtype=np.float32, copy=True)
            return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 127.5 - 1.
        total, semantic, perceptual = self.compute_reward(tensor(original), tensor(edited), target_prompt,
                                                         weight_sem, weight_stealth)
        cosine, distance = float(semantic[0]) / weight_sem, -float(perceptual[0]) / weight_stealth
        a, b = np.asarray(original.convert("RGB")), np.asarray(edited.convert("RGB"))
        return {"CLIP_cosine": cosine, "LPIPS": distance, "training_reward": float(total[0]),
                "common_reward": 2. * cosine - distance,
                "SSIM": float(structural_similarity(a, b, channel_axis=2, data_range=255)),
                "exact_noop": bool(np.array_equal(a, b)), "semantic_validity": "UNREVIEWED",
                "ASR": "NOT TESTED", "Clean_accuracy": "NOT TESTED"}
