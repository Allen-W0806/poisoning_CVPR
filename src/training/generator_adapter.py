import hashlib
import torch


def add_attention_lora(pipe, rank=8, alpha=16):
    from peft import LoraConfig
    targets = ["to_q", "to_k", "to_v", "to_out.0"]
    found = [n for n, m in pipe.unet.named_modules() if any(n.endswith(t) for t in targets) and isinstance(m, torch.nn.Linear)]
    if not found:
        raise RuntimeError("No compatible UNet attention linear modules found")
    pipe.unet.requires_grad_(False)
    pipe.unet.add_adapter(LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights="gaussian", target_modules=targets))
    for p in pipe.unet.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return {"targets": targets, "matched_modules": found,
            "generator_parameters": sum(p.numel() for p in pipe.unet.parameters()),
            "trainable_parameters": sum(p.numel() for p in pipe.unet.parameters() if p.requires_grad),
            "pipeline_parameters": sum(p.numel() for model in [pipe.unet, pipe.vae, pipe.text_encoder] for p in model.parameters())}


def frozen_hashes(pipe):
    result = {}
    for group, model in [("unet", pipe.unet), ("vae", pipe.vae), ("text_encoder", pipe.text_encoder)]:
        for name, value in model.named_parameters():
            if not value.requires_grad:
                raw = value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                result[group + "." + name] = hashlib.sha256(raw).hexdigest()
    return result


def adapter_state(unet):
    from peft import get_peft_model_state_dict
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(unet).items()}


def load_adapter_state(unet, state):
    from peft import set_peft_model_state_dict
    # PEFT's missing_keys includes frozen base weights, so checking that list
    # wholesale is wrong. Compare the complete adapter namespace before loading.
    expected = adapter_state(unet)
    if set(state) != set(expected):
        raise ValueError("Adapter checkpoint key mismatch: missing=%s unexpected=%s" %
                         (sorted(set(expected)-set(state)), sorted(set(state)-set(expected))))
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.shape != expected[name].shape
                or value.dtype != expected[name].dtype or not torch.isfinite(value).all()):
            raise ValueError(f"Invalid adapter checkpoint tensor: {name}")
    result = set_peft_model_state_dict(unet, state)
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys}")
    restored = adapter_state(unet)
    if any(not torch.equal(restored[name], value.cpu()) for name, value in state.items()):
        raise RuntimeError("Adapter checkpoint did not restore exactly")
