import gc
import json
import random
import numpy as np
import torch
import yaml
import _bootstrap
from _phase2 import phase2_run, mark_latest
from src.data.dataset import VQADataset, sha256
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.training.generator_adapter import adapter_state, frozen_hashes
from src.training.generator_trainer import GeneratorTrainer
from src.training.generator_optimizer_interface import GeneratorOptimizerContract
from src.utils.logging import write_json


def state_equal(a, b):
    if isinstance(a, torch.Tensor):
        return torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(state_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(state_equal(x, y) for x, y in zip(a, b))
    return a == b


if __name__ == "__main__":
    root = _bootstrap.ROOT
    config = yaml.safe_load((root / "configs/generator_lora_smoke.yaml").read_text())
    dataset_path = root / "data/processed/vqav2_seed42_v1/train.json"
    data = VQADataset(dataset_path, root, "train")
    indices = list(range(len(data)))
    random.Random(42).shuffle(indices)
    records = [data[i] for i in indices[:10]]
    with phase2_run("generator_lora_smoke", config, ("train",), editor=True) as run:
        editor = load_pretrained_editor(root)
        trainer = GeneratorTrainer(editor, records, config)
        contract = {"model": editor.config.to_dict(), "train_manifest_sha256": sha256(dataset_path),
                    "sample_ids": [r["sample_id"] for r in records], "objective": config["objective"],
                    "target_generation": "brightness_1.02", "adapter_targets": trainer.parameters["targets"]}
        before = adapter_state(editor.pipe.unet)
        frozen_before = frozen_hashes(editor.pipe)
        write_json(run.path / "frozen_before.json", frozen_before)
        history = []
        for i in range(3):
            update = trainer.update()
            history.append(update)
            run.event("supervised_update", update)
            if i == 1:
                trainer.save(run.path / "checkpoints/step_0002", contract)
        after = adapter_state(editor.pipe.unet)
        delta = max(float((after[k] - before[k]).abs().max()) for k in before)
        assert delta > 0, "Actual generator adapter parameters did not change"
        frozen_after = frozen_hashes(editor.pipe)
        write_json(run.path / "frozen_after.json", frozen_after)
        assert frozen_before == frozen_after, "Frozen generator/VAE/text weights changed"
        trainer.save(run.path / "checkpoints/step_0003", contract)
        original_optimizer = __import__('copy').deepcopy(trainer.optimizer.state_dict())
        original_scheduler = trainer.scheduler.state_dict()
        editor.pipe.unet.eval()
        original_image = editor.edit(records[0]["image"], trainer.instruction, records[0]["image_id"], 999,
                            run.path / "images/after_training.png", steps=10).image
        parameter_counts = trainer.parameters
        # Free all old model storage before loading a genuinely fresh instance.
        del trainer, editor
        gc.collect()
        torch.cuda.empty_cache()
        fresh = load_pretrained_editor(root)
        resumed = GeneratorTrainer(fresh, records, config)
        resumed.restore(run.path / "checkpoints/step_0003", contract)
        fresh.pipe.unet.eval()
        reloaded = fresh.edit(records[0]["image"], resumed.instruction, records[0]["image_id"], 999,
                              run.path / "images/reloaded.png", steps=10).image
        max_pixel_difference = int(np.abs(np.asarray(original_image).astype(int) - np.asarray(reloaded).astype(int)).max())
        assert max_pixel_difference == 0, "Fresh-model checkpoint inference diverged"
        resumed.restore(run.path / "checkpoints/step_0002", contract)
        next_result = resumed.update()
        resume_state = adapter_state(fresh.pipe.unet)
        differences = {k: float((after[k] - resume_state[k]).abs().max()) for k in after}
        diagnostic = {"resume_adapter_max_abs_difference": max(differences.values()),
            "original_loss_step3": history[2]["loss"], "resumed_loss_step3": next_result["loss"],
            "largest_differences": sorted(differences.items(), key=lambda x: x[1], reverse=True)[:10],
            "optimizer_equal": state_equal(original_optimizer, resumed.optimizer.state_dict()),
            "scheduler_equal": state_equal(original_scheduler, resumed.scheduler.state_dict())}
        write_json(run.path / "resume_diagnostic.json", diagnostic)
        resumed.save(run.path / "checkpoints/resumed_step_0003", contract)
        run.event("resume_diagnostic", diagnostic)
        assert state_equal(after, adapter_state(fresh.pipe.unet)), "Resumed next-update adapter differs"
        assert state_equal(original_optimizer, resumed.optimizer.state_dict()), "Optimizer restore differs"
        assert state_equal(original_scheduler, resumed.scheduler.state_dict()), "Scheduler restore differs"
        assert next_result["loss"] == history[2]["loss"]
        # Verify the public diffusers adapter loading path too, separately from training state.
        del resumed, fresh
        gc.collect()
        torch.cuda.empty_cache()
        exported = load_pretrained_editor(root, adapter_checkpoint=str(run.path / "checkpoints/step_0003/diffusers_adapter"))
        exported_configs = list(exported.pipe.unet.peft_config.values())
        assert len(exported_configs) == 1
        assert exported_configs[0].r == config["lora_rank"] and exported_configs[0].lora_alpha == config["lora_alpha"]
        exported_state = adapter_state(exported.pipe.unet)
        export_tensor_difference = max(float((after[k] - exported_state[k]).abs().max()) for k in after)
        assert export_tensor_difference == 0 and all(v.dtype == torch.float32 for v in exported_state.values()), "Public adapter load changed FP32 A/B values"
        external_image = exported.edit(records[0]["image"], contract.get("instruction", "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."),
            records[0]["image_id"], 999, run.path / "images/export_reloaded.png", steps=10).image
        export_difference = int(np.abs(np.asarray(original_image).astype(int) - np.asarray(external_image).astype(int)).max())
        assert export_difference <= 2, f"Exported adapter inference difference {export_difference} exceeds declared tolerance 2 uint8 levels"
        adapter_bytes = sum(p.stat().st_size for p in (run.path / "checkpoints/step_0003/diffusers_adapter").glob("*"))
        run.metrics({"parameters": parameter_counts, "trainable_percent": 100 * parameter_counts["trainable_parameters"] / parameter_counts["generator_parameters"],
            "updates": history, "adapter_max_parameter_change": delta, "frozen_tensor_count": len(frozen_before),
            "frozen_weights_unchanged": True, "fresh_model_reload_pixel_difference": max_pixel_difference,
            "export_reload_pixel_difference": export_difference, "resume_next_update_equal": True,
            "optimizer_resume_equal": True, "scheduler_resume_equal": True, "adapter_bytes": adapter_bytes,
            "exported_lora_rank": exported_configs[0].r, "exported_lora_alpha": exported_configs[0].lora_alpha,
            "public_adapter_tensor_difference": export_tensor_difference, "public_adapter_dtype": "float32",
            "resume_diagnostic": diagnostic,
            "mean_step_seconds": float(np.mean([x["step_seconds"] for x in history])),
            "generator_training_kind": "supervised_NOT_RL", "generator_rl_contract": GeneratorOptimizerContract().__dict__})
        run.finish(summary="Actual pretrained UNet attention LoRA updated with supervised DDPM denoising loss; not reward-based generator training.")
        mark_latest("training", run)
        print(run.path)
