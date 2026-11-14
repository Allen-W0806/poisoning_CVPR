import json
import random
import _bootstrap
from _phase2 import phase2_run, mark_latest
from src.data.dataset import VQADataset
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.evaluation.image_quality import image_quality
from src.utils.contact_sheet import contact_sheet
from src.utils.logging import write_json

INSTRUCTION = "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."

if __name__ == "__main__":
    data = VQADataset(_bootstrap.ROOT / "data/processed/vqav2_seed42_v1/validation.json", _bootstrap.ROOT, "validation")
    indices = list(range(len(data)))
    random.Random(42).shuffle(indices)
    with phase2_run("editor_inference", {"samples": 5, "resolution": 512, "steps": [10, 20, 30, 10, 20],
                                         "instruction": INSTRUCTION}, ("validation",), editor=True) as run:
        editor = load_pretrained_editor(_bootstrap.ROOT)
        outputs, pairs = [], []
        for j, index in enumerate(indices[:5]):
            row = data[index]
            source = editor.prepare_image(row["image"])
            source_path = run.path / f"images/source_{j}.png"
            source.save(source_path)
            result = editor.edit(row["image"], INSTRUCTION, row["image_id"], seed=42 + j,
                                output_path=run.path / f"images/edited_{j}.png", steps=[10, 20, 30, 10, 20][j])
            item = {**result.metadata, "sample_id": row["sample_id"], "question": row["question"], "answers": row["answers"],
                    "source_path": str(source_path), "quality": image_quality(source, result.image),
                    "semantic_status": "unverified_pending_question_aware_review"}
            outputs.append(item)
            run.event("generated", item)
            pairs.append((source, result.image, f"{row['sample_id']} | {row['question']}"))
        contact_sheet(pairs, run.path / "images/comparison_grid.png")
        run.metrics({"outputs": outputs, "generated_count": len(outputs), "model_load_seconds": editor.load_seconds,
                     "total_generation_seconds": sum(x["generation_seconds"] for x in outputs),
                     "peak_gpu_memory_bytes": max(x["peak_gpu_memory_bytes"] for x in outputs),
                     "valid_image_files": True, "semantic_preservation_claim": False})
        run.finish(summary="Five actual pretrained image-conditioned edits. Semantic validity assessed separately; no quality improvement claim.")
        mark_latest("editor", run)
        print(run.path)
