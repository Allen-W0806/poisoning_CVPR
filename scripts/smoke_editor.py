import _bootstrap
from _runs import new_run
import numpy as np
from src.data.prepare_vqa import prepare_fixture
from src.data.dataset import VQADataset
from src.models.editor_loader import load_editor
from src.editing.semantic_validator import validate_semantics
from src.evaluation.image_quality import image_quality


def main():
    path = prepare_fixture(_bootstrap.ROOT) / "train.json"
    data = VQADataset(path, _bootstrap.ROOT)
    with new_run("editor_smoke", {"backend": "pillow", "samples": 2}, path) as run:
        editor = load_editor()
        records = [data[i] for i in range(2)]
        edited = editor([r["image"] for r in records], [{"brightness": .97}, {"brightness": 1.03}], seed=42)
        parameters = editor.last_parameters.copy()
        metrics = []
        for i, (r, image) in enumerate(zip(records, edited)):
            assert image.size == r["image"].size and np.asarray(image).dtype == np.uint8
            editor.save(image, run.path / f"edited_{i}.png")
            q = image_quality(r["image"], image)
            assert all(np.isfinite(v) for k, v in q.items() if k != "ssim_kind")
            s = validate_semantics(r["image"], image, r)
            assert s.verified and s.score == 1
            metrics.append({**q, "semantic_score": s.score, "verification_method": s.method, "parameters": parameters[i]})
        run.metrics({"samples": metrics, "semantic_validity_rate": 1.0})
        run.finish(summary="Two benign edits saved; shapes, uint8, finite quality and synthetic semantics checked.")
        print(run.path)


if __name__ == "__main__":
    main()
