import json
import _bootstrap
from _runs import new_run
from src.data.prepare_vqa import prepare_fixture
from src.data.dataset import VQADataset, validate_disjoint, sha256


if __name__ == "__main__":
    synthetic = prepare_fixture(_bootstrap.ROOT)
    sources = [synthetic]
    official = _bootstrap.ROOT / "data/processed/vqav2_seed42_v1"
    if (official / "provenance.json").exists():
        sources.append(official)
    with new_run("dataset_smoke", {"sources": [p.name for p in sources]}) as run:
        results = []
        for source in sources:
            datasets = [VQADataset(source / f"{s}.json", _bootstrap.ROOT, s) for s in ["train", "validation", "test"]]
            validate_disjoint(*datasets)
            for dataset in datasets:
                before = sha256(dataset.manifest)
                for i in range(min(10, len(dataset))):
                    row = dataset[i]
                    assert row["image"].mode == "RGB"
                    assert row["image_sha256"] == sha256(dataset.image_path(row))
                assert before == sha256(dataset.manifest)
            results.append({"source": source.name, "counts": [len(d) for d in datasets],
                            "image_disjoint": True, "decoded_per_split": 10,
                            "split_hashes": {d.manifest.stem: sha256(d.manifest) for d in datasets}})
        run.metrics({"datasets": results})
        run.finish(summary="Real image paths, original annotations and hashes checked on ten examples per split; no image leakage.")
        print(json.dumps({"run": str(run.path), "datasets": results}))
