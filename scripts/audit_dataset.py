"""Compare all selected official records against unmodified source archives."""
import json
import zipfile
import _bootstrap
from _runs import new_run
from src.data.prepare_vqa import OFFICIAL, select_subsets
from src.data.dataset import sha256
from pathlib import Path

if __name__ == "__main__":
    root = _bootstrap.ROOT
    processed = root / "data/processed/vqav2_seed42_v1"
    with new_run("official_data_audit", {"seed": 42, "all_selected_records": 1500}) as run:
        source = {}
        for kind, url in OFFICIAL.items():
            with zipfile.ZipFile(root / "data/raw/vqav2" / Path(url).name) as z:
                member = next(n for n in z.namelist() if n.endswith(".json") and not n.startswith("__MACOSX"))
                source[kind] = json.loads(z.read(member))
        repeat = select_subsets(source["annotations"]["annotations"], source["questions"]["questions"])
        checked = 0
        for split, rows in repeat.items():
            saved = json.loads((processed / f"{split}.json").read_text())
            for original, selected in zip(rows, saved):
                assert all(selected[k] == v for k, v in original.items()), "Selected records differ from source/replay"
                assert selected["image_sha256"] == sha256(root / selected["image_path"])
                checked += 1
            assert len(rows) == len(saved)
        assert checked == 1500
        run.metrics({"records_matched_to_official_sources": checked, "deterministic_selection": True,
                     "all_selected_image_hashes_verified": True})
        run.finish(summary="All 1500 questions, answers, image IDs and selections match immutable official source archives.")
        print(run.path)
