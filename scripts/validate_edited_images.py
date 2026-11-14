import json
import random
from pathlib import Path
from PIL import Image, ImageEnhance
import _bootstrap
from _phase2 import phase2_run, mark_latest, latest
from _runs import model_options
from src.models.qwen_loader import QwenVQA
from src.data.dataset import VQADataset
from src.evaluation.answer_consistency import answer_consistency
from src.evaluation.semantic_validation import semantic_decision, review_pair
from src.evaluation.edit_localization import edit_localization
from src.evaluation.image_quality import image_quality
from src.utils.contact_sheet import contact_sheet
from src.utils.logging import write_json


if __name__ == "__main__":
    root = _bootstrap.ROOT
    data = VQADataset(root / "data/processed/vqav2_seed42_v1/validation.json", root, "validation")
    by_id = {r["sample_id"]: i for i, r in enumerate(data.records)}
    editor_metrics = json.loads((latest("editor") / "metrics.json").read_text())
    with phase2_run("semantic_validation", {"pairs": 20, "automatic_judges": "same Qwen in two prompting modes",
                    "human_reviewed": False, "independent_human_gate": "pending"}, ("validation",), qwen=True, editor=True) as run:
        model = QwenVQA(**model_options("cuda"))
        decisions, pairs = [], []
        for j, output in enumerate(editor_metrics["outputs"]):
            record = data[by_id[output["sample_id"]]]
            source = Image.open(output["source_path"]).convert("RGB")
            generated = Image.open(output["output_path"]).convert("RGB")
            variants = [("identity", source.copy(), None),
                        ("brightness", ImageEnhance.Brightness(source).enhance(1.02), None),
                        ("invalid_occluded", Image.new("RGB", source.size, "white"), "All original scene content occluded; original question no longer answerable from the image"),
                        ("pretrained", generated, None)]
            original_answer = model.answer([source], [record["question"]])[0]
            for kind, edited, invalid in variants:
                edited_answer = model.answer([edited], [record["question"]])[0]
                consistency = answer_consistency(original_answer, edited_answer, record)
                judgment = review_pair(model, source, edited, record)
                verdict = semantic_decision(source, edited, record, consistency, judgment, invalid)
                path = run.path / f"images/{j}_{kind}.png"
                edited.save(path)
                result = {"pair_id": f"{record['sample_id']}_{kind}", "kind": kind, "question": record["question"],
                    "answers": record["answers"], "source_path": output["source_path"], "edited_path": str(path),
                    **verdict, "answer_consistency": consistency, "structured_judgment": judgment,
                    "visual_quality": image_quality(source, edited), "localization": edit_localization(source, edited),
                    "eligible_for_answer_preserving_accuracy": verdict["valid"] and verdict["verified"]}
                decisions.append(result)
                run.event("semantic_pair", result)
                pairs.append((source, edited, f"{kind} valid={verdict['valid']} proxy={verdict.get('provisionally_valid',False)} | {record['question']}"))
        assert len(decisions) == 20
        assert sum(d["valid"] for d in decisions if d["kind"] == "identity") == 5
        assert all(not d["valid"] for d in decisions if d["kind"] == "invalid_occluded")
        contact_sheet(pairs, run.path / "images/human_review_contact_sheet.png", thumbnail=256)
        write_json(run.path / "individual_decisions.json", decisions)
        review_template = [{"pair_id": d["pair_id"], "question": d["question"], "human_valid": None, "reviewer": None,
                            "notes": ""} for d in decisions]
        write_json(run.path / "human_review_template.json", review_template)
        run.metrics({"pairs": len(decisions), "verified_identity_valid": 5, "verified_constructed_invalid": 5,
            "nonidentity_human_verified": 0, "automatic_provisional_valid": sum(d.get("provisionally_valid", False) for d in decisions),
            "automatic_disagreements": sum(bool(d["disagreements"]) for d in decisions),
            "model_calls": model.calls, "all_decisions": decisions, "natural_semantic_oracle_ready": False})
        run.finish(summary="20 real image/question pairs with identity/occluded controls. Natural-image edited validity remains conservative and human-unverified.")
        mark_latest("semantics", run)
        print(run.path)
