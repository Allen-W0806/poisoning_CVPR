import csv
import json
from pathlib import Path
from PIL import Image
import _bootstrap
from _phase2 import latest
from _phase3 import phase3_run,mark_latest
from _runs import model_options
from src.models.qwen_loader import QwenVQA
from src.evaluation.semantic_review import describe_image,compare_inventories
from src.evaluation.semantic_inventory_schema import parse_inventory_v2
from src.evaluation.image_edit_consistency import numerical_edit_checks
from src.utils.contact_sheet import contact_sheet
from src.utils.logging import write_json

if __name__=="__main__":
    root=_bootstrap.ROOT
    old=json.loads((latest("semantics")/"individual_decisions.json").read_text())
    rows=[r for r in old if r["kind"] in {"brightness","pretrained"}]
    assert len(rows)==10
    with phase3_run("semantic_review",{"seed":42,"pairs":10,"human_labels":False,"method":"separate_single_image_inventories_and_numerical_checks"},("validation",)) as run:
        qwen=QwenVQA(**model_options("cuda"));sources={};decisions=[];pairs=[]
        for i,item in enumerate(rows):
            source=Image.open(item["source_path"]).convert("RGB");edited=Image.open(item["edited_path"]).convert("RGB")
            # Portable copies retain exact pixels; no new editor generation.
            original_path=run.path/f"images/{i}_original.png";edited_path=run.path/f"images/{i}_edited.png"
            source.save(original_path);edited.save(edited_path)
            key=item["sample_id"]
            if key not in sources:sources[key]=parse_inventory_v2(describe_image(qwen,source,item["question"])['raw_output'])
            inventory=parse_inventory_v2(describe_image(qwen,edited,item["question"])['raw_output'])
            checks=compare_inventories(sources[key],inventory,item["question"])
            instruction="Deterministic brightness multiply 1.02" if item["kind"]=="brightness" else "Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged."
            differences=[k for k,v in checks["checks"].items() if v["status"]=="DESCRIPTION_DIFFERENCE_REVIEW"]
            uncertainty="INVENTORY_PARSE_FAILURE" if not inventory.get("parse_ok") or not sources[key].get("parse_ok") else "AUTOMATIC_DESCRIPTION_DISAGREEMENT" if differences else "AWAITING_INDEPENDENT_REVIEW"
            row={"pair_id":item["pair_id"],"source_experiment":latest("semantics").name,"source_path":str(original_path),"edited_path":str(edited_path),
                "question":item["question"],"reference_answers":item["answers"],"instruction":instruction,
                "independent_validity_judgment":"","judgment_reason":"","reviewer":"","human_status":"UNVERIFIED",
                "source_inventory":sources[key],"edited_inventory":inventory,"automatic_checks":checks,
                "numerical_checks":numerical_edit_checks(source,edited,item["question"]),"uncertainty_category":uncertainty,
                "automatic_disagreement_fields":differences,"prior_proxy_provisionally_valid":item.get("provisionally_valid",False),
                "cross_validator_status":"prior_proxy_agreement_but_inventory_review_flag" if item.get("provisionally_valid",False) and differences else "unresolved_no_ground_truth",
                "prior_same_model_disagreements":item["disagreements"],"eligible_for_semantic_accuracy":False}
            decisions.append(row);run.event("semantic_review",row)
            pairs.append((source,edited,f"{i}: UNVERIFIED | {item['question']}"))
        write_json(run.path/"semantic_validity_table.json",decisions)
        with (run.path/"review_form.jsonl").open("w") as f:
            for row in decisions:f.write(json.dumps({k:row[k] for k in ["pair_id","source_path","edited_path","question","reference_answers","instruction","independent_validity_judgment","judgment_reason","reviewer","human_status"]})+"\n")
        keys=["pair_id","source_path","edited_path","question","reference_answers","instruction","independent_validity_judgment","judgment_reason","reviewer","human_status"]
        with (run.path/"review_form.csv").open("w",newline="") as f:
            writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows([{k:r[k] for k in keys} for r in decisions])
        contact_sheet(pairs,run.path/"images/review_contact_sheet.png",thumbnail=384)
        run.metrics({"pairs":10,"source_images":len(sources),"human_verified_changed_edits":0,"automatic_human_agreement":None,
            "inventory_model_calls":qwen.calls,"model_independence":False,"independent_inference_outputs":True,
            "description_disagreement_pairs":sum(bool(r["automatic_disagreement_fields"]) for r in decisions),
            "unverified":10,"independent_semantic_truth_ready":False})
        run.finish(summary="Ten changed prior pairs packaged with blank reviewer labels and separate-prompt numerical/inventory diagnostics; all remain UNVERIFIED.")
        mark_latest("semantics",run);print(run.path)
