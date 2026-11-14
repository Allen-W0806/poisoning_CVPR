"""Separate inference records for review; predictions never become human labels."""
import json
import torch
from .semantic_validation import question_attributes

FIELDS=["object_presence","object_identity","color","count","spatial","attributes","plausibility","artifacts"]

def parse_inventory(raw):
    try:
        result=json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        if not all(isinstance(result.get(k),str) and result[k].strip() for k in FIELDS):
            raise ValueError("Missing inventory descriptions")
        return {**result,"parse_ok":True,"raw_output":raw}
    except (ValueError,TypeError):
        return {"parse_ok":False,"raw_output":raw,"reason":"Unparseable independent-prompt inventory"}

@torch.inference_mode()
def describe_image(model,image,question):
    prompt=("Describe ONLY this single image with respect to this question: "+question+
        "\nReturn JSON with string fields object_presence, object_identity, color, count, spatial, attributes, plausibility, artifacts. "
        "For each field provide a concise factual description of the question-relevant objects/property. "
        "For count give visible integer count if clear. For color name actual visible colors. "
        "For spatial describe relative locations. For plausibility say plausible or describe anomaly. "
        "For artifacts say none_visible or describe visible editing defects. Use unknown when uncertain; irrelevant when irrelevant. "
        "Do not infer another image or assume semantic preservation. No markdown.")
    messages=[{"role":"user","content":[{"type":"image","image":image},{"type":"text","text":prompt}]}]
    inputs=model.processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors="pt").to(model.device)
    output=model.model.generate(**inputs,do_sample=False,max_new_tokens=256)
    raw=model.processor.batch_decode(output[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
    model.calls+=1
    return parse_inventory(raw)

def compare_inventories(original,edited,question):
    attrs=question_attributes(question)
    checks={}
    for key in FIELDS:
        relevant=attrs.get(key,True)
        if not relevant: verdict="NOT_APPLICABLE"
        elif not original.get("parse_ok") or not edited.get("parse_ok"):verdict="UNKNOWN"
        else:
            a,b=original[key].strip().casefold(),edited[key].strip().casefold()
            verdict="UNKNOWN" if any(s in a or s in b for s in ["unknown","uncertain","irrelevant"]) else "DESCRIPTION_MATCH" if a==b else "DESCRIPTION_DIFFERENCE_REVIEW"
        checks[key]={"status":verdict,"original":original.get(key),"edited":edited.get(key)}
    return {"checks":checks,"method":"separate_single_image_descriptions_exact_text_comparison",
        "independent_of_training_output":True,"independent_model_family":False,"human_ground_truth":False,
        "limitation":"Wording differences are review flags, not proven semantic changes; matches do not establish validity"}
