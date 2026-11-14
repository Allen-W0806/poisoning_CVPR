import json
import re
import numpy as np


def question_attributes(question):
    q = question.lower()
    return {"object_identity": True, "color": bool(re.search(r"colou?r", q)),
            "count": bool(re.search(r"how many|number of|count", q)),
            "spatial": bool(re.search(r"where|left|right|above|below|behind|front|direction", q)),
            "attributes": True, "answerable": True}


def parse_judgment(text):
    try:
        raw = text.strip()
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
        value = json.loads(raw)
        keys = ["object_identity", "color", "count", "spatial", "attributes", "answerable", "answer_preserved"]
        if any(not isinstance(value.get(k), bool) for k in keys):
            raise ValueError("Missing boolean judgments")
        return {**value, "parse_ok": True}
    except (ValueError, TypeError):
        return {"parse_ok": False, "raw": text, "reason": "Malformed/ambiguous structured verifier output"}


def semantic_decision(original, edited, record, consistency, judgment, known_invalid_reason=None):
    sample_id = record["sample_id"]
    attrs = question_attributes(record["question"])
    common = {"sample_id": sample_id, "question_attributes": attrs, "human_verified": False,
              "verifier_version": "phase2_conservative_v1", "disagreements": []}
    if original.size != edited.size:
        return {**common, "valid": False, "verified": True, "score": 0., "reason": "Geometry mismatch", "verifier": "geometry_guard"}
    if known_invalid_reason:
        return {**common, "valid": False, "verified": True, "score": 0., "reason": known_invalid_reason, "verifier": "constructed_negative_control"}
    if np.array_equal(np.asarray(original), np.asarray(edited)):
        return {**common, "valid": True, "verified": True, "score": 1., "reason": "Exact original-image identity, question unchanged", "verifier": "identity_guard"}
    required = [k for k, required in attrs.items() if required]
    structured = judgment.get("parse_ok", False) and all(judgment.get(k) is True for k in required + ["answer_preserved"])
    consistent = consistency["score"] == 1
    if structured != consistent:
        common["disagreements"].append("Qwen short-answer consistency and structured pair judgment disagree")
    # Same-model automatic agreement is a useful proxy, never proof of unchanged GT.
    return {**common, "valid": False, "verified": False, "score": .5 if structured and consistent else 0.,
            "provisionally_valid": bool(structured and consistent), "verifier": "Qwen_pair_and_answer_proxy",
            "reason": "Automatic proxy agreement; independent human verification pending" if structured and consistent else "Automatic contradiction, ambiguity or verifier parse failure"}


def review_pair(model, original, edited, record):
    import torch
    prompt = ("Compare image 1 (original) and image 2 (edited) only with respect to this question: "
        + record["question"] + "\nReturn JSON only with boolean keys object_identity, color, count, spatial, attributes, "
        "answerable, answer_preserved and a short reason. Each boolean says the question-relevant property/answer is preserved. "
        "For a property irrelevant to the question use true. If uncertain use false. Do not use image realism or similarity as proof.")
    messages = [{"role": "user", "content": [{"type": "image", "image": original},
        {"type": "image", "image": edited}, {"type": "text", "text": prompt}]}]
    inputs = model.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
        return_dict=True, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        output = model.model.generate(**inputs, do_sample=False, max_new_tokens=192)
    text = model.processor.batch_decode(output[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
                                        clean_up_tokenization_spaces=False)[0]
    model.calls += 1
    return {**parse_judgment(text), "raw_output": text}
