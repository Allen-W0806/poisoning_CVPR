from .vqa_official_compatible import normalize_answer, score_answer


def answer_consistency(original_answer, edited_answer, record):
    original_score = score_answer(original_answer, record)["score"]
    edited_score = score_answer(edited_answer, record)["score"]
    same = normalize_answer(original_answer) == normalize_answer(edited_answer)
    return {"original_answer": original_answer, "edited_answer": edited_answer,
            "same_normalized_answer": same, "original_reference_score": original_score,
            "edited_reference_score": edited_score,
            "score": float(same and original_score > 0 and edited_score > 0),
            "scope": "diagnostic until semantic validity is verified; consistency alone is not correctness"}
