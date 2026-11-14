"""Use the pinned official evaluator; preserve original answers and raw output.

Phase 1's preliminary scorer is deliberately unchanged.
"""
from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path

PROTOCOL = "official_vqa_upstream_py3_v1"


def reference_metadata():
    return json.loads((Path(__file__).parent / "_vendor/REFERENCE.json").read_text())


def _reference():
    from ._vendor.official_vqa import VQAEval
    class Stub:
        def __init__(self, qa):
            self.qa = qa
        def getQuesIds(self):
            return list(self.qa)
    return VQAEval, Stub


def normalize_answer(answer):
    VQAEval, Stub = _reference()
    evaluator = VQAEval(Stub({}), Stub({}))
    return evaluator.processDigitArticle(evaluator.processPunctuation(str(answer).replace("\n", " ").replace("\t", " ").strip()))


def score_answer(raw_answer, record):
    VQAEval, Stub = _reference()
    answers = list(record["answers"])
    if len(answers) != 10:
        raise ValueError("Official VQA scoring requires all ten annotator answers")
    # No label-aware extraction: preserve complete unexpected verbose outputs.
    short = str(raw_answer).strip()
    gt = {1: {"answers": [{"answer": a, "answer_id": i + 1} for i, a in enumerate(answers)],
              "question_type": record.get("question_type", "unknown"),
              "answer_type": record.get("answer_type", "unknown")}}
    prediction = {1: {"answer": short}}
    evaluator = VQAEval(Stub(copy.deepcopy(gt)), Stub(prediction), n=10)
    with redirect_stdout(io.StringIO()):
        evaluator.evaluate()
    return {"raw_answer": raw_answer, "extracted_short_answer": short,
            "normalized_answer": normalize_answer(short), "human_reference_answers": answers,
            "score": evaluator.evalQA[1] / 100.0,
            "scoring_protocol": PROTOCOL,
            "normalization_note": "Actual scoring follows upstream conditional normalization, including unanimous-answer behavior"}
