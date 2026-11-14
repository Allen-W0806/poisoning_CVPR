"""Execute the pinned official VizWiz accuracy path, without caption metrics.

The original API class and normalization are used directly. AST projection
only stops evaluate after its official setAccuracy call, avoiding unrelated
caption/answerability dependencies. VQAv2's distinct scorer is unchanged.
"""
import ast
from contextlib import redirect_stdout
from functools import lru_cache
import hashlib
import io
from pathlib import Path
import re
import sys

SOURCE = Path(__file__).parent / "_vendor/vizwiz_vqaEval_original.py"
SOURCE_SHA256 = "d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694"
PROTOCOL = "official_vizwiz_API_accuracy_d26ef7ac_v1"


def reference_metadata():
    actual = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    if actual != SOURCE_SHA256:
        raise RuntimeError("Official VizWiz reference source changed")
    return {"protocol": PROTOCOL, "source_SHA256": actual,
            "source_url": "https://vizwiz.cs.colorado.edu/VizWiz_final/vqa_data/API.zip",
            "accuracy_path": "original VQAEval.evaluate through setAccuracy; caption metrics omitted",
            "references": "all ten original answers, unchanged as in official VQA JSON loader",
            "prediction_normalization": "official unconditional punctuation/digit/article/contraction processing"}


@lru_cache(maxsize=1)
def _reference():
    reference_metadata()
    tree = ast.parse(SOURCE.read_bytes(), filename=str(SOURCE))
    definition = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "VQAEval")
    evaluate = next(node for node in definition.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    end = next(i for i, node in enumerate(evaluate.body) if isinstance(node, ast.Expr)
               and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
               and node.value.func.attr == "setAccuracy")
    evaluate.body = evaluate.body[:end + 1]
    module = ast.Module(body=[definition], type_ignores=[])
    namespace = {"sys": sys, "re": re, "__name__": __name__ + ".reference"}
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["VQAEval"]


def score_answer(raw_answer, record):
    answers = list(record["answers"])
    if len(answers) != 10 or any(not isinstance(a, str) for a in answers):
        raise ValueError("VizWiz accuracy requires ten unchanged human answer strings")
    class Stub:
        def __init__(self, rows): self.imgToQA = rows
        def getImgs(self): return list(self.imgToQA)
    image = record.get("image", "one_image")
    ground_truth = {image: {"image": image, "answer_type": record.get("answer_type", "unknown"),
                            "answers": [{"answer": value} for value in answers]}}
    prediction = {image: {"image": image, "answer": str(raw_answer)}}
    evaluator = _reference()(Stub(ground_truth), Stub(prediction), n=10)
    with redirect_stdout(io.StringIO()): evaluator.evaluate()
    normalized = str(raw_answer).replace("\n", " ").replace("\t", " ").strip()
    normalized = evaluator.processDigitArticle(evaluator.processPunctuation(normalized))
    return {"raw_answer": raw_answer, "normalized_answer": normalized,
            "human_reference_answers": answers, "score": evaluator.evalQA[image] / 100.,
            "scoring_protocol": PROTOCOL}
