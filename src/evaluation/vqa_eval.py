"""Leave-one-reference-out VQA consensus and separately named exact-match score.

Normalization v1 is explicit and tested; full official toolkit string-parity is
not claimed. Never use a generative model to rewrite benchmark questions/labels.
"""
import re
import random
from ..editing.semantic_validator import synthetic_count

NUMBER_WORDS = dict(zip("zero one two three four five six seven eight nine ten".split(), map(str, range(11))))
CONTRACTIONS = {"dont": "don't", "cant": "can't", "isnt": "isn't", "arent": "aren't",
                "wont": "won't", "doesnt": "doesn't", "didnt": "didn't", "its": "it's",
                "youre": "you're", "theyre": "they're", "thats": "that's", "theres": "there's"}


def normalize_answer(answer):
    answer = str(answer).lower().strip().replace("\n", " ").replace("\t", " ")
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)
    answer = re.sub(r"[;\/\[\]\"{}()=+\\_\-><@`,?!]", " ", answer)
    answer = re.sub(r"(?<!\d)\.|\.(?!\d)", "", answer)
    return " ".join(CONTRACTIONS.get(w, NUMBER_WORDS.get(w, "0" if w == "none" else w))
                    for w in answer.split() if w not in {"a", "an", "the"})


def consensus_score(prediction, answers):
    if len(answers) != 10:
        raise ValueError("VQA consensus requires the original ten annotator answers")
    prediction = normalize_answer(prediction)
    references = [normalize_answer(a) for a in answers]
    return sum(min(1., sum(a == prediction for j, a in enumerate(references) if j != i) / 3.)
               for i in range(10)) / 10.


class SyntheticOracleVQA:
    """Engineering oracle, never a replacement for Qwen evaluation."""
    def answer(self, images, questions):
        return [str(synthetic_count(image)) for image in images]


def evaluate_vqa(model, dataset, batch_size=1, limit=None, seed=42):
    rows = []
    n = min(len(dataset), limit) if limit else len(dataset)
    if batch_size < 1 or n == 0:
        raise ValueError("Empty evaluation or invalid batch size")
    indices = list(range(len(dataset)))
    if limit and limit < len(dataset):
        random.Random(seed).shuffle(indices)
    indices = indices[:n]
    for start in range(0, n, batch_size):
        records = [dataset[i] for i in indices[start:min(start + batch_size, n)]]
        answers = model.answer([r["image"] for r in records], [r["question"] for r in records])
        if len(answers) != len(records):
            raise ValueError("Model answer count mismatch")
        for record, answer in zip(records, answers):
            rows.append({"sample_id": record["sample_id"], "image_id": record["image_id"],
                         "question": record["question"], "prediction": answer,
                         "answers": record["answers"], "split": record["split"],
                         "consensus_score": consensus_score(answer, record["answers"]),
                         "exact_any": float(normalize_answer(answer) in [normalize_answer(a) for a in record["answers"]])})
    return {"clean_vqa_consensus": sum(r["consensus_score"] for r in rows) / n,
            "clean_exact_any": sum(r["exact_any"] for r in rows) / n, "samples": n,
            "normalization": "preliminary_v1; official toolkit string-parity not yet established",
            "selection": "seeded_shuffle_prefix" if limit and limit < len(dataset) else "full_split",
            "selection_seed": seed,
            "dataset_source": dataset.records[0].get("source"), "predictions": rows}
