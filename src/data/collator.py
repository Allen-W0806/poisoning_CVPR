def collate_vqa(records):
    return {"images": [r["image"] for r in records],
            "questions": [r["question"] for r in records],
            "answers": [list(r["answers"]) for r in records],
            "records": records}
