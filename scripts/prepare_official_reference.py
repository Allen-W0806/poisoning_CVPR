"""Fetch the licensed official evaluator at an immutable upstream commit.

Run with Python 3.11 (lib2to3). Changes are mechanical Python-3 syntax only.
"""
import hashlib
import json
from pathlib import Path
import urllib.request
from lib2to3.refactor import RefactoringTool, get_fixers_from_package
import _bootstrap

if __name__ == "__main__":
    headers = {"User-Agent": "RSI-VQA-evaluator-reference"}
    def get(url):
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
            return r.read()
    commit = json.loads(get("https://api.github.com/repos/GT-Vision-Lab/VQA/commits/master"))["sha"]
    base = f"https://raw.githubusercontent.com/GT-Vision-Lab/VQA/{commit}/"
    original = get(base + "PythonEvaluationTools/vqaEvaluation/vqaEval.py")
    license_text = get(base + "license.txt")
    tool = RefactoringTool(get_fixers_from_package("lib2to3.fixes"))
    converted = str(tool.refactor_string(original.decode().expandtabs(4) + "\n", "vqaEval.py"))
    out = _bootstrap.ROOT / "src/evaluation/_vendor"
    out.mkdir(exist_ok=True)
    (out / "__init__.py").touch()
    (out / "official_vqa.py").write_text(converted, encoding="utf-8")
    (out / "LICENSE_VQA.txt").write_bytes(license_text)
    (out / "REFERENCE.json").write_text(json.dumps({"repository": "GT-Vision-Lab/VQA", "commit": commit,
        "source_url": base + "PythonEvaluationTools/vqaEvaluation/vqaEval.py",
        "original_sha256": hashlib.sha256(original).hexdigest(),
        "converted_sha256": hashlib.sha256(converted.encode()).hexdigest(),
        "conversion": "lib2to3 all standard fixes; tabs expanded to 4 spaces; no scoring changes",
        "license": "upstream BSD-style license retained as LICENSE_VQA.txt"}, indent=2))
    print((out / "REFERENCE.json").read_text())
