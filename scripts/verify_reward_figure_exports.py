"""Read-only checks of final PDF/PPTX exports, plus a separate visual-review record."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import zipfile
from pypdf import PdfReader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign", required=True, type=Path)
    p.add_argument("--previews", required=True, type=Path)
    p.add_argument("--visual-review-completed", action="store_true")
    a = p.parse_args()
    c = a.campaign.resolve()
    state = json.loads((c / "internal_pilot/run_state.json").read_text())
    if state["status"] != "COMPLETED":
        raise ValueError("Final exports require completed measurements")
    folder = c / "figures"
    deck = folder / "reward_debug_editable_figures.pptx"
    receipt = json.loads((c / "evidence/editable_figures_validation.json").read_text())
    if digest(deck) != receipt["finalSha256"]:
        raise ValueError("PPTX differs from first-party finalization receipt")
    ns = {"p": "http://schemas.openxmlformats.org/presentationml/2006/main"}
    with zipfile.ZipFile(deck) as z:
        slides = [name for name in z.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml", name)]
        charts = [name for name in z.namelist() if re.fullmatch(r"ppt/(?:slides/)?charts/chart\d+\.xml", name)]
        workbooks = [name for name in z.namelist() if name.startswith("ppt/embeddings/") and name.endswith(".xlsx")]
        pictures = sum(len(ET.fromstring(z.read(name)).findall(".//p:pic", ns)) for name in slides)
    if (len(slides), len(charts), len(workbooks), pictures) != (7, 2, 2, 100):
        raise ValueError("Wrong native object counts")
    pdfs = {}
    for name, expected in [("reward_calibration.pdf", 1), ("editing_parameter_audit.pdf", 1),
                           ("lighting_v3_validation.pdf", 1), ("final_editing_assessment.pdf", 1),
                           ("final_editing_assessment_full16.pdf", 4)]:
        path = folder / name
        reader = PdfReader(path)
        if len(reader.pages) != expected:
            raise ValueError("Wrong PDF page count: " + name)
        if name == "final_editing_assessment_full16.pdf":
            for block, page in enumerate(reader.pages):
                text = page.extract_text()
                for row in state["assessment"]["frozen"][4 * block:4 * block + 4]:
                    if str(row["sample_id"]) not in text:
                        raise ValueError("Full16 PDF source label missing")
        pdfs[name] = {"pages": len(reader.pages), "bytes": path.stat().st_size, "SHA256": digest(path)}
    previews = []
    for number in range(1, 8):
        path = a.previews / f"editable_figures_slide{number}.png"
        previews.append({"slide": number, "SHA256": digest(path)})
    result = {"status": "PASS", "native_PPTX_slides": len(slides), "native_charts": len(charts),
              "embedded_literal_workbooks": len(workbooks), "native_picture_objects": pictures,
              "PPTX_SHA256": digest(deck), "PDFs": pdfs, "preview_hashes": previews,
              "primary_assistant_unblinded_visual_review": a.visual_review_completed,
              "visual_review_scope": "all7 full-size artifact-tool previews; not independent human semantic scoring",
              "native_PowerPoint_execution_verified": False,
              "pypdf_version": importlib.metadata.version("pypdf"), "worker_SHA256": digest(Path(__file__)),
              "source_run_state_SHA256": digest(c / "internal_pilot/run_state.json"),
              "claim_boundary": "Export structure, source-label coverage and declared preview inspection; not a new model or efficacy test."}
    (c / "evidence/figure_export_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
