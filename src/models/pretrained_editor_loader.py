import json
from pathlib import Path
from ..editing.editor_config import PretrainedEditorConfig
from .editor_loader import load_editor


def pretrained_config(root, **overrides):
    metadata = json.loads((Path(root) / "reports/phase2_model_metadata.json").read_text())
    item = metadata["timbrooks/instruct-pix2pix"]
    return PretrainedEditorConfig(revision=item["revision"], variant=item.get("variant"), **overrides)


def load_pretrained_editor(root, strict_backward=False, **kwargs):
    """Optional experimental setting; it does not certify reproducibility."""
    if strict_backward:
        from ..rl.cuda_execution_guard import configure_strict_diffusion_backward
        configure_strict_diffusion_backward()
    return load_editor("instructpix2pix", config=pretrained_config(root), **kwargs)
