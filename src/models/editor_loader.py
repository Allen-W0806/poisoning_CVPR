from ..editing.transforms import BenignEditor


def load_editor(backend="pillow", **kwargs):
    if backend == "instructpix2pix":
        from ..editing.pretrained_editor import PretrainedEditor
        return PretrainedEditor(**kwargs)
    if backend != "pillow":
        raise NotImplementedError("Supported backends: pillow, instructpix2pix")
    return BenignEditor(**kwargs)
