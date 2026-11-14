import numpy as np


def edit_localization(original, edited, allowed_mask=None, threshold=.03):
    a, b = np.asarray(original.convert("RGB"), dtype=np.float32), np.asarray(edited.convert("RGB"), dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("Localization requires matching geometry")
    change = np.max(np.abs(a - b), axis=2) / 255 > threshold
    ys, xs = np.where(change)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1] if len(xs) else None
    outside = None
    if allowed_mask is not None:
        mask = np.asarray(allowed_mask, dtype=bool)
        if mask.shape != change.shape:
            raise ValueError("Allowed mask size mismatch")
        outside = float((change & ~mask).sum() / max(1, change.sum()))
    return {"changed_pixel_fraction": float(change.mean()), "changed_bbox_xyxy": bbox,
            "outside_allowed_fraction": outside,
            "interpretation": "pixel-change localization; no object/semantic region claim without a reviewed mask"}
