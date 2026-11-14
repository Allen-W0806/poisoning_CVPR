import numpy as np


def image_quality(original, edited):
    a = np.asarray(original.convert("RGB"), dtype=np.float32) / 255.0
    b = np.asarray(edited.convert("RGB"), dtype=np.float32) / 255.0
    if a.shape != b.shape:
        raise ValueError("Image dimensions must match")
    mse = float(np.mean((a - b) ** 2))
    mae = float(np.mean(np.abs(a - b)))
    try:
        from skimage.metrics import structural_similarity
        ssim = float(structural_similarity(a, b, data_range=1.0, channel_axis=-1))
        ssim_kind = "windowed_skimage"
    except ImportError:
        # Explicitly named fallback, never reported as standard windowed SSIM.
        ma, mb, va, vb = a.mean(), b.mean(), a.var(), b.var()
        covariance = ((a - ma) * (b - mb)).mean()
        ssim = float(((2 * ma * mb + .01 ** 2) * (2 * covariance + .03 ** 2)) /
                     ((ma ** 2 + mb ** 2 + .01 ** 2) * (va + vb + .03 ** 2)))
        ssim_kind = "global_ssim_fallback"
    distortion = float(np.clip((mae + (1 - ssim) / 2) / 2, 0, 1))
    return {"mse": mse, "mae": mae, "ssim": ssim, "ssim_kind": ssim_kind, "distortion": distortion}
