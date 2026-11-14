"""Independent numerical image diagnostics; no object-grounding claim."""
import numpy as np

def numerical_edit_checks(original,edited,question):
    a=np.asarray(original.convert("RGB"),dtype=np.float32)/255
    b=np.asarray(edited.convert("RGB"),dtype=np.float32)/255
    if a.shape!=b.shape:raise ValueError("Geometry mismatch")
    color_shift=np.abs(a.mean((0,1))-b.mean((0,1))).tolist()
    saturated=float(((b<=0.005)|(b>=.995)).mean())
    roughness=float(np.abs(np.diff(b,axis=0)).mean()+np.abs(np.diff(b,axis=1)).mean())
    return {"rgb_channel_mean_shift":color_shift,"saturated_channel_fraction":saturated,
        "adjacent_pixel_roughness":roughness,"color_question":("color" in question.lower() or "colour" in question.lower()),
        "semantic_validity":"UNKNOWN","scope":"global photometric and artifact diagnostics; no object/color/count/spatial truth without grounding"}
