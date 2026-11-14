"""CPU rescoring of saved development pixels/CLIP/LPIPS; no new model calls."""
import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import write_json
from src.mini_pilot.lighting_reward_v3 import mean_lighting_terms
from scripts.audit_padding_reward_20261104 import content_box


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign",required=True,type=Path)
    args=p.parse_args();c=args.campaign.resolve();old=c/"internal_pilot_v2_saturation_probe"
    from PIL import Image
    cohort=json.loads((old/"cohort.json").read_text());records={r["sample_id"]:r for rows in cohort.values() for r in rows}
    rows=[]
    for line in (old/"observations.jsonl").read_text().splitlines():
        obs=json.loads(line)
        if "/train/step_" not in obs["phase"]:continue
        record=records[obs["sample_id"]]
        with Image.open(ROOT/record["image_path"]) as image:box=content_box(image.convert("RGB"))[0]
        with Image.open(old/obs["source_visual_path"]) as source,Image.open(old/obs["image_path"]) as edited:
            terms=mean_lighting_terms(source.convert("RGB"),edited.convert("RGB"),box)
        rows.append({"sample_id":obs["sample_id"],"phase":obs["phase"],"v2_progress":obs["brightness_progress"],
                     "v3_progress":terms["brightness_progress"],"v3_counterfactual_reward":
                         2*(obs["CLIP_delta"]+terms["brightness_progress"])-obs["color_cost"]-obs["LPIPS"],**terms})
    controls=[]
    debug=json.loads((c/"diagnostics/debug_results.json").read_text())
    for obs in debug["calibration"]:
        record=records[obs["sample_id"]]
        with Image.open(ROOT/record["image_path"]) as image:box=content_box(image.convert("RGB"))[0]
        directory=c/"diagnostics/calibration_images"/obs["sample_id"]
        with Image.open(directory/"identity.png") as source,Image.open(directory/f"{obs['variant']}.png") as edited:
            terms=mean_lighting_terms(source.convert("RGB"),edited.convert("RGB"),box)
        controls.append({"sample_id":obs["sample_id"],"variant":obs["variant"],
            "common_reward":2*(obs["CLIP_delta"]+terms["brightness_progress"])-obs["color_cost"]-obs["LPIPS"],**terms})
    progress=[r["brightness_progress"] for r in rows]
    gate={"identity_zero":all(abs(r["common_reward"])<1e-6 for r in controls if r["variant"]=="identity"),
          "positive_control_above_identity":all(r["common_reward"]>0 for r in controls if r["variant"]=="exposure_plus3pct"),
          "negative_controls_below_identity":all(r["common_reward"]<0 for r in controls if r["variant"] in {"exposure_minus3pct","channel_swap"}),
          "real_output_progress_SD_exceeds_0_001":statistics.pstdev(progress)>.001,
          "no_hard_saturation":all(-1<x<1 for x in progress)}
    result={"scope":"CPU-only rescore of rejected-v2 development generations; NOT trained-v3 outcomes",
        "real_output_n":len(rows),"v2_hard_saturation_fraction":sum(r["v2_progress"]==-1 for r in rows)/len(rows),
        "v2_progress_SD":statistics.pstdev([r["v2_progress"] for r in rows]),
        "v3_progress_SD":statistics.pstdev(progress),"v3_progress_range":[min(progress),max(progress)],
        "gate":gate,"all_gates_pass":all(gate.values()),"controls":controls,"rows":rows}
    write_json(c/"lighting_v3_calibration.json",result,exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k not in {"rows","controls"}},indent=2))
    if not result["all_gates_pass"]:raise RuntimeError("Mean-property objective failed calibration; no new training admitted")


if __name__=="__main__":main()
