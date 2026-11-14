"""Read-only validation diagnostic: does the common proxy prefer the unedited source?"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected an existing mini-pilot output")
    import torch
    from PIL import Image
    from src.mini_pilot.reward import ProxyRewardCalculator
    from src.mini_pilot.data import write_json
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.monotonic()
    protocol = json.loads((output / "protocol.json").read_text())
    cohort = json.loads((output / "cohort.json").read_text())
    directory = output / "identity_validation_audit"
    directory.mkdir(exist_ok=False)
    reward = ProxyRewardCalculator(ROOT, device="cpu")
    observations = []
    for record in cohort["validation"]:
        # Validation only, with no mini-test reward access before checkpoint lock.
        path = output / "visuals/sources" / f"{record['sample_id']}.png"
        with Image.open(path) as image:
            original = image.convert("RGB")
        metric = reward.measure(original, original, protocol["target_prompt"])
        if abs(metric["LPIPS"]) > 1e-7 or abs(metric["SSIM"] - 1.) > 1e-7:
            raise ValueError("The real metric's identity witness failed")
        observations.append({"sample_id": record["sample_id"], **metric})
    write_json(directory / "results.json", {"scope": "READ-ONLY VALIDATION DIAGNOSTIC; not a sixth trained arm",
                                             "observations": observations, "mean_common_reward": sum(r["common_reward"] for r in observations)/len(observations),
                                             "GPU_seconds": 0, "CPU_wall_seconds": time.monotonic() - started,
                                             "reward_resources": reward.receipt, "Slurm_job_ID": os.getenv("SLURM_JOB_ID"),
                                             "scientific_training_settings_changed": False}, exclusive=True)
    (directory / "README.md").write_text(
        "# Identity validation audit\n\nReal CLIP and calibrated LPIPS were run on the 16 validation sources "
        "against themselves, on CPU. This diagnostic checks whether the stated proxy rewards no editing. "
        "It is not a sixth trained arm, ASR, or a measurement of brightness-instruction correctness. "
        "No mini-test rewards, training settings or controller feedback were changed.\n", encoding="utf-8")
    print(json.dumps({"identity_validation_common_reward": sum(r["common_reward"] for r in observations)/len(observations),
                      "CPU_wall_seconds": time.monotonic() - started, "GPU_seconds": 0}), flush=True)


if __name__ == "__main__":
    main()
