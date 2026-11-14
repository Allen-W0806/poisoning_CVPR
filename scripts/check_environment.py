"""Read-only environment discovery and optional seeded CUDA kernel witness."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def inspect_environment():
    result = {"hostname": platform.node(), "platform": platform.platform(),
              "python": sys.version, "executable": sys.executable, "cwd": str(Path.cwd()),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "packages": {}}
    for package in ["torch", "torchvision", "transformers", "accelerate", "qwen-vl-utils",
                    "Pillow", "numpy", "PyYAML", "huggingface-hub", "scikit-image"]:
        try:
            result["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result["packages"][package] = None
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    result["torch_cuda"] = torch.version.cuda
    result["gpus"] = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            result["gpus"].append({"index": i, "name": p.name, "memory_bytes": p.total_memory})
        torch.manual_seed(42)
        x = torch.randn(8, 8, device="cuda")
        y = x @ x.T
        torch.cuda.synchronize()
        assert torch.isfinite(y).all()
        result["kernel_witness"] = {"shape": list(y.shape), "sum": y.sum().item(),
                                      "bf16_supported": torch.cuda.is_bf16_supported()}
    from transformers import AutoModelForImageTextToText, AutoProcessor, Qwen3VLConfig
    result["qwen3_api_import"] = True
    for name in ["nvidia-smi", "sinfo", "squeue"]:
        if shutil.which(name):
            args = [name] if name == "nvidia-smi" else ([name, "-o", "%P %a %l %G"] if name == "sinfo" else [name, "-u", os.environ.get("USER", "")])
            p = subprocess.run(args, capture_output=True, text=True, timeout=20, errors="replace")
            result[name] = {"code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}
    usage = shutil.disk_usage(Path.cwd())
    result["disk"] = dict(zip(["total", "used", "free"], usage))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/environment.json")
    args = parser.parse_args()
    result = inspect_environment()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
