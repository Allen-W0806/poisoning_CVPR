import time
import torch


class CostTracker:
    def __init__(self, device="cpu"):
        self.cuda = torch.device(device).type == "cuda"
        self.model_calls = 0
        self.evaluation_calls = 0
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self.started = time.perf_counter()

    def snapshot(self, samples=0):
        if self.cuda:
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self.started
        return {"wall_seconds": elapsed, "gpu_hours": elapsed / 3600 if self.cuda else 0.0,
                "gpu_hours_scope": "component wall time on one GPU; not scheduler allocation",
                "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated() if self.cuda else 0,
                "samples_per_second": samples / elapsed if elapsed else 0,
                "model_calls": self.model_calls, "evaluation_calls": self.evaluation_calls}
