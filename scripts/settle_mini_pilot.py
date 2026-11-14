"""Settle only this run's actual Slurm spend and refresh its reproducibility inventory."""
import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import sha256, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected this project's output")
    state_path = output / "run_state.json"
    completion_record = datetime.fromtimestamp(state_path.stat().st_mtime, timezone.utc).isoformat()
    state = json.loads(state_path.read_text())
    if state["status"] not in {"COMPLETED", "FAILED"}:
        raise RuntimeError("Cannot settle an active experiment")
    ids = [json.loads((ROOT / "outputs" / f"{output.name}_submission/submission.json").read_text())["job_id"]]
    for folder in ["continuation_submission", "identity_validation_submission", "final_audit_submission"]:
        path = output / folder / "submission.json"
        if path.exists():
            receipt = json.loads(path.read_text())
            ids.append(receipt.get("job_id") or receipt["stdout"].split(";")[0])
    process = subprocess.run(["sacct", "-X", "-nP", "-j", ",".join(ids),
                              "--format=JobIDRaw,State,ElapsedRaw,AllocTRES,ExitCode,NodeList"],
                             check=True, text=True, capture_output=True)
    records = {}
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL"}
    for line in process.stdout.splitlines():
        cells = line.split("|")
        if cells[0] not in ids:
            continue
        status = cells[1].split()[0].rstrip("+")
        if status not in terminal:
            raise RuntimeError(f"Job is not terminal: {cells[0]} {status}")
        match = re.search(r"(?:^|,)gres/gpu=(\d+)(?:,|$)", cells[3])
        count = int(match[1]) if match else 0
        if count > 1:
            raise ValueError("Unexpected multi-GPU allocation")
        records[cells[0]] = {"job_id": cells[0], "state": status, "elapsed_seconds": int(cells[2]),
                             "allocated_TRES": cells[3], "exit_code": cells[4], "nodes": cells[5],
                             "GPU_count": count, "allocated_GPU_seconds": int(cells[2]) * count}
    if set(records) != set(ids):
        raise RuntimeError("Missing scheduler accounting; do not invent spend")
    total = sum(record["allocated_GPU_seconds"] for record in records.values())
    if total > 5400:
        raise RuntimeError("Actual cumulative GPU allocation exceeded the authorized cap")
    protocol = json.loads((output / "protocol.json").read_text())
    protection = {name: sha256(ROOT / name) == expected for name, expected in protocol["source_SHA256"].items()}
    patch_path = output / "reporting_repair/post_run_patch.json"
    patches = json.loads(patch_path.read_text())["files"] if patch_path.exists() else {}
    for name, same in protection.items():
        if not same and not (name in patches and patches[name]["before_SHA256"] == protocol["source_SHA256"][name]
                             and patches[name]["after_SHA256"] == sha256(ROOT / name)
                             and name == "src/mini_pilot/reporting.py"):
            raise RuntimeError(f"An unrecorded executed source changed: {name}")
    state["settled_UTC"] = datetime.now(timezone.utc).isoformat()
    if state.get("continuation"):
        state.setdefault("initial_process_finished_UTC", state.get("finished_UTC"))
        state.setdefault("comparison_finished_UTC", completion_record)
        state["finished_UTC"] = state["comparison_finished_UTC"]
        state.setdefault("initial_process_full_UNet_calls", state.get("actual_forward_UNet_calls", 0))
        if "full_UNet_calls" in state["continuation"]:
            state["actual_forward_UNet_calls"] = state["initial_process_full_UNet_calls"] + state["continuation"]["full_UNet_calls"]
        else:
            state["actual_forward_UNet_calls"] = None
            state["continuation_full_UNet_calls_status"] = "NOT RECORDED"
    state["original_source_hash_checks"] = protection
    state["post_run_reporting_patch"] = str(patch_path.relative_to(output)) if patches else None
    write_json(state_path, state)
    training = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    counts = {method: sum(row.get("scope") == "arm_training" and row.get("arm") == method for row in training)
              for method in protocol["methods"]}
    ledger = {"scope": output.name, "user_authorized_cumulative_cap_GPU_seconds": 5400,
              "actual_allocated_GPU_seconds": total, "actual_allocated_GPU_hours": total / 3600.,
              "remaining_unspent_cap_seconds": 5400 - total, "jobs": list(records.values()),
              "actual_arm_training_updates": counts, "actual_generator_optimizer_calls_including_witnesses": len(training),
              "training_sources_unchanged": all(same for name, same in protection.items() if name != "src/mini_pilot/reporting.py"),
              "post_run_reporting_dependency_guard": bool(patches),
              "unrelated_project_jobs_included": False, "historical_budgets_reopened": False,
              "cluster_disk_free_bytes_after": shutil.disk_usage(ROOT).free}
    write_json(output / "RESOURCE_ACCOUNTING.json", ledger)
    initial_submission = ROOT / "outputs" / f"{output.name}_submission"
    (output / "initial_submission").mkdir(exist_ok=True)
    for source in initial_submission.iterdir():
        if source.is_file():
            shutil.copy2(source, output / "initial_submission" / source.name)
    code = [ROOT / "train_and_evaluate_pilot.py", ROOT / "run_mini_pilot.sh", ROOT / "docs/MINI_PILOT.md",
            *sorted((ROOT / "src/mini_pilot").glob("*.py")),
            *[ROOT / "scripts" / name for name in ["submit_mini_pilot.py", "submit_mini_pilot_continuation.py",
                                                   "continue_mini_pilot.py", "report_mini_pilot_cn.py", "settle_mini_pilot.py",
                                                   "audit_mini_pilot_identity.py", "audit_mini_pilot_final.py"]],
            *sorted((ROOT / "tests").glob("test_mini_pilot*.py"))]
    for source in code:
        target = output / "reproduction_code" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    lines = ["# Resource settlement", "", f"Actual allocated GPU time: **{total}/5400 seconds ({total/3600.:.6f} GPU hours)**.", "",
             "Failed/time-capped allocations and numerical-witness updates are included. CPU-only audits consume zero GPU seconds. "
             "The original historical budgets and unrelated projects are not charged or modified.", "",
             "| Job | State | Wall seconds | GPUs | GPU seconds | Exit |", "|---|---|---:|---:|---:|---|"]
    for record in records.values():
        lines.append(f"| {record['job_id']} | {record['state']} | {record['elapsed_seconds']} | {record['GPU_count']} | {record['allocated_GPU_seconds']} | {record['exit_code']} |")
    lines += ["", f"Actual arm training updates: `{json.dumps(counts)}`. "
              f"Physical generator optimizer calls including separate numerical witnesses: {len(training)}.", "",
              "Large checkpoint/gradient/image files are retained as experiment evidence on cluster storage. "
              "The artifact inventory records exact bytes and SHA256. Local delivery contains reports, figures, logs and source, "
              "while model weights and full checkpoint evidence remain in the canonical cluster directory.", ""]
    (output / "reports/RESOURCE_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    files = [path for path in output.rglob("*") if path.is_file() and path.name != "artifact_inventory.json"]
    retained = sum(path.stat().st_size for path in files)
    write_json(output / "artifact_inventory.json", {"files": [{"path": str(path.relative_to(output)), "bytes": path.stat().st_size,
                                                              "SHA256": sha256(path)} for path in files],
                                                    "retained_bytes_excluding_inventory": retained})
    print(json.dumps({"actual_GPU_seconds": total, "retained_bytes": retained, "jobs": list(records.values())}, indent=2))


if __name__ == "__main__":
    main()
