import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import _bootstrap
from _runs import new_run
from src.utils.logging import Experiment, write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--include-qwen", action="store_true")
    args = p.parse_args()
    active = os.environ.get("POISONING_JOB_RUN")
    run = Experiment.attach_active(_bootstrap.ROOT, active) if active else new_run("smoke_suite", vars(args))
    results = []
    started = time.perf_counter()
    with run:
        run.set_status("RUNNING", slurm_job_id=os.environ.get("SLURM_JOB_ID"))
        tests = [
            ("A", "environment", ["scripts/check_environment.py", "--output", str(run.path / "environment_probe.json")], None),
            ("C", "dataset", ["scripts/smoke_dataset.py"], None),
            ("D", "editor", ["scripts/smoke_editor.py"], None),
            ("E", "rl", ["scripts/smoke_rl.py", "--device", args.device], None),
            ("F", "rsi", ["scripts/smoke_rsi.py"], None),
            ("G", "evaluation_units", ["-m", "unittest", "tests.test_evaluation_reward", "-v"], None),
            ("G_SECURITY", "security_fixture", ["scripts/run_evaluation.py", "--fixture", "tests/fixtures/security_synthetic.json",
                "--fixture-id", "synthetic-observations-v1", "--synthetic"], None),
            ("G_ORACLE", "clean_fixture", ["scripts/run_clean_baseline.py", "--oracle-fixture"], None),
            ("H", "logging", ["scripts/smoke_logging.py"], None),
            ("UNIT", "all_units", ["-m", "unittest", "discover", "-v"], None),
        ]
        if args.include_qwen:
            ready = (_bootstrap.ROOT / "reports/model_ready.json").exists()
            official = (_bootstrap.ROOT / "data/processed/vqav2_seed42_v1/provenance.json").exists()
            tests += [("B", "qwen", ["scripts/smoke_qwen.py", "--device", args.device], None if ready else "Model cache not ready"),
                      ("G_REAL", "clean_vqav2", ["scripts/run_clean_baseline.py", "--device", args.device, "--limit", "10"],
                       None if ready and official else "Model or official dataset not ready")]
        else:
            tests += [("B", "qwen", [], "Qwen not requested in this CPU suite")]
        for code, name, command, blocker in tests:
            if code == "G_REAL":
                official_now = (_bootstrap.ROOT / "data/processed/vqav2_seed42_v1/provenance.json").exists()
                ready_now = (_bootstrap.ROOT / "reports/model_ready.json").exists()
                blocker = None if official_now and ready_now else "Model or official dataset not ready"
            log = run.path / f"test_{code}_{name}.log"
            cmd = [sys.executable, *command]
            if blocker:
                result = {"test": code, "name": name, "status": "BLOCKED", "command": shlex.join(cmd), "reason": blocker}
                log.write_text(blocker, encoding="utf-8")
            else:
                print(f"START {code}: {shlex.join(cmd)}", flush=True)
                t = time.perf_counter()
                try:
                    proc = subprocess.run(cmd, cwd=_bootstrap.ROOT, capture_output=True, text=True, timeout=720, errors="replace")
                    log.write_text(proc.stdout + "\nSTDERR:\n" + proc.stderr, encoding="utf-8")
                    result = {"test": code, "name": name, "status": "PASS" if proc.returncode == 0 else "FAIL",
                              "command": shlex.join(cmd), "exit_code": proc.returncode, "wall_seconds": time.perf_counter() - t}
                except subprocess.TimeoutExpired as exc:
                    log.write_text(str(exc), encoding="utf-8")
                    result = {"test": code, "name": name, "status": "FAIL", "command": shlex.join(cmd), "reason": "720s timeout"}
            result["output_path"] = str(log)
            results.append(result)
            run.event("test", result)
            print(f"END {code}: {result['status']} ({log})", flush=True)
        job = os.environ.get("SLURM_JOB_ID")
        results.append({"test": "I", "name": "slurm", "status": "PASS" if job else "BLOCKED",
                        "command": "python scripts/submit_smoke.py", "job_id": job,
                        "output_path": str(run.path), "reason": "Executed inside allocated GPU job" if job else "Not inside Slurm"})
        run.metrics({"tests": results, "wall_seconds": time.perf_counter() - started})
        write_json(_bootstrap.ROOT / "reports/latest_smoke_results.json", {"run": str(run.path), "tests": results})
        rows = ["# Smoke test report", "", f"Suite: `{run.path}`", "",
                "| Test | Status | Actual command | Evidence |", "|---|---|---|---|"]
        for r in results:
            rows.append(f"| {r['test']} {r['name']} | {r['status']} | `{r['command']}` | `{r['output_path']}` |")
        rows += ["", "Synthetic fixtures validate engineering only. Real VQAv2 and model inference are separately named.",
                 "All earlier failures and fixes are retained in reports/FAILURES.md and immutable failed runs."]
        (_bootstrap.ROOT / "reports/SMOKE_TEST_REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
        status = "FAIL" if any(r["status"] == "FAIL" for r in results) else ("PARTIAL" if any(r["status"] == "BLOCKED" for r in results) else "PASS")
        run.finish(status, f"Smoke suite {status}. No large experiments or security optimization.")
    if any(r["status"] == "FAIL" for r in results):
        return 1
    # A requested full GPU acceptance run must not pass on skipped dependencies.
    return 2 if args.include_qwen and any(r["status"] == "BLOCKED" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
