import json
import subprocess
from pathlib import Path
import torch
import _bootstrap
from src.utils.logging import write_json


def capture(command):
    p = subprocess.run(command, capture_output=True, text=True, timeout=60)
    return {"command": command, "exit_code": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def state_equal(left, right):
    if torch.is_tensor(left):
        return torch.is_tensor(right) and torch.equal(left, right)
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(state_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(state_equal(a, b) for a, b in zip(left, right))
    return left == right


if __name__ == "__main__":
    root = _bootstrap.ROOT
    jobs = [json.loads(s) for s in (root / "reports/phase2_jobs.jsonl").read_text().splitlines()]
    evidence = {"jobs": jobs, "scheduler": capture(["sacct", "-X", "-j", ",".join(j["job_id"] for j in jobs),
        "-P", "--format=JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocTRES"]),
        "disk": capture(["du", "-sh", str(root / ".venv-editor"), str(root / "experiments"),
            "/storage/home/ydk297/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix"]),
        "quota": capture(["/usr/lpp/mmfs/bin/mmlsquota", "-u", "ydk297", "--block-size", "auto"]),
        "queue": capture(["squeue", "-u", "ydk297"]), "runs": []}
    for env, label in [(".venv", "qwen"), (".venv-editor", "editor")]:
        frozen = capture([str(root / env / "bin/python"), "-m", "pip", "freeze"])
        (root / f"reports/phase2_freeze_{label}.txt").write_text(frozen["stdout"])
        evidence[label + "_python"] = str(root / env / "bin/python")
    for path in sorted((root / "experiments").glob("EXP_*phase2_*")):
        manifest = json.loads((path / "manifest.json").read_text())
        status = json.loads((path / "status.json").read_text())
        submission = json.loads((path / "job_submission.json").read_text()) if (path / "job_submission.json").exists() else {}
        evidence["runs"].append({"id": path.name, "path": str(path), "status": status["status"],
            "job_id": manifest.get("slurm_job_id") or submission.get("job_id"), "git_commit": manifest["git_commit"],
            "required_files_present": all((path / name).exists() for name in ["config.yaml", "manifest.json", "metrics.json",
                "metrics.csv", "train.log", "run.log", "environment.txt", "git_commit.txt", "status.json", "summary.md", "images", "checkpoints"])})
    registry = [json.loads(s) for s in (root / "experiments/registry.jsonl").read_text().splitlines()]
    ids = [r["experiment_id"] for r in registry]
    evidence["registry_unique"] = len(ids) == len(set(ids))
    evidence["phase2_all_registered"] = all(r["id"] in ids for r in evidence["runs"])
    evidence["all_phase2_required_files_present"] = all(r["required_files_present"] for r in evidence["runs"])
    pointer = json.loads((root / "reports/phase2_latest_controller.json").read_text())
    controller_run = root / "experiments" / pointer["id"]
    metrics = json.loads((controller_run / "metrics.json").read_text())
    revision = metrics["revision"]
    selected_name = "candidate" if revision["accepted"] else "base"
    checkpoints = {name: torch.load(controller_run / f"checkpoints/{name}_controller.pt", map_location="cpu", weights_only=True)
                   for name in ["base", "candidate", "selected"]}
    state = json.loads((controller_run / "rsi/state.json").read_text())
    evidence["checkpoint_selection_audit"] = {
        "experiment_id": pointer["id"], "loaded_from_disk": True, "selected": selected_name,
        "selected_policy_and_optimizer_equal": state_equal(checkpoints["selected"], checkpoints[selected_name]),
        "candidate_policy_differs_from_base": not state_equal(checkpoints["base"]["policy"], checkpoints["candidate"]["policy"]),
        "configuration_matches_selection": state["training_config"] == revision["proposed_config" if revision["accepted"] else "previous_config"],
        "rejected_trial_cost_retained": state["cost_gpu_hours"] >= revision["cost_gpu_hours"],
    }
    assert all(evidence["checkpoint_selection_audit"][k] for k in ["selected_policy_and_optimizer_equal",
        "candidate_policy_differs_from_base", "configuration_matches_selection", "rejected_trial_cost_retained"])
    feedback_run = root / "experiments" / metrics["shared_precomputation_run"]
    feedback = json.loads((feedback_run / "feedback.json").read_text())
    validation_feedback = [r for r in feedback if r["split"] == "validation"]
    assert len(validation_feedback) == 30
    evidence["feedback_table_audit"] = {
        "experiment_id": feedback_run.name, "records": len(feedback),
        "random_uniform_expected_validation_reward": sum(r["reward"]["reward"] for r in validation_feedback) / 30,
        "automatic_provisional_agreements": sum(bool(r["semantic"].get("provisionally_valid")) for r in feedback),
        "independently_verified_changed_edits": sum(bool(r["semantic"]["verified"]) for r in feedback),
        "note": "Derived read-only from saved feedback; original single-draw random baseline and immutable run metrics retained",
    }
    evidence["git_status"] = capture(["git", "status", "--short"])
    evidence["report_source_commit"] = capture(["git", "rev-parse", "HEAD"])["stdout"].strip()
    write_json(root / "reports/phase2_evidence_inventory.json", evidence)
    print(json.dumps({"phase2_runs": len(evidence["runs"]), "registry_unique": evidence["registry_unique"],
                      "all_required_files": evidence["all_phase2_required_files_present"]}))
