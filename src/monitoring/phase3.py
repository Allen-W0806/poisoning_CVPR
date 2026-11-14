"""CPU-only Phase3 witnesses. Never load models/unpickle tensors or alter science."""
from .common import read_json,digest,TERMINAL
from .diagnostics import issue

PROTECTED=["configs/phase3_generator_rl.yaml","src/rl/diffusion_policy.py","src/rl/diffusion_trajectory.py",
    "src/rl/diffusion_rl_trainer.py","src/rl/reward_normalization.py","src/rl/reward.py",
    "scripts/smoke_generator_rl.py","src/evaluation/semantic_review.py","src/evaluation/image_edit_consistency.py"]
PROTECTED += ["src/evaluation/semantic_inventory_schema.py"]

def phase3_checks(root,runs,state,snapshot,protect=True):
    findings=[]
    if protect:
        baseline=state.setdefault("phase3_protected_hashes_v1",{})
        for name in PROTECTED:
            path=root/name
            if not path.is_file():continue
            current=digest(path)
            if name not in baseline:baseline[name]=current
            elif baseline[name]!=current:findings.append(issue("phase3_scientific_source_changed",severity="error",path=name))
    witnesses=[];one_witnesses=[]
    for run in runs:
        if "_phase3_generator_rl_" not in run["id"]:continue
        m=run["metrics"]; path=root/"experiments"/run["id"]
        for u in m.get("updates",[]):
            errors=[]
            if not u.get("finite_gradients") or u.get("gradient_norm",0)<=0:errors.append("gradient")
            if not u.get("probabilities_finite") or u.get("behavior_logp_replay_max_difference",1)>1e-6:errors.append("likelihood")
            if not u.get("conditioning_verified"):errors.append("conditioning")
            if u.get("changes",{}).get("max_abs_change",0)<=0:errors.append("adapter")
            if errors:findings.append(issue("phase3_optimizer_invalid",run["id"],"error",step=u.get("step"),fields=errors))
            if u.get("reward_variance",0)<1e-10:findings.append(issue("phase3_low_reward_variance",run["id"],"info",step=u.get("step"),action="diagnose only; do not rescale reward"))
        if run["status"]=="PASS":
            missing=[str(p) for p in [path/f"checkpoints/step_{i:04d}/training_state.pt" for i in range(len(m.get("updates",[]))+1)] if not p.is_file()]
            required=["adapter_exact","optimizer_exact","scheduler_exact","rng_exact","next_step_loss_exact","rollout_seeds_exact","reward_components_exact"]
            valid=bool(m.get("updates")) and all(m.get("restart",{}).get(k) is True for k in required) and m.get("frozen_weights_unchanged") is True
            if missing or not valid:findings.append(issue("phase3_incomplete_restart_witness",run["id"],"error",missing=missing))
            elif len(m["updates"])==3 and all(u.get("logical_batch_size")==10 for u in m["updates"]):witnesses.append(run["id"])
            elif len(m["updates"])==1:one_witnesses.append(run["id"])
    ledger=root/"reports/phase3_jobs.jsonl"
    jobs=[__import__("json").loads(s) for s in ledger.read_text().splitlines()] if ledger.is_file() else []
    charged=0
    for job in jobs:
        row=snapshot.get("jobs",{}).get(str(job["job_id"]))
        charged+=int(row["ElapsedRaw"]) if row and row["State"] in TERMINAL else job["reserved_seconds"]
    if charged>7200:findings.append(issue("phase3_budget_exceeded",severity="error",charged_seconds=charged))
    return {"findings":findings,"full_smoke_witnesses":witnesses,"generator_rl_implementation_ready":bool(witnesses),
        "one_update_witnesses":one_witnesses,
        "charged_or_reserved_gpu_seconds":charged,"maximum_gpu_seconds":7200,
        "source_baseline_version":"phase3_protected_hashes_v1; append only, old Phase2 baseline unchanged",
        "checkpoint_scope":"Runtime execution witnesses and archive integrity, no idle tensor replay",
        "gpu_memory_trends":[{"id":r["id"],"peak_bytes_by_update":[u.get("peak_gpu_memory_bytes") for u in r["metrics"].get("updates",[])]} for r in runs if "_phase3_generator_rl_" in r["id"]]}
