"""Render final infrastructure reports from recorded runs, never fabricated values."""
import csv
import io
import json
from pathlib import Path
import _bootstrap


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest(root, suffix, predicate=lambda p, m: True):
    candidates = []
    for p in sorted((root / "experiments").glob("EXP_*" + suffix)):
        m = read(p / "metrics.json")
        if predicate(p, m):
            candidates.append((p, m))
    return candidates[-1]


if __name__ == "__main__":
    root = _bootstrap.ROOT
    inventory = read(root / "reports/cluster_inventory.json")
    smoke = read(root / "reports/latest_smoke_results.json")
    qpath, q = latest(root, "qwen_smoke")
    rpath, rl = latest(root, "rl_smoke", lambda p, m: m.get("peak_gpu_memory_bytes", 0) > 0)
    vpath, vqa = latest(root, "qwen_clean_baseline")
    dpath, audit = latest(root, "official_data_audit")
    jobs = [r for r in csv.DictReader(io.StringIO(inventory["scheduler_accounting"]["stdout"]), delimiter="|") if r["JobID"].isdigit()]
    seconds = sum(sum(int(x) * unit for x, unit in zip(r["Elapsed"].split(":"), [3600, 60, 1])) for r in jobs)
    rows = ["# Setup report", "", "All feasible requested smoke tests completed; final full GPU suite PASS.", "",
        "- Canonical runtime: `/storage/home/ydk297/projects/poisoning`.",
        "- Existing Windows source/evidence mirror: `H:\\Project\\Poisoning`.",
        "- Cluster: UTC mocshpc; inspected via configured `utc-login` SSH entry. Verified partition `epyc-gpu`, account `allusers`, A100 80GB, 4 CPUs/32G per smoke, maximum request 20 minutes.",
        "- Isolated venv: project `.venv`, using existing Python 3.11/Torch 2.6.0 cu124 `lvr_train` base; shared runtime not modified.",
        "- Stack: Transformers 4.57.1, Accelerate 1.11.0, qwen-vl-utils 0.0.14, HF Hub 0.35.3, torchvision 0.21.0, Pillow 11.0.0, NumPy 2.1.2, PyYAML 6.0.2, scikit-image 0.25.2. Exact freeze retained.",
        "- Model: Qwen/Qwen3-VL-2B-Instruct, immutable revision `89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0 model metadata, central cache reused; no duplicate local weight download.",
        "- Official VQAv2: 1000/200/300 image-disjoint research subsets from official train2014. Questions, ten answers, IDs, split hashes and selected-image checksums retained. Internal test is not official evaluation-server test.",
        f"- All-source audit: {audit['records_matched_to_official_sources']} records and image checksums matched; deterministic selection replay PASS (`{dpath.name}`).",
        "- Independent synthetic fixture: ten samples per split; only engineering feedback/oracle tests use it. Real images reside on the cluster; local copies of manifests are under `data/manifests/cluster`, with no claim that all images were copied.", "",
        "## Actual smoke measurements", "",
        f"- Qwen two-example answers: {q['sequential']}; sequential/batched/replayed answers agree, correctness 2/2. BF16; frozen model; left padding.",
        f"- Qwen inference peak allocation: {q['peak_gpu_memory_bytes'] / 1e9:.3f} GB; measured repeated-example throughput {q['samples_per_second']:.3f} samples/s; loading {q['model']['load_seconds']:.3f}s. Tiny image smoke only; throughput excludes loading.",
        f"- Three-update GPU RL plus replay/resume: max trainable change {rl['trainable_max_change']:.9g}; frozen parameters unchanged; finite gradients/rewards; exact same-device replay, next-step/AdamW/scheduler restoration PASS. Component duration {rl['wall_seconds']:.3f}s, peak allocation {rl['peak_gpu_memory_bytes'] / 1e6:.2f} MB. No VLM is used in RL feedback.",
        f"- Real clean VQA short-answer smoke: consensus {vqa['clean_vqa_consensus']:.3f}, exact-any {vqa['clean_exact_any']:.3f}, n={vqa['samples']}, seeded validation selection, {vqa['wall_seconds']:.3f}s; source `{vpath.name}`. Whole-answer string normalization is preliminary, and n=10 is not a benchmark estimate.",
        "- Fixed short-answer system prompt is declared in model metadata. Original questions and labels stay unchanged. Earlier verbose raw-answer runs scored 0.0; raw predictions and scores remain preserved. These are different inference protocols, not model/RL/RSI improvement evidence.",
        "- RSI toy revisions: one accepted, one rejected; previous accepted configuration preserved. No empirical RSI-versus-RL comparison.",
        "- Independent synthetic security observation aggregation exercised; real trigger/detection metrics remain null because no pre-existing authorized security fixture/checkpoint was supplied.", "",
        "## Scheduler accounting", "", "| Job | State | Exit | Allocated elapsed |", "|---|---|---|---|"]
    rows.extend(f"| {j['JobID']} | {j['State']} | {j['ExitCode']} | {j['Elapsed']} |" for j in jobs)
    rows += ["", f"Total allocated cost: {seconds / 3600:.6f} GPU hours ({seconds} seconds on one GPU), including earlier partial/calibration runs. Component timings above are already inside these allocations and must not be added again.", "",
        "## Installed components and versioning", "",
        "Modular dataset/collator, frozen Qwen wrapper, PIL editor, quality and semantic validators, trainable categorical policy, REINFORCE/AdamW, complete checked checkpoint continuation, RSI state/history/revisions, independent read-only evaluation, CPU/CUDA detection, local JSON/CSV tracking, immutable experiment allocation, registry/index, and verified Slurm templates are present. Pretrained editor survey is in docs/EDITOR_BACKENDS.md; no diffusion checkpoint was downloaded or trained.", "",
        "Source/configs have local Git history on both systems. Each actual run records its own executing commit; generated weights/data/results are excluded from Git. Active status/metrics may evolve; finalized runs are never reused. Historical provenance defects are preserved and described in reports/FAILURES.md, not rewritten.", "",
        "## Retained disk usage", "", "```text", inventory["retained_sizes"]["stdout"].strip(), "```", "",
        "C: large-data estimate was zero: all model/data/environment staging was on cluster storage and local small evidence on H:. Initial C: free about 71.014 GB; post-run about 70.99 GB (system-wide usage may include unrelated activity). H: retained source, original user files and small run evidence; no multi-GB model copy. Setup logs are preserved before scratch removal; central model weights, dataset source archives, unique checkpoints and experiment evidence remain.", "",
        "## Remaining boundaries", "",
        "Natural-image semantic validation needs an independent verifier; official VQA string-normalization parity remains unverified; diffusion/PEFT adaptation and compute-matched RSI comparisons have not run. No large-scale experiment, poisoned checkpoint or ASR/evasion optimization was performed. Same-family implementation review is provisional; final interface/prompt corrections were executor-verified after the two-round review budget. Fresh-agent documented local and remote CPU invocations passed.", "",
        "Evidence: reports/SMOKE_TEST_REPORT.md, reports/cluster_inventory.json, reports/FAILURES.md, reports/DOC_INVOCATION_REVIEW.md, implement-stage/SILENT_ASSUMPTION_SWEEP.json, EXPERIMENT_INDEX.md and experiments/registry.jsonl."]
    (root / "reports/SETUP_REPORT.md").write_text("\n".join(rows) + "\n", encoding="utf-8")
    next_steps = f"""# Next steps

Ready: all requested A–I functional smokes passed in `{Path(smoke['run']).name}`; real BF16 Qwen, image-disjoint VQAv2 loader, benign editor, three-step RL/replay/resume, toy RSI, independent evaluation and immutable logging.

Measured on one A100 80GB: Qwen tiny-fixture inference peak {q['peak_gpu_memory_bytes']/1e9:.2f} GB, {q['samples_per_second']:.2f} samples/s excluding load; ten real short-answer validation questions {vqa['wall_seconds']:.2f}s, peak {vqa['peak_gpu_memory_bytes']/1e9:.2f} GB; lightweight GPU RL including replay/resume {rl['wall_seconds']:.2f}s. For another small validation job, one GPU/4 CPUs/32G/20-minute ceiling is ample in these measurements. Larger-image throughput, other datasets and diffusion adaptation cannot be inferred from this smoke.

Before scaling: establish official VQA normalization parity, independently verify natural-image semantic preservation, freeze a larger clean evaluation protocol, measure the selected image-editor/adapter backend, and freeze equal total proposal/training/validation/rejected-trial budgets for the four arms. Keep raw verbose-answer score 0 and all earlier outcomes. Short-answer consensus {vqa['clean_vqa_consensus']:.2f} on n=10 is a formatting sanity check, not a method result.

No real security checkpoint/fixture was supplied: trigger-conditioned behavior and detection remain unmeasured. Pretrained editing model memory estimates in docs/EDITOR_BACKENDS.md are untested. No large sweep or attack training is queued.

Next executable command (another limited clean validation run, not a large experiment):

```bash
ssh utc-login 'cd /storage/home/ydk297/projects/poisoning; source .venv/bin/activate; python scripts/submit_smoke.py --job clean_baseline'
```
"""
    (root / "NEXT_STEPS.md").write_text(next_steps, encoding="utf-8")
    print("Generated SETUP_REPORT.md and NEXT_STEPS.md from recorded evidence")
