# Setup report

All feasible requested smoke tests completed; final full GPU suite PASS.

- Canonical runtime: `/storage/home/ydk297/projects/poisoning`.
- Existing Windows source/evidence mirror: `H:\Project\Poisoning`.
- Cluster: UTC mocshpc; inspected via configured `utc-login` SSH entry. Verified partition `epyc-gpu`, account `allusers`, A100 80GB, 4 CPUs/32G per smoke, maximum request 20 minutes.
- Isolated venv: project `.venv`, using existing Python 3.11/Torch 2.6.0 cu124 `lvr_train` base; shared runtime not modified.
- Stack: Transformers 4.57.1, Accelerate 1.11.0, qwen-vl-utils 0.0.14, HF Hub 0.35.3, torchvision 0.21.0, Pillow 11.0.0, NumPy 2.1.2, PyYAML 6.0.2, scikit-image 0.25.2. Exact freeze retained.
- Model: Qwen/Qwen3-VL-2B-Instruct, immutable revision `89644892e4d85e24eaac8bacfd4f463576704203`, Apache-2.0 model metadata, central cache reused; no duplicate local weight download.
- Official VQAv2: 1000/200/300 image-disjoint research subsets from official train2014. Questions, ten answers, IDs, split hashes and selected-image checksums retained. Internal test is not official evaluation-server test.
- All-source audit: 1500 records and image checksums matched; deterministic selection replay PASS (`EXP_20261027T034426_acc87f5a9ee5_official_data_audit`).
- Independent synthetic fixture: ten samples per split; only engineering feedback/oracle tests use it. Real images reside on the cluster; local copies of manifests are under `data/manifests/cluster`, with no claim that all images were copied.

## Actual smoke measurements

- Qwen two-example answers: ['1', '2']; sequential/batched/replayed answers agree, correctness 2/2. BF16; frozen model; left padding.
- Qwen inference peak allocation: 4.317 GB; measured repeated-example throughput 4.779 samples/s; loading 3.887s. Tiny image smoke only; throughput excludes loading.
- Three-update GPU RL plus replay/resume: max trainable change 2.96682119e-05; frozen parameters unchanged; finite gradients/rewards; exact same-device replay, next-step/AdamW/scheduler restoration PASS. Component duration 1.892s, peak allocation 68.18 MB. No VLM is used in RL feedback.
- Real clean VQA short-answer smoke: consensus 0.680, exact-any 0.900, n=10, seeded validation selection, 2.744s; source `EXP_20261027T035147_f743daabf286_qwen_clean_baseline`. Whole-answer string normalization is preliminary, and n=10 is not a benchmark estimate.
- Fixed short-answer system prompt is declared in model metadata. Original questions and labels stay unchanged. Earlier verbose raw-answer runs scored 0.0; raw predictions and scores remain preserved. These are different inference protocols, not model/RL/RSI improvement evidence.
- RSI toy revisions: one accepted, one rejected; previous accepted configuration preserved. No empirical RSI-versus-RL comparison.
- Independent synthetic security observation aggregation exercised; real trigger/detection metrics remain null because no pre-existing authorized security fixture/checkpoint was supplied.

## Scheduler accounting

| Job | State | Exit | Allocated elapsed |
|---|---|---|---|
| 86254 | COMPLETED | 0:0 | 00:00:59 |
| 86256 | COMPLETED | 0:0 | 00:01:30 |
| 86259 | COMPLETED | 0:0 | 00:01:57 |
| 86260 | COMPLETED | 0:0 | 00:00:59 |

Total allocated cost: 0.090278 GPU hours (325 seconds on one GPU), including earlier partial/calibration runs. Component timings above are already inside these allocations and must not be added again.

## Installed components and versioning

Modular dataset/collator, frozen Qwen wrapper, PIL editor, quality and semantic validators, trainable categorical policy, REINFORCE/AdamW, complete checked checkpoint continuation, RSI state/history/revisions, independent read-only evaluation, CPU/CUDA detection, local JSON/CSV tracking, immutable experiment allocation, registry/index, and verified Slurm templates are present. Pretrained editor survey is in docs/EDITOR_BACKENDS.md; no diffusion checkpoint was downloaded or trained.

Source/configs have local Git history on both systems. Each actual run records its own executing commit; generated weights/data/results are excluded from Git. Active status/metrics may evolve; finalized runs are never reused. Historical provenance defects are preserved and described in reports/FAILURES.md, not rewritten.

## Retained disk usage

```text
987M	/storage/home/ydk297/projects/poisoning/.venv
339M	/storage/home/ydk297/projects/poisoning/data
14M	/storage/home/ydk297/projects/poisoning/experiments
4.0G	/storage/home/ydk297/.cache/huggingface/hub/models--Qwen--Qwen3-VL-2B-Instruct
```

C: large-data estimate was zero: all model/data/environment staging was on cluster storage and local small evidence on H:. Initial C: free about 71.014 GB; post-run about 70.99 GB (system-wide usage may include unrelated activity). H: retained source, original user files and small run evidence; no multi-GB model copy. Setup logs are preserved before scratch removal; central model weights, dataset source archives, unique checkpoints and experiment evidence remain.

## Remaining boundaries

Natural-image semantic validation needs an independent verifier; official VQA string-normalization parity remains unverified; diffusion/PEFT adaptation and compute-matched RSI comparisons have not run. No large-scale experiment, poisoned checkpoint or ASR/evasion optimization was performed. Same-family implementation review is provisional; final interface/prompt corrections were executor-verified after the two-round review budget. Fresh-agent documented local and remote CPU invocations passed.

Evidence: reports/SMOKE_TEST_REPORT.md, reports/cluster_inventory.json, reports/FAILURES.md, reports/DOC_INVOCATION_REVIEW.md, implement-stage/SILENT_ASSUMPTION_SWEEP.json, EXPERIMENT_INDEX.md and experiments/registry.jsonl.
