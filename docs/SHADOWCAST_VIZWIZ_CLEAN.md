# Shadowcast full VizWiz clean utility

The official VizWiz site released full-test reference answers in April2026:
https://vizwiz.org/tasks-and-datasets/vqa/ . Local self-evaluation uses all8000
official test images and10 human references each. The image package is
3975272799bytes, staged and decoded/hashed on cluster storage, not C:.

Original LLaVA eval.zip is pinned by source question SHA256
25b71577dafc1e4049b1a273b8e58d0d8bca510510b512e4925730905c3eceac.
The original question/prompt text stays unchanged.1773 questions differ only
in capitalization from the2026 references; casefold comparison verified all8000,
and the initial over-strict exact-string staging failure is retained.

Canonical root `/storage/home/ydk297/projects/poisoning`.
CPU preflight, no CUDA or job submission:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-shadowcast-20261104/bin/python scripts/evaluate_shadowcast_vizwiz.py --campaign outputs/reward_debug_20261104T054605Z_s42 --stage preflight
```

GPU stages are named vizwiz_clean0 and vizwiz_poison50, one A100 each, maximum
120minutes each, submitted once through the72000GPU-second campaign accountant.
Require both final victim checkpoint hashes already locked. Use the original
pinned Shadowcast `llava.eval.model_vqa_loader.eval_model`, conv_mode vicuna_v1,
temperature0,num_beams1,max128newtokens,physicalbatch1,four data workers. The only
I/O change is line buffering of raw answers. No altered questions, answer
extraction, dropped cases, benchmark tuning or model selection by these scores.
The original builder infers architecture from the model-path basename. Its
multimodal LoRA branch requires both llava and lora in that name. The saved
poison_0/50_seed42_lora directories lack llava, so evaluation uses named symlink
aliases vizwiz/llava-v1.5-7b-poison0/50-lora pointing to the same locked files.
No checkpoint is changed or copied and no model/loader source is patched.
CPU routing test intercepts the actual builder before loading any weights:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-shadowcast-20261104/bin/python -m unittest tests.test_shadowcast_vizwiz_loader -v
```

Human reference answers are not present in model input JSONL and are accessed
for scoring only after all8000 responses have been generated.

Primary score executes the pinned official VizWiz API VQAEval accuracy path,
matching its documented10choose9 metric. The original11856-byte vqaEval.py has
SHA256 d26ef7ac2db06351444c3531f8d95fd9c454423fbcb9fd5dd984f12281146694.
The dedicated src/evaluation/vizwiz_official.py extracts the original class AST
and stops evaluate immediately after its setAccuracy call, excluding unrelated
caption metrics. All normalization, contraction mapping and accuracy statements
execute unchanged. Human references stay literal, as in the official JSON
loader. Predictions are normalized unconditionally, including unanimous human
answers. The historical VQAv2 wrapper has different semantics and is not changed.
The scorer correction and source audit occur before any model response exists;
the initial preregistration is archived with the dated correction. CPU check:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-shadowcast-20261104/bin/python -m unittest tests.test_vizwiz_official -v
```

Save every raw response and
per-question score. Report poison50 minus clean0 as a paired full-test utility
difference. Source bootstrap is conditional on these two fixed trained models,
not an independent-training-seed CI. GQA remains unmeasured.

This is the full8000 public test. The original paper Table3 does not identify
which historical EvalAI partition produced its numbers; the service offered
4000-case test-dev and8000-case standard/challenge partitions. Exact partition
matching is unverified. Do not claim equality to its published VizWiz number.
Both models use the original configured one-epoch recipe, whose actual update
counts can differ when50 poison samples are appended; this is not an isolated
equal-update causal comparison. Internal sealed evaluation pools remain closed.
