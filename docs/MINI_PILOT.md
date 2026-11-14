# Benign 32/16/16 mini-pilot

This is an isolated lighting-editing pilot using the cached InstructPix2Pix,
CLIP ViT-L/14 and calibrated AlexNet LPIPS. It creates no poisoned victim
checkpoint and measures no ASR or Delta Clean. Production training admission,
historical numerical findings, sealed test data and finalized budgets remain
unchanged. The current user request authorizes this separate bounded pilot.

From `/storage/home/ydk297/projects/poisoning`, use the existing `.venv-editor`.
No dependency installation or model/data download is needed. LPIPS is imported
from the already verified vendored directory; its calibration and trunk hashes
are checked. The recorded editor ledger is `.aris/compute/utc-editor.md`.

Figure export additionally requires Matplotlib and Pillow in the CPU reporting
runtime. The measured editor environment lacks Matplotlib. A post-run dependency
guard now preserves completed measurements and records pending figures instead
of turning a plotting-only import error into a failed learning result. For this
run the figures were exported from saved observations/images using the existing
Windows `H:\Anaconda\python.exe` Matplotlib runtime, without installing packages
or rerunning any model. The original executed source and plotting failure are
retained alongside the exact post-run reporting-only patch. All training,
controller, reward, transition and metric formulas remain as executed.

CPU checks (the directory is disposable scratch):

```bash
export PYTHONDONTWRITEBYTECODE=1
export PILOT_TEST_TMP="/storage/home/ydk297/projects/poisoning/tmp/mini_pilot_checks_$(date -u +%Y%m%dT%H%M%SZ)"
.venv-editor/bin/python -m unittest tests.test_mini_pilot -v
```

Environment and cohort check (choose a new timestamped output name):

```bash
.venv-editor/bin/python train_and_evaluate_pilot.py --preflight_only --output_dir outputs/pilot_run_UNIQUE_preflight
```

Exactly one A100 allocation, at most 3600 allocated GPU seconds, a cooperative
3300-second guard at sampling/update boundaries, no retry, five arm records.
Slurm enforces the hard 3600-second limit; the cooperative guard cannot interrupt
an individual long CUDA operation:

```bash
.venv-editor/bin/python scripts/submit_mini_pilot.py --run-id pilot_run_UNIQUE
```

The wrapper reserves an exclusive submission directory before calling Slurm.
Keep Slurm's device mapping; do not force `CUDA_VISIBLE_DEVICES=0`. CUDA runs
only inside the one-GPU allocation. A preflight-only result is not an experiment.
Read `submission.json`, the Slurm log and `run_state.json` for actual status.
If sbatch fails ambiguously, check the queue; never blindly repeat submission.

For the current run, the user subsequently approved at most 90 cumulative
single-GPU minutes and one checkpoint continuation if the initial time cap is
reached. Only after the initial allocation has ended with a cooperative
`TimeoutError`, the following command reserves at most 1800 additional GPU
seconds and verifies that actual prior spend plus the reservation is <=5400:

```bash
.venv-editor/bin/python scripts/submit_mini_pilot_continuation.py --run-id pilot_run_UNIQUE
```

It does not retry model/numerical failures. It preserves the failed-run receipt,
restores actual adapter/Adam/RNG and any pending learned action, verifies a real
next-update repeat from the partial checkpoint, and executes only remaining
training steps. The baseline, semantic weight, pending action and controller RNG
are restored; the settings and per-step seeds remain unchanged. A single second
failure does not admit a third allocation. Continuation-specific CPU regression:

```bash
.venv-editor/bin/python -m unittest tests.test_mini_pilot_continuation -v
```

`max_inner_steps=100` means 100 total real optimizer updates per trained arm,
divided 34/33/33 over three rounds, with physical batch one. Each round visits
all 32 train images, then cycles the preregistered shuffled ordering. All trained
arms start from identical adapter/Adam/RNG state, training sources and per-step
sampling seeds. Stochastic DDIM uses 10 transitions, eta=1, at 512×512; this is
DDPO score-function optimization, not DDPO-PPO. The actual UNet LoRA is optimized.
UNet targets are `to_q`, `to_k`, `to_v`, `to_out.0` (the pasted q_proj names do
not match this UNet). Rank=8, alpha=16, AdamW LR=1e-5, eps=1e-8.

Full FP32 base computation upcasts the existing cached FP16-variant weights;
it does not recover precision absent from that cached checkpoint. LoRA storage/linear
compute/Adam moments are FP32. This is not a fix claim for the older BF16 path.
Before arm training, five real numerical-witness updates test an ordinary
two-update repeat and disk-reloaded next update, with exact and tolerance
checks separately covering raw/postclip gradients, adapter, Adam and CUDA/Torch
RNG. Training stops if either fails. Failure evidence and the partial optimizer
checkpoint are retained. The witness is a local gate, not universal numerical
certification. No existing acceptance threshold is weakened.

Reward is `weight_sem * CLIP_cosine - LPIPS`, with both terms required. LPIPS
receives [-1,1]; CLIP uses PIL RGB through its processor. Training begins with
weights 2 and 1. All comparisons and controller feedback use the SAME common
evaluator `2*CLIP_cosine - LPIPS`, preventing gains from merely changing units.
The learned controller samples four bounded actions before each round and
actually learns with REINFORCE from that round's validation gain. Its policy,
Adam, RNG and experience memory are saved. Three rounds give only three controller
updates; they do not establish a reliable controller-learning curve.

Frozen and Fixed are an explicitly user-approved equivalence control: same
prompt, weights, sampler and seed. Their generation/observations/checkpoint are
shared and counted once physically. A frozen editor can still change image
pixels. Rule RSI halves LR only after a common validation-score decline.

The mini split references 64 existing images without copying them. Image IDs,
sample IDs, SHA256 and decoding are validated across all three splits. Its
parent is the historically exposed VQAv2 training pool, so the 16 mini-test
images are DEVELOPMENT assessment, not an untouched final test. The historical
test and prospective sealed pool are never read. Final checkpoints are locked
before assessment and are not selected by assessment scores. No victim training
or actual ASR feedback is represented by the validation memory.

Every run writes the seven requested CSV/Markdown/TeX reports, protocol, exact
source snapshots, observations, updates, model-resource receipts and artifact
hash inventory. Missing measurements say NOT TESTED. Figures show the common
validation reward and the first four assessment sources (no visual cherry
picking). Source-bootstrap intervals do not estimate training-seed uncertainty.
Retained evidence is estimated below 1.5 GB on cluster storage; C: receives no
weights or large intermediates. Preserve checkpoints and numerical witnesses.

Report regeneration uses existing observations without running models:

```bash
.venv-editor/bin/python train_and_evaluate_pilot.py --reports_only --output_dir outputs/pilot_run_UNIQUE
```

Metric references: [LPIPS input contract](https://github.com/richzhang/PerceptualSimilarity),
[CLIP processor/model](https://huggingface.co/docs/transformers/model_doc/clip),
[InstructPix2Pix pipeline](https://huggingface.co/docs/diffusers/api/pipelines/pix2pix).
