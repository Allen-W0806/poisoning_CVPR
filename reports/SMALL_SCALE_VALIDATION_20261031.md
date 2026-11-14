# Small-scale benign validation — 2026-10-31

Updated 2026-10-31T19:41:25.706707+00:00; immutable run `small_scale_validation_20261031T1928Z_v1`.

CPU auditing and real GPU work were executed; data/metric gaps remain blocked. These are engineering measurements, not a backdoor or autonomous-RSI efficacy result.

|Stage|Executed|Run ID|Slurm job|Result|Numerical checks|Scientific conclusion|Blockers|
|---|---|---|---|---|---|---|---|
|CPU source/provenance|True|small_scale_validation_20261031T1928Z_v1|87920|80 unique source/content; 32/16/32|PASS|Source-disjoint engineering data|Untouched-test status BLOCKED|
|Image controls|True|small_scale_validation_20261031T1928Z_v1|87920|400 valid traced outputs|PASS: decode/SSIM/pixel|Transformation engineering only|LPIPS missing; edited-label validity unreviewed|
|Frozen IP2Pix|True|small_scale_validation_20261031T1928Z_v1|87920|160 existing outputs/hashes audited|PASS: existing traceability|Descriptive reuse only|Independent human review0/160; BF16 continuation unresolved|
|Clean Qwen LoRA|True|small_scale_validation_20261031T1928Z_v1|87921|3 seeds complete|PASS|No inferred generalization benefit|historical development assessment; not an untouched scientific test|
|Benign RL exact/MC|True|small_scale_validation_20261031T1928Z_v1|87920|finite categorical/group reference|PASS|Synthetic mathematical consistency only||
|Production DDPO optimizer|True|small_scale_validation_20261031T1928Z_v1|87921|Tiny random UNet; pretrained ownership ZERO STEPS|PASS|No production GRPO or pretrained reliability claim|Full pretrained BF16 gate remains blocked|
|Synthetic RSI|True|small_scale_validation_20261031T1928Z_v1|87920|2 reproducible accept/reject/regression/rollback paths|PASS|Integrity oracle improvement only; not autonomous scientific RSI||
|Benign augmentation training|False|small_scale_validation_20261031T1928Z_v1|87920|NOT RUN|BLOCKED|No claim|Label validity not independently established|
|Attack reproduction / ASR|False|small_scale_validation_20261031T1928Z_v1|87920|NOT PERFORMED|N/A|No efficacy claim|Outside this benign pilot|

## Clean VQA measurements

|Seed|Split|n|Before|After|Paired95%CI pp|Changed answers|
|---|---|---:|---:|---:|---|---:|
|42|development|16|93.125%|93.125%|[+0.000,+0.000]|0|
|42|independent_evaluation|32|78.438%|78.438%|[+0.000,+0.000]|0|
|123|development|16|93.125%|93.125%|[+0.000,+0.000]|0|
|123|independent_evaluation|32|78.438%|80.625%|[+0.000,+6.562]|1|
|3407|development|16|93.125%|93.125%|[+0.000,+0.000]|0|
|3407|independent_evaluation|32|78.438%|80.625%|[+0.000,+6.562]|1|

Five thousand paired image-bootstrap draws per split/seed; shared sources across seeds are not96 independent test images. A zero empirical difference interval does not prove population equivalence. The two improving seeds change the same question362793004 from yes(score0.3) to no(score1.0); these are not two independent improved examples. Their+2.1875pp difference has empirical95% interval[0,+6.5625]pp and does not establish a reliable population benefit. Per-category accuracy, latency, invalid rates, actual parameter counts and checkpoint hashes are in the linked statistics. No seed/config selection used final scores.

Data scope: **historical development assessment; not an untouched scientific test**. All80 source IDs/hashes differ, selected pairs pass dHash64 Hamming>4; this limited screen cannot exclude all semantic/perceptual duplicates. Two old unparseable records prevent an untouched-test certification. Cached official questions/all ten answers retained; majority-normalized supervision requires6/10 consensus. No images or labels were downloaded or invented.

## Image controls

{
  "original": {
    "n": 80,
    "mean_SSIM": 1.0,
    "mean_MAE": 0.0,
    "processing_seconds": 4.899182181805372,
    "LPIPS": null
  },
  "identity": {
    "n": 80,
    "mean_SSIM": 1.0,
    "mean_MAE": 0.0,
    "processing_seconds": 11.95321193151176,
    "LPIPS": null
  },
  "brightness": {
    "n": 80,
    "mean_SSIM": 0.9987124382228147,
    "mean_MAE": 2.606712558531439,
    "processing_seconds": 11.98843969590962,
    "LPIPS": null
  },
  "contrast": {
    "n": 80,
    "mean_SSIM": 0.9913415206050192,
    "mean_MAE": 1.4841470826885268,
    "processing_seconds": 11.966238394379616,
    "LPIPS": null
  },
  "geometry": {
    "n": 80,
    "mean_SSIM": 1.0,
    "mean_MAE": 0.0,
    "processing_seconds": 3.4061366245150566,
    "LPIPS": null
  }
}

400 current original/identity/brightness1.03/contrast1.03/geometry256 outputs have hashes, dimensions, SSIM, pixel MAE and CPU latency. Geometry SSIM/MAE use the geometry-matched reference, so zero difference is not a certificate against native-resolution loss. Existing frozen IP2Pix generation latency/memory/seed metadata are copied with their parent hash, not charged as new generation. LPIPS remains **BLOCKED**: package absent; no substitution called LPIPS. All nonidentity image-answer validity remains unreviewed; label-changing and ambiguous population rates are unknown. No benign augmentation training is eligible yet.

## RL and RSI boundaries

Actual categorical code is REINFORCE; production diffusion optimizer is single-pass on-policy DDPO-SF, not GRPO. Exact-vs-Monte-Carlo and group-normalized reference tests retain frozen tolerances. Tiny random-UNet LoRA is the only diffusion model updated by this run; pretrained IP2Pix ownership is audited with zero steps. Existing guard rejects empty, false and true eligibility receipts and its historical hash matches. No target-error/security signal enters the benign objective. Historical uninstrumented pretrained BF16 continuation FAIL and instrumented witness PASS remain separate; general continuation reliability is NOT VALIDATED.

Synthetic RSI uses the actual memory, investigator, proposal, fixed Gate and VersionManager twice: fixed/no-revision score0.5, revised integrity-oracle score1.0, ineffective/regression revisions rejected, explicit rollback successful; every diagnostic/evaluator operation is recorded. These deliberately constructed oracle cases do not establish natural-image semantic validity or autonomous research improvement.

## Resources, monitor and evidence

{
  "prior_gpu_seconds": 1344,
  "new_gpu_seconds": 332,
  "total_validation_gpu_seconds": 1676,
  "remaining_validation_gpu_seconds": 1924,
  "new_stage_cap": 900,
  "global_validation_cap": 3600,
  "new_jobs": [
    {
      "job_id": "87920",
      "state": "COMPLETED",
      "elapsed": 94,
      "gpu_count": 0,
      "gpu_seconds": 0,
      "exit_code": "0:0",
      "start_EDT": "2026-10-31T15:27:20",
      "end_EDT": "2026-10-31T15:28:54"
    },
    {
      "job_id": "87921",
      "state": "COMPLETED",
      "elapsed": 332,
      "gpu_count": 1,
      "gpu_seconds": 332,
      "exit_code": "0:0",
      "start_EDT": "2026-10-31T15:29:41",
      "end_EDT": "2026-10-31T15:35:13"
    },
    {
      "job_id": "87922",
      "state": "COMPLETED",
      "elapsed": 3,
      "gpu_count": 0,
      "gpu_seconds": 0,
      "exit_code": "0:0",
      "start_EDT": "2026-10-31T15:35:19",
      "end_EDT": "2026-10-31T15:35:22"
    }
  ],
  "prior_jobs": [
    {
      "job_id": "86736",
      "state": "FAILED",
      "elapsed": 138,
      "gpu_count": 1,
      "gpu_seconds": 138,
      "exit_code": "1:0"
    },
    {
      "job_id": "86738",
      "state": "COMPLETED",
      "elapsed": 28,
      "gpu_count": 0,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "86739",
      "state": "FAILED",
      "elapsed": 6,
      "gpu_count": 1,
      "gpu_seconds": 6,
      "exit_code": "1:0"
    },
    {
      "job_id": "86741",
      "state": "COMPLETED",
      "elapsed": 20,
      "gpu_count": 1,
      "gpu_seconds": 20,
      "exit_code": "0:0"
    },
    {
      "job_id": "86742",
      "state": "COMPLETED",
      "elapsed": 24,
      "gpu_count": 1,
      "gpu_seconds": 24,
      "exit_code": "0:0"
    },
    {
      "job_id": "86745",
      "state": "COMPLETED",
      "elapsed": 10,
      "gpu_count": 1,
      "gpu_seconds": 10,
      "exit_code": "0:0"
    },
    {
      "job_id": "86793",
      "state": "COMPLETED",
      "elapsed": 129,
      "gpu_count": 1,
      "gpu_seconds": 129,
      "exit_code": "0:0"
    },
    {
      "job_id": "86794",
      "state": "COMPLETED",
      "elapsed": 32,
      "gpu_count": 0,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "86795",
      "state": "FAILED",
      "elapsed": 436,
      "gpu_count": 1,
      "gpu_seconds": 436,
      "exit_code": "1:0"
    },
    {
      "job_id": "86796",
      "state": "COMPLETED",
      "elapsed": 19,
      "gpu_count": 1,
      "gpu_seconds": 19,
      "exit_code": "0:0"
    },
    {
      "job_id": "86804",
      "state": "FAILED",
      "elapsed": 33,
      "gpu_count": 1,
      "gpu_seconds": 33,
      "exit_code": "1:0"
    },
    {
      "job_id": "86805",
      "state": "FAILED",
      "elapsed": 31,
      "gpu_count": 1,
      "gpu_seconds": 31,
      "exit_code": "1:0"
    },
    {
      "job_id": "86806",
      "state": "COMPLETED",
      "elapsed": 60,
      "gpu_count": 1,
      "gpu_seconds": 60,
      "exit_code": "0:0"
    },
    {
      "job_id": "86807",
      "state": "COMPLETED",
      "elapsed": 96,
      "gpu_count": 1,
      "gpu_seconds": 96,
      "exit_code": "0:0"
    },
    {
      "job_id": "86846",
      "state": "COMPLETED",
      "elapsed": 79,
      "gpu_count": 1,
      "gpu_seconds": 79,
      "exit_code": "0:0"
    },
    {
      "job_id": "86848",
      "state": "COMPLETED",
      "elapsed": 79,
      "gpu_count": 1,
      "gpu_seconds": 79,
      "exit_code": "0:0"
    },
    {
      "job_id": "86851",
      "state": "COMPLETED",
      "elapsed": 78,
      "gpu_count": 1,
      "gpu_seconds": 78,
      "exit_code": "0:0"
    },
    {
      "job_id": "86853",
      "state": "FAILED",
      "elapsed": 53,
      "gpu_count": 1,
      "gpu_seconds": 53,
      "exit_code": "1:0"
    },
    {
      "job_id": "86856",
      "state": "COMPLETED",
      "elapsed": 53,
      "gpu_count": 1,
      "gpu_seconds": 53,
      "exit_code": "0:0"
    }
  ],
  "raw_sacct": "87920|COMPLETED|94|billing=4,cpu=4,mem=8G,node=1|0:0|2026-10-31T15:27:20|2026-10-31T15:28:54\n87921|COMPLETED|332|billing=4,cpu=4,gres/gpu=1,mem=48G,node=1|0:0|2026-10-31T15:29:41|2026-10-31T15:35:13\n87922|COMPLETED|3|billing=2,cpu=2,mem=8G,node=1|0:0|2026-10-31T15:35:19|2026-10-31T15:35:22\n",
  "cpu_only_allocated_seconds": 97,
  "billing_dollars": null
}

Existing two-hour monitor was reused without installation or duplicate recurrence. Existing finalizer is an afterany CPU report hook, not another timer. Dollar pricing is unavailable. New-stage900 A100-second and cumulative3600-second caps apply to this validation scope; historical unrelated training is not silently charged to or authorized by it. Existing scheduler enabled=True, next registered job87909, eligibility UTC 2026-10-31T20:32:35+00:00. The old replication-stage heartbeat still reports its own1344-second subtotal; use RESOURCE_ACCOUNTING_20261031.json for the aggregate including this stage. Dashboard refresh preserves this new-stage payload.

Evidence: [immutable protocol](../results/small_scale_validation_20261031T1928Z_v1/protocol.json), [split manifest](../results/small_scale_validation_20261031T1928Z_v1/split_manifest.json), [data checks](../results/small_scale_validation_20261031T1928Z_v1/audit/data_integrity.json), [raw normalized predictions](../results/small_scale_validation_20261031T1928Z_v1/clean_evaluation.csv), [statistics/checkpoint hashes](../results/small_scale_validation_20261031T1928Z_v1/clean_statistics.json), [image pairs](../results/small_scale_validation_20261031T1928Z_v1/image_controls.csv), [synthetic RSI](../results/small_scale_validation_20261031T1928Z_v1/synthetic_rsi/results.json), [guard tests](../results/small_scale_validation_20261031T1928Z_v1/audit/reward_guard_verified.json), [source freeze](../results/small_scale_validation_20261031T1928Z_v1/source_manifest.json), [logs](../results/small_scale_validation_20261031T1928Z_v1/logs/), [literature protocol audit](LITERATURE_BASELINE_PROTOCOL_20261031.md). Full adapters/checkpoints stay on cluster; Windows sync transfers small evidence only.

Next decision: retain no-go for generator learning; obtain genuine independent reviews and a verifiable private assessment split. Do not expand training or retry unexplained numerical failures to chase a positive result.
