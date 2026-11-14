# Phase 2 setup and execution

Canonical project `/storage/home/ydk297/projects/poisoning`; existing mirror `H:\Project\Poisoning`. Starting commits are recorded in reports/PHASE2_START.json; existing experiments remain immutable. New real editor environment `.venv-editor` is separate from original Qwen `.venv`, which ran the full baseline successfully. Exact dependency freezes are phase2_freeze_editor.txt and phase2_freeze_qwen.txt.

Report-generation source commit: `b78323954a5af4ea49e4692cd0382634eaa86c3c`. Each scientific run retains its own executing commit below.

New components: native pretrained editor/config/output metadata, inspected UNet rank-8 LoRA supervision/restart, explicit unsupported generator-RL contract, real cached-editor categorical RL, conservative question-aware semantic/answer/localization checks, licensed pinned official VQA evaluator, frozen full-split baseline, actual-observation RSI history/rollback, budget guard and experiment records.

Resource policy: rechecked epyc-gpu/allusers/A100 80GB; one GPU, four CPUs, 48G host RAM per short job. Wrapper refuses active/unconfirmed previous Phase 2 allocations and checks completed/failed elapsed seconds plus active reservations against 7200 GPU-seconds. No large sweep launched.

| Job | State | Exit | Allocated seconds |
|---|---|---|---|
| 86292 | COMPLETED | 0:0 | 41 |
| 86293 | FAILED | 1:0 | 46 |
| 86299 | FAILED | 1:0 | 77 |
| 86300 | COMPLETED | 0:0 | 63 |
| 86301 | FAILED | 1:0 | 44 |
| 86302 | FAILED | 1:0 | 42 |
| 86303 | FAILED | 1:0 | 15 |
| 86304 | FAILED | 1:0 | 49 |
| 86305 | COMPLETED | 0:0 | 73 |
| 86306 | COMPLETED | 0:0 | 48 |
| 86307 | COMPLETED | 0:0 | 146 |
| 86308 | COMPLETED | 0:0 | 543 |

Total allocated Phase 2 GPU cost: **0.329722 GPU-hours** (1187s), including all failures/loading/evaluation; maximum budget 2 GPU-hours. Optimizer component timings are inside this total and must not be added again.

Retained storage snapshot:
```text
1.2G	/storage/home/ydk297/projects/poisoning/.venv-editor
468M	/storage/home/ydk297/projects/poisoning/experiments
2.0G	/storage/home/ydk297/.cache/huggingface/hub/models--timbrooks--instruct-pix2pix
```

No multi-GB download on C:. Pre-staging estimate ≤15GB new cluster data/environment/artifacts; actual model download selected 2.134GB required FP16 component files. Central cache reused. Failed adapter/optimizer evidence and generated comparison images remain; scratch/install archives are removed only after mirroring. Shared runtimes and original model weights are untouched.

All new canonical experiment IDs:

| Experiment | Status | Job | Executing commit |
|---|---|---|---|
| `EXP_20261027T045854_00a2bc07659a_phase2_slurm_editor` | PASS | 86292 | `1bc230fc4bd6c30527e883a0303ad74538958818` |
| `EXP_20261027T045901_48eded21ef7d_phase2_editor_inference` | PASS | 86292 | `1bc230fc4bd6c30527e883a0303ad74538958818` |
| `EXP_20261027T050226_b587248a1238_phase2_slurm_training` | FAIL | 86293 | `67f5aeecc4c700b87141f8d4d8baccbf864b6772` |
| `EXP_20261027T050233_d1a8189e815b_phase2_generator_lora_smoke` | FAIL | 86293 | `67f5aeecc4c700b87141f8d4d8baccbf864b6772` |
| `EXP_20261027T050443_d70eeb41829e_phase2_slurm_training` | FAIL | 86299 | `d88c0e66bc72a09a4644b6cf1aa427df9856fff3` |
| `EXP_20261027T050455_62f4806ace20_phase2_generator_lora_smoke` | FAIL | 86299 | `d88c0e66bc72a09a4644b6cf1aa427df9856fff3` |
| `EXP_20261027T050734_4883b0f3fbde_phase2_slurm_vqa_dev` | PASS | 86300 | `0b2c8e44d36d2e001c9e5f57bd329a88255fc465` |
| `EXP_20261027T050746_4556a0cd5c4e_phase2_qwen_baseline_validation` | PASS | 86300 | `0b2c8e44d36d2e001c9e5f57bd329a88255fc465` |
| `EXP_20261027T051104_149fb4487202_phase2_slurm_training` | FAIL | 86301 | `4af6804840c7052fb09b65c9a75fa86987b58a1f` |
| `EXP_20261027T051111_9c5d2763200c_phase2_generator_lora_smoke` | FAIL | 86301 | `4af6804840c7052fb09b65c9a75fa86987b58a1f` |
| `EXP_20261027T051338_de75b22010ca_phase2_slurm_training` | FAIL | 86302 | `1c4e7189976d758b5c494aa3436cc9fe4c9d3670` |
| `EXP_20261027T051344_4868aef599e7_phase2_generator_lora_smoke` | FAIL | 86302 | `1c4e7189976d758b5c494aa3436cc9fe4c9d3670` |
| `EXP_20261027T051624_8506a65aa0ca_phase2_slurm_training` | FAIL | 86303 | `2bcabf927dd83a0dc8b5b6f87224949d498a31fd` |
| `EXP_20261027T051632_b374e74bd043_phase2_generator_lora_smoke` | FAIL | 86303 | `2bcabf927dd83a0dc8b5b6f87224949d498a31fd` |
| `EXP_20261027T051832_ba75b1635cca_phase2_generator_rl_contract` | BLOCKED | None | `056a777cab404080fc885509e67f843e2d6e21fb` |
| `EXP_20261027T051835_6a0c63595002_phase2_slurm_training` | FAIL | 86304 | `056a777cab404080fc885509e67f843e2d6e21fb` |
| `EXP_20261027T051841_f4c308c19fc1_phase2_generator_lora_smoke` | FAIL | 86304 | `056a777cab404080fc885509e67f843e2d6e21fb` |
| `EXP_20261027T052236_c5cc0d4d4aab_phase2_slurm_vqa_test` | PASS | 86305 | `056a777cab404080fc885509e67f843e2d6e21fb` |
| `EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test` | PASS | 86305 | `056a777cab404080fc885509e67f843e2d6e21fb` |
| `EXP_20261027T052537_301c8d9917c9_phase2_slurm_training` | PASS | 86306 | `cceb467df32c5c7120a348ab512fa944abf4e8dc` |
| `EXP_20261027T052543_a20882fb60de_phase2_generator_lora_smoke` | PASS | 86306 | `cceb467df32c5c7120a348ab512fa944abf4e8dc` |
| `EXP_20261027T052704_89747bf85f00_phase2_slurm_semantics` | PASS | 86307 | `cceb467df32c5c7120a348ab512fa944abf4e8dc` |
| `EXP_20261027T052713_2e70f825d4bd_phase2_semantic_validation` | PASS | 86307 | `cceb467df32c5c7120a348ab512fa944abf4e8dc` |
| `EXP_20261027T053035_485bf0d613af_phase2_slurm_controller` | PASS | 86308 | `09bf28c2ca7f0aa73e7c2c45680502d6570c4ace` |
| `EXP_20261027T053043_7b4b5ce3fe61_phase2_real_editor_feedback` | PASS | 86308 | `09bf28c2ca7f0aa73e7c2c45680502d6570c4ace` |
| `EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation` | PASS | 86308 | `09bf28c2ca7f0aa73e7c2c45680502d6570c4ace` |

Registry unique: True; all new runs registered: True; all required files present: True. Raw source revisions are in every manifest/git_commit.txt. Old source changes extend modules; Phase 1 scorer/results are not replaced. Final mirror import checks existing same-ID experiment bytes before accepting new evidence.

Read the seven focused reports plus PHASE2_FAILURES.md. Completion here means feasible Phase 2 engineering stages executed with honest blockers; it does not claim a learned backdoor, generator RL, RSI advantage, or fully verified natural-image semantics.

Final mirror and cleanup verification: 59 changed source files match after CRLF-to-LF normalization; prior same-ID experiment files passed exact byte-equality checks before import. Local final commit `c577c27d61836d7aaa9d0e79a515ffdea7b99d69`; canonical final commit `b78323954a5af4ea49e4692cd0382634eaa86c3c`; both source trees clean. The local registry contains 78 unique runs, including preserved independent Phase 1 history.

Run-specific scratch archives and 12 project bytecode-cache directories on each machine were removed after successful import. All weights, environments, experiment images, failures and checkpoints remain. Local retained experiment files total 462052610 bytes. Post-cleanup free space: C: 70963273728 bytes; H: 235770863616 bytes. No multi-gigabyte artifact was placed on C:. Detailed receipts: `phase2_final_verification.json`, `phase2_final_cleanup_cluster.json`, and `phase2_source_mirror_audit.json`.
