# Published baseline protocols and comparison admission

Project: /storage/home/ydk297/projects/poisoning. Evidence review: 2026-11-01 UTC. Internal recomputation: CPU Slurm **88317**, completed 22:57:34–22:57:54 UTC, exit 0:0.

**No external method below was reproduced in this project. None is directly comparable to our current measured endpoints.** Similar dataset names do not establish identical sources, victim checkpoint, training, labels, prompts, scoring or denominator. Published attack results are not inserted into the internal CSVs. No backdoor training or attack-reward optimization was performed.

## A. Internally measured results

| Evidence | Executed scope and denominator | Permitted use | Excluded interpretation |
|---|---|---|---|
| C0–C5 natural-image controls | Frozen Qwen3-VL-2B-Instruct; 160 matched VQAv2 sources; original question and ten references; 960 primary records, 72 dependent repeats excluded | Original-reference agreement, paired source intervals, prediction changes, aligned SSIM and saved costs | Edited-image label validity, untouched-test generalization, training-induced attack success |
| Fixed DDPO / ordinary scheduler / RSI | Tiny FP32; seeds 42/123/3407; 32 shared synthetic test conditions; four updates and 320 forward NFE per branch | Controlled benign-task ablation and negative/exploratory RSI contrasts | Natural-image learning or backdoor efficacy |
| Historical pretrained M0/M1/M2 | M0 frozen: zero updates, two evaluation seeds; M1 DDPO/M2 RSI: two training seeds, eight updates each. Legacy proxy; trained states differ before intervention | Descriptive history and confounding evidence only | Causal RSI improvement |

Reproducible values, raw-record paths, scores, CIs and cost distinctions are in reports/BASELINE_REAL_IMAGE_RESULTS.{md,csv,tex} and reports/RL_RSI_ABLATION_RESULTS.{md,csv,tex}. All final-test claims remain blocked.

## B. External protocols — published/reported, not reproduced

NE means not established in the inspected primary sections or linked assets, **not** evidence of absence. Seed count means independent training runs, not generation seeds, trigger variants or human assessors. Checkpoints were not downloaded or executed.

### Required core papers

| Method / primary source | Task, dataset and victim | Training / modality / supervision | Metrics and denominator | Seeds; public assets verified | Match to project |
|---|---|---|---|---|---|
| **BadSem (2025)** — [paper, Experiments](https://arxiv.org/html/2506.07214v1) | VQAv2/GQA; LLaVA-1.5-7B, Qwen2-VL-2B/7B, Llama-3.2-Vision-11B | 5,000 clean triplets/dataset plus injected data; LoRA victim fine-tuning. Textual or visual semantic inconsistency; targeted responses rather than unchanged original-answer supervision | Clean accuracy on 2,000 validation items/dataset. Overall ASR: any successful candidate per original item; Full ASR: successes / all candidate attempts | Independent training seed count NE; official code/checkpoint availability NE in inspected paper | VQAv2 task only. Different victims, trained conditions, sources and ASR denominator; **not matched** |
| **CBV (ICML 2026)** — [official proceedings](https://proceedings.mlr.press/v306/guo26q.html), [paper §5.1](https://arxiv.org/html/2605.02202) | MSCOCO captioning/VQAv2; LLaVA-1.5-7B, MiniGPT-v2, InstructBLIP-7B, Qwen2.5-VL-7B | Victim training on clean plus poisoned images; nominally retained text labels. Image modification with image/text construction guidance; not our learned DDPO-SF policy | Target-word occurrence / triggered evaluations; clean BLEU-4, CIDEr or VQA accuracy. Exact train/test sample counts NE here | Independent seeds NE; official runnable code and checkpoints NE | Diffusion editing/VQAv2 partially similar. Different generation, victim training and estimands; **not matched** |
| **TrojVLM (ECCV 2024)** — [official paper](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/08278.pdf), [§4.1](https://arxiv.org/html/2409.19232) | Flickr8k/30k/COCO captioning; OK-VQA/VQAv2. BLIP-2, extensions MiniGPT-4/InstructBLIP | Access to training data/process; clean task fine-tuning then backdoor training, primarily Q-Former. Visual trigger and modified output supervision; question unchanged | Target text occurs / generated triggered outputs. VQA/caption metrics; target text removed before poisoned-output quality scoring | Independent seeds and official code/checkpoints NE in inspected primary text | VQA only; victim, labels and target-stripped scoring differ; **not matched** |
| **Shadowcast (NeurIPS 2024)** — [paper §4](https://arxiv.org/html/2402.06659), [official code](https://github.com/umd-huang-lab/VLM-Poisoning) | Four concept tasks; 3,500 cc-sbu-align clean pairs; 200 test images/concept. Mainly LLaVA-1.5-7B; MiniGPT-v2 and transfer settings | Contaminated image/text pairs for victim fine-tuning; generated/refined captions. Both image construction and text-pair manipulation; not unchanged VQA labels | Label ASR requires destination mention and absence of original concept; persuasion uses judged responses. Clean utility on GQA/VizWiz. Task-specific response denominator | Independent training seeds NE; public code verified; checkpoint release NE | Neither our matched VQA score nor our no-training C4; **not matched** |

### Other relevant methods already cited in the supplied manuscript

Frozen manuscript: results/method_code_audit_20261101T0510Z_v1/audit/supplied_method_snapshot/manuscript_original.tex. The nominal original manuscript path on the cluster is absent; this snapshot and the existing local template are retained rather than silently replacing manuscript sources.

| Method / verifiable source | Task / data / victim | Training, modalities, label assumptions | Metric unit / denominator | Seeds / code / checkpoints | Comparability |
|---|---|---|---|---|---|
| **VLOOD (ICLR 2025)** — [official proceedings](https://proceedings.iclr.cc/paper_files/paper/2025/file/da37068018f8f7185720b83ea6f48377-Paper-Conference.pdf), [paper](https://arxiv.org/html/2410.01264) | Captioning/VQA; Flickr/COCO/OK-VQA/VQAv2; BLIP-2/MiniGPT-4/InstructBLIP | OOD shadow data without original training data; victim updated; visual trigger plus changed text outputs | Target-string inclusion per poisoned output; caption/VQA quality after target removal on poisoned inputs | Seeds NE; official method code/checkpoints NE; cited defense framework is not a method release | Different data access, victim and supervision |
| **TokenSwap (ICML 2026)** — [official proceedings](https://proceedings.mlr.press/v306/zhang26fr.html), [paper §3.1](https://arxiv.org/html/2509.24566) | Flickr8k/30k/MSCOCO; BLIP-2, InstructBLIP-7B, LLaVA-1.5-7B/13B, Qwen2.5-VL-7B | Shadow-data victim training; visual input plus subject/object-swapped text supervision | Successful semantic role swaps / evaluated generated responses; BLEU/ROUGE utility | Independent seeds NE; method code/checkpoints NE in inspected sources | Composition/caption endpoint differs from original-answer VQA agreement |
| **IAG (CVPR 2026)** — [paper](https://arxiv.org/html/2508.09456), [official code](https://github.com/lijunxian111/IAG) | RefCOCO/+/g, Flickr30k Entities, ShowUI; LLaVA-1.5-7B, Ferret-7B, InternVL-2.5-8B | Grounding victim training; image changes and changed target-box supervision; text-guided generator | ASR/BA/CA at IoU **>0.5** per grounding query; ShowUI uses predicted-box center inclusion | Seeds NE; public code verified, checkpoints NE | Grounding/box labels differ; a UNet does not imply DDPO equivalence |
| **Phantasia (2026 arXiv)** — [paper §6.1](https://arxiv.org/html/2604.08395) | Captioning/VQA; Flickr8k/30k, OKVQA/VQAv2; BLIP/BLIP-2/LLaVA; reports 1,000 evaluation samples/task | Malicious-provider/shadow-data setting; teacher/student fine-tuning; visual triggers and changed task-dependent answers | BERTScore-based ASR and LAVE compliance over responses; clean text/VQA quality | Seeds NE. [Paper-linked code](https://github.com/nduongw/Phantasia) returned 404 during this check; checkpoints NE | Different training/judges. Manuscript “CVPR Findings” venue not independently confirmed |
| **ReShift (2026)** — [paper](https://arxiv.org/html/2607.00361), [author venue record](https://sumonbis.github.io/publication/eccv26/) | A-OKVQA/ScienceQA, OOD MathVista/MMMU; Qwen2.5-VL-7B/InternVL3.5-8B; 500-item test sets | Victim supervised/reinforcement training with changed reasoning/answers; visual triggers | Target-answer fraction of triggered samples; clean accuracy/false-trigger rate; judged reasoning quality | Seeds NE; [linked repository](https://github.com/AlbertZhaoCA/ReShift) accessible, runnable implementation/checkpoints not verified | Victim reasoning optimization differs from generator DDPO |
| **BEAT (ICLR 2026)** — [author project](https://zqs1943.github.io/BEAT/), [conference paper](https://openreview.net/pdf/8bcc994c41815c824b90f9b6cdec94f9e1e3d872.pdf) | VisualAgentBench/EmbodiedBench; Qwen2-VL-7B-Instruct/InternVL3-8B agents | Backbone fine-tuned on benign/triggered embodied trajectories; object triggers and changed action supervision | Episode/task success, backdoor activation F1/ASR; exact evaluated episode counts NE here | Seeds NE; project links code/data, linked assets/checkpoints not verified here | Multi-step episodes rather than VQA items |
| **ASO (CVPR 2026)** — [official proceedings](https://openaccess.thecvf.com/content/CVPR2026/html/Luo_Adversarial_Style_Optimization_Enhancing_VLM_Jailbreaks_by_GRPO-based_Stylistic_Triggers_CVPR_2026_paper.html), [official code](https://github.com/bingjunluo/ASO) | MM-SafetyBench/VLBreakBench visual jailbreaks; repository includes Qwen3 victim endpoint | Editing-generator RL with served victim/judge; inference-time jailbreak images, not poisoned-victim training comparison | Released ASR: HarmBench binary Yes / judged responses; HS: mean judge log-probability difference | Seeds NE; public code verified; trained checkpoints NE | Related editing/RL family only; different endpoint |
| **BadVLM (Dang et al.; manuscript citation)** — [author institution](https://espace.inrs.ca/id/eprint/17209/) | Abstract confirms captioning/VQA and two VLMs; exact datasets/victim IDs not established | Abstract describes black-box visual construction and later victim training; precise supervision NE | CA/attack outcomes mentioned; denominator/scorer NE | Seeds/code/checkpoints NE. Thesis embargoed until 2026-12-08; attempted conference page unavailable | Identity/venue/protocol incomplete; do not conflate with same-name papers or Phantasia's BadNets adaptation |

No published numeric ASR is transcribed here. Missing seeds, inaccessible repositories and unresolved denominators are admission blockers, not values to guess. An old manuscript venue label is insufficient evidence.

## Matched-comparison admission

Required: same task/estimand, immutable source/question/reference manifest, victim checkpoint/training lineage, supervision, victim-update budget, generator exposure/query budget, scoring/normalization, validity rule, seed pairing and measured GPU accounting. Currently **0 external baselines meet this full contract**. This matrix is contextual related work, not an empirical leaderboard.

## Progression decision

| Task | Existing evidence / prerequisite | Actual action / next falsifiable check | Resource / status |
|---|---|---|---|
| Matched controls / image quality | Existing 160-source raw outputs | 88317 re-scored 960 primary rows, checked candidate hashes and computed 640 aligned SSIM measurements; reused C4/C5 quality; historical points reconcile | 20 CPU-job seconds; **PASS scoped computation**, 0 new GPU |
| Tiny ablation statistics | Existing 3×3 branches and paired vectors | Original two crossed intervals reproduce exactly; additional scheduler–Fixed interval from saved data; no branch rerun | Bounded CPU supplement; **PASS computation**, superiority **NOT ESTABLISHED** |
| BF16 origin | 88099/88105 plus completed CPU88229 | 32 raw mismatches in first down-block attentions. Different global clip factors exactly reconstruct 256 post-clip tensors; 224 were equal before clipping. Clipping is not the origin. New hypothesis: unobserved upstream adjoints/recomputation versus local LoRA VJP; future decisive check must capture matched adjoint boundaries, absent in current files | Existing evidence reused; **NOT FIXED**; no repeated GPU diagnostic |
| Independent semantics | Existing 32-source blind kits; 0 genuine labels | Deliver A/B packs; independent answer-validity and instruction-following judgments, then adjudication; never auto-fill labels | Human **BLOCKED**; descriptive SSIM can proceed independently |
| LPIPS | No calibrated implementation/weights | Keep null; verify pinned implementation and preprocessing before measurement | **BLOCKED dependency**; no substitute/download |
| Untouched evaluation | Internal source/hash separation but prior exposure; prospective IDs not content-verified | Freeze protocol and verify source/content/near-duplicate/exposure admission before outputs; acknowledge foundation-model exposure uncertainty | **BLOCKED admission**; no new held-out result |
| Full pretrained four-arm benign learning | Needs reliable replay, valid task/reward and independent evaluation | **Not launched**. Local VJP/one replay insufficient; no threshold change; objective revision requires explicit versioned scientific decision | **NO-GO**, 0 GPU reserved |

Scientific coefficients and reward-release guard remain unchanged. Legacy 500-row evidence retains S=A/2, U=1, C=1/3, 230/500 floor-censored rewards and identity preference. Publication tables do not repair these issues. Post-training ASR remains **NOT TESTED/null**.

## Reproduction and evidence

Primary run: results/baseline_comparison_20261101T2255Z_v1/.

- protocol.json, submission_intent.json, submission_reply.json, input_hashes.json, checks.json, cpu-88317.out and .err.
- real_image/per_source_recomputed.csv and summary.json; raw results/matched_feedback_20261028_v2/main_repair_v1/predictions.jsonl.
- ablation/recomputed.json; publication_supplement/protocol.json and receipt.json.
- scripts/build_baseline_comparisons_20261101.py and scripts/finalize_baseline_publication_20261101.py. Use fresh output directories; prior root-report versions are archived.
- BF16: results/next_scientific_20261101T1905Z_v1/ and results/overnight_supervisor_20261101_v1_P1_ABBA_v1_a0/. No threshold relaxation.

The existing unique two-hour CPU Slurm successor is retained. Latest job/heartbeat/budget evidence is saved alongside the publication receipt; no new periodic workflow or cloud dependency was installed.
