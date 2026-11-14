# Method implementation audit

Snapshot 2026-11-01T05:16:56.482704+00:00. Canonical root `/storage/home/ydk297/projects/poisoning`. **Scientific gate NO-GO.** No manuscript, research objective, scientific loss, threshold or reward-release guard was changed.

## Artifacts actually read

Latest project-resident manuscript sources are output/cvpr2027_visuals/source/manuscript_original.tex and manuscript_integrated.tex; compiled manuscript_preview.pdf has7 pages. Algorithm1 is the existing conceptual algorithm1.tex/pdf. The later Figure1 is output/rsi_figure1_final_20261031/RSI_Figure1_Final.pdf/png; it was visually inspected and its text/spec read. Exact copies/hashes are under results/method_code_audit_20261101T0510Z_v1/audit/supplied_method_snapshot/. PDF extraction is retained as audit/pdf_readback.json. No unseen external manuscript version is assumed.

These artifacts are not one coherent current implementation specification. New Figure1 uses one Gpsi/pipsi, but Algorithm1 explicitly retains separate piphi/Gpsi. The manuscript says an editor is not yet committed in Problem Setup yet names IP2Pix in its setup, and suggests GRPO while production executes DDPO-SF. The latest artwork has not been integrated into the existing compiled preview. This audit identifies discrepancies without rewriting them.

## 1. Generator and trainable parameters

**Source/function:** `src/training/generator_adapter.py:5` (`add_attention_lora`); `src/rl/diffusion_policy.py:26` (`__init__`)
**Existing experiment/receipt:** `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL; method_code_audit_20261101T0510Z_v1 /88083,88084`.
**Observed:** Actual production editor is cached InstructPix2Pix31519b5cb02a7fd89b906d88731cd4d6a7bbf88d. editor.pipe and policy.pipe share the same UNet. One trainable vector psi means one collection of256 FP32 LoRA A/B tensors, not one tensor:1,594,368 elements in128 attention linear modules. Rank8, alpha16. Targets to_q/to_k/to_v/to_out.0 cover self/cross attention. Frozen base UNet, VAE and text encoder are separate from this vector. Qwen clean-adapter1,605,632 elements/112 tensors and tiny-UNet4,096 elements are separate experiments, never generator parameter counts.
**Unresolved:** The latest Figure1 single Gpsi/pipsi agrees with this implementation; Algorithm1 separate phi/psi and manuscript abstract decision policy require notation/implementation reconciliation.

## 2. Actual RL algorithm

**Source/function:** `src/rl/diffusion_rl_trainer.py:27` (`update`); `src/rl/reward_normalization.py:4` (`fixed_advantage`); `src/rl/diffusion_rl_trainer_v2.py:20` (`update`)
**Existing experiment/receipt:** `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL; method_code_audit_20261101T0510Z_v1 vjp_probe/summary.json`.
**Observed:** Executed core is single-pass on-policy DDPO score-function (DDPO_SF_v1). Gaussian DDIM transition actions, detached terminal coefficient, coordinate/time/trajectory averaging, AdamW, norm cap1. No PPO ratio/surrogate clipping, no KL/reference penalty, no group mean/std normalization. Separate categorical trainer is REINFORCE. Current Phase5 guarded wrapper delegates to the core only after receipt validation, but all unreleased scientific rewards are denied.
**Unresolved:** GRPO is only a manuscript baseline suggestion and an isolated reference check; it is not the executed production algorithm. No delayed post-training security credit assignment is implemented.

## 3. Instruction versus question

**Source/function:** `src/rl/diffusion_policy.py:52` (`condition`); `src/rl/diffusion_policy.py:61` (`predict`); `scripts/smoke_generator_rl.py:34` (`collect`)
**Existing experiment/receipt:** `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL; matched_feedback_20261028_v2/main_repair_v1`.
**Observed:** instruction is a distinct argument, fixed to the benign slight-lighting command for historical production runs. condition encodes the image and instruction, not record.question. The unchanged VQA question is sent separately to Qwen. Stochastic actions are latent transitions, not learned textual instructions.
**Unresolved:** Manuscript state includes q/history/budget and separate abstract decisions; these are not all actual generator conditioning variables. Do not silently add question conditioning or instruction optimization.

## 4. Original answers and validity

**Source/function:** `src/data/semantic_contract_v2.py:19` (`from_original`); `src/evaluation/semantic_validation.py:27` (`semantic_decision`); `src/evaluation/semantic_validation_v2.py:22` (`decision_v2`)
**Existing experiment/receipt:** `method_code_audit_20261101T0510Z_v1 original_fields_verified.json; next_validation_20261101T0040Z_v1/semantic_review32`.
**Observed:** 1040 rows independently matched official cached question/image join and all ten original answers:80 latest-split records plus960 matched candidate-condition records. Original reference answers remain unchanged. Phase5 keeps edited_factual_answer UNKNOWN rather than copying an inferred factual answer. Verified pixel identity is a narrow exception.
**Unresolved:** 0 independently resolved edited pairs. A retained original label is not a verified edited-image factual label; same-Qwen proxy agreement is not independent review.

## 5. Metric unit and normalization

**Source/function:** `src/rl/reward.py:14` (`validation_reward`); `src/evaluation/vqa_official_compatible.py:34` (`score_answer`); `src/evaluation/security_metrics.py:9` (`security_metrics`)
**Existing experiment/receipt:** `EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL; matched_feedback_20261028_v2; audit_remediation_20261101T0120Z_v1_threeway_repair_v1`.
**Observed:** Per-image/pair semantic and VQA diagnostics become terminal trajectory coefficients in legacy DDPO. Loss averages transitions and latent coordinates; batch reward mean is logging, not a shared per-image reward. Dataset and post-training outcomes have different units and denominators; see METRIC_AND_NORMALIZATION_AUDIT.md.
**Unresolved:** ASR, dataset-level paired fine-tuning outcomes, per-image agreement and cost proxies must not be substituted for each other.

## 6. Actual RSI revisions

**Source/function:** `src/rsi/phase4_schedule.py:14` (`revise_schedule`); `src/rsi/controller.py:21` (`revise`); `src/rsi/revision.py:6` (`validate_revision`); `src/rsi/harness/revision_proposer.py:4` (`propose`)
**Existing experiment/receipt:** `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI; EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI; audit_remediation_20261101T0120Z_v1_threeway_repair_v1; small_scale_validation_20261031T1928Z_v1/synthetic_rsi`.
**Observed:** Historical natural-image Phase4 executes one validation-driven LR cooldown after update4,1e-5 to5e-6. Generic controller allows batch_size10/20/32, learning_rate<=1e-3, uniform/easy_first curriculum; this is API capacity, not evidence all settings were exercised. New tiny comparison executes LR proposals, own-prefix rollback and persistent versions. Separate deterministic integrity harness actually revises workflow checks by adding content_hash; it cannot rewrite objectives or generator weights.
**Unresolved:** Additional workflow revision exists in a synthetic engineering harness, not demonstrated autonomous natural-image policy/curriculum improvement. Manuscript broad Hk components exceed executed generator changes.

## 7. Preintervention comparability

**Source/function:** `scripts/run_phase4_rsi.py:55` (`main`); `scripts/benign_threeway_20261101_v2.py:25` (`main`)
**Existing experiment/receipt:** `EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI; EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI; audit_remediation_20261101T0120Z_v1_threeway_repair_v1`.
**Observed:** Historical step4 before LR intervention already differs: adapter maxabs4.34328e-5(seed42),6.64559e-5(seed123), despite same initial parameters/order/seeds. This invalidates causal attribution of old final gains. New FP32 tiny comparison independently checks full parameters, optimizer/scheduler, CPU/CUDA/NumPy/Python RNG, last gradients, contract/version/consumed IDs and predictions before adaptation;12 saved initial/prefix cross-branch comparisons pass.
**Unresolved:** The new tiny repair cannot retroactively certify historical pretrained BF16 comparability. Configuration/contract differences during deliberate LR revision are expected and must be versioned.

## 8. Data overlap and exposure

**Source/function:** `src/data/dataset.py:49` (`validate_disjoint`); `results/method_code_audit_20261101T0510Z_v1/actual_source_hash_recheck.json` (CPU artifact)
**Existing experiment/receipt:** `phase45_recompute.json; method_code_audit_20261101T0510Z_v1 data_versions_verified.json; audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json`.
**Observed:** v1 image-ID disjoint train/test still shares one exact content hash. v2 content split999/104/96/301 is internally ID/hash-disjoint, matching the manuscript counts, but is development-exposed. Latest32/16/32 split is internally disjoint yet all80 exposure flags are true; assessment32 includes4 earlier inference sources and10 old opened-test sources. Old synthetic_v1 also shares exact rendered content across nominal splits. Actual source bytes were rehashed in this run.
**Unresolved:** No genuinely unused natural-image final test is certified. Internal content separation is not historical feedback isolation or proof of absence from foundation-model pretraining. Training-manifest overlap alone does not prove each source was used in a gradient.

## Exact pretrained optimizer ownership

{
  "source": "results/small_scale_validation_20261031T1928Z_v1_ddpo/pretrained_IP2Pix_optimizer_groups_ZERO_STEPS.json",
  "sha256": "a5068482123588314c0dcfc687f5a205648efb18599ebab8ebf9f1b46bfef44f",
  "trainable_tensors": 256,
  "trainable_elements": 1594368,
  "modules": 128,
  "target_counts": {
    "to_q": {
      "tensors": 64,
      "elements": 399360,
      "modules": 32
    },
    "to_k": {
      "tensors": 64,
      "elements": 397824,
      "modules": 32
    },
    "to_v": {
      "tensors": 64,
      "elements": 397824,
      "modules": 32
    },
    "to_out.0": {
      "tensors": 64,
      "elements": 399360,
      "modules": 32
    }
  },
  "optimizer_groups": [
    {
      "index": 0,
      "lr": 1e-05,
      "weight_decay": 0.0,
      "betas": [
        0.9,
        0.999
      ],
      "eps": 1e-08
    }
  ],
  "only_LoRA": true,
  "parameter_order": [
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.0.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "down_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.1.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.2.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.1.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "up_blocks.3.attentions.2.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_q.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_k.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_v.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn1.to_out.0.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_q.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_k.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.lora_B.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_A.default.weight",
    "mid_block.attentions.0.transformer_blocks.0.attn2.to_out.0.lora_B.default.weight"
  ],
  "scope": "actual pretrained IP2Pix optimizer inventory; not Qwen adapters or random tiny UNet",
  "rank": 8,
  "alpha": 16
}

One AdamW group: lr1e-5, betas(.9,.999), eps1e-8, weight_decay0; constant LambdaLR1. Max gradient norm1. Diffusion base compute BF16, FP32 states/guidance/adapters, math SDPA and sequential CFG. Effective diagnostic overrides (batch2/steps2/checkpointing true) are distinct from the default production batch10/steps10/checkpointing false, and are recorded rather than silently merged.

## Scope labels

| Observation | Engineering | Actual model update | Task learning | Scientific improvement |
|---|---|---|---|---|
| New CPU provenance/field/parameter audit | PASS | None | NOT TESTED | NOT VALIDATED |
| Local VJP numerical probe | PASS local only | 256 LoRA tensors changed once | NOT VALIDATED: constant coefficient | NOT VALIDATED |
| Existing clean Qwen3 seed study | PASS updates/reload | 112 tensors per seed | No reliable clean-score gain | NOT VALIDATED; development-only |
| Existing tiny three-way cosine comparison | PASS controlled engineering | Real tiny LoRA updates | Differences compatible with zero | No RSI advantage established |
| Four-arm real image-edit comparison | BLOCKED | Not launched | NOT TESTED | NOT VALIDATED |

## Direct historical checkpoint/RNG follow-up

A distinct read-only CPU task compared the preserved initial and pre-LR-step4 checkpoints directly. Evidence: results/method_code_audit_20261101T0510Z_v1/historical_prefix_state_comparison.json. Matching data order and RNG do not imply matched model/optimizer trajectories.

{
  "status": "COMPLETE_HISTORICAL_DIFFERENCES_PRESERVED",
  "rows": [
    {
      "seed": 42,
      "step": 0,
      "baseline": "EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": true,
        "optimizer": true,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 0.0,
      "different_optimizer_moment_tensors": {
        "exp_avg": 0,
        "exp_avg_sq": 0
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL/checkpoints/step_0000/training_state.pt": "6b2446ae5b14b4033a493beafc6d661e940cfacb990d563c72ef35a0c45bf31c",
        "experiments/EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI/checkpoints/step_0000/training_state.pt": "d47bbb1a322743d0393dcd2c06590dc3eeacdd232898f0b99e4bdcbfa6e14b8d"
      }
    },
    {
      "seed": 42,
      "step": 4,
      "baseline": "EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": false,
        "optimizer": false,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 4.343278851592913e-05,
      "different_optimizer_moment_tensors": {
        "exp_avg": 256,
        "exp_avg_sq": 256
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL/checkpoints/step_0004/training_state.pt": "18c82539a230ddcbe5d82028ac69b903052bfded9d77c21c381719e3768162bc",
        "experiments/EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI/checkpoints/step_0004/training_state.pt": "2a77e297fdd793c6c1456f1a76201c412c250ba1f530c99e21a83bcd68ad2348"
      }
    },
    {
      "seed": 123,
      "step": 0,
      "baseline": "EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": true,
        "optimizer": true,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 0.0,
      "different_optimizer_moment_tensors": {
        "exp_avg": 0,
        "exp_avg_sq": 0
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL/checkpoints/step_0000/training_state.pt": "5d6fcefab465d7d0e275eb5497425dff804438da1f1056b56c9f796e7e0f6cdc",
        "experiments/EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI/checkpoints/step_0000/training_state.pt": "e15d426b3f7b375e4990e140ea7c1082cca9e2b44158cb7e782f0d4f5f32be1e"
      }
    },
    {
      "seed": 123,
      "step": 4,
      "baseline": "EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL",
      "rsi": "EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI",
      "fields_equal": {
        "adapter": false,
        "optimizer": false,
        "scheduler": true,
        "step": true,
        "version": true,
        "contract": false,
        "consumed": true,
        "rng_torch": true,
        "rng_cuda": true,
        "rng_numpy": true,
        "rng_python": true
      },
      "adapter_max_abs_difference": 6.645589019171894e-05,
      "different_optimizer_moment_tensors": {
        "exp_avg": 256,
        "exp_avg_sq": 256
      },
      "baseline_lr": 1e-05,
      "rsi_lr": 1e-05,
      "first_four_image_order_and_rollout_seed_draws_equal": true,
      "source_sha256": {
        "experiments/EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL/checkpoints/step_0004/training_state.pt": "de7023da79fe70f116974b30ad3a20d0c28c636db283c3bcdf7dedfdbd7f6ddb",
        "experiments/EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI/checkpoints/step_0004/training_state.pt": "1f3d1a0b886b47f86d392e936fa6c94a21ae669e717c7d26e2ae0a95cb33e055"
      }
    }
  ],
  "scope": "saved checkpoints step0 and step4 before LR intervention; matching RNG/order is not proof of equal backward execution"
}

## Final execution and claim boundary

{
  "run_id": "method_code_audit_20261101T0510Z_v1",
  "updated_UTC": "2026-11-01T05:25:12.799177+00:00",
  "scientific_gate": "NO-GO",
  "jobs": [
    {
      "job_id": "88083",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:08:44",
      "end_EDT": "2026-11-01T01:08:55",
      "elapsed_seconds": 11,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "88084",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:10:02",
      "end_EDT": "2026-11-01T01:10:37",
      "elapsed_seconds": 35,
      "gpu_seconds": 35,
      "exit_code": "0:0"
    },
    {
      "job_id": "88085",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:16:53",
      "end_EDT": "2026-11-01T01:16:56",
      "elapsed_seconds": 3,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    },
    {
      "job_id": "88090",
      "state": "COMPLETED",
      "start_EDT": "2026-11-01T01:20:32",
      "end_EDT": "2026-11-01T01:20:34",
      "elapsed_seconds": 2,
      "gpu_seconds": 0,
      "exit_code": "0:0"
    }
  ],
  "budget": {
    "cap": 3600,
    "prior": 2246,
    "new": 35,
    "consumed": 2281,
    "reserved": 0,
    "remaining": 1319,
    "scope": "validation-only subtotal including all earlier failures, not lifetime whole-project cost"
  },
  "local_VJP": "PASS_LOCAL_VJP_ONLY",
  "full_graph_BF16_restart": "NOT VALIDATED",
  "independent_edited_labels": 0,
  "actual_task_learning": "NOT VALIDATED",
  "scientific_improvement": "NOT VALIDATED",
  "four_arm_image_training": "BLOCKED_NOT_LAUNCHED",
  "monitor": {
    "method": "slurm_cpu_successor",
    "last_success_UTC": "2026-11-01T04:32:51.929261+00:00",
    "next_job_id": "88064",
    "next_due_UTC": "2026-11-01T06:32:35+00:00",
    "scheduler_evidence": "88064|PENDING|2026-11-01T01:32:35|poisoning-monitor-v1:f916f24e0de6:ec957b65b56e40c1b1d83b61080d5d4b",
    "duplicate_created": false
  },
  "tensor_integrity_files": 27
}

Direct initial checkpoint contract values are retained in results/method_code_audit_20261101T0510Z_v1/historical_initial_contracts.json. Both seeds initially share parameters, optimizer, scheduler and RNG. Before LR intervention at step4, all256 exp_avg and256 exp_avg_sq tensors differ, although saved RNG, data order, rollout seeds and LR match. This is a confounded historical comparison, not evidence of RSI benefit.

## Figure1 / Algorithm1 execution coverage

| Stage | Execution evidence | Remaining boundary |
|---|---|---|
| Shared generator/policy | Actual pretrained IP2Pix LoRA and DDPO-SF | Algorithm1 separate phi/psi is not reconciled |
| Original q/y retained | Official cached join verified1040 records | Edited factual labels require independent review |
| External victim training | Clean Qwen LoRA engineering replication | Matched modified/clean victim lineage not established |
| Delayed development feedback | Metric interface and conceptual diagram | Post-training credit to generator not implemented/released |
| RSI revision | Historical LR cooldown; separate workflow-check harness | No demonstrated natural-image recursive scientific gain |
| Untouched final test | Diagram requirement | Current real-image data exposed; no independent final claim |

## Retained reporting failures and bounded repair

CPU88093 and88094 failed the final dashboard-contract assertion: legacy Latest HTML used an implicit body and had no embedded JSON. Chinese/English data already matched. The passive renderer now makes that existing body explicit and adds the authoritative JSON/banner; the original payload-equality/unique-banner assertions pass. Evidence: results/method_code_audit_20261101T0510Z_v1/dashboard_data_QA.json. This is an interface repair; no numerical test, reward or threshold was changed.

## Final scheduler receipt and initial contract explanation

Initial contract differences for both seeds are metadata config.arm=rl versusrsi only; initial optimizer/RNG/adapter states match. Step4 optimizer/adapter differences precede LR intervention and remain a scientific confound.

| Job | State | Start EDT | End EDT | Seconds | Exit |
|---|---|---|---|---|---|
|88083|COMPLETED|2026-11-01T01:08:44|2026-11-01T01:08:55|11|0:0|
|88084|COMPLETED|2026-11-01T01:10:02|2026-11-01T01:10:37|35|0:0|
|88085|COMPLETED|2026-11-01T01:16:53|2026-11-01T01:16:56|3|0:0|
|88090|COMPLETED|2026-11-01T01:20:32|2026-11-01T01:20:34|2|0:0|
|88093|FAILED|2026-11-01T01:25:10|2026-11-01T01:25:13|3|1:0|
|88094|FAILED|2026-11-01T01:26:24|2026-11-01T01:26:25|1|1:0|
|88095|COMPLETED|2026-11-01T01:27:24|2026-11-01T01:27:25|1|0:0|

The two failures are reporting-interface jobs only;88095 passes the unchanged HTML/data assertions. All model evidence is unchanged, and no new GPU job is running.
