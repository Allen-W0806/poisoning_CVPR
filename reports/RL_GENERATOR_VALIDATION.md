# Generator versus controller RL

**Reward-based generator training is BLOCKED.** Contract execution: `EXP_20261027T051832_ba75b1635cca_phase2_generator_rl_contract`. InstructPix2Pix's native scheduler/CFG path has no validated trainable trajectory log-probability implementation in this project. A DDPO-style stochastic transition density, per-step likelihood/reference agreement, masking and gradient tests are missing. No arbitrary categorical REINFORCE update was applied to diffusion latents.

Actual generator LoRA updates are supervised denoising, documented in GENERATOR_TRAINING_REPORT.md. They do not count as generator RL.

Separate controller RL evidence: `EXP_20261027T053938_b24f177856e1_phase2_rl_rsi_validation`. A categorical policy samples one of three benign instructions for a **frozen real editor**, with REINFORCE `-mean(log p(action|image) * detached sample reward)` and AdamW. Real outputs and Qwen/proxy measurements are precomputed for a fixed source/seed/action mapping. This is a valid stochastic categorical bandit, not on-policy diffusion trajectory learning or generator adaptation.

Each observation retains semantic proxy S, answer consistency A, distortion D, resource C and invalid/unverified penalty. The existing reward clips components [0,1] and totals [-1,1.5]; unknown semantic validity remains penalized. C is 10 steps / 30-step fixed reference, not a fabricated wall-clock measure. Actual generation/Qwen/selection runtime is charged separately. No held-out test or security metric enters training.

Frozen action-0 editor, random actions, trained controller and RSI-selected controller are measured on the same validation table. The unsupported reward-trained-generator arm is explicitly absent. No ASR, trigger search or evasion optimization was implemented.
