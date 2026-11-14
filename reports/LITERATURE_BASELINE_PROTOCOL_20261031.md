# Literature baseline protocol audit — 2026-10-31

This is a primary-source protocol comparison, not an attack reproduction. No published attack-success number is a measurement of this project. No attack optimizer, poisoning recipe or adversarial reward was implemented. Unknown counts and unverified artifacts remain explicitly unknown.

The PMLR publication pages for CBV and TokenSwap were accessible. The supplied CVF pages for ASO/BPGO returned HTTP403; author arXiv manuscripts were read instead, so identity with the final conference PDFs is not certified. IAG was checked against its author repository and manuscript. Access date: 2026-10-31.

## BadSem — training-time VQA

[Primary manuscript, arXiv2506.07214v1](https://arxiv.org/html/2506.07214v1), experimental setup and Table2.

|Field|Verified protocol / limitation|
|---|---|
|Targets|LLaVA1.5-7B, Qwen2-VL2B/7B, Llama3.2-Vision11B; vision encoder plus language decoder|
|Data|VQAv2/GQA; per dataset5,000 clean fine-tuning triplets,2,000 clean validation examples|
|Semantic subsets|SIMBad Table2 distinguishes consistent versus inconsistent color/object cases and textual versus visual interventions; their counts are separate from clean samples|
|Changed fields|Textual question or image semantics, and attack-target answers; not uniformly image-only or clean supervision|
|Label validity|Inconsistent inputs/target answers cannot be assumed to preserve original VQA truth|
|Unit / metrics|Clean accuracy; overall success per image-question with any candidate success differs from success per individual trigger attempt|
|Fidelity|Semantic interventions; no universal image-distance certificate identified|
|Compute / repeats|Single L40S; three epochs; independent training-seed repetitions and GPUh not established in inspected setup|
|Assets|Paper describes SIMBad construction; verified downloadable complete evaluation package not established here|
|Compatibility|Relevant training-time VQA motivation, but its full multimodal intervention is not an image-only, unchanged-label control|

## CBV — training-time VQA and captioning

[ICML2026 publication](https://proceedings.mlr.press/v306/guo26q.html), [author manuscript2605.02202v1, §5.1–5.5](https://arxiv.org/html/2605.02202).

|Field|Verified protocol / limitation|
|---|---|
|Targets|LLaVA1.5-7B, MiniGPT-v2, InstructBLIP7B, Qwen2.5-VL7B|
|Data / sizes|MSCOCO captioning and VQAv2; exact train/test record counts not specified in retrieved experimental setup; do not substitute full-dataset population sizes|
|Changed fields|Generated image; nominal original textual supervision retained under a clean-label claim|
|Label validity|Unchanged answer strings alone do not independently establish semantic correctness after image generation|
|Evaluation|Target-word occurrence per triggered example; clean caption BLEU4/CIDEr and VQA accuracy separately|
|Visual measurements|Naturalness and perturbation/computational-overhead analyses are separate from task accuracy; no human semantic certificate inferred|
|Architecture / cost|CLIP surrogate and SD1.4 generator; target training remains separate. Total GPUh and independent seed count not established here|
|Assets|Publication/manuscript accessible; complete code/evaluation-asset availability not verified in this audit|
|Compatibility|Closest image-only nominal-clean-label VQA comparison, requiring independent label preservation checks and actual post-training evaluation|

## TokenSwap — training-time caption compositionality

[ICML2026 publication](https://proceedings.mlr.press/v306/zhang26fr.html), [author manuscript2509.24566v2, §3 and AppendicesB/C](https://arxiv.org/html/2509.24566).

|Field|Verified protocol / limitation|
|---|---|
|Models|BLIP2 OPT2.7B, InstructBLIP7B, LLaVA1.5-7B/13B, Qwen2.5-VL7B|
|Data|Flickr8k8,000 images; Flickr30k31,783; MSCOCO over120,000; these are population sizes, not experimental split counts|
|Selected set|3,000 filtered image-caption pairs described in AppendixC; exact clean evaluation counts not established here|
|Changed fields|Image trigger and subject/object roles in answer captions; labels intentionally altered|
|Validity|Object names may remain correct while relations become false; not label-preserving VQA supervision|
|Evaluation|Compositional-error judgments; clean ROUGE1/ROUGEL/BLEU; evaluation information supplied to judge changes detectability|
|Fidelity / cost|Visual patch intervention; complete perceptual-distance reporting, total GPUh and independent seed count not verified|
|Assets|Paper and publication accessible; complete evaluation package not verified here|
|Compatibility|Captioning/compositional backdoor evidence; incompatible with an image-only VQA pipeline that preserves official answers|

## ASO — inference-time jailbreak enhancement

[Author manuscript2607.21619v1, §5.1](https://arxiv.org/html/2607.21619), [official repository](https://github.com/bingjunluo/ASO).

|Field|Verified protocol / limitation|
|---|---|
|Targets|Qwen3-VL, LLaVA-OneVision1.5, GPT4.1/GPT4.1-mini, Gemini2.5/Gemini2.5-Flash; exact versions differ between prose and table headings|
|Data / splits|MM-SafetyBench and VLBreakBench with existing attack inputs; exact train/development/evaluation counts not established in inspected manuscript|
|Modified fields|Style-edited attack image; harmful prompt supplied to a fixed target|
|Training distinction|An editing policy is trained; target victim fine-tuning is not the evidence being measured|
|Labels|No ordinary clean VQA reference-answer protocol; content/style preservation is not independently verified VQA truth|
|Evaluation|HarmBench binary judgments and harmfulness scores per response; distinct from clean VQA accuracy and training-time backdoor activation|
|Fidelity / compute|Stylistic preservation claims; complete LPIPS/SSIM, GPUh and independent replication counts not established here|
|Assets|Author repository provides model/evaluation workflow; not executed or imported|
|Compatibility|Inference-time jailbreak, not a training-time clean-victim VQA baseline; ASR must not become this project's generator reward|

## BPGO — benign visual-generation preference learning

[Author manuscript2511.18919v1, §4 and AppendixB](https://arxiv.org/html/2511.18919).

|Field|Verified protocol / limitation|
|---|---|
|Models|Wan2.1-1.3B T2V, Wan2.2-14B I2V, FLUX T2I; diffusion generators, not a VQA target|
|Data|Video iStock10,000 training/100 test samples; T2I DanceGRPO training data and HPSv2 evaluation3,200 prompts|
|Fields / supervision|Prompts/conditioning images and generated outputs; no VQA question/answer-label intervention. I2V removes an object from the conditioning first frame|
|Evaluation|Video alignment/preference and VBench dimensions; not VQA accuracy or backdoor ASR|
|Visual measures|Generation alignment and consistency metrics; not independent confirmation of original VQA answers|
|Compute|200 steps; stated T2V32H100, I2V16H100, T2I8H100. Wall-time/GPUh and independent training repetitions not established|
|Assets|Inspected manuscript accessible; complete release/evaluation-asset availability not verified here|
|Compatibility|Relevant reward-uncertainty methodology only. Production DDPO-SF in this project must not be relabeled BPGO/GRPO|

## IAG — training-time visual grounding

[Official MIT repository](https://github.com/lijunxian111/IAG), [author manuscript2508.09456, §4 and AppendicesE/N/O](https://arxiv.org/html/2508.09456).

|Field|Verified protocol / limitation|
|---|---|
|Models|LLaVA1.5-7B, InternVL2.5-8B, Ferret7B; text-conditioned UNet input module|
|Image splits, train/validation/test|RefCOCO16,994/1,406/(715+691); RefCOCO+16,992/1,406/(715+689); RefCOCOg21,899/816/1,580; Flickr30kEntities28,475/941/921; ShowUI-Web7,581/300|
|Changed fields|Image plus target grounding-box supervision. AppendixN VQA extension changes answer supervision; not clean-label VQA|
|Validity|Attacker-selected boxes/answers differ from original requested truth|
|Unit / metrics|Grounding expression and IoU0.5 success; benign versus clean accuracy; ShowUI uses predicted-center containment|
|Fidelity|L1, LPIPS, PSNR; these do not certify semantic validity|
|Compute / repetitions|RTX A6000; about2,000 training steps. Total GPUh/independent seeds not established; ten-question timing sample is inference timing, not training replication|
|Assets|Author code and demo assets released; complete benchmark images remain external|
|Compatibility|Grounding is not ordinary VQA; its VQA appendix uses changed target answers and is incompatible with this benign pilot|

## Application to the present validation

Keep image controls, clean supervised VQA, mathematical RL and synthetic RSI in distinct result tables. Any unreviewed edit remains ambiguous. A frozen-model answer change is an inference observation, not a training effect. An engineering PASS, published result or perceptual score does not establish attack efficacy. Attack-specific reproduction and current-project post-training ASR are **NOT PERFORMED**.
