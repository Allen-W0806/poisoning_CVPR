"""Render Phase4.5 audit deliverables from preserved CPU receipts and source anchors.

This creates new audit reports only. Findings are audit judgments, not automatic
scientific corrections or permission to submit training.
"""
import json
from pathlib import Path
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports'
def read(p):return json.loads((ROOT/p).read_text(encoding='utf-8-sig'))
receipt=read('reports/phase45_recompute.json')
results=read('reports/phase4_results.json')
monitor=read('reports/phase45_monitor_snapshot.json')
def ref(path,needle=None):
    p=ROOT/path;line=1
    if needle:
        lines=p.read_text(encoding='utf-8').splitlines()
        line=next(i for i,s in enumerate(lines,1) if needle in s)
    target='H:/Project/Poisoning/'+path
    return f'[{path}:{line}]({target}:{line})'
def run(name):return f'`{name}`'
ids={
 'B':'EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR',
 'R42':'EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL',
 'R123':'EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL',
 'S42':'EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI',
 'S123':'EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI',
 'D':'EXP_20261027T164820_1b8b6e4a33e5_phase4_PH4_RESTART_DIAGNOSTIC',
 'DA':'EXP_20261027T165452_2d038326075f_phase4_PH4_RESTART_STATE_AUDIT',
 'O':'EXP_20261027T175526_5aee05413ae5_phase4_PH4_RESTART_OUTCOMES',
 'H':'EXP_20261027T165331_14e29d03962b_phase4_PH4_SEMANTIC_REVIEW',
 'P3':'EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke',
 'T':'EXP_20261027T052245_7a312217676f_phase2_qwen_baseline_test',
 'SEC':'EXP_20261027T035119_0e11fe79ae3d_read_only_security_eval'}
F=[]
def finding(key,level,title,sources,experiments,observed,correction,status='VERIFIED'):
    F.append(dict(id=key,classification=level,title=title,sources=sources,
        experiments=[ids[k] for k in experiments],evidence_status=status,observed=observed,proposed_correction=correction))
finding('F01','CRITICAL','The implemented estimand is benign preservation, not BadSem reproduction',
 [ref('src/rl/diffusion_policy.py','INSTRUCTION ='),ref('scripts/smoke_generator_rl.py','def collect'),ref('docs/METHOD.md','The [BadSem paper]')],
 ['B','R42','R123','S42','S123'],
 'The policy receives an image and one fixed lighting instruction. The question is used by the reward, not as policy text conditioning. Qwen is frozen. No victim fine-tuning or real post-training backdoor evaluation occurs. Semantic changes that break proxy preservation reduce reward; therefore this objective opposes introducing question-relevant semantic inconsistencies. This is a mismatch if results are presented as completing the broader BadSem objective, rather than a bug in the declared benign pilot.',
 'Freeze and name the benign preservation estimand explicitly; separately specify any future research question and label-validity contract before training. No reward changes or poisoning implementation are authorized by this audit.')
finding('F02','HIGH','Reward provider and semantic verifier are the same model',
 [ref('scripts/smoke_generator_rl.py','def collect'),ref('src/evaluation/semantic_validation.py','def review_pair'),ref('src/models/qwen_loader.py','class QwenVQA')],
 ['B','R42','R123','S42','S123','H'],
 'Frozen Qwen supplies original answers, edited answers and pair judgments. All 500 observations remain independently unverified. Agreement between its two prompt formats is not independent verification; official original-image answers do not certify an edited image.',
 'Obtain blinded independent semantic labels and assess verifier errors on all relevant conditions. Keep proxy metrics and independent validity separate; do not tune against a purported independent evaluator that also supplies reward.')
finding('F03','HIGH','The reward collapses to one proxy gate with extensive floor censoring',
 [ref('src/rl/reward.py','def validation_reward'),ref('src/evaluation/semantic_validation.py','def semantic_decision'),ref('scripts/smoke_generator_rl.py','consistency["score"] if')],
 ['B','R42','R123','S42','S123'],
 'Recomputed 500/500 observations have U=1 and S=0.5*A_reward. For changed edits with gate G, R=clip(G-0.2D-1/30-1,-1,1.5). G=0 is always -1, discarding D and C differences. 165/320 training and 65/180 development observations are at -1. All observed rewards are negative. U dominates the level and clipping; proxy-gate flips dominate between-arm changes. U itself is constant and does not rank changed samples.',
 'Review construct definitions, uncertainty treatment, redundant terms and clipping analytically under a new versioned protocol. Preserve current coefficients and all raw/clipped values. No coefficient search or automated reward repair in this phase.')
finding('F04','HIGH','Instruction success is missing from the objective',
 [ref('src/rl/diffusion_policy.py','INSTRUCTION ='),ref('src/evaluation/image_quality.py','def image_quality'),ref('src/rl/reward.py','total =')],
 ['B','R42','R123','S42','S123'],
 'There is no explicit measure that lighting was slightly brightened. When the proxy gate passes (G=1), reducing D increases reward. When G=0, clipping makes reward flat at -1 regardless of D. Identity has S=1,U=0 and is strongly preferred. Thus preservation can improve by approaching no edit; reward improvement would not demonstrate instruction following.',
 'Before further training, define independently assessable task success alongside preservation, and document how no-op controls are interpreted. Do not silently add a brightness objective now.')
finding('F05','HIGH','Executed RSI is a validation-driven LR schedule extension',
 [ref('src/rsi/phase4_schedule.py','class Phase4ScheduleController'),ref('src/rsi/controller.py','def revise('),ref('scripts/run_phase4_rsi.py','controller.revise_schedule')],
 ['S42','S123'],
 'Both arms make one LR change 1e-5 to 5e-6 after update4. The extension records accepted_schedule but explicitly quality_improvement_accepted=false. It bypasses the legacy candidate-gain acceptance/rollback semantics. Persistent revision history is implemented, but ordinary adaptive scheduling and a novel recursive improvement mechanism are not experimentally distinguished.',
 'Describe this result as an exploratory validation-driven cooldown. Define the proposed RSI mechanism and compare with a predeclared ordinary scheduler/control before a novelty claim. Do not relabel historical acceptance as quality acceptance.')
finding('F06','CRITICAL','RSI causal benefit is confounded by divergence before intervention',
 [ref('scripts/run_phase4_rsi.py','trainer.save(run.path'),ref('scripts/build_phase4_reports.py','pre_schedule')],
 ['R42','R123','S42','S123'],
 'Initial adapters, data order and noise seeds match. Before either LR change, step4 adapters already differ by max 4.34328e-5 (42) and 6.64559e-5 (123). Seed123 step4 M2-M1 reward is +0.0932581, comparable to final +0.0937077. Both final paired intervals include zero. Each final +~0.094 is driven by one gate flip. Matching seeds therefore does not isolate the intervention.',
 'Withdraw causal RSI-gain claims. A future authorized design must establish comparable pre-intervention behavior and separate numerical execution variability from adaptation effects; preserve all pre-intervention evidence and negative outcomes.')
finding('F07','HIGH','Two seeds and ten repeatedly reused development images cannot establish generalization',
 [ref('scripts/run_phase4_budgeted.py','dev_indices=subset'),ref('src/evaluation/phase4_statistics.py','def paired_summary'),ref('src/evaluation/phase4_statistics.py','def rsi_plateau')],
 ['B','R42','R123','S42','S123'],
 'There are 180 validation trajectories but only 10 unique development images. Four looks per trained arm reuse them; M2 uses these same looks to revise LR. Percentile source bootstrap CIs are conditional on a fitted seed, without repeated-selection correction. Two seeds mix training and evaluation-noise variation. Treating 180 trajectories as independent images would be invalid.',
 'Keep these estimates exploratory. Predeclare a separate selection/evaluation design and source-clustered analysis before any new campaign; do not reopen the old held-out subset to rescue this comparison.')
finding('F08','HIGH','ID-disjoint splits contain an exact train/test image duplicate',
 [ref('src/data/dataset.py','def validate_disjoint'),ref('src/data/prepare_vqa.py','def select_subsets'),ref('data/manifests/cluster/vqav2_seed42_v1/train.json','397819015'),ref('data/manifests/cluster/vqav2_seed42_v1/test.json','388662037')],
 ['T','R42','R123','S42','S123'],
 'Canonical image hashing verifies train image397819/sample397819015 and test image388662/sample388662037 share SHA256 fcc1fd100ed50e30ea87b9d84597cb40876c76d955716fc9dd6e936787faf181. The questions differ. All split image IDs are disjoint, but bytes are not. No exact-byte train/validation overlap exists. Image397819 was not used in any of the four Phase4 training arms; no current Phase4 train/dev contamination is established. Frozen Qwen clean-test evaluation did not train on this internal train split, so this is not proof its historical accuracy was inflated.',
 'For a future dataset version, group exact duplicates before assigning splits and audit near duplicates separately. Preserve v1 and the already-opened test marker; restrict v1 independence claims. Do not silently delete a test record or recompute a favorable score.')
finding('F09','HIGH','Exact stochastic checkpoint continuation remains unaccepted',
 [ref('src/rl/diffusion_rl_trainer.py','def restore'),ref('scripts/diagnose_phase4_restart.py','restoration='),ref('scripts/audit_phase4_restart.py','rng_scope'),ref('scripts/phase4_restart_outcomes.py')],
 ['P3','D','DA','O'],
 'Fixed-trajectory replay has exact checked restored fields and exact behavior likelihoods, but backward gradients and post-update adapters/optimizer states differ. Adapter max differences are 2.87986e-6 and 2.86177e-6. Two outcome sources produce unequal RGB images (max channel differences 103/37/93/40). The fixed-trajectory replay bypasses ten rollout seed draws; it does not establish full stochastic RNG continuation.',
 'Keep exact equality as the acceptance criterion. Permit no resume-equivalence claim until a separately authorized diagnostic closes it. No new numerical fix or threshold relaxation is made here.')
finding('F10','MEDIUM','Question-relative semantic checking is narrower than the global preservation instruction',
 [ref('src/evaluation/semantic_validation.py','def question_attributes'),ref('src/evaluation/semantic_validation.py','Compare image 1'),ref('src/rl/diffusion_policy.py','INSTRUCTION =')],
 ['B','R42','R123','S42','S123','H'],
 'The verifier asks only about question-relevant properties and explicitly allows irrelevant properties to be true. The instruction asks to retain all objects, colors, counts and positions. Unchanged answers cannot validate all image semantics, nor guarantee the original labels remain valid after a question-relevant change.',
 'Specify separate question-answer validity, global semantic preservation and instruction-following rubrics for a future protocol; keep current labels diagnostic until verified.')
finding('F11','MEDIUM','C is a fixed denoising-step proxy, not measured compute cost',
 [ref('scripts/smoke_generator_rl.py','consistency["score"] if'),ref('src/rl/reward.py','cost_weight')],
 ['B','R42','R123','S42','S123'],
 'C=clip(10/30,0,1)=1/3 for every Phase4 trajectory, so -0.1C=-1/30. It excludes Qwen calls, model loading, diagnostics, validation and RSI bookkeeping. It cannot demonstrate compute savings. Slurm separately records actual total cost.',
 'Call C a normalized step-count proxy. Retain separate end-to-end Slurm accounting and define any future cost objective before comparing efficiency.')
finding('F12','MEDIUM','Verified identity loses its answer-consistency credit at the caller',
 [ref('src/evaluation/semantic_validation.py','Exact original-image identity'),ref('scripts/smoke_generator_rl.py','consistency["score"] if')],
 ['B','R42','R123','S42','S123'],
 'The verified-identity branch omits provisionally_valid; collect gates A solely on that key. With raw consistency1, identity is passed A=0 and R=0.9666667 (C=1/3), rather than 1.4666667 if the declared consistency component were credited. This is a definite branch inconsistency, not an observed cause of Phase4 results: all 500 current edits were nonidentity/unverified.',
 'Under a future versioned implementation correction, explicitly define eligibility for verified-valid and provisional cases and add an identity branch test. Do not rewrite any historical reward.')
finding('F13','MEDIUM','Exposure matching does not establish equal realized compute',
 [ref('scripts/submit_phase4.py'),ref('reports/phase4_sacct_final.log'),ref('reports/phase4_results.json','actual_gpu_seconds')],
 ['R42','R123','S42','S123'],
 'Each arm has 8 updates and 80 training exposures. M1/M2 elapsed GPU seconds are 3227/3211 for42 and 2707/3216 for123; nodes differ. No repeated same-node latency comparison was done. The entire campaign uses 13122 seconds, within 14400, including failures/diagnostics and evaluation.',
 'Report matched exposures plus actual cost, rather than strict compute equivalence or generation speedup. A future efficiency study needs a predeclared comparable hardware/timing protocol.')
finding('F14','MEDIUM','Saved images show suspected instruction-following and padding failures',
 [ref('src/editing/pretrained_editor.py','def prepare_image'),ref('src/evaluation/image_quality.py','def image_quality')],
 ['H','S123'],
 'Direct executor inspection of existing Phase3 contact sheet and Phase4 seed123 eval8 sources4/6 shows large warm/darker shifts, altered white padding and suspected shape/detail changes. Banana source6 and skating source4 are visibly not a simple uniform slight brightening. This is a selected qualitative review, not blinded human ground truth or a failure-rate estimate. Pixel scores include padding, which can affect D independently of semantic content.',
 'Keep these as suspected failures and obtain blinded content/padding/instruction labels. Retain full-frame measurements; any content-only metric must be added as a separately documented derived metric, not replace old D.')
finding('F15','MEDIUM','Checkpoint configuration revisions require reconstructing the exact active contract',
 [ref('src/rl/diffusion_rl_trainer.py','Checkpoint contract mismatch'),ref('scripts/run_phase4_rsi.py',"active_revision_experiment_id"),ref('scripts/run_phase4_rsi.py',"config_revision_step")],
 ['S42','S123'],
 'LR revision updates optimizer LR, scheduler base_lrs and policy contract and creates a child revision record. The parent manifest/config remains the initial version; subsequent checkpoints carry the active revision. Reconstructing a parent-only default policy would fail the contract check. No production resume across this revision was accepted or used.',
 'Document the active revision lineage and reconstruction procedure; retain immutable parent and child records. Validate it before any future authorized resume; do not treat an intentional contract rejection as successful continuation.')
finding('F16','LOW','The top-level method document describes an obsolete stage',
 [ref('docs/METHOD.md','`BenignEditor'),ref('docs/METHOD.md','preliminary_v1')],
 ['P3','R42'],
 'docs/METHOD.md still describes only a PIL brightness controller and preliminary normalization, while Phase3/4 use real UNet LoRA DDPO and the official-compatible scorer. Historical Phase4 reports are more accurate. This can confuse a new reader about which implementation generated the results.',
 'Add a versioned current-method entry point linking this audit and retain historical descriptions with explicit stage labels. No historical report has been edited during this audit.')
finding('F17','UNRESOLVED','The specific backward numerical divergence mechanism is not localized',
 [ref('scripts/diagnose_phase4_restart.py','fresh_repeat_gradient'),ref('scripts/audit_phase4_restart.py','first_observed_difference')],
 ['P3','D','DA'],
 'The earliest inspected difference is in backward gradients after exact forward log-likelihoods. That does not identify the first differing CUDA primitive, exclude hidden execution-state differences, or prove an unavoidable universal nondeterminism mechanism.',
 'Retain competing hypotheses and exact diagnostics. Only separately authorized controlled diagnostics may identify a root cause; do not invent a fix or declare all GPU determinism impossible.',status='UNRESOLVED')
finding('F18','UNRESOLVED','Independent semantic validity and visual improvement remain unknown',
 [ref('reports/PHASE4_SEMANTIC_REVIEW.md'),ref('experiments/'+ids['H']+'/review_form.csv')],
 ['H','R42','R123','S42','S123'],
 'The ten-pair portable review form is blank; all new training/development edits are also unverified. Inventory wording flags are not semantic error labels. No independent human/model agreement rate, valid-edit rate, or quality improvement can be computed.',
 'Complete a blinded review with explicit uncertainty and disagreement adjudication before scientific quality claims. Do not manufacture labels from the executor observations in this audit.',status='UNRESOLVED')
finding('F19','MEDIUM','Reward configuration metadata is not the executed coefficient source',
 [ref('configs/phase3_generator_rl.yaml','reward:'),ref('scripts/smoke_generator_rl.py','reward=validation_reward'),ref('src/rl/reward.py','weights or RewardWeights')],
 ['R42','R123','S42','S123'],
 'collect calls validation_reward without a weights object, so dataclass defaults, not the YAML reward block, determine executed coefficients. They numerically match this campaign. Several YAML names also differ from dataclass field names. Editing YAML alone would silently fail to change the actual reward.',
 'Before future protocol changes, choose one explicit validated coefficient source and assert it equals recorded metadata. Current calculations are correct for the defaults; no coefficient or scientific source is changed in this audit.')
finding('F20','MEDIUM','Runtime dataset guards are weaker than provenance audit guards',
 [ref('src/data/dataset.py','class VQADataset'),ref('scripts/run_phase4_budgeted.py',"hash drift")],
 ['R42','R123','S42','S123'],
 'The loader checks presence, basic schema and sample-ID uniqueness; it does not check each image SHA or the original question/annotation image join on every read. Training validates manifest SHA and train/dev image-ID disjointness. The new CPU audit closes these checks for current files; it does not turn the loader into a general corruption/duplicate detector.',
 'Use audited immutable dataset versions and record verified input hashes; add explicit source-join/image-hash/duplicate guards for future versions without changing v1.')
finding('F21','UNRESOLVED','Near-duplicate and pretraining exposure are outside the present verification',
 [ref('src/data/prepare_vqa.py','def select_subsets'),ref('reports/phase45_recompute.json','cross_split_identical_bytes_groups')],
 ['T','R42','R123','S42','S123'],
 'Exact file hashing detects one train/test duplicate and no train/dev duplicate. It cannot exclude re-encoded/cropped near duplicates or COCO/VQAv2 exposure during foundation-model pretraining. Internal image-ID disjointness is not evidence of foundation-model unseen data.',
 'Qualify the internal held-out claim; separately audit near duplicates and available training provenance if needed. Do not claim either zero pretraining exposure or definite memorization without evidence.',status='UNRESOLVED')
finding('F22','HIGH','Real controlled post-training security assessment is not ready',
 [ref('src/evaluation/security_metrics.py','def security_metrics'),ref('experiments/'+ids['SEC']+'/metrics.json','synthetic_fixture')],
 ['SEC','T','R42','R123','S42','S123'],
 'Existing security rows are labeled synthetic fixture observations. Post-training behavior-change denominator is zero and its rate is null. Qwen remains frozen in this campaign. There is no evaluated victim checkpoint lineage or independently verified real post-training fixture. Inference answer changes on edited images are not backdoor ASR.',
 'Keep real security readiness NO-GO. First close task/label/evaluator/provenance gates and define a separately reviewed controlled assessment of fixed artifacts; never connect ASR or evasion feedback to automated optimization.')
finding('F23','LOW','Generalized freshness checks have an unexercised duplicate-within-batch gap',
 [ref('src/rl/diffusion_trajectory.py','def assert_fresh'),ref('src/rl/diffusion_rl_trainer.py','for trajectory,condition,reward in episodes:'),ref('src/rl/diffusion_rl_trainer.py','self.consumed.update')],
 ['R42','R123','S42','S123'],
 'The trainer checks against previously consumed IDs before the batch, then adds all IDs only after the update. It does not independently reject two copies of the same fresh trajectory inside one input list. Actual collectors generate unique IDs and distinct sources, and all 32 recorded batches were verified distinct; no current duplicate update is found.',
 'Add an explicit per-batch trajectory/source uniqueness contract before generalizing this trainer. Do not infer contamination of current recorded batches.')

def block(keys):
    chunks=[]
    for f in F:
        if f['id'] not in keys:continue
        chunks.append(f"## {f['id']} — {f['classification']}: {f['title']}\n\n"
          f"Evidence status: **{f['evidence_status']}**. Source/function: "+'; '.join(f['sources'])+'.\n\n'
          'Supporting experiment IDs: '+', '.join(run(i) for i in f['experiments'])+'.\n\n'
          f"Observed: {f['observed']}\n\nProposed correction: {f['proposed_correction']}\n")
    return '\n'.join(chunks)
def save(name,text):
    p=REPORTS/name
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
base=f"""Audit date: {receipt['generated_at']} (UTC); local New York date 2026-10-27.
Canonical root: `/storage/home/ydk297/projects/poisoning`; local evidence mirror: `H:\\Project\\Poisoning`.
Audit only: no GPU allocation, model inference, training, fine-tuning, reward change or scientific fix.
Reproducible CPU receipt: {ref('reports/phase45_recompute.json')}; audit source: {ref('scripts/audit_phase45.py')}.
Finding classifications follow the requested CRITICAL/HIGH/MEDIUM/LOW/VERIFIED/UNRESOLVED vocabulary. Evidence status is stated separately: a verified weakness can have HIGH severity; UNRESOLVED never means PASS.
"""
mapping='\n'.join(f'- **{k}**: `{v}`' for k,v in ids.items())
method=f"""# Methodology and generator audit

**Decision for further scientific training: NO-GO.** Core wiring is supported, but the research objective, independent semantic evidence and causal RSI comparison are not ready.

{base}

## Research formulation and literature

[BadSem (2025), Sections 3–4](https://arxiv.org/html/2506.07214v1) studies training-time image/text semantic mismatches as implicit conditions for later malicious model behavior, focusing on color and object semantics. Its defining construct is cross-modal inconsistency. A changed image or a changed answer alone does not establish that construct. The present pilot trains an editor against preservation proxies and leaves Qwen frozen. It does not reproduce that threat model or demonstrate post-training persistence.

| Condition | Precise meaning | Present evidence |
|---|---|---|
| Image-only semantic manipulation | Change an object's identity, color, count, relation or other content in an image | Some saved edits have suspected changes; none independently labeled |
| Image-question semantic mismatch | The paired text asserts or presupposes something inconsistent with the image | Not systematically constructed/labeled here; ordinary questions need not assert an attribute |
| Semantic-preserving edit | An intentional image change that preserves a declared global or question-specific semantic contract | The optimization target; current validity is a same-Qwen proxy |
| Ordinary augmentation | A transformation such as brightness adjustment, independent of a backdoor target | Earlier PIL/controller smokes; not evidence that a diffusion edit preserves labels |
| Training-time data modification | Alter training examples for a learner | Generated trajectories train editor LoRA; they are not used to fine-tune Qwen |
| Inference-time model behavior | A fixed Qwen answer on original versus edited input | Measured diagnostic behavior; distinct from learned backdoor behavior |
| Post-training backdoor evaluation | Assess fixed victim artifacts for condition-dependent behavior after a documented training lineage | Real measurement absent; only explicitly synthetic aggregation fixtures |

M0 is zero-B frozen LoRA on the **same modified stochastic DDIM path** as trained arms; it is not necessarily stock pipeline-default inference. M1 is eight single-pass on-policy updates at 1e-5. M2 shares the exposure/noise protocol and makes one validation-triggered cooldown after update4. Earlier categorical-editor/controller RL and supervised LoRA smokes are separate methods and cannot be pooled as generator RL results.

## Generator source and evidence

`DiffusionPolicy.condition/predict/sample` in {ref('src/rl/diffusion_policy.py','class DiffusionPolicy')} uses full-image aspect-preserving resize and white pad to512, VAE mode **without source-latent scaling**, instruction embeddings in three CFG branches and an8-channel UNet input. CFG is unconditional +7.5(text-image)+1.5(image-unconditional). The VAE-decoded terminal latent is divided by the VAE scaling factor. This source-conditioning convention matches the [pinned Diffusers 0.35.1 IP2Pix implementation](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_instruct_pix2pix.py).

The policy changes the base scheduler to stochastic DDIM epsilon prediction, eta1, 10 leading timesteps901,...,1, offset1, clipping off and set_alpha_to_one=false. This avoids a zero-variance final transition. It is an explicit generation-protocol change, shared by M0/M1/M2, not evidence of equivalence with the pretrained editor's default scheduler. CPU algebra matches installed DDIM step exactly on all ten synthetic transitions.

LoRA attaches rank8/alpha16 adapters to `to_q`, `to_k`, `to_v`, `to_out.0`: 1,594,368 trainable parameters, FP32 adapter parameters/operators, BF16 frozen base. Saved before/after hashes cover1130 frozen named parameters; all agree. Conditioning controls show source-image and instruction influence and exact explicit-seed trajectory replay in the recorded control. These are wiring checks, not instruction-success measurements. No fresh model was loaded for this audit.

## Saved-image review boundary

Inspected the existing ten-row Phase3 contact sheet and Phase4 seed123 final development originals/edits for sources4 (skating) and6 (bananas), plus the exact duplicate source image. The displayed edits show warm/darker shifts and altered padding; fine contour changes are suspected. No count, object-identity or quality verdict is promoted to independent ground truth. See F14 and F18. The portable human-review package remains blank.

{block(['F01','F04','F10','F14','F16','F18','F22'])}

## Experiment map used throughout the audit

{mapping}
"""
save('METHOD_AUDIT.md',method)

reward=f"""# Reward audit

**Reward validity: NO-GO for scientific improvement claims and further scaling.** Arithmetic is reproduced; construct validity remains insufficient. No coefficients changed.

{base}

The executed function is {ref('src/rl/reward.py','def validation_reward')} with defaults(1,0.5,0.2,0.1,1); the actual caller is {ref('scripts/smoke_generator_rl.py','def collect')}. Inputs S,A,D,C are each clamped to[0,1]. Let `valid = semantic.verified and S == 1`; U is0 if valid, otherwise1. Raw reward is S+0.5A−0.2D−0.1C−U; final reward is clipped to[−1,1.5]. With default coefficients, the unconstrained component-wise raw bound is[−1.3,1.5]. No prediction-dependent max normalization is used. `fixed_advantage` returns the bounded reward unchanged (baseline0, scale1).

| Component | Exact definition and range | Dependencies / uncertainty | Overlap and intended construct |
|---|---|---|---|
| S | Identity:1 and verified; invalid geometry/constructed negative:0 and verified; changed image:.5 only if structured checks and consistency both pass, otherwise0; all changed images unverified | {ref('src/evaluation/semantic_validation.py','def semantic_decision')}; Qwen pair judgment, original/edited Qwen answers, original VQA references; malformed or uncertain judgments fail the gate | A conservative proxy, not independent semantic validity; duplicates the gated A for all500 saved observations |
| A | Raw consistency=`same_normalized_answer AND original_reference_score>0 AND edited_reference_score>0`; binary0/1. Caller further gates it on `provisionally_valid` | {ref('src/evaluation/answer_consistency.py','def answer_consistency')}; official original references plus same Qwen; partial VQA consensus merely has to be positive | Not clean accuracy, not continuous edited consensus; verified identity gate defect in F12 |
| D | `clip((MAE+(1−SSIM)/2)/2,0,1)`; MAE uses RGB[0,1], SSIM data_range1; nominal D[0,1] | {ref('src/evaluation/image_quality.py','def image_quality')}; installed windowed skimage SSIM, with explicitly named global fallback only if import unavailable; current receipt verifies skimage0.25.2 | Pixel/structure change, not perceptual quality or semantic validity; includes padding and penalizes intended brightening |
| C | `clip(num_inference_steps/30,0,1)`; current10/30 exactly1/3 | Caller above; independent of actual GPU seconds and reward-model overhead | Constant in this campaign; no ability to reward observed efficiency |
| U |0 only for verified S1; otherwise1 | `validation_reward`; uncertainty and invalidity are conflated | All500 U1, including proxy-pass images. Constant level penalty; induces floor censoring together with negative D/C |

For a changed edit let G be joint proxy agreement. Then S=.5G, A=G, U=1 and C=1/3:

`R = clip(G − .2D − 1/30 − 1, −1, 1.5)`.

G0 yields−1 for every D; G1 yields−1/30−.2D. S and A thus provide one binary signal, not independent semantic and correctness evidence. For G1, D can change reward by at most.2 across its entire allowed range; G flips produce a gap. A single flip contributes about.094 to a ten-image mean. Keeping an unverified image unchanged is not the same as producing a successful edit, and exact identity has privileged verification. All-negative rewards are **not** by themselves a policy-gradient sign error: an action-independent constant is a zero-expectation baseline in the ideal score estimator. The current clipped gate and finite-sample variance still require scrutiny.

## Recomputed distributions

All320 training and180 development observations were recomputed using saved PNGs, raw judgments and original answers; no new Qwen call. Train floor mass165/320=51.5625%; development floor mass65/180=36.1111%. All500 U1, all500 S=.5A, all500 human-unverified. Exact counts and per-arm min/max, means and raw/clipped values are in `phase45_recompute.json` → `saved_observation_recompute.statistics`.

| Arm/seed | Training proxy passes /80 | Mean raw reward | Mean clipped reward |
|---|---:|---:|---:|
"""
for k,v in receipt['saved_observation_recompute']['statistics'].items():
    if 'train' in v:
        s=v['train'];c=s['mean_components'];reward+=f"| {k} | {s['proxy_pass']} | {c['reward_unclipped']:.9f} | {c['reward']:.9f} |\n"
reward+='\nU dominates absolute score level and the censoring regime; observed ranking is driven primarily by the proxy gate. It is inaccurate to say differences are caused by a varying U because U never varies. Semantic false negatives, wrong-but-consistent answers and model prompt instability cannot be disentangled from real preservation without independent labels.\n\n'
reward+=block(['F02','F03','F04','F10','F11','F12','F19'])
reward+='\nTraining uses train records; development images are source-disjoint from them. The same frozen verifier is reused across splits, and development is repeatedly used for adaptation. That is reward/evaluator dependence and adaptive evaluation, not evidence that validation images were directly included in optimizer batches. ASR and evasion do not enter reward or coefficient selection.\n'
save('REWARD_AUDIT.md',reward)

mathtext=r"""# RL mathematical and implementation audit

**Generator RL correctness: CONDITIONAL GO at the formulation/wiring level; no authorization for new training. Strict continuation and scientific reward/evaluation gates remain open.**

"""+base+r"""

## Estimator and assumptions

For source/instruction condition c, a denoising trajectory has density

`pθ(τ|c)=p(zT) ∏t N(zt−1; μθ(zt,t,c), σt²I)`.

For a fixed terminal reward functional, the score-function identity is

`∇θ E[R] = E[R ∑t ∇θ log pθ(zt−1|zt,c)]`.

It requires current-policy trajectories, parameter-independent initial noise, correct transition density, sampled states/actions held constant during the score derivative, positive transition variance, and no unaccounted θ-dependence in reward. The frozen nondifferentiable Qwen reward is permissible. The original [DDPO paper §4.3](https://arxiv.org/html/2305.13301) distinguishes one-step score-function updates from importance-sampled multi-update variants. The implementation here is the former.

The code minimizes `−(1/B) Σi Ri (1/T) Σt (1/d) Σj log Normal(action_ij;mean_ij,std_t)` where B10, T10, d=4×64×64=16384. Thus the expected raw gradient is the standard terminal-reward score gradient scaled by1/(T*d). It is a scaled log-density score, not an unscaled joint trajectory log probability. Scaling is constant for the fixed protocol; varying image dimensions/step counts would change weighting. Adam epsilon, gradient clipping and finite precision mean a constant gradient scale is not a guarantee of identical optimizer behavior or equivalence to another DDPO implementation.

## Transition derivation and CPU verification

Let αt be cumulative alpha and αp the previous timestep cumulative alpha. The implemented variance is `(1−αp)/(1−αt) * (1−αt/αp)`, σ=eta√variance; x0=(zt−√(1−αt)epsilon)/√αt; μ=√αp*x0+√(1−αp−σ²)*epsilon. The replay action is detached before the Gaussian score. Parameters are not optimized through sampled actions or the reward model.

The new CPU audit independently compares all ten synthetic steps to the installed [Diffusers 0.35.1 DDIM scheduler](https://github.com/huggingface/diffusers/blob/v0.35.1/src/diffusers/schedulers/scheduling_ddim.py): max output difference0.0. Gaussian log density and its epsilon gradient agree with `torch.distributions.Normal`. All ten standard deviations are positive, including the final0.0206482187. This checks algebra on CPU, not actual BF16 UNet backward reproducibility.

## Batch and update accounting

| Quantity | Per logical update | Phase4 trained-arm total |
|---|---:|---:|
| Source-image exposures / trajectories |10 |320 |
| Independent unique sources within each batch |10 |154 unique sources across seeds/arms |
| Denoising transitions with backward calls |10×10=100 |3200 |
| Optimizer steps |1 |32 |
| Physical source microbatch |1 |Sequential accumulation |
| CFG UNet branches per transition |3 sequential |Distinct from source batch count |

An additional180 development trajectories contribute1800 denoising transitions without optimizer steps. The CPU audit reads500 stored trajectories and verifies5000 complete, chained, FP32 transitions with expected timesteps, finite fields and positive variances. The two historical diagnostic optimizer replays are separate from32 training steps.

## Code-path assessment

- `DiffusionPolicy.sample` runs no_grad and stores source/image/instruction/seed/version/contract, states, sampled actions, noise and behavior score. The source latent and text embeddings are fixed conditions.
- `assert_fresh` checks version, contract and prior consumption. Each batch is fully validated before the optimizer step. There are no stale epochs or off-policy replay in recorded training.
- `update` recomputes scores at the unchanged behavior parameters; all recorded max likelihood discrepancies are0. It accumulates all100 scaled transition losses before one optimizer step. Terminal reward is assigned once per source and shared across its ten transitions; this is appropriate credit assignment, not100 independent rewards.
- Reward normalization is fixed baseline0/scale1, without batch whitening or self-max normalization. Stored behavior scores are a consistency check, not PPO ratios. There is no frozen-reference KL anchor; that is an explicit on-policy SF choice, not an omitted PPO requirement.
- AdamW uses1e-5 (M2 later5e-6), weight_decay0, constant LambdaLR and max_grad_norm1. Gradients are checked finite/nonzero; each recorded update changes the intended adapters; frozen hashes stay equal. Gradient sign-reversal checks verify wiring but do not prove a useful reward or convergence.
- The finite-gradient result and59 passing CPU tests are supportive engineering evidence. They do not prove an unbiased real-device implementation under every future configuration, statistical gain, semantic correctness or bitwise continuation.

"""+block(['F09','F12','F15','F17','F19','F23'])+"\nNo decisive score-sign, missing-source-conditioning or batch-accounting bug was found in the executed fixed protocol. The known identity reward branch and configuration wiring gaps are distinct from the sampled-action gradient. Numerical divergence remains an observed failure with an unresolved mechanism; it has not been repaired or excused.\n"
save('RL_MATHEMATICAL_AUDIT.md',mathtext)

rsi=f"""# RSI mechanism, novelty and fairness audit

**RSI methodology: NO-GO for causal improvement or novelty claims.** Revision persistence is implemented; benefit is not identified.

{base}

`RSIController.revise` accepts a candidate only for strict validation-score gain within budget, recording rejected costs and optional rollback evidence. Phase4 instead subclasses it with `revise_schedule`: after at least two looks, if both `mean_delta <= (CI_hi−CI_lo)/2`, make a single LR reduction. It records `quality_improvement_accepted=false`. The controller did not learn a proposal policy, edit code, search multiple learning mechanisms or validate a candidate improvement before adoption.

The trigger is a gain-magnitude-versus-width heuristic, not the hypothesis test “CI contains zero”, not a plateau test on successive score slopes, and not evidence of no improvement. For asymmetric percentile intervals the CI can be entirely positive while the rule still fires. Its raw reason string “two unresolved validation gains” is imprecise; the Phase4 assumption ledger already qualifies this. We preserve the raw history.

Both seeds change LR after step4, so updates1–4 use1e-5; updates5–8 use5e-6. Step4 checkpoints are saved **before** revision. Parent manifests stay initial; child config revisions and active checkpoint contracts record the actual new setting. No retroactive schedule tuning is found.

## Pre-intervention and final comparison

| Seed | Step4 M2−M1 mean (before LR revision) | Step8 M2−M1 mean | Final descriptive paired95% interval |
|---|---:|---:|---|
"""
for seed in [42,123]:
    p=receipt['paired'][str(seed)];a=p['4'];b=p['8'];rsi+=f"| {seed} | {a['mean_delta']:.9f} | {b['mean_delta']:.9f} | [{b['ci95'][0]:.9f}, {b['ci95'][1]:.9f}] |\n"
rsi+="""
The seed123 difference largely predates intervention. Final differences each include a single source's roughly.938 proxy-gate change divided by10; remaining small photometric differences cannot establish independent semantic improvement. Both final intervals include zero. A small positive M2-versus-M0 interval for seed123 is retained in the raw results; after repeated development reuse and with only one fitted seed, it is not confirmatory generalization evidence.

Initial adapter tensors are bitwise equal per matched seed, and train source order/noise seeds, validation IDs/seeds, batch counts and model revisions match. They do not force subsequent numerical execution to match. Real cost differs (3227/3211 and2707/3216 seconds), and compute nodes differ. No matched intervention fork from a verified continuation was executed. No counterfactual claim that LR caused the difference is justified.

Supported: revision memory, one-variable validation-conditioned configuration change, budget accounting, fresh trajectories and honest proxy logging. Unverified: self-improvement beyond scheduling, quality-accepted revision selection, superiority to a conventional cooldown, causal RL benefit, generalization, or improved security behavior.

"""+block(['F05','F06','F07','F13','F15','F17'])
save('RSI_NOVELTY_AUDIT.md',rsi)

data=f"""# Dataset, evaluation, statistics and reproducibility audit

**Evaluation readiness: NO-GO for scientific claims; dataset provenance checks mostly pass with one new duplicate-image defect.**

{base}

## Official annotation and sampling checks

The CPU audit reads original annotation/question archives on the cluster. It independently joins all1500 selected question IDs to both archives and confirms matching image IDs, question text and all ten original answers, correct COCO filename, exact selected-image file SHA and exact deterministic sampling replay. It does not infer a person's identity or ethnicity from the reviewed duplicate photo; that question is simply an original dataset annotation.

| Internal split | Questions | Unique image IDs | Official source |
|---|---:|---:|---|
| Train |1000 |989 |VQAv2 official train2014 pool |
| Development |200 |197 |Same pool, image-ID-disjoint allocation |
| Internal held-out |300 |298 |Same pool; **not** official VQAv2 test |

All question IDs are unique and split image-ID intersections are empty. One train/test pair has identical JPEG bytes despite distinct IDs (F08). Train/development has zero exact-byte overlap. Four Phase4 training arms did not expose the duplicated train image. The v1 split is immutable; the audit accesses held-out annotations/file hashes only, with no model call, new metric evaluation, selection or marker change.

After editing, the label contract depends on what changed. Exact identity retains original annotation provenance. Independently verified question-relevant preservation can justify original answer use within that contract. A question-relevant color/count/object/relationship change may invalidate original answers; a mismatch can make the original question's premise false or unanswerable. A real semantic change irrelevant to this question may retain its answer but violate the global editing instruction. Ordinary brightness augmentation has no universal label-validity guarantee. Current edited-reference scores are correctly tagged diagnostic/eligible_for_accuracy=false.

## Metric dictionary

| Quantity | Unit / reference | What can be concluded |
|---|---|---|
| Training reward |320 source exposures, newly changing train batches; same-Qwen gated reward |Optimization observations; cross-update increases mix source difficulty and model change |
| Development reward |180 trajectories but10 distinct images; four looks/arm |Exploratory matched-source proxy comparison |
| Clean VQA accuracy |Official consensus scoring against ten dataset annotator answers on unmodified images |Historical frozen inference protocol accuracy; no Phase4 learned-Qwen improvement |
| Answer consistency |Same normalized answers; raw and positive-reference-gated versions separately stored |Agreement, possibly wrong in both cases; not semantic truth |
| Semantic-validity proxy |Same Qwen's pair check plus answer gate |Automatic agreement only |
| Independent semantic validity |No completed human labels |Unknown, no valid-edit success-rate estimate |
| Post-training security behavior |Real denominator0; synthetic fixtures labeled |No real behavior-change or ASR result |

Phase2 clean development81.15% and once-opened internal-test77.633333% belong to the old frozen baseline protocol. They are not new Phase4 training results. The official-compatible scorer uses dataset answers, not model-created ground truth. The auditor independently rechecks its output for500 saved original/edited answer pairs. No suspicious self-max score normalization or phantom result is identified by the fresh reviewer; the same-model semantic proxy remains a validity weakness.

## Final development reward, exact recomputation

| Condition | Seed42 | Seed123 | Mean across two seeds | Sample SD across two seeds |
|---|---:|---:|---:|---:|
"""
for arm,mean,sd in [('M0',-.3461671304454406,.12947613924652557),('M1',-.48629410471146306,.1957765652803446),('M2',-.3925535009863476,.19573003294687905)]:
    rows=[r for r in results['raw_final_rows'] if r['arm']==arm];v={r['seed']:r['reward'] for r in rows}
    data+=f"| {arm} | {v[42]:.9f} | {v[123]:.9f} | {mean:.9f} | {sd:.9f} |\n"
data+=f"""
Both fixed-RL seeds score below their frozen reference. The average RSI proxy reward is also below frozen. Raw same-answer fraction averaged across seeds is.75/.70/.70 for M0/M1/M2, while reference-positive raw consistency is.70/.65/.65. These differ from semantic-gated A_reward (.70/.55/.65). None should be relabeled clean accuracy or independent preservation.

`paired_summary` resamples ten source-level differences10,000 times with numpy default_rng7 and percentile2.5/97.5%. All16 stored frozen-paired intervals and final/intermediate M2−M1 intervals reproduce exactly. The unit is the source image **within one fitted seed**, not transitions, trajectories, questions on the same image or bootstrap draws. No pooled180-independent-image interval is valid. Repeated LR selection on the same development set, two fitted seeds and common ten images limit all inference; sample SD is descriptive, not a reliable population uncertainty estimate.

## Reproducibility and immutability

Pinned revisions: InstructPix2Pix `{receipt['model_revisions']['timbrooks/instruct-pix2pix']}`; Qwen `{receipt['model_revisions']['Qwen/Qwen3-VL-2B-Instruct']}`. Current environment versions reproduce saved runtime records: Torch2.6.0, Diffusers0.35.1, PEFT0.17.1, Transformers4.57.1, Accelerate1.11.0, hub0.35.3, skimage0.25.2. BF16 base/FP32 adapters, math SDPA, TF32 off and deterministic flags are recorded; flags are not an empirical bitwise guarantee.

CPU audit verifies1872 Phase4 files against their original per-file SHA receipt and2130 historical files before/after with no change. Frozen protocol SHA and once-opened test marker are unchanged. Existing source histories have different local/canonical commits but the Phase4 receipt provides32 normalized-LF source hashes; these are checked again at audit close. Some early wrapper/diagnostic manifests were dirty or mislabeled by parent seed metadata; scientific child configs and real command lines preserve the actual seeds. Those records remain untouched.

Fresh existing CPU tests:59 passed (`phase45_cpu_tests.log`). Read-only monitor snapshot: enabled/installed, interval2h, failures0, uncertain submission false; last successful native tick `{monitor['state']['last_success_utc']}`, one pending owned CPU job `{monitor['state']['next_job_id']}` due `{monitor['state']['next_due_utc']}`. No duplicate scheduler was created and no GPU research job was active at inspection. This is a point-in-time operational check, not a promise of future uptime.

{block(['F07','F08','F09','F20','F21','F22'])}
"""
save('DATA_EVALUATION_AUDIT.md',data)

matrix=f"""# Audit decision matrix

**Overall further-scientific-training decision: NO-GO.** This is an audit recommendation, not a permission request or an automatic change to jobs. No scientific training was launched.

{base}

| Readiness dimension | Decision | Evidence and unmet gate |
|---|---|---|
|1 Engineering readiness |CONDITIONAL GO |Model/data provenance, real updates, logging, budget, monitoring and59 CPU tests supported. Exact continuation, runtime guards and active-contract reconstruction remain limited. CPU audits/reporting can continue; resume or new training is not cleared. |
|2 Generator RL correctness |CONDITIONAL GO |Fixed-contract score-function/DDIM algebra, actual LoRA gradients, B10/T10 and single optimizer pass supported. Real-device backward repeatability and continuation unresolved. This is not statistical or semantic correctness. |
|3 Reward validity |NO-GO |Same-model verification, S/A redundancy, constant uncertainty penalty plus censoring, no instruction-success measure, latent identity branch inconsistency. |
|4 RSI methodology |NO-GO |One LR cooldown; no demonstrated distinction from scheduling; major differences precede intervention; final intervals include zero. |
|5 Evaluation readiness |NO-GO |No independent semantic labels; ten adaptively reused sources/two seeds; ID-disjointness misses exact train/test duplicate. |
|6 Controlled post-training security assessment |NO-GO |No real fixed-victim lineage/independent real assessment; synthetic aggregator fixtures only; upstream task/semantic/evaluation gates unresolved. |

GO would mean all relevant gates are satisfied for the proposed scope; CONDITIONAL GO records supported components with explicit unresolved conditions. It does **not** override this phase's prohibition on new scientific training. No dimension grants approval for a poisoning campaign, automated ASR optimization or evasion tuning.

## All adverse and unresolved findings

| ID | Classification | Evidence status | Finding |
|---|---|---|---|
"""
for f in F:matrix+=f"| {f['id']} | {f['classification']} | {f['evidence_status']} | {f['title']} |\n"
matrix+='\n'+block([f['id'] for f in F])
verified=[
 ('V01','Dataset correspondence and deterministic replay','src/data/prepare_vqa.py','def select_subsets','All1500 selected original question/annotation/image joins, answer arrays, file hashes and sampling replay match. Split independence is qualified by F08.','T'),
 ('V02','Stochastic DDIM/score algebra','src/rl/diffusion_trajectory.py','def transition','Ten CPU transitions match installed DDIM step with max error0; Gaussian score and epsilon gradient match torch.distributions.Normal; all variances positive. This is CPU algebra only.','D'),
 ('V03','Actual generator RL and batch accounting','src/rl/diffusion_rl_trainer.py','def update','32 actual updates,10 source exposures and100 denoising backward transitions each; finite/nonzero gradients and adapter changes; no categorical-head substitution in Phase4.','R42'),
 ('V04','Preserved model revisions/frozen tensors','src/training/generator_adapter.py','def frozen_hashes','Pinned model revisions; 1130 named frozen parameters unchanged per arm; 1594368 intended LoRA parameters; does not imply useful editing.','S42'),
 ('V05','Saved metric and confidence-interval arithmetic','scripts/audit_phase45.py','for key,m in metrics.items()','500 observations/trajectories,5000 transitions, reward/quality/consistency fields and paired CIs reproduced without model calls.','S123'),
 ('V06','Existing two-hour monitor operational at inspection','src/monitoring/scheduler.py',None,'One owned pending native CPU successor; enabled and failures0; no GPU research active. Snapshot records exact ID/times.','D'),
 ('V07','Historical evidence and budget preserved','scripts/audit_phase45.py','def tree_digest','1872 original Phase4 files checked against receipt;2130 historical files unchanged before/after; historical failed run remains failed; actual prior cost13122/14400GPU seconds; audit adds0.','P3'),
 ('V08','Security aggregation is separated from optimization','src/evaluation/security_metrics.py','def security_metrics','No optimizer imports; synthetic fixture labeling and null absent denominators; no ASR/evasion reward path. This does not validate a real security experiment.','SEC')]
matrix+='\n## VERIFIED supported components\n\n'
for key,title,path,needle,obs,eid in verified:
    matrix+=f"### {key} — VERIFIED: {title}\n\nSource/function: {ref(path,needle)}. Supporting experiment: `{ids[eid]}`; derived receipt `phase45_recompute.json` / `phase45_monitor_snapshot.json`.\n\nObserved: {obs}\n\nProposed correction: retain this scoped evidence; do not promote it to a broader scientific claim.\n\n"
matrix+='''## Prioritized correction plan — proposed, not executed

1. **P0 — Freeze the scientific question and claim boundaries (F01/F04/F22).** Decide the precise preservation/augmentation research estimand and distinguish it from a BadSem-inspired security question. Produce a versioned label-validity contract. Keep all current results exploratory; no downstream security training is cleared.
2. **P0 — Validate semantic and task success independently (F02/F03/F10/F18).** Complete blinded review of existing pairs with uncertainty/adjudication; quantify proxy errors and instruction success. No model-generated substitute labels. Analyze reward redundancy, invalidity/uncertainty and clipping on existing evidence before any new objective is approved.
3. **P1 — Correct provenance and reward-interface gaps in a new version (F08/F12/F19/F20).** Group duplicate content for future splits, fix identity eligibility and explicit coefficient wiring, and validate new guards. Preserve v1, old coefficients, markers and all historical metrics; do not delete unfavorable samples.
4. **P1 — Resolve continuation/numerical comparability (F09/F15/F17).** First specify decisive controlled diagnostics. Keep exact acceptance unchanged. This phase authorizes no new GPU diagnostic; do not hypothesize a fix as if demonstrated.
5. **P2 — Define a testable RSI contribution and independent analysis (F05/F06/F07/F13).** Compare a defined adaptation mechanism against an ordinary schedule under a frozen fairness/selection design. State source/seed units and repeated-look handling in advance. Any later training needs its own authorized scope and budget.
6. **P3 — Refresh current-method documentation and only then assess security readiness (F16/F21/F22).** Evaluate fixed, documented artifacts with independent observations after upstream gates close; keep ASR/detection-evasion measurements outside automated reward optimization.

No scientific correction above was implemented by this audit. Existing monitoring continues. A passing engineering component does not waive another gate.
'''
save('AUDIT_DECISION_MATRIX.md',matrix)

zh=f"""# Phase 4.5 全面审计总结

**最终决定：NO-GO，暂不继续科学训练。** 代码确实执行了生成器 LoRA 的奖励梯度更新，已有数字也能重算；目前不足以支持“方法有效”“RSI 带来提升”或“实现了 BadSem 目标”的结论。

审计时间：{receipt['generated_at']} UTC。本阶段只读既有科学证据并写入新审计报告：**新增 GPU 用时0，模型推理0，训练更新0，科学方法修改0**。现有59项 CPU 测试通过。

## 按科学严重性排序的主要问题

1. **CRITICAL：研究目标与实际优化目标不同（F01）。** 当前做的是“轻微提亮并保持语义”的编辑器训练；Qwen 全程冻结。BadSem 关注图文语义不一致及其训练后行为。当前奖励会惩罚破坏保持条件的语义变化，因此不能把这批结果写成 BadSem 实现或安全效果证明。
2. **CRITICAL：RSI 的因果效果无法识别（F06）。** 两个匹配种子虽然初始化、图片顺序和噪声种子相同，但降低学习率之前模型就已分叉。seed123 在第4步已有约+0.09326的差异，第8步约+0.09371；最终两个种子的 RSI−RL 区间都包含0。单张图片的代理判断翻转即可造成约+0.094的十图均值差异。
3. **HIGH：奖励算对了，但测量含义不足（F02–F04）。** 500条样本全部 U=1，且 S=0.5×奖励里的A；两项实质上重复同一代理门控。训练165/320（51.56%）、开发65/180（36.11%）被截断成−1，差异信息丢失。奖励没有明确测量“成功轻微提亮”，不编辑本身会受到偏好。同一Qwen既提供答案又判语义，没有独立语义证据。
4. **HIGH：实际 RSI 仅验证驱动的学习率冷却（F05）。** 两个种子都是第4步后1e−5降至5e−6；记录明确为接受调度，不是接受质量提升。还不能证明区别于普通调度器的递归改进能力或新颖性。
5. **HIGH：验证范围过小且复用（F07/F18）。** 180条开发轨迹实际只来自10张图片，反复用于评估与调度选择；只有两个种子，人工语义标签仍为0。置信区间只描述单个已训练模型在这些源图上的条件不确定性，不能推广到完整数据分布。
6. **HIGH：新发现一组训练/内部测试重复图片（F08）。** image_id分别397819与388662，但文件 SHA 完全相同。原ID划分未阻止内容重复；**训练/验证没有相同字节图片重叠，重复训练图也未进入四个Phase4训练臂**。这不证明当前Phase4比较被泄漏污染，也不证明冻结Qwen的历史测试分数被抬高；它否定了v1在内容层面完全独立的说法。近重复与预训练见过数据仍未知。
7. **HIGH / UNRESOLVED：严格断点续训未通过（F09/F17）。** 已检查状态恢复和前向似然相同，但反向梯度与更新后参数不同，生成图也不完全相同。具体CUDA算子根因未定位。不能把“诊断任务完成”当成“续训已验收”，本次没有降低阈值。
8. **MEDIUM：确定的边界实现缺口（F12/F19）。** 已验证的identity分支因为缺少provisionally_valid键丢失A奖励；YAML奖励配置没有传入执行函数，当前默认值恰好一致。这两点需要未来版本修复，**没有重写历史奖励或修改当前科学代码**。

## 已经核实的部分

- 官方来源的1500条问题、图片ID、十个人工答案、图片文件SHA与确定性抽样均一致；内部划分为1000/200/300条问题、989/197/298个image_id。
- 500条已保存观测与轨迹、5000个去噪转移的对应关系通过；奖励、质量指标、VQA分数和配对区间可复算。
- batch size10确实是每次逻辑更新10个源图曝光、每图10个转移，共100次反向转移累积后1次优化器更新。Phase4共32次真实训练更新，另有2次历史诊断重放。
- CPU上的DDIM转移与安装版本输出差异为0，Gaussian分数及梯度与独立分布实现一致；这不等于GPU续训完全复现。
- 1872个Phase4文件与原SHA收据相符；2130个历史文件在审计前后不变，旧失败记录、一次打开的测试标记与协议哈希保留。

## 结果应如何表述

| 最终开发代理奖励，越高越好 | seed42 | seed123 | 两种子均值 |
|---|---:|---:|---:|
| 冻结 M0 |−0.437721 |−0.254614 |−0.346167 |
| 固定RL M1 |−0.624729 |−0.347859 |−0.486294 |
| RL+调度 M2 |−0.530956 |−0.254151 |−0.392554 |

固定RL两个种子均低于冻结参考；调度臂的两种子平均也低于冻结参考。seed123相对冻结存在很小的正代理差值，保留在原结果中，但不能据此挑种子或宣布收益。原始答案相同率为M0/M1/M2=0.75/0.70/0.70，没有显示答案一致性提升。没有独立语义质量、泛化收益或真实训练后安全行为结论。

## 六项最终判定

| 方面 | 判定 |
|---|---|
| 工程就绪 |CONDITIONAL GO：只认可已验证组件；续训和边界守卫仍有条件 |
| 生成器RL正确性 |CONDITIONAL GO：固定协议下数学/接线受支持，非完整科学验收 |
| 奖励有效性 |NO-GO |
| RSI方法 |NO-GO |
| 评估就绪 |NO-GO |
| 受控训练后安全评估 |NO-GO |

**优先修正顺序：** 先明确科学问题与标签契约；再对既有图片开展独立盲审并审查奖励构念；随后以新版本处理重复图片、identity奖励和配置接线；再解决续训与干预前可比性；最后才制定独立的RSI对照与评估协议。所有科学修改只是建议，本阶段未自动实施，也未启动新GPU实验。

## 监控与交付

既有两小时CPU监控链正常：最后成功 `{monitor['state']['last_success_utc']}`；唯一后继任务 `{monitor['state']['next_job_id']}`，计划资格时间 `{monitor['state']['next_due_utc']}` UTC。连续失败0，没有新增调度器；现场检查无GPU科研任务。排队可能影响实际开始时间。

完整交付：[方法审计](H:/Project/Poisoning/reports/METHOD_AUDIT.md)、[奖励审计](H:/Project/Poisoning/reports/REWARD_AUDIT.md)、[RL数学审计](H:/Project/Poisoning/reports/RL_MATHEMATICAL_AUDIT.md)、[RSI审计](H:/Project/Poisoning/reports/RSI_NOVELTY_AUDIT.md)、[数据与评估审计](H:/Project/Poisoning/reports/DATA_EVALUATION_AUDIT.md)、[全部问题及决策矩阵](H:/Project/Poisoning/reports/AUDIT_DECISION_MATRIX.md)。每个问题均列有源码位置、函数、实验ID、观察与建议；独立结果完整性复核采用同模型家族审阅，其语义结论是provisional，不是跨模型最终认证。
"""
save('AUDIT_SUMMARY_ZH.md',zh)
with (REPORTS/'phase45_findings.json').open('x',encoding='utf-8') as f:
    json.dump({'generated_at':datetime.now(timezone.utc).isoformat(),'overall_decision':'NO-GO','findings':F,
       'verified_components':[x[0] for x in verified],'scientific_fixes_applied':False},f,indent=2,ensure_ascii=False)
print(json.dumps({'reports_created':7,'adverse_or_unresolved_findings':len(F),'decision':'NO-GO'}))
