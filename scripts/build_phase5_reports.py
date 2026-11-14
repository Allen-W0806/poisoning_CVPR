"""Build evidence-bounded Phase 5 deliverables from saved receipts; no experiments."""
import json
from pathlib import Path
import _bootstrap

ROOT = _bootstrap.ROOT
REPORTS = ROOT / 'reports'
LOCAL_IMPLEMENTATION = '82a3febe7049639f2b43ec9a68ed60ff60113619'
CANONICAL_IMPLEMENTATION = 'afd9c1a4159d88b00388b0db7ea23e96d8c2e689'

def read(name):
    return json.loads((REPORTS / name).read_text(encoding='utf-8'))

def link(path, label=None):
    return f'[{label or path}](<{(ROOT / path).as_posix()}>)'

def save(name, title, body, code, tests, evidence):
    footer = '\n\n## Implementation and reproducibility\n\n'
    footer += f'Implementation source commits: local `{LOCAL_IMPLEMENTATION}`; canonical `{CANONICAL_IMPLEMENTATION}`. '
    footer += 'The final documentation commit is recorded separately in ' + link('reports/phase5_completion.json') + '.\n\n'
    footer += 'Code: ' + ', '.join(link(p) for p in code) + '.\n\n'
    footer += 'Tests: ' + ', '.join(link(p) for p in tests) + '.\n\n'
    footer += 'Evidence: ' + ', '.join(link(p) for p in evidence) + '.\n'
    (REPORTS / name).write_text('# ' + title + '\n\n' + body.strip() + footer, encoding='utf-8')

def main():
    data = read('phase5_dataset_v2.json')['provenance']
    harness = read('phase5_harness_smoke.json')
    saved = read('phase5_saved_episode_investigation.json')
    prepared = read('phase5_comparison_prepared_v2.json')
    monitor = read('phase5_monitor_snapshot.json')
    common_tests = ['tests/test_phase5_contracts.py','tests/test_phase5_harness.py','tests/test_phase5_protocol.py']
    literature = '''Engineering implementation is complete within the restricted Phase 5 scope; efficacy remains unverified. These classifications apply to mechanisms, not to transferable performance claims.

| Component | Classification | Evidence and transfer boundary |
|---|---|---|
| Diffusion transition score-function learning | DIRECTLY SUPPORTED BY PRIOR WORK | DDPO; existing SF backend retained, not relabeled PPO/GRPO |
| Image-conditioned IP2Pix UNet attention LoRA | ADAPTED FROM PRIOR WORK | DDPO learning applied to the existing conditional editor; no equivalence to every DDPO implementation claimed |
| Original versus edited facts; SC/SI-V distinction | ADAPTED FROM PRIOR WORK | BadSem supplies semantic distinctions; strict provenance/UNKNOWN contract is our implementation |
| Active observations, reusable program edits, external Gate | ADAPTED FROM PRIOR WORK | Video-RSI; restricted deterministic workflow editor rather than full LLM code editing |
| Training-side integrity curriculum and diagnostic memory edits | NEW RESEARCH HYPOTHESIS | Allowed interfaces exist; effectiveness beyond integrity fixtures is unverified |
| Combined LoRA RL and harness evolution | NEW RESEARCH HYPOTHESIS | No cited paper establishes this combination's efficacy |
| Backdoor effectiveness, improved semantic editing, end-to-end efficiency | CURRENTLY UNVERIFIED | No new scientific training, human labels or post-training security result |

## Primary sources actually inspected

[BadSem, method and evaluation sections 3–5](https://arxiv.org/html/2506.07214): image-only semantic manipulation distinguishes semantic consistency (SC) and visual semantic inconsistency (SI-V). Its predefined-target-word ASR requires the relevant downstream training/evaluation context; a frozen model changing its answer is not this ASR. Category keywords are candidate filters, not independently verified labels.

[Video-RSI, method sections 3.1–3.4 and Algorithm 1](https://arxiv.org/html/2609.37950): an Editor investigates additional observations and revises executable solver procedures. Candidate/incumbent evaluation uses the same private, video-disjoint Gate with accuracy/frame costs and aggregate feedback. Its frame accounting is not total offline search cost; ours reports both selected execution and all attempted revisions. Its 72-video Gate and thresholds are not transplanted as validated natural-image thresholds.

[DDPO, ICLR 2024 paper](https://proceedings.iclr.cc/paper_files/paper/2024/file/14f75513f0f1ca01de1e826b52e6b840-Paper-Conference.pdf): diffusion denoising transitions form a policy with reward at the sampled image. On-policy score-function updates differ from importance-sampled/PPO multi-pass variants. This project preserves its existing single-pass SF path and fixed normalization; no claim of implementing another estimator is made.

[MetaPoison, method and algorithm](https://proceedings.neurips.cc/paper_files/paper/2020/file/8ce6fc704072e351679ac97d4a985574-Paper.pdf): downstream training is an essential part of evaluating training-time effects. This motivates separating frozen inference observations from post-training outcomes. Its poisoning optimization is not implemented or adapted here.

## Official initial/final code inspection

[Official Video-RSI source](https://github.com/bingjunluo/Video-RSI/tree/5ad014499a5e009b75e3ef05ac83595b78adda09), commit `5ad014499a5e009b75e3ef05ac83595b78adda09`, was read without execution. README's default initial workspace is `solver_full_workspace`; the final candidate is `artifacts/final_harness`. Compared agent/main/tools/skill_policy and the initial evolve/investigation/gate_feedback engine. The final tools expand from 8,615 to 39,355 bytes and add graph scope, event occurrence/order handling, observed-window memory and routing. These are changes to reusable execution, not a learning-rate schedule. MIT reference files and license are retained locally. Our interpreter supports bounded check ordering, evidence acquisition, verification, resource limits, memory representation and training-only curriculum. It does not reproduce the complete Video-RSI solver or its results.
'''
    save('PHASE5_LITERATURE_ALIGNMENT.md','Phase 5 literature alignment',literature,
         ['src/rl/diffusion_policy_v2.py','src/rsi/harness/harness_spec.py','src/rsi/harness/failure_investigator.py'],common_tests,
         ['reports/phase5_literature/VideoRSI_SOURCE.json','reports/phase5_literature/VideoRSI/LICENSE','reports/phase5_harness_smoke.json'])

    method = '''The revised architecture is coherent as a separation of four processes. Scientific coherence of its reward and semantic measurement is not yet established.

1. **Semantic environment:** versioned source facts, edited facts, supplied instruction, SC/SI-V relation and validity carry separate provenance. Unknown edited answers are never copied from the original by default.
2. **Generator RL:** the actual image-conditioned IP2Pix diffusion policy learns UNet attention LoRA A/B parameters. Original UNet weights, VAE, text encoder and frozen Qwen services remain fixed. Stochastic latent transitions are actions; categorical editing choices are only an external historical baseline.
3. **RSI harness:** a bounded declarative program controls diagnostic calls/order, retained evidence, resource limits and training-side case ordering. It cannot edit generator tensors, factual answers, objectives, evaluators or held-out records. Its deterministic proposal rules are deliberately limited.
4. **Independent evaluation:** candidate and incumbent must share a frozen generator/service state and paired evaluation inputs. The operator returns aggregate results and costs; the Editor never receives per-item Gate feedback through this interface.

For a behavior-policy rollout, the preserved on-policy SF estimator has the form `L = -mean_i[stopgrad(A_i) mean_t(log p_theta(z_(t-1)|z_t, source, instruction))]`, with the backend's fixed coordinate/time normalization. This describes a gradient estimator, not a demonstrated positive causal effect. A valid independently established scalar reward would be needed before a Phase 5 optimizer step. The guarded trainer currently refuses every unreleased training reward, including raw scalar attempts.

Harness evolution is a different outer process: `trace -> competing causes -> new permitted observations -> diagnosis -> reusable workflow candidate -> frozen paired Gate -> accept/reject -> immutable history`. Header/hash/quality/trajectory queries acquire facts absent from the old reward trace. Failure causes remain unresolved when observations cannot discriminate them. During candidate comparison no concurrent LoRA update is permitted. A joint improvement claim requires separately measuring generator, harness and total search costs.

The harmless engineering Gate uses >=24 distinct sources, >=0.05 accuracy gain with <=100% diagnostic-operation growth, or equal accuracy with >=20% operation saving; no previously correct source may become wrong. The executed test uses 32 sources. These frozen engineering thresholds do not establish natural-image significance or semantic reliability. Same-UID filesystem secrecy is not provided: isolation is a cooperating-operator capability boundary.

Frozen Qwen answer consistency is inference-time feedback. A training-time security outcome additionally requires a fixed authorized trained artifact, parent lineage, training data provenance and independent outcome observations. Security rates cannot enter training reward or Editor acceptance. DDPO supplies diffusion learning, BadSem supplies relevant semantic/evaluation distinctions, Video-RSI motivates workflow evolution; their simultaneous use remains a new, unverified hypothesis.

The proposed manuscript replacement is in ''' + link('docs/PHASE5_METHOD_PROPOSAL_v1.md') + ' and ' + link('docs/phase5_method_patch.tex') + '''. The original draft is retained, and the proposed TeX is not an applied/compiled manuscript.

## Full recorded assumptions

''' + (ROOT/'implement-stage/phase5/ASSUMPTIONS.md').read_text(encoding='utf-8')
    save('PHASE5_METHOD_SPECIFICATION.md','Phase 5 method specification',method,
         ['src/rl/diffusion_policy_v2.py','src/rl/diffusion_rl_trainer_v2.py','src/rsi/harness/harness_executor.py','docs/phase5_method_patch.tex'],common_tests,
         ['reports/phase5_harness_smoke.json','implement-stage/phase5/SWEEP_ROUND1.json','implement-stage/phase5/SWEEP_ROUND2.json'])

    save('PHASE5_DRAFT_CODE_DISCREPANCIES.md','Phase 5 draft/code discrepancies','''No original `.tex` was present at intake. The available original HTML discussion, `项目创新性与实验计划.html`, has SHA256 `f64197beecaaa6979cde3abb7354679aaee53c5b2cb9a403fcfffd0664dfc148` and was read without modification. It is a discussion draft, not evidence of a compiled manuscript.

| Previous statement/behavior | Current implementation / remaining limit |
|---|---|
| External categorical policy with fixed generator | Primary policy is actual IP2Pix UNet LoRA; categorical controller remains a baseline |
| GRPO terminology | Existing executed estimator is DDPO-SF; no GRPO claim |
| LR change described as RSI | M2 is explicitly ordinary scheduling; RSI must revise a reusable executable workflow |
| Semantic validity inferred from answer change | Original/edited facts and independent labels separated; unknown labels stay UNKNOWN |
| Frozen inference described as training-time attack evidence | Read-only evaluation separates inference_only and post_training; no real trained artifact result |
| Reward weights saved but not executed | Versioned diagnostic aggregation enforces exact config; historical arithmetic/evidence retained |
| Identity excluded from answer consistency | Verified exact identity can be valid and consistency-eligible in v2 |
| Distinct IDs treated as distinct source images | Logical batch requires distinct content SHA and runtime correspondence |
| Train/test manifest names imply isolation | v2 groups exact bytes; historical validation exposure remains disclosed |
| Checkpoint/likelihood checks imply causal comparability | Exact fields reported, causal readiness remains false pending independent execution evidence |

Historical Phase 4 had 32 actual LoRA updates and no reliable gain. Final legacy development means were M0 -0.34616713, M1 -0.48629410 and M2 -0.39255350; seed-specific intervals included zero. Pre-intervention differences and strict continuation failures remain negative evidence. In all 500 legacy observations U=1 and S=0.5A; floor saturation was 165/320 training and 65/180 development observations. These figures are historical diagnostics, not recalculated v2 efficacy results. No old reward, draft or result is rewritten to improve the comparison.

The versioned proposal explicitly replaces the disputed architecture and claims while retaining historical source. It makes no Dual-Key/textual trigger, target-label or attack optimization implementation.''',
         ['docs/PHASE5_METHOD_PROPOSAL_v1.md','docs/phase5_method_patch.tex','src/rl/phase5_backend.py'],common_tests,
         ['reports/phase5_draft_inspection.json','reports/phase45_recompute.json','reports/phase4_results.json'])

    save('PHASE5_RSI_HARNESS_IMPLEMENTATION.md','Phase 5 reusable harness implementation',f'''The CPU integration is a real execution, not a mocked acceptance. Run `{harness['id']}` used 32 distinct constructed PNG sources and an exact integrity oracle. A geometry-only workflow accepted altered file content. New header and content-hash observations distinguished a digest discrepancy from a specific wrong-size explanation. The broad quality-measurement hypothesis remained unresolved.

An ineffective decode-only revision was rejected. Reusable geometry+hash verification H0002 was accepted: accuracy 0.5 -> 1.0, false-valid outcomes 16 -> 0, with no new errors on previously correct sources. A regression was rejected; rollback to H0000 and restoration of H0002 were verified. Every attempted program and decision is retained with parent, evidence, affected module, expectation, regression checks, measured cost and accept/reject status.

Selected parent/candidate costs were 32/64 diagnostic operations; candidate execution wall time was 0.00617149 seconds. Total search including rejected attempts cost 259 operations, 25,389 bytes, 0.09014638 wall seconds and 0.07946718 CPU seconds; 0 model calls and 0 GPU seconds. This is not a wall-time speedup claim. Synthetic oracle success establishes only the engineering mechanism.

Run `{saved['id']}` additionally investigated saved real Phase 4 training case 225164001 using four newly executed CPU diagnostics: headers, content hash, pair quality and trajectory summary. Source hash matched; stored arithmetic reproduced; ten transition states were finite. Cost: 0.32622429 wall seconds, 3.01851716 CPU seconds, 3,325,648 bytes, no model call or update. Recomputing the same metric does not verify its implementation or construct validity. Instruction following, factual correctness and verifier disagreement remain unresolved, so no correction was proposed.

The Editor receives exact registered training-case capabilities; caller split tags alone are insufficient. Gate returns aggregates only and stores private per-item evidence outside the Editor response. A strict declarative interpreter bounds tools, files, diagnostics and revision scope. Memory/resource/curriculum interfaces exist, but the actual demonstrated accepted change is content verification. No arbitrary shell/code generation or independent LLM Editor was implemented. Natural-image Gate refuses absent independent labels.''',
         ['src/rsi/harness/trace_collector.py','src/rsi/harness/failure_investigator.py','src/rsi/harness/revision_proposer.py','src/rsi/harness/revision_validator.py','src/rsi/harness/cost_aware_gate.py','src/rsi/harness/version_manager.py','scripts/phase5_smoke.py','scripts/investigate_phase5_episode.py'],
         ['tests/test_phase5_harness.py','tests/test_phase5_protocol.py'],['reports/phase5_harness_smoke.json','reports/phase5_saved_episode_investigation.json',f"experiments/{harness['id']}"])

    rows = '\n'.join(f"| {s} | {v['records']} | {v['unique_images']} | {v['unique_content_groups']} | {v['independently_eligible']} |" for s,v in data['category_counts'].items())
    hashes = '\n'.join(f'- {s}: `{h}`' for s,h in data['split_sha256'].items())
    save('PHASE5_SEMANTIC_DATA_CONTRACT.md','Phase 5 semantic data contract',f'''Version `vqav2_content_v2` preserves all 1,500 records. Every source image byte hash and official question/image/ten-answer join was reverified on the canonical cluster. No edited factual labels were manufactured.

| Split | Records | Image IDs | Exact content groups | Independently eligible |
|---|---:|---:|---:|---:|
{rows}

Original image ID/path/SHA, question and dataset answers are separate from edited-image identity, independently supplied edited factual answers, supplied training annotation, category, SC/SI-V relation, validity and verification provenance. VERIFIED edited answers require nonempty strings; UNKNOWN/DISPUTED canonical answers remain None. Metadata validity is not proof of authentic human review. Eligibility requires complete independent edited facts, visual validity, relation and category; all current records have UNKNOWN semantics, 0 verified SC and 0 verified SI-V. Keyword color/animal/food/vehicle matches only nominate candidates. Counts by category are in the receipt.

Exact content grouping moved train sample397819015 to test alongside old test388662037 (same SHA `fcc1fd100ed50e30ea87b9d84597cb40876c76d955716fc9dd6e936787faf181`). All old test records remain test. Same-ID inconsistencies are rejected. Runtime loader checks manifest SHA, image bytes, source joins and semantic-record correspondence at initialization and access; Editor role only accepts train. Original v1 files remain unchanged.

Development/Gate derive from historically exposed v1 validation. They are engineering partitions, not fresh scientific evidence. Known Phase 4 development images remain development. Old test was already opened historically and cannot acquire new freshness by renaming. Final-test selection must be operator-only after freezing the harness; future confirmatory evaluation needs an independently justified unseen cohort.

A separate 64-bit dHash (9x8 grayscale; Hamming <=4) candidate audit found 0 candidate pairs. This does not prove absence of near duplicates and causes no automatic deletion.

Frozen manifest SHA256:

{hashes}

Portable mirrors are under `data/manifests/cluster/vqav2_content_v2`; canonical manifests remain `data/processed/vqav2_content_v2`. Raw images were not downloaded anew.''',
         ['src/data/semantic_contract_v2.py','src/data/dataset_v2.py','src/data/prepare_v2.py','scripts/prepare_phase5_data.py'],['tests/test_phase5_contracts.py','tests/test_phase5_protocol.py'],
         ['reports/phase5_dataset_v2.json','data/manifests/cluster/vqav2_content_v2/provenance.json','data/manifests/cluster/vqav2_content_v2/near_duplicate_candidates.json'])

    save('PHASE5_REWARD_CORRECTIONS.md','Phase 5 versioned reward corrections','''Confirmed implementation defects are corrected in new entry points, preserving all old scores and protected modules. `RewardConfigV2` requires every named coefficient and exact equality of recorded/executed configuration. Compatibility defaults remain semantic1.0, accuracy0.5, distortion0.2, cost0.1, invalid penalty1.0. Changing these numbers to manufacture gains was not performed.

Validity has strict VALID/INVALID/UNKNOWN/DISPUTED states with boolean/numeric consistency. Malformed verifier fields fail rather than receiving favorable defaults. Verified exact identity is valid and answer-consistency eligible; an unsupported edit remains unknown. Receipts expose raw components, eligibility, unclipped and clipped scores, lower/upper saturation, constant components and component correlations. Duplicate source content is rejected before optimizer activity, even across distinct IDs.

These repairs fix wiring and accounting; they do not fix a semantic construct by assertion. Legacy U=1, S=0.5A and clipping motivate independent validation; no renamed scalar or coefficient adjustment establishes an informative editing reward. Compatibility aggregation is explicitly diagnostic_only and every receipt has training_eligible=false. The guarded trainer calls the release guard before its preserved DDPO optimizer; raw scalar episodes cannot bypass it.

No new benign training reward is released. Release requires an independently labeled pilot with verified identity, valid SC, valid SI-V, invalid edits, UNKNOWN and DISPUTED cases; report class/source counts, missingness, inter-reviewer disagreement, construct/component redundancy and saturation. Lock a release version and its validation criteria before training. Until then all Phase 5 scientific optimizer updates remain blocked. This is an unresolved scientific prerequisite rather than a completed reward-validation result.''',
         ['configs/phase5_reward.yaml','src/rl/reward_v2.py','src/evaluation/semantic_validation_v2.py','src/rl/diffusion_rl_trainer_v2.py','src/rl/phase5_backend.py'],['tests/test_phase5_contracts.py'],
         ['reports/phase5_new_tests_final.log','reports/phase5_canonical_cpu_tests_final.log','reports/phase45_recompute.json'])

    readiness='\n'.join(f"| {k} | {'PASS' if v else 'NOT SATISFIED'} |" for k,v in prepared['readiness'].items())
    save('PHASE5_RL_RSI_EXPERIMENT_PROTOCOL.md','Phase 5 prepared RL/RSI comparison',f'''**NO-GO for scientific launch.** Six reviewed configs are frozen under `configs/phase5_v2`; initial proposals under `configs/phase5` remain retained. All are PREPARED_NOT_RUN with launch_authorized=false. No large comparison was submitted.

| Arm | Intervention |
|---|---|
| M0 | Frozen generator |
| M1 | Standard generator DDPO-SF |
| M2 | Ordinary fixed scheduler: after update4 LR1e-5 ->5e-6 |
| M3 | Traces collected; no harness revision |
| M4 | Diagnosis-guided reusable revisions |
| M5 | Active additional observations plus independently gated revisions |

Planning seeds42/123/2026; eight logical updates for trained arms, batch10 distinct content sources, microbatch1 and accumulation10; LoRA rank8/alpha16. Fixed64-source development observations at updates2/4/6/8 and64 private Gate sources; max2 candidate attempts per M4/M5 seed. No target-model poisoning or security reward. Editor and evaluator must not share per-item Gate feedback. Generator/service snapshots stay frozen within each paired candidate comparison.

Before branching, compare adapters, optimizer, scheduler, step/version, contracts, consumed sources and all Python/NumPy/Torch/CUDA RNG exactly. Report full metadata equality separately; only arm/rsi_mode bookkeeping may be stripped from scientific-contract comparison. No numeric tolerance is relaxed. Exact available fields alone cannot establish causal readiness: frozen weights, content/source order, rollout noise and independent pre-intervention execution require additional proof. Historical strict continuation failures remain unresolved. The checker never auto-clears causal readiness.

Statistical unit is the unique source-image content cluster, not each repeated question or evaluation look. Keep all predeclared seeds and failed runs. Report seed-specific paired candidate/incumbent effects with image-cluster bootstrap 95% intervals (10,000 deterministic resamples, analysis seed20261028), source/record coverage and complete search cost. Any pooling is descriptive; three seeds do not support a broad significance claim. Fixed development looks are descriptive trajectories; no best-look/seed selection. Gate supports bounded operational selection, not a fresh confirmatory efficacy claim. Lock natural-image acceptance criteria using independent labels before opening an unseen evaluation cohort. Existing exposed Gate/test cannot provide that confirmation.

| Readiness item | Status |
|---|---|
{readiness}

Cost is a planning scenario, not an allocation: prior eight-update GPU seconds3227/3211/2707/3216 and median generation+judging15.77772137sec per observation. Five trained arms x3seeds plus expanded development:27.58953 A100 GPUh; frozen arm0.84148h; paired Gate attempts6.73183h. Subtotal35.16284h; +20% overhead = **42.19541 A100 GPUh**. Human review, a changed independent evaluator and clean-target fine-tuning are unmeasured and excluded. Current authorized next engineering work requires0 GPUh. Resolve labels/reward/fresh private Gate/strict branch comparability first; then a separately authorized small clean feasibility check may precede any campaign.''',
         ['scripts/prepare_phase5_protocol.py','configs/phase5_v2/M0.yaml','configs/phase5_v2/M1.yaml','configs/phase5_v2/M2.yaml','configs/phase5_v2/M3.yaml','configs/phase5_v2/M4.yaml','configs/phase5_v2/M5.yaml','src/evaluation/phase5_comparability.py'],['tests/test_phase5_protocol.py'],
         ['reports/phase5_comparison_prepared_v2.json','reports/phase4_results.json','reports/phase5_dataset_v2.json'])

    save('PHASE5_SECURITY_EVALUATION_PROTOCOL.md','Phase 5 fixed-artifact security evaluation','''Implemented a read-only aggregation interface, not a security-result generator. It accepts independently supplied observations and explicit pretrained/clean_finetuned/authorized_fixed_condition lineage. Qwen revision is pinned to `89644892e4d85e24eaac8bacfd4f463576704203`. Every trained node needs a documented parent, data, authorization and artifact manifest. A real artifact must be a local JSON bundle with listed checkpoint files confined under its directory; the interface reads and hashes them. External/remote paths alone are unsupported. Content hashes attest identity, not scientific authorization or authentic annotation.

Inference_only and post_training populations are never pooled. Report clean accuracy and SC accuracy with scored/all row counts, missing scores, available/scored/unscored unique sources. False activation on clean controls and SC-conditioned false activation are distinct. SI-V-conditioned target-observed rate reports the explicit supplied boolean denominator and unique sources. Missing scores are not silently converted into incorrect predictions or complete coverage.

Published BadSem ASR concerns a predefined target word under the applicable trained-artifact/evaluation setting. No target is generated here. A genuine post-training ASR requires real authorized artifact lineage plus post-training rows; missing or synthetic artifacts return None and denominator0. Pretrained observations cannot be labeled post_training. Frozen-Qwen answer changes are inference feedback, not demonstrated backdoor implantation.

Only synthetic fixtures exercised this interface. There is no new clean fine-tuning result, no supplied real security artifact, and no real ASR. Report numerator, denominator, source-cluster coverage, uncertainty, checkpoint revision, dataset lineage and evaluation resources when valid artifacts become available; small/absent denominators remain explicit. Future independent evaluator costs are unmeasured. Security feedback is excluded from generator reward, harness selection and active Editor observations. No poisoned-target training, target-label generator, evasion optimizer or automated ASR maximization is implemented.''',
         ['src/evaluation/security_readonly_v2.py','src/data/semantic_contract_v2.py'],['tests/test_phase5_protocol.py'],
         ['reports/phase5_canonical_cpu_tests_final.log','implement-stage/phase5/ASSUMPTIONS.md'])

    save('PHASE5_TEST_RESULTS.md','Phase 5 tests and evidence preservation',f'''**Scoped engineering GO; scientific comparison NO-GO.**

| Verification | Actual outcome |
|---|---|
| Local new Phase 5 tests |22/22 PASS |
| Canonical full suite, final reviewed source |81/81 PASS,3.500sec |
| Earlier local full suite |75pass +3errors: local Diffusers missing; retained failure log, not reported as full pass |
| Real32-source reusable-harness integration |PASS; actual diagnosis, ineffective rejection, accepted general correction, regression rejection and rollback |
| Saved real training-episode investigation |Four new CPU observations; semantic causes unresolved; no unsupported revision |
| Official data joins/bytes |1500/1500 reverified; exact-content split isolation; independent eligible0 |
| Historical preservation |Local3418 files; canonical3344 files unchanged;32 protected Phase 4 source hashes unchanged on both |
| New scientific execution |0 model calls,0 GPUh,0 optimizer updates; M0–M5 prepared only |

Commands: local `python -B -m unittest tests.test_phase5_contracts tests.test_phase5_harness tests.test_phase5_protocol -v`; canonical `.venv-editor/bin/python -B -m unittest discover -v`. Existing canonical runtime used; no shared runtime installation. Tests include explicit mismatched config, identity/unknown/disputed/malformed verifier, invalid factual answers, runtime image/hash/join mismatch, duplicate-content-before-update, immutable train capability, scope-changing revision rejection, both-direction hard regression, rollback, source coverage, separate inference/post-training and real-artifact requirements. They do not establish human semantic validity or GPU continuation.

Exactly two fresh same-family GPT-6-Astra xhigh assumption-review rounds were performed. Round1 findings led to eligibility, content-batch, security-population, causal-readiness, regression and training-guard fixes. Round2 flagged a bundle-layout interface assumption and stale factual-answer/source-coverage claims. A26 was recorded; string validation and unique-source coverage were corrected and executor-tested afterward. Raw reviews retain their provisional/gaps verdict. No third review, independent acquittal or clean-review claim is made.

Monitor snapshot: enabled={monitor['state']['enabled']}, installed={monitor['state']['installed']}; exactly{len(monitor['owned_jobs'])} owned pending CPU successor, job{monitor['state']['next_job_id']}; next eligibility{monitor['state']['next_due_utc']}; last success{monitor['state']['last_success_utc']}; failures0. It remains the existing two-hour Slurm monitor; no duplicate scheduler was created. Actual starts depend on Slurm. This is a timestamped snapshot, not a guarantee of future state.

All legacy failed/negative experiments, old manifests and original draft remain. Protected historical reward, DDPO backend, drivers and monitor source hashes were not modified. New source, versioned manifests/configs, reference code and audit evidence are retained. Run-specific patch scratch is removed at closure; no checkpoints, model weights or historical evidence are cleanup targets.''',
         ['scripts/phase5_preservation.py','scripts/phase5_smoke.py','scripts/investigate_phase5_episode.py'],common_tests,
         ['reports/phase5_new_tests_final.log','reports/phase5_canonical_cpu_tests_final.log','reports/phase5_local_cpu_tests.log','reports/phase5_preservation_verified.json','reports/phase5_canonical_preservation_verified.json','reports/phase5_monitor_snapshot.json','implement-stage/phase5/SWEEP_ROUND1.json','implement-stage/phase5/SWEEP_ROUND2.json'])

    save('PHASE5_PROGRESS_SUMMARY_ZH.md','Phase 5 实现、实测结果与下一阶段判断',f'''**结论：限定范围的 CPU 工程验证 GO；下一轮大规模 Generator RL/RSI 科学比较 NO-GO。** 本阶段已完成版本化代码、真实 harness 演化执行、数据重分组与回归测试；没有提交科学训练或安全攻击实验。

## 上一阶段的问题与本次修复

过去把普通学习率调整称为 RSI；保存的 reward 权重未完整接入执行；identity 的一致性项被错误排除；不同 ID 仍可能来自同一图像内容；原始答案与编辑后事实没有充分分离；旧验证集已暴露，不能作为新独立 Gate；分支在干预前已不完全相同。旧 reward 还存在常量/冗余项和截断饱和，因此旧增益不能作为方法有效证据。

本次新增 v2 接口修复配置接线、identity 判定、语义 UNKNOWN/DISPUTED、编辑答案校验、逐次内容校验和重复内容 batch 阻断。原 Phase1–4.5 代码、失败及数值均保留。真实 Generator RL 仍以 IP2Pix UNet attention LoRA 为学习对象，保留原 DDPO-SF 后端；没有改成外部分类控制器。由于尚无经独立验证的新训练奖励，Phase5 trainer 明确阻断优化器更新。

## RSI 现在是否真的修改可复用流程

是，但已验证范围仅为无害数据完整性任务。采用 Video-RSI 的追加观察、竞争原因诊断、可复用执行流程修改、独立成本 Gate、不可变版本与回滚。实现是有限诊断规则和声明式工作流解释器，没有宣称复现其完整 LLM Editor。

实际执行 `{harness['id']}`：32个独立构造图像，准确率50%→100%；无效修改被拒，文件内容校验修改获接受，退化修改被拒，回滚成功。总搜索259次诊断、25,389字节、0.09015秒墙钟、0.07947秒CPU、0模型调用、0GPU。获选流程诊断数32→64，故不能称为提速。

对真实旧训练 episode225164001 另执行4项新 CPU 检查：哈希匹配、旧质量数值可重算、10步状态有限；语义/指令/答案原因仍未知，未强行提出修改。数值重算不是指标正确性证明。

## 数据与独立评价是否可靠

保留1500条记录，重新核对官方问题/答案及图片字节。train/dev/Gate/test为999/104/96/301；唯一内容组988/103/94/298。旧 train/test 同字节组整体进入 test，旧 test 成员全部保留。近重复候选检索为0，不能解释为不存在近重复。

**独立可用语义样本0，确认SC/SI-V均为0。** 关键词只是候选；编辑后事实缺失时保持UNKNOWN。dev/Gate来自已暴露旧验证集；旧test也曾打开。当前独立自然图像评价不可靠，需真实独立标注、验证奖励及新隔离评价方案。程序配对或检查点字段相等也不能替代严格分支可比性证据。

## 方法与证据边界

扩散策略学习与 harness 外层演化在数学上可以分离；DDPO提供策略梯度依据，BadSem提供语义/训练后评价区分，Video-RSI提供程序级修订机制。联合流程的语义有效性、收益和效率仍是新研究假设。冻结Qwen推理变化不能称为训练后后门ASR。本阶段只实现固定授权工件的只读安全评价接口，未产生真实安全结果。

## 测试、保留与监控

集群完整测试81/81通过；本地新增22/22通过。本地早期完整测试75通过、3项因缺Diffusers报错，失败日志保留。两轮同族假设复核发现的问题已记录和修复；第二轮后的改动仅执行器测试，没有第三轮独立复核或“审查无问题”结论。

本地3418、集群3344份旧证据文件未改变；两端32项受保护源码哈希一致。已有两小时监控仍启用，快照只有一个自有待运行作业{monitor['state']['next_job_id']}，未新建重复监控。快照下次到期{monitor['state']['next_due_utc']}，实际开始受排队影响。

## 下一步与资源

M0冻结、M1普通Generator RL、M2普通scheduler、M3仅trace、M4诊断修订、M5主动独立Gate修订已准备并冻结，均PREPARED_NOT_RUN。先完成独立编辑事实/有效性审核、奖励构念验证、新私有Gate隔离和干预前严格可比性；这些CPU/标注工作可继续，当前GPU预算0小时。科学门槛满足后，再单独授权小规模干净闭环验证。

三固定seed、每训练臂8步、64图dev/Gate及限定候选次数的完整准备方案，按旧实测估计35.16 A100 GPU小时，含20%余量约 **42.20小时**；不是获批分配。人工标注、未来独立评价器和clean-target微调费用尚未知，未计入。未自动提交大规模比较。

全部10份技术报告见同目录；方法提案与LaTeX替换片段在docs。原始HTML未改，未发现原始LaTeX，因此不存在已编译修订论文。''',
         ['docs/PHASE5_METHOD_PROPOSAL_v1.md','docs/phase5_method_patch.tex','src/rl/phase5_backend.py','src/rsi/harness/harness_executor.py'],common_tests,
         ['reports/phase5_harness_smoke.json','reports/phase5_dataset_v2.json','reports/phase5_comparison_prepared_v2.json','reports/phase5_canonical_cpu_tests_final.log','reports/phase5_preservation_verified.json','reports/phase5_monitor_snapshot.json'])
    print(json.dumps({'reports':10,'scientific_decision':'NO-GO','engineering_decision':'GO_SCOPED'}))

if __name__ == '__main__':
    main()
