"""Read-only reanalysis of historical evidence; all new results in a new run."""
import argparse,collections,csv,hashlib,importlib.util,json,math,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
from PIL import Image,ImageOps
from skimage.metrics import structural_similarity
from src.evaluation.vqa_official_compatible import score_answer,normalize_answer
from src.rl.reward import validation_reward,RewardWeights
from src.editing.semantic_validator import SemanticResult
from src.evaluation.semantic_validation import semantic_decision
from analyze_semantic_clean_stage import validated_reviews
RUN='audit_remediation_20261101T0120Z_v1';OLD='small_scale_validation_20261031T1928Z_v1';NEXT='next_validation_20261101T0040Z_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def table(p,rows):
 with Path(p).open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def stats(x):
 a=np.asarray(x,float);c=collections.Counter(round(float(v),12) for v in a);return dict(n=len(a),mean=float(a.mean()),variance=float(a.var(ddof=1)) if len(a)>1 else 0.,minimum=float(a.min()),maximum=float(a.max()),unique_at_12dp=len(c),duplicate_pairs=sum(n*(n-1)//2 for n in c.values()),pair_tie_fraction=sum(n*(n-1) for n in c.values())/(len(a)*(len(a)-1)) if len(a)>1 else 0.)
def main(run):
 out=run/'cpu_audit';out.mkdir(exist_ok=False);inputs={};findings=[]
 matrix=ROOT/'reports/AUDIT_DECISION_MATRIX.md';inputs[str(matrix.relative_to(ROOT))]=sha(matrix)
 # Exact historical files remain untouched; recompute rather than trust prose.
 names={'M0':'EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR','M1_42':'EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL','M1_123':'EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL','M2_42':'EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI','M2_123':'EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI'}
 rows=[];actual_train_ids=set();actual_eval_ids=set()
 for arm,folder in names.items():
  p=ROOT/'experiments'/folder/'metrics.json';m=read(p);inputs[str(p.relative_to(ROOT))]=sha(p)
  batches=[('train',u['step'],u['observations']) for u in m['updates']]+[('development',u['step'],u['observations']) for u in m['evaluations']]
  for role,step,obs in batches:
   for o in obs:
    rew=o['reward'];s=o['semantic'];recomputed=validation_reward(SemanticResult(s['verified'],s['score'],s['verifier'],s['reason']),rew['answer_consistency'],o['quality']['distortion'],o['settings']['num_inference_steps']/30)
    assert all(math.isclose(float(recomputed[k]),float(rew[k]),rel_tol=1e-12,abs_tol=1e-12) for k in ['semantic','answer_consistency','distortion','cost','invalid_penalty','reward_unclipped','reward'])
    r=dict(arm=arm,role=role,step=step,sample_id=o['sample_id'],image_id=o['source_image_id'],S=rew['semantic'],A=rew['answer_consistency'],D=rew['distortion'],C=rew['cost'],U=rew['invalid_penalty'],reward_unclipped=rew['reward_unclipped'],reward=rew['reward'],generation_seconds=o['generation_seconds'],reward_seconds=o['reward_seconds']);rows.append(r)
    (actual_train_ids if role=='train' else actual_eval_ids).add(str(o['source_image_id']))
 assert len(rows)==500;table(out/'historical_feedback_recomputed.csv',rows)
 feedback=dict(total=500,all_saved_reward_arithmetic_exact=True,reward_definition_unchanged=True,S_equals_half_A=sum(r['S']==.5*r['A'] for r in rows),constant_C=sorted({r['C'] for r in rows}),constant_U=sorted({r['U'] for r in rows}),all_negative=all(r['reward']<0 for r in rows),floor_count=sum(r['reward']==-1 for r in rows),upper_count=sum(r['reward']==1.5 for r in rows),raw_below_floor=sum(r['reward_unclipped']<-1 for r in rows),centered_component_rank=int(np.linalg.matrix_rank(np.array([[r[k] for k in ['S','A','D','C','U']] for r in rows])-np.mean([[r[k] for k in ['S','A','D','C','U']] for r in rows],axis=0))),distribution=stats([r['reward'] for r in rows]),by_role={k:{**stats([r['reward'] for r in rows if r['role']==k]),'floor_count':sum(r['reward']==-1 for r in rows if r['role']==k)} for k in ['train','development']},by_arm={k:stats([r['reward'] for r in rows if r['arm']==k]) for k in names},interpretation='constant U and C cannot rank observations; S/A duplicate the same gate; floor censoring hides D/C differences; no independently verified semantics')
 put(out/'feedback_distribution.json',feedback)
 # Demonstrate the existing caller's identity bookkeeping gap; never modify reward.
 im=Image.new('RGB',(8,8),'white');sem=semantic_decision(im,im,dict(sample_id='SYNTHETIC_IDENTITY_FIXTURE',question='How many objects?'),dict(score=1.),{})
 identity=validation_reward(SemanticResult(sem['verified'],sem['score'],sem['verifier'],sem['reason']),1. if sem.get('provisionally_valid') else 0.,0.,1/3)
 put(out/'identity_and_config_audit.json',dict(constructed_fixture_only=True,human_label_created=False,identity_semantic_verified=sem['verified'],identity_provisionally_valid_key_present='provisionally_valid' in sem,actual_caller_answer_credit=identity['answer_consistency'],actual_caller_reward=identity['reward'],declared_consistency_credit_reference_reward=validation_reward(SemanticResult(True,1.,'identity','fixture'),1.,0.,1/3)['reward'],F12='FAIL: caller still gates solely on provisionally_valid',F19='FAIL: collect still invokes defaults rather than explicit YAML weights',executed_defaults=RewardWeights().__dict__,no_op_preference='identity exceeds every observed changed reward even with the lost credit; instruction success absent, not optimized or fixed here'))
 # Matched conditions share the same160 sources; never compare different difficulty pools as GRPO candidates.
 p=ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/predictions.jsonl';inputs[str(p.relative_to(ROOT))]=sha(p);raw=[json.loads(s) for s in p.read_text().splitlines()];primary=[r for r in raw if r['evaluation_seed']==42];ids=sorted({r['source_image_id'] for r in primary},key=lambda x:next(r['source_index'] for r in primary if r['source_image_id']==x));keys={(r['source_image_id'],r['candidate_id']):r for r in primary};assert len(ids)==160 and len(keys)==len(primary)==960
 mismatches=[]
 for r in raw:
  if r['execution_status']=='PASS':
   score=score_answer(r['raw_vlm_answer'],dict(answers=r['reference_answers']))['score'];exact=float(normalize_answer(r['raw_vlm_answer']) in {normalize_answer(a) for a in r['reference_answers']})
   if not (abs(score-r['official_vqa_consensus_score'])<1e-12 and exact==r['exact_any_reference_score'] and score==r['all_attempted_utility_score']):mismatches.append((r['source_image_id'],r['candidate_id'],r['evaluation_seed']))
 conditions=[];batch=[];source=[]
 for c in ['C0','C1','C2','C3','C4','C5']:
  x=np.array([keys[i,c]['all_attempted_utility_score'] for i in ids]);conditions.append(dict(condition=c,**stats(x),at_zero=float((x==0).mean()),at_one=float((x==1).mean()),invalid=sum(keys[i,c]['execution_status']!='PASS' for i in ids),missing_official=sum(keys[i,c]['official_vqa_consensus_score'] is None for i in ids),ties_original=sum(keys[i,c]['all_attempted_utility_score']==keys[i,'C0']['all_attempted_utility_score'] for i in ids)))
  for b in range(10):
   group=ids[b*16:(b+1)*16];v=np.array([keys[i,c]['all_attempted_utility_score'] for i in group]);delta=np.array([keys[i,c]['all_attempted_utility_score']-keys[i,'C0']['all_attempted_utility_score'] for i in group]);g=np.random.default_rng(20261101+b);idx=g.integers(16,size=(5000,16));batch.append(dict(condition=c,batch_id=b,n=16,sample_ids=';'.join(keys[i,c]['sample_id'] for i in group),source_image_ids=';'.join(group),mean=float(v.mean()),within_variance=float(v.var(ddof=1)),mean_ci95_low=float(np.quantile(v[idx].mean(1),.025)),mean_ci95_high=float(np.quantile(v[idx].mean(1),.975)),paired_difference_C0=float(delta.mean()),paired_ci95_low=float(np.quantile(delta[idx].mean(1),.025)),paired_ci95_high=float(np.quantile(delta[idx].mean(1),.975)),difficulty='same sources across conditions; source_index diagnostic blocks may differ in difficulty; not GRPO candidate batches'))
 for i in ids:
  x=np.array([keys[i,c]['all_attempted_utility_score'] for c in ['C0','C1','C2','C3','C4']]);source.append(dict(source_image_id=i,all_tied=bool(np.ptp(x)==0),variance=float(x.var(ddof=1))))
 put(out/'matched_feedback.json',dict(n_sources=160,raw_records=len(raw),primary_records=len(primary),conditions=conditions,all_candidates_tied_fraction=float(np.mean([r['all_tied'] for r in source])),mean_within_source_variance=float(np.mean([r['variance'] for r in source])),C0_C1_utility_equal=all(keys[i,'C0']['all_attempted_utility_score']==keys[i,'C1']['all_attempted_utility_score'] for i in ids),metric_mismatches=mismatches,between_batch_mean_variance={c:float(np.var([r['mean'] for r in batch if r['condition']==c],ddof=1)) for c in ['C0','C1','C2','C3','C4','C5']},repeats_exact=all(len({x['raw_vlm_answer'] for x in raw if x['source_image_id']==i and x['candidate_id']==c})==1 for i in ids for c in ['C0','C1','C2','C3','C4','C5']),scores_for_C2_C3_C4_C5='inherited-reference agreement, not certified label-valid accuracy'))
 table(out/'matched_batch_feedback.csv',batch);table(out/'matched_source_feedback.csv',source);assert not mismatches
 # Reconstruct stored preprocessing, then measure generated-vs-geometry SSIM.
 parent=read(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json');by={(r['source_image_id'],r['candidate_id']):r for r in parent['records']};quality=[]
 for i in ids:
  images={}
  for c in ['C0','C1','C4','C5']:
   r=by[i,c];path=ROOT/r['candidate_path'];assert sha(path)==r['candidate_image_hash']
   with Image.open(path) as im:images[c]=im.convert('RGB');images[c].load()
  ref=ImageOps.pad(images['C0'],(256,256),color='white',method=3);assert np.array_equal(np.asarray(ref),np.asarray(images['C5']));assert np.array_equal(np.asarray(images['C0']),np.asarray(images['C1']))
  quality.append(dict(source_image_id=i,geometry_reconstruction_exact=True,identity_pixels_exact=True,SSIM_C4_C5=float(structural_similarity(np.array(images['C5']),np.array(images['C4']),channel_axis=2,data_range=255)),original_size=list(images['C0'].size),evaluated_size=[256,256],LPIPS=None))
 table(out/'preprocessing_aligned_quality.csv',quality);put(out/'metric_availability.json',dict(SSIM_available=True,SSIM_reference='decoded C5 == ImageOps.pad(original RGB,256,white,BICUBIC=3); C4 and C5 same 256 geometry; full-frame including padding',n_reconstruction_exact=160,SSIM_mean=float(np.mean([r['SSIM_C4_C5'] for r in quality])),LPIPS=None,modules={m:importlib.util.find_spec(m) is not None for m in ['lpips','piq','torchmetrics']},calibrated_LPIPS_weights='not verified; cached AlexNet classification weights alone are not LPIPS',no_substitution=True))
 # Reuse the existing random32 kit, not a resampled subset selected by performance.
 kit=ROOT/'results'/NEXT/'semantic_review32';selection=read(kit/'selection_receipt.json');cases=read(kit/'blinded_manifest.json')['cases'];review,labels=validated_reviews(sorted((kit/'submitted').glob('*.csv')),cases)
 assert len(cases)==len(selection['selected_ids'])==32;assets=[]
 for c in cases:
  for key in ['original','edited','geometry_control']:
   p=kit/c[key];assets.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
 put(out/'review32_status.json',dict(status='BLOCKED' if not review['final_labels'] else 'REQUIRES_HUMAN_AUDIT',kit=str(kit.relative_to(ROOT)),selection_receipt_sha256=sha(kit/'selection_receipt.json'),selection_seed=20261031032,manifest_sha256=sha(kit/'blinded_manifest.json'),cases=32,assets=assets,existing_review_validator=review,no_reviewer_labels_created=True,disagreements='unknown until paired genuine reviewer exports; preserve separately before adjudication'))
 # Audit historical exposure at both manifest and actually executed-inference levels.
 split=read(ROOT/'results'/OLD/'split_manifest.json')['splits'];seen_inference=set(actual_eval_ids);seen_training=set(actual_train_ids);seen_manifests={}
 for p in (ROOT/'results').rglob('predictions.jsonl'):
  if any(part.startswith((OLD,NEXT,RUN)) for part in p.parts) or 'audit' in p.parts:continue
  try:
   for line in p.read_text(encoding='utf-8-sig').splitlines():
    row=json.loads(line);ident=row.get('image_id',row.get('source_image_id'))
    if ident is not None:seen_inference.add(str(ident))
  except (ValueError,UnicodeError):pass
 for path in ['data/processed/vqav2_seed42_v1/train.json','data/processed/vqav2_seed42_v1/validation.json','data/processed/vqav2_seed42_v1/test.json']:
  pp=ROOT/path;ss=read(pp);seen_manifests[path]={str(x['image_id']) for x in ss};inputs[path]=sha(pp)
 for p in (ROOT/'results').rglob('clean_data_manifest.json'):
  if p.parent.name.startswith((OLD,NEXT,RUN)):continue
  m=read(p)
  if m.get('train'):seen_training.update(str(r['image_id']) for r in m['train']) # Training manifest exposure; not inferred optimizer consumption.
 leakage=[]
 for role,rr in split.items():
  ids2={r['image_id'] for r in rr};leakage.append(dict(split=role,n=len(rr),known_prior_exposure_flags=sum(r['known_prior_project_exposure'] for r in rr),prior_executed_inference_overlap=len(ids2&seen_inference),prior_training_or_training_manifest_overlap=len(ids2&seen_training),prior_dataset_manifest_overlap={k:len(v&ids2) for k,v in seen_manifests.items()}))
 overlaps=[]
 for i,a in enumerate(split):
  for b in list(split)[:i]:
   overlaps.append(dict(a=a,b=b,ids=len({r['image_id'] for r in split[a]}&{r['image_id'] for r in split[b]}),hashes=len({r['image_sha256'] for r in split[a]}&{r['image_sha256'] for r in split[b]})))
 oldduplicate=read(ROOT/'reports/phase45_recompute.json')['dataset']['cross_split_identical_bytes_groups'];assert len(oldduplicate)==1
 for group in oldduplicate:
  for r in group['records_a']+group['records_b']:assert sha(ROOT/r['image_path'])==group['image_sha256']
 put(out/'data_leakage.json',dict(new_split_classification='source-disjoint historical development, not genuinely independent',new_split_counts=leakage,within_new_split_overlap=overlaps,all80_images_match_recorded_hash=all(sha(ROOT/r['image_path'])==r['image_sha256'] for rr in split.values() for r in rr),historical_v1_exact_train_test_duplicates=oldduplicate,notes=['Training-manifest overlap does not prove a particular gradient used each example; actual inference overlap is independently counted.','Prior output scans are a lower-bound positive-exposure witness, not complete absence proof.','All80 prior-exposure flags are true; missing census records strengthen uncertainty but are not the only blocker.','dHash near-duplicate screen across80 is partial, foundation-model pretraining exposure remains unknown.'],historical_unparsed=read(ROOT/'results'/OLD/'audit/data_integrity.json')['exposure_scan_errors']))
 q=read(ROOT/'results'/NEXT/'qwen_recheck.json');pred=[json.loads(s) for s in (ROOT/'results'/NEXT/'qwen_predictions.jsonl').read_text().splitlines()];base={r['sample_id']:r for r in pred if r['seed'] is None and r['repeat']==0};qstats=[]
 for seed in [42,123,3407]:
  after={r['sample_id']:r for r in pred if r['seed']==seed and r['repeat']==0};changed=[k for k in base if base[k]['normalized_answer']!=after[k]['normalized_answer']];nonzero=[after[k]['score']-base[k]['score'] for k in base if after[k]['score']!=base[k]['score']];pvalue=1. if not nonzero or len(nonzero)==1 else None;qstats.append(dict(seed=seed,changed_sample_ids=changed,nonzero_score_pairs=len(nonzero),exact_two_sided_paired_sign_flip_p=pvalue,interpretation='no significant clean-task gain; discrete consensus delta on at most one source, not extra independent replications'))
 put(out/'qwen_statistical_conclusion.json',dict(existing_recheck=q['status'],stats=q['stats'],paired_sign_diagnostics=qstats,same_changed_source_across_123_and_3407=qstats[1]['changed_sample_ids']==qstats[2]['changed_sample_ids'],conclusion='engineering reproducibility PASS; clean-task scientific improvement unsupported; no equivalence claim; development-only',new_inference_calls=0,new_optimizer_steps=0))
 # Source evidence for each historical finding is preserved; scoped remediation is separate.
 put(out/'input_hashes.json',inputs);put(out/'complete.json',dict(status='PASS',scope='CPU arithmetic/provenance/metric audits only; findings remain adverse',historical_observations=500,matched_sources=160,blinded_pairs=32,new_model_calls=0,new_GPU_seconds=0))
 print(json.dumps(dict(status='PASS',feedback=feedback,leakage=leakage,semantic='BLOCKED',quality_SSIM=float(np.mean([r['SSIM_C4_C5'] for r in quality])))))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',default=RUN);a=ap.parse_args();main(ROOT/'results'/a.run_id)
