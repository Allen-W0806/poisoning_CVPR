"""Independent CPU evidence extraction for method/code alignment, not efficacy."""
import argparse,ast,collections,hashlib,importlib.util,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
import torch
from src.rl.phase5_backend import require_training_reward
from analyze_semantic_clean_stage import validated_reviews
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,v):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def main(run):
 torch.set_num_threads(2);files=['src/rl/diffusion_policy.py','src/rl/diffusion_rl_trainer.py','src/rl/diffusion_trajectory.py','src/training/generator_adapter.py','src/rl/trainer.py','src/rl/reward.py','src/rl/reward_normalization.py','src/models/qwen_loader.py','src/rsi/controller.py','src/rsi/revision.py','src/rsi/phase4_schedule.py','src/rsi/harness/revision_proposer.py','src/rsi/harness/version_manager.py','src/evaluation/semantic_validation.py','src/evaluation/image_quality.py','src/evaluation/vqa_official_compatible.py','src/evaluation/security_metrics.py','scripts/smoke_generator_rl.py','scripts/run_phase4_rsi.py','scripts/benign_threeway_20261101_v2.py','src/data/dataset.py'];source={}
 for name in files:
  p=ROOT/name;text=p.read_text(encoding='utf-8');tree=ast.parse(text);source[name]=dict(sha256=sha(p),symbols=[dict(name=n.name,line=n.lineno,end=n.end_lineno) for n in ast.walk(tree) if isinstance(n,(ast.ClassDef,ast.FunctionDef))]);copy=run/'audit/executed_source'/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(p.read_bytes())
 put(run/'source_index.json',source)
 invpath=ROOT/'results/small_scale_validation_20261031T1928Z_v1_ddpo/pretrained_IP2Pix_optimizer_groups_ZERO_STEPS.json';inventory=read(invpath);params=[r for g in inventory for r in g['parameters']];names=[r['name'] for r in params];assert len(params)==256 and sum(r['numel'] for r in params)==1594368
 counts={}
 for suffix in ['to_q','to_k','to_v','to_out.0']:
  rr=[r for r in params if suffix+'.lora_' in r['name']];counts[suffix]=dict(tensors=len(rr),elements=sum(r['numel'] for r in rr),modules=len({r['name'].split('.lora_')[0] for r in rr}))
 put(run/'generator_ownership.json',dict(source=str(invpath.relative_to(ROOT)),sha256=sha(invpath),trainable_tensors=len(params),trainable_elements=sum(r['numel'] for r in params),modules=len({r['name'].split('.lora_')[0] for r in params}),target_counts=counts,optimizer_groups=[{k:v for k,v in g.items() if k!='parameters'} for g in inventory],only_LoRA=all('lora_' in n for n in names),parameter_order=names,scope='actual pretrained IP2Pix optimizer inventory; not Qwen adapters or random tiny UNet',rank=8,alpha=16))
 failed='results/next_validation_20261101T0040Z_v1_BF16_instrumented/diagnostic';path=ROOT/failed/'raw_gradient_observations.pt';captured=torch.load(path,map_location='cpu',weights_only=False);ref,resume=captured[1]['raw_gradients'],captured[2]['raw_gradients'];assert len(ref)==len(params);details=[]
 for p,a,b in zip(params,ref,resume):
  assert a is not None and b is not None and list(a.shape)==p['shape'];details.append(dict(name=p['name'],shape=p['shape'],exact=torch.equal(a,b),nonfinite=not bool(torch.isfinite(a).all() and torch.isfinite(b).all()),different_elements=int((a!=b).sum()),max_abs=float((a-b).abs().max()),relative_L2=float((a.double()-b.double()).norm()/(a.double().norm()+1e-300))))
 first=next(r for r in details if not r['exact']);put(run/'named_backward_forensics.json',dict(source=failed,source_sha256=sha(path),first_in_parameter_order=first,temporal_kernel_order='NOT DETERMINED by parameter registration order',different_tensors=sum(not r['exact'] for r in details),raw_before_clipping=True,gradient_layout_equal=captured[1]['layout']==captured[2]['layout'],clip_norms=[captured[1]['preclip_norm'],captured[2]['preclip_norm']],all_finite=all(not r['nonfinite'] for r in details),details=details,interpretation='raw mismatch excludes clipping as sole origin; no unique GEMM/attention/accumulation cause follows'))
 # Verify immutable original question/answer fields directly against cached official archives.
 raw=ROOT/'data/raw/vqav2';official={}
 for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
  with zipfile.ZipFile(raw/filename) as z:member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));official[key]={str(r['question_id']):r for r in json.loads(z.read(member))[key]}
 checked=0
 def verify(r):
  key=str(r.get('sample_id',r.get('question_id')));q=official['questions'][key];a=official['annotations'][key];assert str(q['image_id'])==str(a['image_id'])==str(r.get('image_id',r.get('source_image_id')));assert r['question']==q['question'];assert r['answers']==[v['answer'] for v in a['answers']]
 split=read(ROOT/'results/small_scale_validation_20261031T1928Z_v1/split_manifest.json')['splits']
 for rows in split.values():
  for r in rows:verify(r);checked+=1
 candidates=read(ROOT/'results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json')['records']
 for r in candidates:verify(r);checked+=1
 put(run/'original_fields_verified.json',dict(status='PASS_FIELD_RETENTION_ONLY',official_join_and_question_and_all_ten_answers=checked,latest_split=80,matched_conditions=len(candidates),edited_visual_answer_validity='NOT VALIDATED; retaining original labels does not prove them visually correct'))
 kit=ROOT/'results/next_validation_20261101T0040Z_v1/semantic_review32';reviews,_=validated_reviews(sorted((kit/'submitted').glob('*.csv')),read(kit/'blinded_manifest.json')['cases']);put(run/'independent_semantic_status.json',dict(status='BLOCKED' if len(reviews['final_labels'])<32 else 'REQUIRES_INDEPENDENCE_AUDIT',reviewed_pairs=len(reviews['final_labels']),review=reviews,kit=str(kit.relative_to(ROOT)),selection_sha256=sha(kit/'selection_receipt.json'),no_generated_labels=True))
 versions=[]
 for directory in (ROOT/'data/processed').iterdir():
  if not directory.is_dir():continue
  ds={}
  for p in directory.glob('*.json'):
   try:v=read(p)
   except ValueError:continue
   if isinstance(v,list) and v and isinstance(v[0],dict) and 'image_id' in v[0] and 'answers' in v[0]:ds[p.stem]=v
  if ds:
   overlaps=[]
   for i,(name,rr) in enumerate(ds.items()):
    for other,oo in list(ds.items())[:i]:overlaps.append(dict(a=name,b=other,image_ids=len({str(r['image_id']) for r in rr}&{str(r['image_id']) for r in oo}),recorded_content_hashes=len({r.get('image_sha256') for r in rr if r.get('image_sha256')}&{r.get('image_sha256') for r in oo if r.get('image_sha256')})))
   versions.append(dict(version=directory.name,counts={n:len(v) for n,v in ds.items()},manifest_sha256={n:sha(directory/(n+'.json')) for n in ds},overlaps=overlaps))
 put(run/'data_versions_verified.json',dict(versions=versions,latest80_exposure=read(ROOT/'results/audit_remediation_20261101T0120Z_v1/cpu_audit/data_leakage.json'),warning='Internal source/content disjointness is distinct from prior project exposure and foundation-model nonexposure.'))
 rejects=[]
 for r in [{},{'training_eligible':False},{'training_eligible':True}]:
  try:require_training_reward(r)
  except RuntimeError:rejects.append(True)
  else:rejects.append(False)
 assert all(rejects);assert sha(ROOT/'src/rl/phase5_backend.py')=='73b480ae552990e6d41c560713f2c47137379d7f7ec7049b64fe6c7709552712'
 supplied=run/'audit/supplied_method_snapshot';man=(supplied/'manuscript_integrated.tex').read_text(encoding='utf-8');alg=(supplied/'algorithm1.tex').read_text(encoding='utf-8');fig=read(supplied/'figure_spec.json');texts=[r.get('text','') for r in fig['items']];put(run/'manuscript_alignment.json',dict(source_hashes={p.name:sha(p) for p in supplied.iterdir() if p.is_file()},manuscript_GRPO_baseline='We can use GRPO' in man,algorithm_separate_policy_editor='\\pi_\\phi' in alg and 'G_\\psi' in alg,figure_text=texts,figure_has_one_generator='A  RL-Trained Image Generator' in texts,unified_current_snapshot=False,reason='latest Figure1 unifies G_psi/pi_psi; conceptual Algorithm1 and manuscript retain separate phi/psi and proposed GRPO; latest figure is not yet integrated into existing manuscript source'))
 result=dict(status='PASS_CPU_VERIFICATION_ONLY',failed_parent=failed,first_different_parameter=first,optimizer_parameter_order=names,reward_guards_all_rejected=rejects,independent_semantic_pairs=len(reviews['final_labels']),LPIPS_available=importlib.util.find_spec('lpips') is not None,first_divergence='raw_backward_before_clipping; not localized to primitive',GPU_probe_eligible=True,controlled_image_edit_training_eligible=False,blocked_reasons=['No genuine independent edited-image labels','Ordinary full-graph pretrained continuation not accepted','No unused real-image final test'],no_new_model_calls=True)
 put(run/'cpu_verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='optimizer_parameter_order'}))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();main(ROOT/'results'/a.run_id)
