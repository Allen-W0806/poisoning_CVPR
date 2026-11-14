"""New mathematical reward stratification, isolated human handoff and prospective protocol.

No network, model inference, image generation, training, or monitor installation.
Candidate IDs are metadata reservations, never certified independent test results.
"""
import argparse,ast,collections,csv,hashlib,io,json,math,os,random,re,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
KIT=ROOT/'results/next_validation_20261101T0040Z_v1/semantic_review32'
OLD=ROOT/'results/audit_remediation_20261101T0120Z_v1/cpu_audit'
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def digest(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def csvout(p,rows):
 with Path(p).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def reward(run):
 import yaml
 rows=list(csv.DictReader((OLD/'historical_feedback_recomputed.csv').open(encoding='utf-8')));assert len(rows)==500
 for row in rows:
  for k in ['S','A','D','C','U','reward_unclipped','reward']:row[k]=float(row[k])
 assert {x['A'] for x in rows}=={0.,1.} and all(x['S']==x['A']/2 and x['U']==1 and x['C']==1/3 for x in rows)
 errors=[]
 for x in rows:
  simplified=x['A']-x['D']/5-31/30
  if not math.isclose(simplified,x['reward_unclipped'],rel_tol=1e-12,abs_tol=1e-12):errors.append(x['sample_id'])
 assert not errors
 strata=[]
 for a in [0.,1.]:
  group=[x for x in rows if x['A']==a];strata.append(dict(A=a,n=len(group),lower_clipped=sum(x['reward']==-1 for x in group),D_min=min(x['D'] for x in group),D_max=max(x['D'] for x in group),raw_min=min(x['reward_unclipped'] for x in group),raw_max=max(x['reward_unclipped'] for x in group),reward_min=min(x['reward'] for x in group),reward_max=max(x['reward'] for x in group),distinct_D=len({x['D'] for x in group}),distinct_reward=len({x['reward'] for x in group})))
 src=ROOT/'scripts/smoke_generator_rl.py';tree=ast.parse(src.read_text());calls=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='validation_reward':calls.append(dict(line=n.lineno,positional_arguments=len(n.args),keywords=[k.arg for k in n.keywords],explicit_weights=any(k.arg=='weights' for k in n.keywords) or len(n.args)>=5))
 configs={}
 for name in ['configs/generator_rl_smoke.yaml','configs/phase3_generator_rl.yaml','configs/rl.yaml']:
  p=ROOT/name
  if p.exists():configs[name]=dict(sha256=sha(p),recorded_reward=yaml.safe_load(p.read_text()).get('reward'))
 defaults={}
 for n in ast.walk(ast.parse((ROOT/'src/rl/reward.py').read_text())):
  if isinstance(n,ast.ClassDef) and n.name=='RewardWeights':
   defaults={x.target.id:ast.literal_eval(x.value) for x in n.body if isinstance(x,ast.AnnAssign)}
 inputs=read(OLD/'input_hashes.json');metrics={p:h for p,h in inputs.items() if p.startswith('experiments/') and p.endswith('metrics.json')};assert len(metrics)==5 and all(sha(ROOT/p)==h for p,h in metrics.items())
 identity=read(OLD/'identity_and_config_audit.json');best=max(x['reward'] for x in rows)
 result=dict(status='PASS_NEW_MATHEMATICAL_DECOMPOSITION_ONLY',UTC=now(),n=500,assumptions_verified=dict(A_binary=True,S_equals_A_over_2=True,U_is_1=True,C_is_1_over_3=True),reduced_unclipped_formula='A - D/5 - 31/30',A0_formula='clip(-31/30-D/5,-1,1.5) = -1 for D in [0,1]',A1_formula='-1/30-D/5, range [-7/30,-1/30]',strata=strata,every_lower_clipped_row_has_A0=all(x['A']==0 for x in rows if x['reward']==-1),every_A0_row_lower_clipped=all(x['reward']==-1 for x in rows if x['A']==0),loss_of_D_ranking_rows=sum(x['A']==0 for x in rows),max_saved_arithmetic_error_accepted='existing 1e-12 arithmetic recheck',identity_actual_caller_reward=identity['actual_caller_reward'],identity_declared_credit_reward=identity['declared_consistency_credit_reference_reward'],largest_changed_reward=best,identity_actual_gap_over_best_changed=identity['actual_caller_reward']-best,identity_correct_credit_gap_over_best_changed=identity['declared_consistency_credit_reference_reward']-best,actual_call_sites=calls,defaults=defaults,YAML_records=configs,config_wiring='Four-positional-argument call uses RewardWeights defaults; current YAML agreement does not prove YAML is loaded into reward; no coefficients changed',old_raw_metrics_sha_verified=metrics,reward_construct_validated=False,no_op_instruction_success='NOT MEASURED; no independent labels',source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [src,ROOT/'src/rl/reward.py',ROOT/'src/rl/reward_normalization.py']},new_model_calls=0,new_GPU_seconds=0)
 put(run/'reward_mathematical_decomposition.json',result);csvout(run/'reward_strata.csv',strata);return result

def handoff(run):
 manifest=read(KIT/'blinded_manifest.json');cases=manifest['cases'];assert len(cases)==32 and not manifest['predictions_included']
 forbidden={'raw_vlm_answer','official_vqa_consensus_score','all_attempted_utility_score','risk_darkening','risk_low_SSIM','reviewer_labels'}
 assert not any(forbidden.intersection(c) for c in cases)
 html=(KIT/'index.html').read_text();assert "'independent-semantic-32-next_validation_20261101T0040Z_v1:'" in html
 selection=read(KIT/'selection_receipt.json');assert {c['source_image_id'] for c in cases}==set(selection['selected_ids'])
 templates=[list(csv.DictReader((KIT/f'reviewer_{i}_template.csv').open(encoding='utf-8'))) for i in [1,2]]
 assert all(len(x)==32 and not any(r['reviewer_id'] or r['semantic_label'] or r['reviewed_utc'] for r in x) for x in templates)
 assert all({x['source_image_id'] for x in rr}==set(selection['selected_ids']) for rr in templates)
 submission_files=[p for p in (KIT/'submitted').rglob('*') if p.is_file()]
 assets={}
 for c in cases:
  for k in ['original','edited','geometry_control']:
   n=c[k];p=KIT/n;assert p.resolve().is_relative_to(KIT.resolve())
   if k!='geometry_control':assert sha(p)==c[k+'_sha256']
   assets[n]=p
 out=run/'review_handoff';out.mkdir(exist_ok=False);bundles=[]
 readme='''# 32对图像独立人工盲审交接

解压后打开index.html；此材料只供一名真实审阅者。另一名审阅者使用另一个独立包。
1. 使用不同的真实稳定Reviewer ID；只有确实独立工作时才能勾选独立性声明。
2. 每例查看原图、编辑图及原始VQA题目/十答案；需要时打开全分辨率和几何控制。
3. 逐项判断场景、对象、身份/数量、颜色/空间、原参考答案是否仍有效、指令是否完成，标明valid/invalid/ambiguous及置信度；不可用模型输出替代判断。
4. 无效/歧义必须给出原因。不确定就标ambiguous，不能为了通过而标valid。
5. 每例点击Save assessment再导航。完成后导出CSV和JSON，检查32行、Reviewer ID及时间齐全。浏览器本地保存不是向研究项目提交。
6. 将原始导出文件分别交协调者；审阅结束前不看另一人的标签、模型预测、分数、风险flag或实验报告。不要用同一人两次填写冒充双审。

两个包只隔离浏览器存储命名空间和下载文件名，不改变样本、原始答案、审核rubric或默认值；全部人工判断字段初始为空。包中原始十答案是数据集参考，不是新的人类编辑审核结论。
盲法范围：隐藏模型预测/评分/自动风险和对方标签；原图/编辑图角色与编辑指令可见，以便完成任务。

协调者把两个原始导出文件分别保存到原项目
results/next_validation_20261101T0040Z_v1/semantic_review32/submitted/
不修改或覆盖空模板。不同审阅者必须有不同ID并真实独立。冲突保留两份原始意见，交第三名独立人工裁决；不得由模型生成裁决。原有监控只在真实证据内容变化时重审，不应重跑已完成GPU诊断。
不要给审阅者实验负面结论、预测、统计图或另一人的导出。
'''
 for i,label in enumerate(['reviewer_A','reviewer_B']):
  prefix='independent-semantic-32-'+run.name+'-'+label+':'
  page=html.replace('independent-semantic-32-next_validation_20261101T0040Z_v1:',prefix).replace("'semantic_review_'+", "'semantic_review_"+label+"_'+")
  members={'index.html':page.encode(),'READ_ME_ZH.md':readme.encode(),'blinded_manifest.json':(KIT/'blinded_manifest.json').read_bytes(),'reviewer_template.csv':(KIT/f'reviewer_{i+1}_template.csv').read_bytes()}
  checksums={n:digest(b) for n,b in members.items()};checksums.update({n:sha(p) for n,p in assets.items()})
  info=dict(role=label,parent_manifest_sha256=sha(KIT/'blinded_manifest.json'),selection_receipt_sha256=sha(KIT/'selection_receipt.json'),cases=32,human_labels_generated=0,storage_namespace=prefix,members=checksums)
  target=out/(label+'.zip')
  with zipfile.ZipFile(target,'x',compression=zipfile.ZIP_DEFLATED) as z:
   for n,b in members.items():z.writestr(n,b)
   for n,p in assets.items():z.write(p,n)
   z.writestr('bundle_manifest.json',json.dumps(info,indent=2))
  with zipfile.ZipFile(target) as z:
   assert z.testzip() is None
   assert all(digest(z.read(n))==h for n,h in checksums.items())
  bundles.append(dict(role=label,path=str(target.relative_to(ROOT)),bytes=target.stat().st_size,sha256=sha(target),storage_namespace=prefix))
 (out/'COORDINATOR_HANDOFF_ZH.md').write_text(readme)
 result=dict(status='READY_FOR_TWO_REAL_HUMANS',created_UTC=now(),cases=32,unchanged_image_assets=len(assets),image_asset_bytes=sum(p.stat().st_size for p in assets.values()),genuine_labels_received=0 if not submission_files else 'REQUIRES_VALIDATOR',submission_files=[str(p.relative_to(ROOT)) for p in submission_files],new_human_labels_generated=0,blinding='Predictions/scores/risk flags/other labels excluded; original versus edited role and instruction visible',UI_change='Only independent storage namespaces and exported filename prefixes; original kit untouched',bundles=bundles,parent_manifest_sha256=sha(KIT/'blinded_manifest.json'),coordinator_workflow='Two independent original exports; preserve disagreement; third independent human adjudication; no invented labels')
 put(out/'handoff_receipt.json',result);return result

def census(run):
 from src.evaluation.vqa_official_compatible import normalize_answer
 raw=ROOT/'data/raw/vqav2'
 def archive(name,key):
  with zipfile.ZipFile(raw/name) as z:
   n=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'));return json.loads(z.read(n))[key]
 q=archive('v2_Questions_Train_mscoco.zip','questions');question_image={int(x['question_id']):str(x['image_id']) for x in q};del q
 ids=set();questions=set();hashes=set();files=[];errors=[];markdown_files=0
 idkeys={'image_id','source_image_id','source_image_ids','image_ids','source_ids','original_image_id','used_image_ids','selected_ids'}
 qkeys={'sample_id','question_id','sample_ids','question_ids','used_question_ids'}
 hkeys={'image_sha256','source_image_sha256','original_image_hash','candidate_image_hash','original_sha256','expected_sha256','used_content_hashes'}
 def collect(v,key=''):
  if isinstance(v,dict):
   for k,x in v.items():collect(x,k)
  elif isinstance(v,list):
   for x in v:collect(x,key)
  elif isinstance(v,(str,int)):
   s=str(v)
   if key in idkeys|qkeys:
    for word in re.split(r'[;,\s]+',s):
     if word.isdigit():
      if key in idkeys:ids.add(str(int(word)))
      else:questions.add(int(word))
   if key in hkeys and re.fullmatch('[a-fA-F0-9]{64}',s):hashes.add(s.lower())
   if s.startswith(('[','{')):
    try:collect(json.loads(s),key)
    except ValueError:pass
 def lexical(text):
  ids.update(str(int(x)) for x in re.findall(r'COCO_(?:train|val)2014_(\d{12})',text))
  ids.update(str(int(x)) for x in re.findall(r'(?:source_image_id|image_id)[\s\x22\x27:=]+(\d+)',text))
  questions.update(int(x) for x in re.findall(r'(?:question_id|sample_id|question)[\s\x22\x27:=]*(\d{7,})',text))
 for top in ['data/processed','data/manifests','experiments','reports','results','dashboard','dashboard_package','chatgpt_reports']:
  for p in sorted((ROOT/top).rglob('*')):
   if not p.is_file() or run in p.parents or p.suffix.lower() not in {'.json','.jsonl','.csv','.md','.html'}:continue
   if 'drive_auth_recovery' in p.parts or 'source_snapshot' in p.parts or 'source_versions' in p.parts or 'DRIVE_SYNC' in p.name:continue
   b=p.read_bytes();record=dict(path=p.relative_to(ROOT).as_posix(),bytes=len(b),sha256=digest(b),parse='OK')
   try:
    text=b.decode('utf-8-sig');lexical(text)
    if p.suffix=='.json':collect(json.loads(text))
    elif p.suffix=='.jsonl':
     for line in text.splitlines():
      if line.strip():collect(json.loads(line))
    elif p.suffix=='.csv':
     for row in csv.DictReader(io.StringIO(text)):collect(row)
    else:
     markdown_files+=1
     # Embedded JSON in HTML is positive evidence even when no standalone JSON is present.
     for block in re.findall(r'<script[^>]*type=[\x22\x27]application/json[\x22\x27][^>]*>(.*?)</script>',text,re.S):collect(json.loads(block))
   except Exception as e:record['parse']='UNRESOLVED';errors.append(dict(path=record['path'],bytes=len(b),sha256=record['sha256'],error=type(e).__name__,positive_lexical_evidence_still_collected=True))
   files.append(record)
 ids.update(question_image[x] for x in questions if x in question_image)
 annotations=archive('v2_Annotations_Train_mscoco.zip','annotations');buckets=collections.defaultdict(dict)
 for a in annotations:
  iid=str(a['image_id'])
  if iid in ids:continue
  if len(a['answers'])!=10 or max(collections.Counter(normalize_answer(x['answer']) for x in a['answers']).values())<6:continue
  cat=a['answer_type'];qid=int(a['question_id']);old=buckets[cat].get(iid)
  if old is None or qid<old:buckets[cat][iid]=qid
 del annotations
 # Scope, seed, quotas and power gate are written before choosing candidate IDs.
 protocol=dict(schema='prospective_benign_matched_evaluation_v1',created_UTC=now(),status='DRAFT_LOCKED_ADMISSION_NOT_RELEASED',objective='Same original VQA and inherited-answer-valid benign edit evaluation; no attack/ASR optimization and no scientific loss change',sampling='Preserve existing diagnostic category quotas53/53/54 and >=6/10 original-answer consensus; not a population representative VQA benchmark',seed=202611011905,planned_unique_sources=160,planned_batches=10,batch_size=16,conditions='Original; any benign edits require independently confirmed answer preservation and instruction labels before semantic ranking',question_rule='Smallest eligible official question_id within each image/category; source sampled without model scores; preserve original question/ten answers at future acquisition',statistical_unit='source image; paired common sources for arms; repeated inference/training seeds are not extra independent images',score='Pinned official-compatible VQA leave-one-out consensus; exact_any separate; no label-based extraction',confidence='Predeclare paired source bootstrap and seed clustering before new inference; final sample size/power and minimum meaningful effect must be frozen separately before release',power_status='BLOCKED_NOT_SPECIFIED_BY_CURRENT_METHOD;160 is diagnostic size, not established scientific power',admission_gates=['reconcile every unreadable historical record or documented complete independent witness','download/freeze actual image bytes, license/provenance and SHA; exclude ID AND content duplicates','complete perceptual near-duplicate screen and human confirmation where required','bind every historical training/inference/feedback/model-selection job to its exposure manifest','freeze model/checkpoint, preprocessing, seeds, decoding, denominators, budget and acceptance before inference','separate feedback/development sources from sealed final-test sources','no selection/tuning using final-test predictions; retire exposed final sets','independent valid edited labels required; no fabricated reviewers'],census_limitations=['Positive exposure is auditable; absence in logs is not proof of global nonexposure','Raw upstream annotation archive membership is not itself an executed project split','Full foundation-model pretraining contamination unknown','Binary checkpoint data provenance still requires job/manifest lineage'],source_archives={n:sha(raw/n) for n in ['v2_Questions_Train_mscoco.zip','v2_Annotations_Train_mscoco.zip']},new_images_downloaded=0,new_VLM_calls=0,new_GPU_seconds=0,scientific_loss_unchanged=True)
 out=run/'prospective_evaluation';out.mkdir(exist_ok=False);put(out/'protocol.json',protocol)
 rng=random.Random(protocol['seed']);used=set(ids);chosen=[]
 for category,quota in [('yes/no',53),('number',53),('other',54)]:
  pool=sorted(buckets[category],key=int);rng.shuffle(pool);count=0
  for iid in pool:
   if iid in used:continue
   used.add(iid);chosen.append(dict(source_image_id=iid,question_id=str(buckets[category][iid]),answer_type=category,eligibility='CANDIDATE_METADATA_ONLY_NOT_CERTIFIED',prior_known_source_or_question_overlap=False,image_sha256=None,image_downloaded=False));count+=1
   if count==quota:break
  assert count==quota
 rng.shuffle(chosen)
 for i,x in enumerate(chosen):x['diagnostic_batch_id']=i//16
 assert len(chosen)==160 and not {x['source_image_id'] for x in chosen}&ids
 put(out/'candidate_reservations_METADATA_ONLY.json',dict(protocol_sha256=sha(out/'protocol.json'),created_UTC=now(),status='NOT_AN_INDEPENDENCE_CERTIFICATE_NO_INFERENCE_ALLOWED',records=chosen))
 csvout(out/'exposure_file_audit.csv',files)
 put(out/'unresolved_historical_records.json',errors)
 put(out/'known_exposure_exclusions.json',dict(source_image_ids=sorted(ids,key=int),question_ids=sorted(questions),content_sha256=sorted(hashes)))
 result=dict(status='BLOCKED_INDEPENDENCE_ADMISSION',created_UTC=now(),scanned_files=len(files),text_or_HTML_files=markdown_files,known_source_ids=len(ids),known_question_ids=len(questions),known_content_hashes=len(hashes),unresolved_records=errors,unresolved_nonempty_records=sum(e['bytes']>0 for e in errors),candidate_reservations=160,known_ID_overlap=0,content_independence='NOT CHECKED_NO_NEW_IMAGE_BYTES',power='NOT SPECIFIED',new_heldout_results=0,new_images_downloaded=0,new_model_calls=0,new_GPU_seconds=0,protocol_sha256=sha(out/'protocol.json'),reservation_sha256=sha(out/'candidate_reservations_METADATA_ONLY.json'))
 put(out/'admission_status.json',result);return result

def main(run):
 started=time.perf_counter();assert os.getenv('CUDA_VISIBLE_DEVICES')==''
 csv.field_size_limit(32*1024*1024)
 protocol=dict(created_UTC=now(),scope='New mathematical stratification, independent reviewer packaging and prospective exposure admission',script_sha256=sha(Path(__file__)),Slurm_job_id=os.getenv('SLURM_JOB_ID'),no_GPU_no_model_calls=True)
 put(run/'preparation_protocol.json',protocol)
 result=dict(reward=reward(run),review=handoff(run),evaluation=census(run),seconds=time.perf_counter()-started,new_GPU_seconds=0)
 put(run/'preparation_summary.json',result);print(json.dumps(dict(status='CPU_PREPARATION_COMPLETE_SCIENTIFIC_GATES_BLOCKED',seconds=result['seconds'],floor_equivalence=result['reward']['every_A0_row_lower_clipped'],review_bundles=len(result['review']['bundles']),candidate_reservations=160,unresolved_records=len(result['evaluation']['unresolved_records']),new_GPU_seconds=0)),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('run');a=p.parse_args();main(Path(a.run))
