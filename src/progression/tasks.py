"""Whitelisted CPU stages. No GPU/model service/attack optimization entry points."""
import copy
import csv
import hashlib
import html
import json
import random
import shutil
import statistics
import subprocess
import time
import zipfile
from collections import Counter,defaultdict
from pathlib import Path
import yaml
from ..utils.logging import Experiment,write_json,utcnow
from ..data.dataset import sha256
from ..data.dataset_v2 import VerifiedVQADataset,validate_content_disjoint
from ..data.semantic_contract_v2 import candidate_categories,eligible_for_semantic_study
from .controller import policy

PARENTS=['EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL',
 'EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL',
 'EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI',
 'EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI',
 'EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR']
DIMENSIONS=['instruction_completion','image_realism','content_preservation','question_answer_factual_validity',
 'image_question_semantic_relationship','visible_artifacts','evaluation_uncertainty']

def saved_observations(root,train_only=False):
    rows=[]
    for name in PARENTS:
        metrics=json.loads((root/'experiments'/name/'metrics.json').read_text())
        for section in ['updates','evaluations']:
            if train_only and section!='updates':continue
            for block in metrics.get(section,[]):
                for index,o in enumerate(block.get('observations',[])):
                    rows.append({**o,'parent_experiment_id':name,'observation_population':section,
                        'update':block['step'],'episode_index':index})
    return rows

def execute(root,stage,function):
    config={'phase':6,'stage':stage,'seed':42,'gpu_seconds':0,'new_model_calls':0}
    source_ids=PARENTS[:4] if stage=='S3_RL_ENGINEERING' else PARENTS if stage=='S1_REWARD_AUDIT' else PARENTS[:2] if stage=='S1_REVIEW_PACKAGE' else PARENTS[:1] if stage=='S2_REAL_HARNESS' else []
    source_provenance=[{'experiment_id':name,'manifest':json.loads((root/'experiments'/name/'manifest.json').read_text())} for name in source_ids]
    prior_runs=sorted((root/'experiments').glob('EXP_*phase6_'+stage.lower()))
    parent=prior_runs[-1].name if prior_runs else None
    provenance={'stage':stage,'dataset_version':'vqav2_content_v2' if stage=='S1_DATA' else 'historical_sources_recorded_separately',
       'reward_version':'legacy_phase4_saved_arithmetic' if stage=='S1_REWARD_AUDIT' else 'not_executed_cpu_audit',
       'harness_version':'phase6_workflow_v1','evaluation_protocol':'phase6_cpu_audit_v1',
       'model_revisions':{'Qwen/Qwen3-VL-2B-Instruct':'89644892e4d85e24eaac8bacfd4f463576704203',
        'timbrooks/instruct-pix2pix':'31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'}}
    start=time.perf_counter();cpu=time.process_time()
    with Experiment(root,'phase6_'+stage.lower(),config,provenance,parent=parent) as run:
        write_json(root/f'reports/phase6/attempts/{stage}.json',{'stage':stage,'experiment_id':run.id,
           'created_at':run.manifest['created_at'],'slurm_job_id':run.manifest['slurm_job_id']})
        try:
            output=function(root,run)
        except Exception as error:
            error.phase6_experiment_id=run.id
            run.metrics({'failure_reason':type(error).__name__+': '+str(error),'gpu_seconds':0})
            write_json(run.path/'phase6_run_contract.json',{'experiment_id':run.id,'stage':stage,
              'parent_experiment_id':parent,'source_experiment_ids':source_ids,'start_time':run.manifest['created_at'],
              'end_time':utcnow(),'status':'FAIL','model_revision':provenance['model_revisions'],
              'dataset_version':provenance['dataset_version'],'reward_version':provenance['reward_version'],
              'harness_version':provenance['harness_version'],'git_commit':run.manifest['git_commit'],'seed':42,
              'slurm_job_id':run.manifest['slurm_job_id'],'gpu_seconds':0,'configuration':config,'metrics':'metrics.json',
              'failure_reason':type(error).__name__+': '+str(error),'checkpoint_path':None,'evaluation_protocol':provenance['evaluation_protocol']})
            raise
        resources={'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,
                   'gpu_seconds':0,'model_calls':0}
        output['resources']=resources
        output['source_provenance']=source_provenance
        output['audit_context_versions']={'dataset':'vqav2_content_v2','reward':'reward_compat_v2_diagnostic_only, not released or executed as training reward'}
        run.metrics(output)
        write_json(run.path/'phase6_run_contract.json',{'experiment_id':run.id,'stage':stage,
          'parent_experiment_id':parent,'source_experiment_ids':source_ids,'start_time':run.manifest['created_at'],
          'model_revision':provenance['model_revisions'],'dataset_version':provenance['dataset_version'],
          'reward_version':provenance['reward_version'],'harness_version':provenance['harness_version'],
          'git_commit':run.manifest['git_commit'],'seed':42,'slurm_job_id':run.manifest['slurm_job_id'],
          'gpu_seconds':0,'configuration':config,'metrics':'metrics.json','failure_reason':None,
          'checkpoint_path':None,'evaluation_protocol':provenance['evaluation_protocol'],
          'end_time':utcnow(),'status':'PASS'})
        artifact_hashes={p.relative_to(run.path).as_posix():sha256(p) for p in run.path.rglob('*') if p.is_file()
            and p.name not in {'run.log','train.log','status.json','summary.md'}}
        write_json(run.path/'output_manifest.json',{'experiment_id':run.id,'files':artifact_hashes,'required_metrics':list(output)})
        run.finish(summary='Phase6 bounded CPU audit; semantic/scientific readiness not implied.')
        receipt={'experiment_id':run.id,'path':str(run.path),'results':output,'output_manifest_sha256':sha256(run.path/'output_manifest.json')}
        write_json(root/f'reports/phase6/{stage}.json',receipt)
    return {'state':'PASSED','experiment_id':run.id,'evidence':f'reports/phase6/{stage}.json',
       'evaluation_outputs':output,'completion_evidence_state':'PENDING_ALLOCATION_FINALIZATION',
       'scope':'provisional in-process CPU task checks; final accounting/output verification remains separate'}

def validate_data(root,run):
    base=root/'data/processed/vqav2_content_v2';provenance=json.loads((base/'provenance.json').read_text())
    manifests={};counts={}
    for split,h in provenance['split_sha256'].items():
        dataset=VerifiedVQADataset(base/f'{split}.json',root,split,h)
        rows=dataset.records;manifests[split]=rows
        hashes=Counter(r['image_sha256'] for r in rows);ids=Counter(r['image_id'] for r in rows)
        counts[split]={'records':len(rows),'unique_image_ids':len(ids),'content_groups':len(hashes),
          'repeated_content_records':sum(v-1 for v in hashes.values()),'repeated_id_records':sum(v-1 for v in ids.values()),
          'independently_eligible':sum(eligible_for_semantic_study(r['semantic_contract']) for r in rows)}
    validate_content_disjoint(manifests)
    from ..data.prepare_vqa import OFFICIAL
    archives={}
    for kind,url in OFFICIAL.items():
        with zipfile.ZipFile(root/'data/raw/vqav2'/Path(url).name) as z:
            member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
            archives[kind]=json.loads(z.read(member))
    questions={str(q['question_id']):q for q in archives['questions']['questions']}
    answers={str(a['question_id']):a for a in archives['annotations']['annotations']}
    for rows in manifests.values():
        for r in rows:
            q=questions[r['sample_id']];a=answers[r['sample_id']]
            assert str(q['image_id'])==str(a['image_id'])==r['image_id'] and q['question']==r['question']
            assert [x['answer'] for x in a['answers']]==r['answers']
    # Fixed sampling groups content before drawing ten distinct sources.
    unique={r['image_sha256']:r for r in manifests['train']}
    def sample():return random.Random(42).sample(sorted(unique),10)
    assert sample()==sample() and len(set(sample()))==10
    oldtest=json.loads((root/'data/processed/vqav2_seed42_v1/test.json').read_text())
    assert {r['sample_id'] for r in oldtest}<={r['sample_id'] for r in manifests['test']}
    return {'counts':counts,'official_joins_reverified':sum(len(v) for v in manifests.values()),
      'content_split_isolation':True,'deterministic_source_batch10':sample(),
      'old_test_members_preserved':True,'independent_semantic_truth_available':False,
      'edited_answers_not_inferred':True,'new_manifest_needed':False,'historically_exposed_dev_gate':True}

def training_selection(root):
    train=json.loads((root/'data/processed/vqav2_content_v2/train.json').read_text())
    allowed={r['sample_id']:r for r in train};selected=[];seen=set()
    for o in saved_observations(root,True):
        r=allowed.get(o['sample_id'])
        if r and r['image_sha256'] not in seen:
            assert r['question']==o['question'] and r['answers']==o['answers'] and r['image_id']==o['source_image_id']
            selected.append((o,r));seen.add(r['image_sha256'])
    return selected

def review_package(root,run):
    rows=training_selection(root)
    def priority(pair):
        o,r=pair
        return (-int(bool(o['semantic'].get('disagreements'))),-int(o['reward']['reward']==-1),-o['quality']['distortion'],o['sample_id'])
    chosen=sorted(rows,key=priority)[:policy(root)['review_pair_limit']]
    cards=[];queue=[]
    for i,(o,r) in enumerate(chosen):
        item={'pair_id':f'REVIEW_{i:03d}','rank':i+1,'sample_id':r['sample_id'],'source_image_sha256':r['image_sha256'],
          'parent_experiment_id':o['parent_experiment_id'],'split':'train','question':o['question'],
          'original_dataset_answers':o['answers'],'instruction':o['instruction'],
          'automatic_proxies':{'semantic':o['semantic'],'quality':o['quality'],'reward':o['reward'],
            'answer_consistency':o['answer_consistency'],'structured_judgment':o.get('structured_judgment')},
          'priority_reasons':{'disagreement':bool(o['semantic'].get('disagreements')),'clipped':o['reward']['reward']==-1,'distortion':o['quality']['distortion']},
          'independent_reviewer':{k:None for k in DIMENSIONS+['edited_factual_answer','reviewer_id','second_reviewer_id','disagreement','adjudication','reason','timestamp']},
          'human_status':'NEEDS_HUMAN_REVIEW','edited_factual_answer_status':'UNKNOWN'}
        for name,source in [('original',o['source_path']),('edited',o['output_path'])]:
            rel=f'images/{i}_{name}.png';shutil.copy2(source,run.path/rel);item[name+'_image']=rel
            item[name+'_image_sha256']=sha256(run.path/rel)
        queue.append(item)
        fields=''.join(f'<label>{html.escape(k)} <input name="{k}" placeholder="Unrated / 未审核" aria-label="{k}"></label>' for k in DIMENSIONS+['edited_factual_answer','reviewer_id','disagreement','reason'])
        cards.append(f'<article id="{item["pair_id"]}"><h2>{i+1}. {html.escape(o["question"])}</h2><p>{html.escape(o["instruction"])}</p><p>Original answers: {html.escape(str(o["answers"]))}</p><div class="pair"><figure><img src="{item["original_image"]}" alt="Original"><figcaption>Original</figcaption></figure><figure><img src="{item["edited_image"]}" alt="Edited"><figcaption>Edited</figcaption></figure></div><p>Edited factual answer: UNKNOWN. Priority is diagnostic only.</p><details><summary>Automatic proxies (not independent truth)</summary><pre>{html.escape(json.dumps(item["automatic_proxies"],indent=2))}</pre></details><form data-pair="{item["pair_id"]}">{fields}</form></article>')
    write_json(run.path/'review_queue.json',queue)
    with (run.path/'review_form.csv').open('x',newline='',encoding='utf8') as f:
        fields=['pair_id','human_status']+DIMENSIONS+['edited_factual_answer','reviewer_id','second_reviewer_id','disagreement','adjudication','reason','timestamp']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for item in queue:w.writerow({'pair_id':item['pair_id'],'human_status':'NEEDS_HUMAN_REVIEW'})
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Independent image edit review</title><style>body{font:17px system-ui;max-width:1200px;margin:auto;padding:24px;background:#f4f7fa;color:#142638}article{background:white;padding:24px;margin:28px 0;border:1px solid #bbcbd9;border-radius:16px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%}label{display:block;margin:10px}input{width:95%;padding:8px}pre{white-space:pre-wrap;overflow-wrap:anywhere}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><h1>Independent semantic review / 独立语义审核</h1><p>All human fields are blank. Score instruction completion, realism and preservation independently. Supply edited factual answers from the edited image; do not assume original answers remain correct. Use UNKNOWN when unsure. Two reviewers should work independently before adjudication. Do not infer SC/SI-V from Qwen agreement. Automatic proxies are hidden by default to reduce anchoring.</p><p>These input boxes are an export aid only; nothing is submitted or certified automatically. Use <a href="review_form.csv">CSV</a> or export JSON, keep source evidence and obtain independent adjudication.</p><button id="export">Export entered review JSON</button>'''+''.join(cards)+'''<script>document.getElementById('export').onclick=()=>{const rows=[...document.querySelectorAll('form')].map(f=>({pair_id:f.dataset.pair,human_status:'UNADJUDICATED',...Object.fromEntries(new FormData(f))}));const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(rows,null,2)],{type:'application/json'}));a.download='unadjudicated_reviews.json';a.click();};</script></html>'''
    (run.path/'index.html').write_text(document,encoding='utf8')
    return {'pairs':len(queue),'unique_sources':len(queue),'human_verified':0,'all_independent_fields_blank':True,
      'dimensions':DIMENSIONS,'queue':'review_queue.json','review_page':'index.html','selection':'prioritized training-only diagnostic queue, not representative test estimate'}

def distribution(values):
    values=[float(x) for x in values if isinstance(x,(int,float)) and not isinstance(x,bool)]
    if not values:return {'n':0,'mean':None}
    return {'n':len(values),'min':min(values),'max':max(values),'mean':statistics.mean(values),'population_sd':statistics.pstdev(values)}

def reward_audit(root,run):
    import numpy as np
    observations=saved_observations(root);names=['semantic','answer_consistency','distortion','cost','invalid_penalty','reward_unclipped','reward']
    rows=[o['reward'] for o in observations];dist={k:distribution([r.get(k) for r in rows]) for k in names}
    missing={k:sum(r.get(k) is None for r in rows) for k in names+['instruction_success']}
    correlations={}
    for i,a in enumerate(names):
        for b in names[:i]:
            pairs=[(r[a],r[b]) for r in rows if r.get(a) is not None and r.get(b) is not None]
            x,y=np.asarray(pairs).T
            correlations[a+':'+b]=None if np.std(x)==0 or np.std(y)==0 else float(np.corrcoef(x,y)[0,1])
    def arithmetic(r):return r['semantic']+.5*r['answer_consistency']-.2*r['distortion']-.1*r['cost']-r['invalid_penalty']
    first=[arithmetic(r) for r in rows];second=[arithmetic(r) for r in rows]
    assert first==second
    max_error=max(abs(a-r['reward_unclipped']) for a,r in zip(first,rows));assert max_error<1e-12
    category=defaultdict(list);pop=defaultdict(list);repeat=defaultdict(list)
    for o in observations:
        for c in candidate_categories(o['question']) or ['unclassified_keyword']:category[c].append(o['reward']['reward'])
        pop[o['observation_population']].append(o['reward']['reward'])
        repeat[(o['sample_id'],o['instruction'],o['observation_population'])].append(o['reward']['reward'])
    result={'observations':len(rows),'component_distributions':dist,'missing_counts':missing,'correlations':correlations,
       'by_edit_instruction':{instruction:distribution([o['reward']['reward'] for o in observations if o['instruction']==instruction]) for instruction in sorted({o['instruction'] for o in observations})},
       'edit_category_coverage':'one benign lighting instruction only; cross-edit-category efficacy unavailable',
       'constant_components':[k for k in names if dist[k]['population_sd']==0],
       'clip_lower_count':sum(r['reward']==-1 for r in rows),'clip_upper_count':sum(r['reward']==1.5 for r in rows),
       'by_population':{p:{'n':len(v),'lower_clip_count':v.count(-1),'mean':statistics.mean(v)} for p,v in pop.items()},
       'by_keyword_category':{c:distribution(v) for c,v in category.items()},
       'fixed_component_recompute_exact_twice':True,'max_saved_arithmetic_error':max_error,
       'repeat_source_variation_diagnostic':{'groups':len(repeat),'groups_with_multiple_observations':sum(len(v)>1 for v in repeat.values()),
          'max_within_source_reward_range':max(max(v)-min(v) for v in repeat.values()),'scope':'different seeds/checkpoints/outputs, not fixed-output evaluator reliability'},
       'independent_label_agreement':None,'independent_label_count':0,'instruction_success_measurement':'MISSING',
       'same_model_dependency':'Qwen supplies answer and semantic proxies; not independent',
       'scientific_objective_proposal':'Improve independently verified instruction completion while preserving factual/visual constraints; not yet a released scalar reward',
       'reward_release':'BLOCKED','coefficients_changed':False,'scope':'saved-observation construct audit, no retraining'}
    write_json(run.path/'component_audit.json',result);return result

def rl_checks(root,run):
    command=[str(Path(__import__('sys').executable)),'-B','-m','unittest','tests.test_diffusion_rl','tests.test_phase5_contracts','tests.test_phase5_protocol','-v']
    process=subprocess.run(command,cwd=root,capture_output=True,text=True,timeout=180)
    (run.path/'cpu_regression.log').write_text(process.stdout+process.stderr,encoding='utf8')
    assert process.returncode==0,process.stderr[-2500:]
    witnesses=[]
    for name in PARENTS[:4]:
        m=json.loads((root/'experiments'/name/'metrics.json').read_text())
        for u in m['updates']:
            assert u['logical_batch_size']==10 and u['gradient_accumulation_steps']==10 and u['optimizer_updates']==1
            assert u['finite_gradients'] and u['conditioning_verified'] and u['trainable_max_change']>0
            witnesses.append({'experiment':name,'step':u['step'],'gradient_norm':u['gradient_norm'],
                'adapter_change':u['trainable_max_change'],'likelihood_replay_max_difference':u['behavior_logp_replay_max_difference'],
                'logical_sources':10,'trajectories':10,'denoising_transitions':100,'backward_calls':u['backward_transition_calls'],
                'physical_microbatch':u['physical_micro_batch_size'],'optimizer_updates':1})
    return {'cpu_regression_exit':process.returncode,'historical_actual_update_witnesses':witnesses,
      'new_updates':0,'frozen_integrity_scope':'historical frozen-weight evidence retained; no new GPU forward',
      'strict_stochastic_continuation':'UNRESOLVED, previous bitwise criterion unchanged',
      'scientific_training_ready':False,'reward_release':'BLOCKED'}

def comparison(root,run):
    configs=[yaml.safe_load((root/f'configs/phase5_v2/M{i}.yaml').read_text()) for i in range(6)]
    same=['generator_config','split_sha256','dataset_version','seeds','development_sources','private_gate_sources','training_reward_config','backend']
    equal={k:all(v[k]==configs[0][k] for v in configs) for k in same};assert all(equal.values())
    assert all(c['training_source_exposures']==80 and c['optimizer_updates']==8 for c in configs[1:])
    assert configs[2]['scheduler_baseline']['kind']=='predeclared_one_cooldown'
    destination=run.path/'prepared_configs';destination.mkdir()
    for i,c in enumerate(configs):
        v=copy.deepcopy(c);v.update(schema_version='phase6_comparison_proposal_v1',supersedes=f'configs/phase5_v2/M{i}.yaml',
           status='PREPARED_NOT_RUN',launch_authorized=False,independent_evaluation_protocol='phase6_review_protocol_pending_validation',
           initialization='same saved pristine LoRA/optimizer/RNG artifact per matched seed, not yet verified',
           selection='predeclared final update; independent frozen Gate; no best-seed/step selection')
        v['condition']='trace_and_diagnosis_without_revision' if i==3 else v['condition']
        v['rsi_mode']=v['condition'] if i>=3 else 'none'
        (destination/f'M{i}.yaml').write_text(yaml.safe_dump(v,sort_keys=False))
    pilot={'status':'PREPARED_NOT_RUN','scientific_gpu_budget_seconds':0,'launch_authorized':False,
       'matched_arms':['frozen','standard_RL','ordinary_schedule','harness_revision'],'seed':42,'logical_batch_size':10,
       'updates_per_trained_arm':4,'scope':'benign lighting edit only, no security optimization',
       'unresolved':['independent instruction/factual validity','accepted training reward','pristine matched initialization','fresh private evaluation','explicit GPU budget'],
       'runtime_estimate':'requires re-estimation after reward/evaluator release; historical42.20h is not allocation'}
    write_json(run.path/'pilot_proposal.json',pilot)
    return {'static_equalities':equal,'prepared_config_version':'phase6_comparison_proposal_v1',
      'M3_wording_corrected':'trace collection and diagnosis, no revision','equal_source_exposure_trained_arms':True,
      'conventional_scheduler_present':True,'runtime_initialization_comparable':False,
      'natural_image_gate_independent':False,'scientific_comparison_ready':False,'pilot':pilot}

def blocked(reason,state='BLOCKED'):
    return lambda root,stage:{'state':state,'reason':reason,'evaluation_outputs':{'passed':False,'missing_evidence':reason}}

def handlers():
    from .real_harness import real_harness
    functions={'validate_data':validate_data,'review_package':review_package,'reward_audit':reward_audit,
               'real_harness':real_harness,'rl_checks':rl_checks,'comparison':comparison}
    result={name:(lambda root,stage,f=f:execute(root,stage,f)) for name,f in functions.items()}
    result.update(human_gate=blocked('Independent reviewed edited factual answers and seven-dimensional adjudication absent','NEEDS_HUMAN_REVIEW'),
      reward_gate=blocked('Independent reward construct validation and release absent'),
      scientific_gate=blocked('Fresh independent Gate and exact pre-intervention execution comparability absent'),
      pilot_gate=blocked('Explicit GPU budget authorization and scientific gates absent','NEEDS_AUTHORIZATION'))
    return result
