"""New scoped Phase6 follow-up audits; completed Phase6 stages are never rerun."""
import hashlib
import html
import json
import statistics
from pathlib import Path
from PIL import Image,ImageOps
from ..data.dataset import sha256
from ..utils.logging import write_json
from ..monitoring.common import read_json
from .review_workflow import DIMENSIONS

def canonical_path(root,value):
    path=Path(value)
    if path.is_relative_to(Path('/storage/home/ydk297/projects/poisoning')):
        return root/path.relative_to('/storage/home/ydk297/projects/poisoning')
    return path if path.is_absolute() else root/path

def prepare_review_workflow(root):
    root=Path(root);receipt=read_json(root/'reports/phase6/S1_REVIEW_PACKAGE.json');package=root/'experiments'/receipt['experiment_id']
    queue=read_json(package/'review_queue.json');assert len(queue)==24
    trainpath=root/'data/processed/vqav2_content_v2/train.json'
    if not trainpath.exists():trainpath=root/'data/manifests/cluster/vqav2_content_v2/train.json'
    train={r['sample_id']:r for r in read_json(trainpath)}
    parents={};pairs=[];pixel_checked=0
    for item in queue:
        parent=item['parent_experiment_id']
        if parent not in parents:
            p=root/'experiments'/parent
            parents[parent]=(read_json(p/'manifest.json'),read_json(p/'metrics.json'))
        model,metrics=parents[parent];matches=[]
        for update in metrics['updates']:
            for observation in update['observations']:
                if observation['sample_id']==item['sample_id'] and sha256(canonical_path(root,observation['output_path']))==item['edited_image_sha256']:
                    matches.append((update['step'],observation))
        assert len(matches)==1,(item['pair_id'],'ambiguous exact edited episode')
        step,observation=matches[0];record=train[item['sample_id']]
        assert observation['question']==item['question']==record['question']
        assert observation['answers']==item['original_dataset_answers']==record['answers']
        assert observation['instruction']==item['instruction'] and observation['source_image_id']==record['image_id']
        assert record['image_sha256']==item['source_image_sha256']
        for name,key in [('original','source_path'),('edited','output_path')]:
            assert sha256(package/item[name+'_image'])==item[name+'_image_sha256']==sha256(canonical_path(root,observation[key]))
        raw=canonical_path(root,record['image_path'])
        if raw.exists():
            with Image.open(raw) as a,Image.open(package/item['original_image']) as b:
                expected=ImageOps.pad(a.convert('RGB'),b.size,color='white',method=3)
                assert expected.tobytes()==b.convert('RGB').tobytes()
                pixel_checked+=1
        assert all(v is None for v in item['independent_reviewer'].values())
        revisions=model['model_revisions'];assert revisions['Qwen/Qwen3-VL-2B-Instruct']==observation['qwen_checkpoint']
        pair={**item,'model_metadata':revisions,'episode':{'step':step,'seed':observation['seed'],'policy_version':observation['policy_version'],
          'trajectory_id':observation['trajectory_id'],'settings':observation['settings'],'qwen_inference':observation.get('qwen_inference')},
          'original_image':'../../experiments/'+receipt['experiment_id']+'/'+item['original_image'],
          'edited_image':'../../experiments/'+receipt['experiment_id']+'/'+item['edited_image'],
          'automatic_metrics':{**observation['quality'],'semantic_proxy':observation['reward']['semantic'],
            'answer_consistency_proxy':observation['reward']['answer_consistency'],'historical_scalar_reward':observation['reward']['reward']},
          'automatic_edited_answer':observation['answer_consistency'].get('edited_answer'),
          'edited_factual_answer':{'status':'UNKNOWN','value':None},'semantic_label':'UNKNOWN'}
        pairs.append(pair)
    manifest={'schema_version':'independent_review_packet_v1','source_package_experiment':receipt['experiment_id'],'dimension_choices':DIMENSIONS,
      'pairs':pairs,'audit':{'pairs':24,'image_files_verified':48,'exact_episode_matches':24,'raw_original_pixel_associations_checked':pixel_checked,
       'all_reference_fields_match':True,'all_unreviewed_answers_unknown':True,'new_model_calls':0}}
    base=root/'reviews/workflow';base.mkdir(parents=True,exist_ok=True);write_json(base/'manifest.json',manifest)
    render_review(base,manifest)
    return manifest['audit']

def render_review(base,manifest):
    cards=[]
    for pair in manifest['pairs']:
        fields=''.join('<label>'+html.escape(name)+'<select name="'+name+'">'+''.join('<option>'+v+'</option>' for v in choices)+'</select></label>' for name,choices in DIMENSIONS.items())
        cards.append(f'''<article data-pair="{pair['pair_id']}"><h2>{pair['rank']}. {html.escape(pair['question'])}</h2><p>{html.escape(pair['instruction'])}</p><p>Original reference answers: {html.escape(str(pair['original_dataset_answers']))}</p><div class="pair"><figure><img src="{pair['original_image']}" alt="Original"><figcaption>Original</figcaption></figure><figure><img src="{pair['edited_image']}" alt="Edited"><figcaption>Edited</figcaption></figure></div><form>{fields}<label>Edited factual answers, one per line; leave empty for UNKNOWN<textarea name="edited_answer"></textarea></label><label>Reason, uncertainty or disagreement<textarea name="reason"></textarea></label></form><details><summary>Model and exact episode provenance</summary><pre>{html.escape(json.dumps({'models':pair['model_metadata'],'episode':pair['episode']},indent=2))}</pre></details><details><summary>Automatic proxies: diagnostic only, open after independent assessment</summary><pre>{html.escape(json.dumps(pair['automatic_proxies'],indent=2))}</pre></details></article>''')
    encoded=json.dumps(manifest,ensure_ascii=False).replace('<','\\u003c')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Independent image review workflow</title><style>*{box-sizing:border-box}body{font:17px system-ui;background:#f0f5f8;color:#182e40;max-width:1200px;margin:auto;padding:24px}article,header{padding:24px;border-radius:16px;background:white;margin:20px 0}label{display:block;margin:12px 0}select,input,textarea,button{font:inherit;padding:8px}input,textarea{width:100%}.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px}figure{margin:0}img{width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.toolbar{position:sticky;top:0;background:#dbeaf2;padding:14px;border-radius:12px;z-index:2}@media(max-width:650px){.pair{grid-template-columns:1fr}}</style><header><h1>Independent review / 独立审核</h1><p>Seven dimensions are separate. UNKNOWN is a valid unresolved response. Original references are not edited facts. Assess without automatic/model assistance, record reasons, then export. Two reviewers should work separately; disagreements require explicit adjudication. Draft save stays in this browser; exports are not uploaded or certified.</p><label>Reviewer ID<input id="reviewer" autocomplete="off"></label><label><input id="independent" type="checkbox" style="width:auto"> I am an independent human reviewer; I did not use automated/model assistance.</label><p>Submit exported JSON to <code>reviews/incoming/</code> through the project operator. Operator provenance confirmation uses <code>reviews/reviewer_provenance.json</code>; metadata alone is not human truth.</p></header><nav class="toolbar"><button id="save">Save local draft</button> <button id="restore">Restore local draft</button> <button id="export">Export review JSON</button><span id="notice"></span></nav>'''+''.join(cards)+'''<script type="application/json" id="packet">'''+encoded+'''</script><script>
const packet=JSON.parse(document.getElementById('packet').textContent);const key='rsi-review-'+packet.source_package_experiment;
function collect(){return {reviewer:document.getElementById('reviewer').value,independent:document.getElementById('independent').checked,rows:[...document.querySelectorAll('article')].map(a=>({pair_id:a.dataset.pair,values:Object.fromEntries(new FormData(a.querySelector('form')))}))};}
document.getElementById('save').onclick=()=>{localStorage.setItem(key,JSON.stringify(collect()));document.getElementById('notice').textContent=' Draft saved locally; not submitted.';};
document.getElementById('restore').onclick=()=>{const data=JSON.parse(localStorage.getItem(key)||'null');if(!data)return;document.getElementById('reviewer').value=data.reviewer;document.getElementById('independent').checked=data.independent;for(const row of data.rows){const article=[...document.querySelectorAll('article')].find(a=>a.dataset.pair===row.pair_id);for(const [name,value]of Object.entries(row.values)){article.querySelector(`[name="${name}"]`).value=value;}}};
document.getElementById('export').onclick=()=>{const data=collect();if(!data.reviewer.trim()||!data.independent){document.getElementById('notice').textContent=' Reviewer identity and independent-human declaration are required; no export certified.';return;}const out={schema_version:'independent_edit_review_v1',provenance:{reviewer_id:data.reviewer.trim(),review_session_id:crypto.randomUUID(),reviewed_at:new Date().toISOString(),reviewer_type:'human',independence_attestation:true,model_assisted:false},reviews:data.rows.map(row=>{const pair=packet.pairs.find(p=>p.pair_id===row.pair_id);const answer=row.values.edited_answer.trim();return {pair_id:row.pair_id,original_image_sha256:pair.original_image_sha256,edited_image_sha256:pair.edited_image_sha256,dimensions:Object.fromEntries(Object.keys(packet.dimension_choices).map(k=>[k,row.values[k]])),edited_factual_answer:{status:answer?'REVIEWED':'UNKNOWN',value:answer?answer.split(/\\n/).map(a=>a.trim()).filter(Boolean):null},reason:row.values.reason};})};const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,2)],{type:'application/json'}));a.download='independent_review_'+data.reviewer.replace(/[^a-zA-Z0-9_-]/g,'_')+'.json';a.click();};
</script></html>'''
    from .dashboard import atomic_text
    atomic_text(base/'index.html',page)
    write_json(base/'submission_template.json',{'schema_version':'independent_edit_review_v1',
       'provenance':{'reviewer_id':'','review_session_id':'','reviewed_at':'','reviewer_type':'UNDECLARED','independence_attestation':False,'model_assisted':None},
       'reviews':[{'pair_id':p['pair_id'],'original_image_sha256':p['original_image_sha256'],'edited_image_sha256':p['edited_image_sha256'],
        'dimensions':{k:'UNKNOWN' for k in DIMENSIONS},'edited_factual_answer':{'status':'UNKNOWN','value':None},'reason':''} for p in manifest['pairs']]})

def separated_measurements(root,output):
    from .tasks import saved_observations
    rows=[]
    for o in saved_observations(root):
        rows.append({'source_experiment_id':o['parent_experiment_id'],'sample_id':o['sample_id'],'population':o['observation_population'],
          'step':o['update'],'instruction':o['instruction'],'quality_measurements':o['quality'],
          'generation_seconds':o['generation_seconds'],'automatic_judgment_seconds':o['reward_seconds'],
          'answer_consistency_proxy':o['answer_consistency'],'semantic_proxy':o['semantic'],
          'instruction_success':'UNKNOWN','independent_factual_answer':{'status':'UNKNOWN','value':None},
          'calibrated_semantic_uncertainty':'UNKNOWN','historical_scalar':o['reward'],
          'uncertainty_handling':'historical U is the invalid/unverified penalty; not a measured calibrated uncertainty'})
    with (output/'raw_measurements.jsonl').open('x',encoding='utf8') as f:
        for row in rows:f.write(json.dumps(row,allow_nan=False)+'\n')
    assert len(rows)==500
    reward=[r['historical_scalar'] for r in rows]
    summary={'n':500,'lower_clipped':sum(r['reward']==-1 for r in reward),'upper_clipped':sum(r['reward']==1.5 for r in reward),
      'constant_historical_U':len({r['invalid_penalty'] for r in reward})==1,'constant_normalized_C':len({r['cost'] for r in reward})==1,
      'S_equals_half_A':all(r['semantic']==.5*r['answer_consistency'] for r in reward),
      'instruction_success_missing':500,'calibrated_uncertainty_missing':500,'independent_labels':0,
      'generation_seconds':{'min':min(r['generation_seconds'] for r in rows),'max':max(r['generation_seconds'] for r in rows),'mean':statistics.mean(r['generation_seconds'] for r in rows)},
      'automatic_judgment_seconds':{'min':min(r['automatic_judgment_seconds'] for r in rows),'max':max(r['automatic_judgment_seconds'] for r in rows),'mean':statistics.mean(r['automatic_judgment_seconds'] for r in rows)},
      'training_reward_released':False,'coefficients_changed':False,'raw_scalar_separated':True,'new_model_calls':0}
    write_json(output/'reward_separation.json',summary);return summary
