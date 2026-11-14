"""Read-only CPU audit of immutable evidence; no model loading or training.

Run on the canonical checkout with its existing editor environment. Writes one
exclusive derived receipt, never an experiment or an existing result file.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import collections
import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import _bootstrap
from src.data.dataset import sha256
from src.data.prepare_vqa import select_subsets, OFFICIAL
from src.editing.semantic_validator import SemanticResult
from src.evaluation.answer_consistency import answer_consistency
from src.evaluation.image_quality import image_quality
from src.evaluation.semantic_validation import semantic_decision
from src.evaluation.phase4_statistics import paired_summary
from src.rl.reward import validation_reward
from src.rl.diffusion_trajectory import transition

ROOT = _bootstrap.ROOT
OUT = ROOT / 'reports/phase45_recompute.json'
assert not OUT.exists(), 'Derived audit receipt already exists; use a new version'
torch.set_num_threads(2)
def read(p): return json.loads(Path(p).read_text())
def close(a,b):
    assert math.isclose(float(a),float(b),rel_tol=1e-7,abs_tol=1e-8),(a,b)
def tree_digest():
    # Historical manifests/status/metrics plus all files in the frozen Phase4 receipt.
    paths=set()
    for run in evidence['phase4_runs']:
        for row in run['files']:
            p=ROOT/'experiments'/run['id']/row['path']
            assert p.stat().st_size==row['bytes'] and sha256(p)==row['sha256'],str(p)
            paths.add(p)
    for name in ['manifest.json','status.json','metrics.json']:
        paths.update((ROOT/'experiments').glob('*/'+name))
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(paths)}

evidence=read(ROOT/'reports/phase4_evidence.json')
before=tree_digest()
report={'generated_at':datetime.now(timezone.utc).isoformat(),'mode':'CPU_READ_ONLY_NO_MODEL_CALLS',
        'new_gpu_seconds':0,'new_training_updates':0,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
        'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()}
protocol=read(ROOT/'configs/phase2_vqa_protocol.json')
marker=sha256(ROOT/'reports/phase2_test_opened.json')
assert marker==evidence['heldout_marker_sha256']
assert sha256(ROOT/'configs/phase2_vqa_protocol.json')==evidence['locked_protocol_sha256']
source={}
for kind,url in OFFICIAL.items():
    with zipfile.ZipFile(ROOT/'data/raw/vqav2'/Path(url).name) as z:
        member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
        source[kind]=json.loads(z.read(member))
qmap={str(x['question_id']):x for x in source['questions']['questions']}
amap={str(x['question_id']):x for x in source['annotations']['annotations']}
repeated=select_subsets(source['annotations']['annotations'],source['questions']['questions'])
data={};audit={};image_sets=[];byte_sets=[];record_map={}
for split in ['train','validation','test']:
    p=ROOT/f'data/processed/vqav2_seed42_v1/{split}.json'
    assert sha256(p)==protocol['split_hashes'][split]
    rows=read(p);data[split]=rows
    assert len(rows)==len(repeated[split])
    for row,replay in zip(rows,repeated[split]):
        assert all(row[k]==v for k,v in replay.items())
        q=qmap[row['sample_id']];a=amap[row['sample_id']]
        assert str(q['image_id'])==str(a['image_id'])==row['image_id']
        assert q['question']==row['question'] and [v['answer'] for v in a['answers']]==row['answers']
        assert len(row['answers'])==10 and row['split']==split
        assert Path(row['image_path']).name==f"COCO_train2014_{int(row['image_id']):012d}.jpg"
        assert sha256(ROOT/row['image_path'])==row['image_sha256']
        record_map[row['sample_id']]=row
    ids={r['image_id'] for r in rows};hashes={r['image_sha256'] for r in rows}
    assert len({r['sample_id'] for r in rows})==len(rows)
    audit[split]={'records':len(rows),'unique_image_ids':len(ids),'sha256':sha256(p),'ten_answers_each':True}
    image_sets.append(ids);byte_sets.append(hashes)
overlaps=[]
splits=['train','validation','test']
for i in range(3):
    for j in range(i):
        assert not image_sets[i]&image_sets[j]
        for digest in sorted(byte_sets[i]&byte_sets[j]):
            overlaps.append({'split_a':splits[j],'split_b':splits[i],'image_sha256':digest,
              'records_a':[{k:r[k] for k in ['sample_id','image_id','question','answers','image_path']} for r in data[splits[j]] if r['image_sha256']==digest],
              'records_b':[{k:r[k] for k in ['sample_id','image_id','question','answers','image_path']} for r in data[splits[i]] if r['image_sha256']==digest]})
report['dataset']={'splits':audit,'official_question_annotation_image_join_verified':1500,
    'deterministic_selection_exact':True,'all_selected_file_hashes_verified':1500,
    'cross_split_image_id_overlap':0,'cross_split_identical_bytes_groups':overlaps,
    'test_access':'annotation/hash audit only; no model inference, selection or new test performance'}
print('DATASET VERIFIED',flush=True)

names={'M0':'EXP_20261027T165424_eb9045420a43_phase4_PH4_FROZEN_GENERATOR',
 'M1_42':'EXP_20261027T170121_a6ed0fa129ca_phase4_PH4_GENERATOR_RL',
 'M1_123':'EXP_20261027T170250_c7c7385c8814_phase4_PH4_GENERATOR_RL',
 'M2_42':'EXP_20261027T175741_3b880a4a5fce_phase4_PH4_GENERATOR_RSI',
 'M2_123':'EXP_20261027T175744_255141d217c5_phase4_PH4_GENERATOR_RSI'}
metrics={key:read(ROOT/'experiments'/value/'metrics.json') for key,value in names.items()}
stats={};checked=0;trajectories=0;transitions=0;source_paths=set();reward_errors=[]
for key,m in metrics.items():
    bykind=collections.defaultdict(list)
    batches=[('train',u['step'],u['observations']) for u in m['updates']]
    batches += [('development',v['step'],v['observations']) for v in m['evaluations']]
    for kind,step,obs in batches:
        assert len(obs)==10 and len({o['source_image_id'] for o in obs})==10
        label=Path(obs[0]['output_path']).stem.rsplit('_',1)[0]
        for index,o in enumerate(obs):
            r=record_map[o['sample_id']]
            assert r['split']==('train' if kind=='train' else 'validation')
            assert o['question']==r['question'] and o['answers']==r['answers'] and o['source_image_id']==r['image_id']
            image=Image.open(o['source_path']).convert('RGB');edited=Image.open(o['output_path']).convert('RGB')
            consistency=answer_consistency(o['answer_consistency']['original_answer'],o['answer_consistency']['edited_answer'],r)
            assert consistency==o['answer_consistency']
            sem=semantic_decision(image,edited,r,consistency,o['structured_judgment'])
            assert sem==o['semantic']
            quality=image_quality(image,edited)
            for k in ['mse','mae','ssim','distortion']:close(quality[k],o['quality'][k])
            reward=validation_reward(SemanticResult(sem['verified'],sem['score'],sem['verifier'],sem['reason']),
                  consistency['score'] if sem.get('provisionally_valid') else 0.,quality['distortion'],10/30)
            for k,v in reward.items():
                if isinstance(v,bool):assert v==o['reward'][k]
                else:close(v,o['reward'][k])
            tp=Path(o['output_path']).parents[1]/'checkpoints'/f'{label}_trajectory_{index}.pt'
            t=torch.load(tp,map_location='cpu',weights_only=False)
            assert t['sample_id']==o['sample_id'] and t['image_id']==r['image_id'] and t['seed']==o['seed']
            assert t['reward']==o['reward'] and t['policy_version']==o['policy_version']
            assert [s['t'] for s in t['steps']]==list(range(901,0,-100))
            assert len(t['steps'])==10
            for i,s in enumerate(t['steps']):
                assert s['state'].shape==s['action'].shape==s['noise'].shape==(1,4,64,64)
                assert s['std']>0 and all(torch.isfinite(s[k]).all() for k in ['state','action','noise','logp'])
                assert s['state'].dtype==s['action'].dtype==torch.float32
                if i:assert torch.equal(t['steps'][i-1]['action'],s['state'])
            if kind=='train':assert o['policy_version']==step-1
            else:assert o['policy_version']==step
            bykind[kind].append(o);checked+=1;trajectories+=1;transitions+=len(t['steps'])
    stats[key]={}
    for kind,obs in bykind.items():
        def mean(k):return float(np.mean([o['reward'][k] for o in obs]))
        stats[key][kind]={'n_exposures':len(obs),'unique_images':len({o['source_image_id'] for o in obs}),
            'proxy_pass':sum(o['semantic'].get('provisionally_valid',False) for o in obs),
            'human_verified':sum(o['human_status']!='UNVERIFIED' for o in obs),
            'clipped_to_minus_one':sum(o['reward']['reward']==-1 for o in obs),
            'mean_components':{k:mean(k) for k in ['semantic','answer_consistency','distortion','cost','invalid_penalty','reward_unclipped','reward']},
            'min_reward':min(o['reward']['reward'] for o in obs),'max_reward':max(o['reward']['reward'] for o in obs),
            'structured_parse_failures':sum(not o['structured_judgment']['parse_ok'] for o in obs),
            'S_equals_half_A_all':all(o['reward']['semantic']==.5*o['reward']['answer_consistency'] for o in obs)}
        print(key,kind,stats[key][kind]['n_exposures'],'verified',flush=True)
report['saved_observation_recompute']={'observations':checked,'trajectories':trajectories,'denoising_transitions':transitions,
 'logical_train_updates':32,'source_exposures_per_update':10,'backward_transitions_per_update':100,
 'all_reward_semantic_consistency_and_quality_fields_reproduced':True,'statistics':stats}
report['paired']={};report['final']={};report['checkpoint_matching']={}
for seed in [42,123]:
    ref=next(x['observations'] for x in metrics['M0']['evaluations'] if x['seed']==seed)
    for arm in ['M1','M2']:
        m=metrics[f'{arm}_{seed}'];final=m['evaluations'][-1]['observations']
        report['final'][f'{arm}_{seed}']={'reward_mean':float(np.mean([o['reward']['reward'] for o in final])),
             'versus_frozen':paired_summary(final,ref)}
        for v in m['evaluations']:
            assert paired_summary(v['observations'],ref)==v['paired_frozen']
    a,b=metrics[f'M1_{seed}'],metrics[f'M2_{seed}']
    report['paired'][str(seed)]={str(look['step']):paired_summary(other['observations'],look['observations'])
        for look,other in zip(a['evaluations'],b['evaluations'])}
    assert [[o['sample_id'] for o in x['observations']] for x in a['updates']]==[[o['sample_id'] for o in x['observations']] for x in b['updates']]
    assert [[o['seed'] for o in x['observations']] for x in a['updates']]==[[o['seed'] for o in x['observations']] for x in b['updates']]
    def checkpoint(arm,step):return torch.load(ROOT/'experiments'/names[f'{arm}_{seed}']/f'checkpoints/step_{step:04d}/training_state.pt',map_location='cpu',weights_only=False)
    a0,b0=checkpoint('M1',0),checkpoint('M2',0);a4,b4=checkpoint('M1',4),checkpoint('M2',4)
    assert all(torch.equal(a0['adapter'][k],b0['adapter'][k]) for k in a0['adapter'])
    diff=max(float((a4['adapter'][k]-b4['adapter'][k]).abs().max()) for k in a4['adapter'])
    report['checkpoint_matching'][str(seed)]={'initial_adapter_exact':True,'step4_before_intervention_adapter_max_difference':diff,
         'training_image_order_and_noise_seeds_match':True,'recorded_updates_lr_M1':[u['learning_rate'] for u in a['updates']],
         'recorded_updates_lr_M2':[u['learning_rate'] for u in b['updates']]}

# Independent algebraic CPU check against the installed Diffusers DDIM step.
from diffusers import DDIMScheduler
scheduler=DDIMScheduler.from_pretrained('timbrooks/instruct-pix2pix',subfolder='scheduler',
 revision='31519b5cb02a7fd89b906d88731cd4d6a7bbf88d',local_files_only=True,
 set_alpha_to_one=False,steps_offset=1,clip_sample=False,timestep_spacing='leading')
scheduler.set_timesteps(10,device='cpu')
g=torch.Generator().manual_seed(4500);x=torch.randn(1,4,8,8,generator=g);deltas=[];stds=[]
for t in scheduler.timesteps:
    eps=torch.randn(x.shape,generator=g)
    noise=torch.randn(x.shape,generator=g)
    action,lp,mean,std=transition(scheduler,eps,t,x,eta=1.,noise=noise)
    official=scheduler.step(eps,t,x,eta=1.,variance_noise=noise).prev_sample
    deltas.append(float((action-official).abs().max()));stds.append(float(std));assert torch.allclose(action,official,atol=1e-6,rtol=1e-6)
    ep=eps.clone().requires_grad_();_,replay,mu,sig=transition(scheduler,ep,t,x,action=action)
    manual=torch.distributions.Normal(mu,sig).log_prob(action.detach()).flatten(1).mean(1)
    assert torch.allclose(replay,manual,atol=1e-5,rtol=1e-6)
    gg=torch.autograd.grad(replay.sum(),ep,retain_graph=True)[0];gm=torch.autograd.grad(manual.sum(),ep)[0]
    assert torch.allclose(gg,gm,atol=1e-5,rtol=1e-5)
report['cpu_transition_algebra']={'installed_diffusers_ddim_step_max_error':max(deltas),'all_ten_variances_positive':all(v>0 for v in stds),
 'stds':stds,'gaussian_logdensity_and_gradient_match_independent_torch_distribution':True,
 'scope':'synthetic CPU arithmetic only, not a new GPU continuation acceptance or production-gradient determinism test'}
report['environment']={k:importlib.metadata.version(k) for k in ['torch','diffusers','peft','transformers','accelerate','huggingface-hub','scikit-image']}
report['model_revisions']=read(ROOT/'experiments'/names['M1_42']/'manifest.json')['model_revisions']
after=tree_digest();assert before==after
assert sha256(ROOT/'reports/phase2_test_opened.json')==marker
report['immutability']={'historical_files_checked_before_and_after':len(before),'all_identical':True,
 'phase4_files_verified_against_original_receipt':sum(len(r['files']) for r in evidence['phase4_runs']),
 'heldout_marker_unchanged':True,'locked_protocol_unchanged':True,
 'snapshot_sha256':hashlib.sha256(json.dumps(before,sort_keys=True).encode()).hexdigest()}
report['audited_source_sha256']={str(p.relative_to(ROOT)):sha256(p) for folder in ['src','configs'] for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in str(p)}
with OUT.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2)
print(json.dumps({'receipt':str(OUT),'observations':checked,'historical_files_unchanged':len(before)}),flush=True)
