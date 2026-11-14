"""Fresh-initialization genuine generator RL, fixed development protocol."""
import argparse
import copy
import json
import random
import time
from pathlib import Path
import torch
import yaml
import _bootstrap
from _phase4 import phase4_run
from _runs import model_options
from smoke_generator_rl import collect,conditioning_and_reward_test
from src.rl.benign_reward_v3 import require_episode_reward_release
from src.data.dataset import VQADataset,validate_disjoint,sha256
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.models.qwen_loader import QwenVQA
from src.training.generator_adapter import adapter_state,frozen_hashes
from src.rl.diffusion_policy import DiffusionPolicy
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.rl.optimizer_validation import state_equal,tensor_changes
from src.evaluation.phase4_statistics import paired_summary,rsi_plateau
from src.utils.seed import seed_everything
from src.utils.logging import write_json

class Collection:
    def __init__(self,run,label):
        self.path=run.path/'collections'/label;self.path.mkdir(parents=True,exist_ok=False)
        (self.path/'images').mkdir();(self.path/'checkpoints').mkdir();self.run=run
    def event(self,*args):self.run.event(*args)

def subset(data,seed,count):
    indices=list(range(len(data)));random.Random(seed).shuffle(indices)
    selected=[];seen=set()
    for i in indices:
        key=str(data.records[i]['image_id'])
        if key in seen:continue
        seen.add(key);selected.append(i)
        if len(selected)==count:return selected
    raise ValueError('Insufficient independent sources')

@torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
def obtain(policy,data,indices,qwen,run,label,seed):
    records=[data[i] for i in indices];conditions=[policy.condition(r) for r in records]
    answers=qwen.answer([c['source'] for c in conditions],[r['question'] for r in records])
    before=adapter_state(policy.pipe.unet);rng=torch.get_rng_state();cuda=torch.cuda.get_rng_state_all()
    torch.manual_seed(seed)
    try:episodes,observations=collect(policy,conditions,records,qwen,answers,Collection(run,label),label)
    finally:torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda)
    assert state_equal(before,adapter_state(policy.pipe.unet)),'Collection modified policy'
    return episodes,observations,records,conditions

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['frozen','rl','rsi'],required=True)
    parser.add_argument('--seed',type=int,choices=[42,123],required=True);a=parser.parse_args();root=_bootstrap.ROOT
    config=yaml.safe_load((root/'configs/phase3_generator_rl.yaml').read_text());config.update(phase=4,seed=a.seed,arm=a.arm,initial_optimizer_updates=10)
    protocol=json.loads((root/'configs/phase2_vqa_protocol.json').read_text())
    for s in ['train','validation']:
        assert sha256(root/f'data/processed/vqav2_seed42_v1/{s}.json')==protocol['split_hashes'][s],f'{s} hash drift'
    train=VQADataset(root/'data/processed/vqav2_seed42_v1/train.json',root,'train')
    dev=VQADataset(root/'data/processed/vqav2_seed42_v1/validation.json',root,'validation');validate_disjoint(train,dev)
    dev_indices=subset(dev,42,10);train_indices=subset(train,a.seed,100)
    config.update(validation_indices=dev_indices,training_indices=train_indices,evaluation_seed_formula='torch.randint stream initialized by seed*10000; per-image seeds saved',evaluation_updates=[2,4,6,8,10],
        continuation='fresh initialization only; no restart acceptance inferred',rsi_rule='two unresolved positive-gain looks then one LR halving')
    if a.arm!='frozen':
        pointer=json.loads((root/'reports/phase4_latest_frozen.json').read_text());baseline_path=Path(pointer['path'])
        assert json.loads((baseline_path/'status.json').read_text())['status']=='PASS'
        baseline=json.loads((baseline_path/'metrics.json').read_text())
    if a.arm=='rsi':
        for seed in [42,123]:
            p=json.loads((root/f'reports/phase4_latest_rl{seed}.json').read_text())
            assert json.loads((Path(p['path'])/'status.json').read_text())['status']=='PASS','Both fixed pilots required'
    category={'frozen':'PH4_FROZEN_GENERATOR','rl':'PH4_GENERATOR_RL','rsi':'PH4_GENERATOR_RSI'}[a.arm]
    with phase4_run(category,config) as run:
        started=time.perf_counter();editor=load_pretrained_editor(root);seed_everything(a.seed);policy=DiffusionPolicy(editor,config)
        trainer=DiffusionRLTrainer(policy);initial=adapter_state(policy.pipe.unet);frozen=frozen_hashes(policy.pipe)
        assert all(torch.count_nonzero(v)==0 for k,v in initial.items() if 'lora_B' in k),'Frozen identity requires zero B'
        torch.save(initial,run.path/'checkpoints/initial_adapter.pt');write_json(run.path/'frozen_before.json',frozen)
        qwen=QwenVQA(**model_options('cuda'));trainer.save(run.path/'checkpoints/step_0000')
        history=[];evaluations=[];decisions=[];revised=False
        if a.arm=='frozen':
            for seed in [42,123]:
                _,obs,_,_=obtain(policy,dev,dev_indices,qwen,run,f'baseline{seed}',seed*10000)
                evaluations.append({'seed':seed,'step':0,'observations':obs})
            assert state_equal(initial,adapter_state(policy.pipe.unet))
        else:
            reference=next(x['observations'] for x in baseline['evaluations'] if x['seed']==a.seed)
            for step in range(1,11):
                lr=trainer.optimizer.param_groups[0]['lr']
                episodes,obs,records,conditions=obtain(policy,train,train_indices[(step-1)*10:step*10],qwen,run,f'train{step}',a.seed*100000+step)
                if step==1:
                    tests=conditioning_and_reward_test(policy,conditions[0],records[0],episodes[0][0]);write_json(run.path/'conditioning_tests.json',tests)
                require_episode_reward_release(episodes)
                result=trainer.update(episodes);trainer.save(run.path/f'checkpoints/step_{step:04d}')
                result.update(learning_rate=lr,observations=obs,source_ids=[x['sample_id'] for x in obs],checkpoint=str(run.path/f'checkpoints/step_{step:04d}'),
                    cumulative_allocation_seconds_estimate=time.perf_counter()-started,cumulative_gpu_hours_estimate=(time.perf_counter()-started)/3600)
                assert torch.isfinite(torch.tensor([result['loss'],result['gradient_norm'],result['reward_mean']])).all()
                assert all(torch.isfinite(p).all() for p in trainer.params),'Nonfinite adapter'
                history.append(result);run.event('generator_rl_update',result)
                if step%2==0:
                    _,validation,_,_=obtain(policy,dev,dev_indices,qwen,run,f'eval{step}',a.seed*10000)
                    paired=paired_summary(validation,reference)
                    evaluations.append({'seed':a.seed,'step':step,'observations':validation,'paired_frozen':paired})
                    if a.arm=='rsi' and not revised and step<10:
                        t=time.perf_counter();accept=rsi_plateau([x['paired_frozen'] for x in evaluations])
                        decision={'step':step,'accept_schedule_revision':accept,'reason':'two unresolved validation gains' if accept else 'insufficient plateau evidence',
                            'old_lr':lr,'new_lr':5e-6 if accept else lr,'diagnostic_seconds':time.perf_counter()-t}
                        if accept:
                            revision_config={**config,'learning_rate':5e-6,'parent_training_run':run.id,'revision_at_step':step,'decision':decision}
                            with phase4_run('PH4_RSI_CONFIG_REVISION',revision_config) as revision:
                                revision.metrics({'decision':decision,'training_parent':run.id,'no_additional_optimizer_updates':True})
                                revision.finish(summary='Immutable scientific configuration version; one-variable LR revision, same ongoing fresh training lineage.')
                                decision['revision_experiment_id']=revision.id
                            for group in trainer.optimizer.param_groups:group['lr']=5e-6
                            trainer.scheduler.base_lrs=[5e-6 for _ in trainer.scheduler.base_lrs];revised=True
                            policy.config['learning_rate']=5e-6
                            policy.contract['config']['learning_rate']=5e-6
                            policy.contract['active_revision_experiment_id']=revision.id
                            write_json(run.path/f'config_revision_step{step}.json',{'parent_config':config,'decision':decision,'reward_unchanged':True})
                        decisions.append(decision);run.event('rsi_decision',decision)
                run.metrics({'arm':a.arm,'seed':a.seed,'updates':history,'evaluations':evaluations,'rsi_decisions':decisions,'in_progress':step<10})
        assert frozen==frozen_hashes(policy.pipe),'Frozen tensors changed';write_json(run.path/'frozen_after.json',frozen)
        metrics={'arm':a.arm,'seed':a.seed,'updates':history,'evaluations':evaluations,'rsi_decisions':decisions,
            'generator_training_kind':'reward_based_generator_RL' if history else 'frozen_reference',
            'adapter_changes':tensor_changes(initial,adapter_state(policy.pipe.unet)),'frozen_weights_unchanged':True,
            'parameters':policy.parameters,'training_source_exposures':len(history)*10,'training_trajectories':len(history)*10,
            'unique_training_sources':len({x for h in history for x in h['source_ids']}),'evaluation_trajectories':sum(len(x['observations']) for x in evaluations),
            'wall_seconds':time.perf_counter()-started,'qwen':qwen.metadata(),'unet_calls':policy.unet_calls,
            'human_verified_changed_edits':0,'restart_claim':False,'semantic_truth_ready':False}
        run.metrics(metrics);run.finish(summary='Completed matched development pilot; proxy evaluation only, independent human semantics pending.')
        key='frozen' if a.arm=='frozen' else a.arm+str(a.seed)
        write_json(root/f'reports/phase4_latest_{key}.json',{'id':run.id,'path':str(run.path)});print(run.id)

if __name__=='__main__':main()
