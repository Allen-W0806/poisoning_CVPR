"""Fixed batch10 witness; do not relax the historical exact criterion."""
import copy
import gc
import json
import torch
import _bootstrap
from _phase4 import phase4_run
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.rl.diffusion_policy import DiffusionPolicy
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.rl.optimizer_validation import state_equal,tensor_changes
from src.training.generator_adapter import adapter_state
from src.utils.logging import write_json

def diff(a,b):
    valid=[(x,y) for x,y in zip(a,b) if x is not None and y is not None]
    n=sum(x.double().square().sum() for x,y in valid)
    return {'exact':state_equal(a,b),'max_abs':max(float((x-y).abs().max()) for x,y in valid),
        'relative_l2':float((sum((x-y).double().square().sum() for x,y in valid)/(n+1e-300)).sqrt())}

if __name__=='__main__':
    root=_bootstrap.ROOT
    origin=root/'experiments/EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke'
    old=torch.load(origin/'checkpoints/step_0002/training_state.pt',weights_only=False,map_location='cpu')
    final=torch.load(origin/'checkpoints/step_0003/training_state.pt',weights_only=False,map_location='cpu')
    saved=torch.load(origin/'checkpoints/pre_update_3.pt',weights_only=False,map_location='cpu')
    original_grad=torch.load(origin/'checkpoints/gradients_3.pt',weights_only=False,map_location='cpu')
    trajectories=[torch.load(origin/f'checkpoints/step3_trajectory_{i}.pt',weights_only=False,map_location='cpu') for i in range(10)]
    with phase4_run('PH4_RESTART_DIAGNOSTIC',{'seed':42,'origin':origin.name,'criterion':'bitwise exact; numerical measures descriptive only','config':old['contract']['config']}) as run:
        results=[];states=[];grads=[]
        for repeat in range(2):
            editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,old['contract']['config']);policy.contract=copy.deepcopy(old['contract'])
            trainer=DiffusionRLTrainer(policy);trainer.restore(origin/'checkpoints/step_0002')
            restoration={'adapter':state_equal(old['adapter'],adapter_state(policy.pipe.unet)),
                'optimizer':state_equal(old['optimizer'],trainer.optimizer.state_dict()),'scheduler':state_equal(old['scheduler'],trainer.scheduler.state_dict()),
                'rng_torch':torch.equal(old['rng_torch'],torch.get_rng_state()),'version':policy.version==old['version'],
                'consumed':trainer.consumed==old['consumed'],
                'parameter_order':[n for n,p in policy.pipe.unet.named_parameters() if p.requires_grad]}
            assert all(v for k,v in restoration.items() if k!='parameter_order'),restoration
            conditions=[{**{k:v.to('cuda') for k,v in c.items()},'sample_id':t['sample_id'],'image_id':t['image_id'],'instruction':t['instruction']} for c,t in zip(saved['conditions'],trajectories)]
            update=trainer.update([(t,c,t['reward']['reward']) for t,c in zip(trajectories,conditions)])
            grads.append(trainer.last_gradients);states.append({'adapter':adapter_state(policy.pipe.unet),'optimizer':copy.deepcopy(trainer.optimizer.state_dict())})
            result={'repeat':repeat,'restoration':restoration,'update':update,'gradient_vs_original':diff(original_grad,grads[-1]),
                'adapter_vs_original':tensor_changes(final['adapter'],states[-1]['adapter']),'adapter_exact_original':state_equal(final['adapter'],states[-1]['adapter'])}
            trainer.save(run.path/f'checkpoints/replay_{repeat}');run.event('replay',result);results.append(result);run.metrics({'replays':results})
            del trainer,policy,editor,conditions;gc.collect();torch.cuda.empty_cache()
        diagnostic={'restoration_exact':True,'fresh_repeat_gradient':diff(grads[0],grads[1]),
            'fresh_repeat_adapter_exact':state_equal(states[0]['adapter'],states[1]['adapter']),
            'fresh_repeat_optimizer_exact':state_equal(states[0]['optimizer'],states[1]['optimizer']),
            'historical_continuation_exact':all(x['adapter_exact_original'] for x in results),
            'criterion':'bitwise exact; no tolerance relaxation','first_observed_difference':'gradient if likelihood replay exact; stored rewards/states fixed',
            'classification':'exact training state restored; investigate floating point execution if gradients differ',
            'resume_authorized':False,'fresh_initialization_authorized':True}
        write_json(run.path/'restart_diagnostic.json',diagnostic);run.metrics({'replays':results,**diagnostic})
        run.finish(summary='Diagnostic executed; PASS denotes completed investigation, not accepted historical continuation. See historical_continuation_exact.');print(json.dumps({'id':run.id,**diagnostic}))
