"""Bounded diagnostic, fixed saved states; no new images/reward or optimizer update."""
import copy
import gc
import json
import torch
import _bootstrap
from _phase3 import phase3_run,mark_latest
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.training.generator_adapter import load_adapter_state
from src.rl.diffusion_policy import DiffusionPolicy
from src.rl.diffusion_trajectory import transition
from src.rl.optimizer_validation import state_equal
from src.utils.logging import write_json

def gradients(policy,trajectory,condition):
    params=[p for p in policy.pipe.unet.parameters() if p.requires_grad]
    policy.pipe.unet.zero_grad(set_to_none=True)
    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
        for step in trajectory['steps']:
            state=step['state'].to('cuda');action=step['action'].to('cuda')
            _,lp,_,_=transition(policy.pipe.scheduler,policy.predict(state,step['t'],condition),step['t'],state,action=action)
            (-lp.mean()/10).backward()
    return [p.grad.detach().cpu().clone() for p in params]

def difference(a,b):
    return {'max_abs':max(float((x-y).abs().max()) for x,y in zip(a,b)),
        'relative_l2':float(torch.sqrt(sum((x-y).double().square().sum() for x,y in zip(a,b)))/torch.sqrt(sum(x.double().square().sum() for x in a))),
        'exact':state_equal(a,b)}

if __name__=='__main__':
    root=_bootstrap.ROOT
    origin=root/'experiments/EXP_20261027T142212_e407797b1bef_phase3_generator_rl_one'
    checkpoint=torch.load(origin/'checkpoints/step_0000/training_state.pt',weights_only=False,map_location='cpu')
    saved=torch.load(origin/'checkpoints/pre_update_1.pt',weights_only=False,map_location='cpu')
    trajectory=torch.load(origin/'checkpoints/step1_trajectory_0.pt',weights_only=False,map_location='cpu')
    with phase3_run('backward_diagnostic',{'seed':42,'origin':origin.name,'no_optimizer_update':True}) as run:
        results={}
        for checkpointing in [True,False]:
            config=copy.deepcopy(checkpoint['contract']['config']);config['gradient_checkpointing']=checkpointing
            editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,config)
            load_adapter_state(policy.pipe.unet,checkpoint['adapter'])
            condition={k:v.to('cuda') for k,v in saved['conditions'][0].items()}
            first=gradients(policy,trajectory,condition);second=gradients(policy,trajectory,condition)
            del policy,editor;gc.collect();torch.cuda.empty_cache()
            editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,config)
            load_adapter_state(policy.pipe.unet,checkpoint['adapter'])
            third=gradients(policy,trajectory,condition)
            results[str(checkpointing)]={'same_instance':difference(first,second),'fresh_instance':difference(first,third)}
            run.event('backward_reproducibility',{'gradient_checkpointing':checkpointing,**results[str(checkpointing)]})
            del policy,editor,first,second,third;gc.collect();torch.cuda.empty_cache()
        run.metrics(results);run.finish(summary='Fixed-state backward diagnostic only, no optimizer step and no scientific retry.')
        mark_latest('backward_diagnostic',run);print(json.dumps(results,indent=2))
