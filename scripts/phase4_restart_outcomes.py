"""Outcome-level numerical witness, no training or reward/model-label inference."""
import gc
import json
import numpy as np
import torch
from PIL import Image
import _bootstrap
from _phase4 import phase4_run
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.rl.diffusion_policy import DiffusionPolicy
from src.training.generator_adapter import load_adapter_state
from src.utils.logging import write_json

if __name__=='__main__':
    root=_bootstrap.ROOT
    origin=root/'experiments/EXP_20261027T143506_e1db622c6126_phase3_generator_rl_smoke'
    diagnostic=sorted((root/'experiments').glob('*phase4_PH4_RESTART_DIAGNOSTIC'))[-1]
    paths=[origin/'checkpoints/step_0003/training_state.pt']+[diagnostic/f'checkpoints/replay_{i}/training_state.pt' for i in range(2)]
    saved=torch.load(origin/'checkpoints/pre_update_3.pt',map_location='cpu',weights_only=False)
    trajectories=[torch.load(origin/f'checkpoints/step3_trajectory_{i}.pt',map_location='cpu',weights_only=False) for i in range(2)]
    with phase4_run('PH4_RESTART_OUTCOMES',{'seed':42,'source':origin.name,'diagnostic':diagnostic.name,'sources':2,
            'models':['historical_step3','replay0_step3','replay1_step3'],'acceptance':'exact PNG equality only; no semantic or full-restart claim'}) as run:
        images=[];details=[]
        for model_index,path in enumerate(paths):
            state=torch.load(path,map_location='cpu',weights_only=False)
            editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,state['contract']['config']);load_adapter_state(policy.pipe.unet,state['adapter']);policy.version=state['version']
            outputs=[]
            with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                for i,t in enumerate(trajectories):
                    source=Image.open(origin/f'images/source_{i}.png').convert('RGB')
                    c={**{k:v.to('cuda') for k,v in saved['conditions'][i].items()},'source':source,
                        'sample_id':t['sample_id'],'image_id':t['image_id'],'instruction':t['instruction']}
                    source.save(run.path/f'images/source_{i}.png')
                    output,trajectory=policy.sample(c,t['seed'],f'model{model_index}_source{i}')
                    output.save(run.path/f'images/model{model_index}_source{i}.png');torch.save(trajectory,run.path/f'checkpoints/model{model_index}_source{i}_trajectory.pt')
                    outputs.append(np.asarray(output).astype(np.int16))
                    details.append({'model':model_index,'checkpoint':str(path),'source_id':t['sample_id'],'seed':t['seed'],'generation_seconds':trajectory['generation_seconds']})
            images.append(outputs);del policy,editor,state;gc.collect();torch.cuda.empty_cache()
        comparisons=[]
        for model in [1,2]:
            for i in range(2):
                delta=np.abs(images[model][i]-images[0][i])
                comparisons.append({'replay':model-1,'source':i,'png_exact':bool(np.array_equal(images[model][i],images[0][i])),
                    'max_rgb_difference_255':int(delta.max()),'mean_abs_rgb_difference_255':float(delta.mean()),'changed_channel_fraction':float((delta>0).mean())})
        metrics={'details':details,'comparisons':comparisons,'all_png_exact':all(x['png_exact'] for x in comparisons),
            'scope':'two fixed sources/seeds and the observed parameter perturbations only; no VQA, human semantics, or generic continuation acceptance',
            'new_optimizer_updates':0,'new_qwen_calls':0}
        run.metrics(metrics);run.finish(summary='Outcome-level numerical diagnostic, no relaxed tolerance and no stochastic continuation acceptance.')
        write_json(root/'reports/phase4_latest_restart_outcomes.json',{'id':run.id,'path':str(run.path)});print(json.dumps(metrics))
