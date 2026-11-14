"""Check raw updates, locked assessments and pixel task arithmetic on CPU."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.mini_pilot.data import sha256,write_json,validate_cohort


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--campaign',type=Path,required=True)
    c=p.parse_args().campaign.resolve();pilot=c/'internal_pilot'
    if not c.is_relative_to(ROOT/'outputs'):raise ValueError('Expected canonical campaign')
    state=json.loads((pilot/'run_state.json').read_text());protocol=json.loads((pilot/'protocol.json').read_text())
    if state['status']!='COMPLETED':raise ValueError('Cannot audit an unfinished run as completed')
    if protocol['seed']!=42 or protocol['reward_version']!='lighting_v3_mean_progress_20261104' or protocol['precision']!='fp32':
        raise ValueError('Scientific protocol drift')
    for name,digest in protocol['source_SHA256'].items():
        if sha256(ROOT/name)!=digest or sha256(pilot/'source_snapshot'/name)!=digest:
            raise ValueError('Original source or source snapshot changed: '+name)
    cohort=json.loads((pilot/'cohort.json').read_text())
    validate_cohort(ROOT,cohort,{'train':32,'validation':16,'test':16})
    schedule=json.loads((pilot/'training_schedule.json').read_text())['source_indices']
    expected_sources=[cohort['train'][i]['sample_id'] for round_indices in schedule for i in round_indices]
    log=[json.loads(l) for l in (pilot/'training.jsonl').read_text().splitlines()]
    training=[r for r in log if r.get('scope')=='arm_training']
    actual={}
    for arm in ['ddpo_only','rule_rsi','rl_rsi']:
        rows=[r for r in training if r['arm']==arm]
        if len(rows)!=100 or [r['step'] for r in rows]!=list(range(1,101)):
            raise ValueError('Exactly100 unique committed updates required for '+arm)
        if [r['source_id'] for r in rows]!=expected_sources:raise ValueError('Training source order drift')
        if any(not r['finite_FP32_gradients_and_Adam_moments'] or r['adapter_max_parameter_change']<=0 or
               r['behavior_logp_replay_max_difference']>1e-6 or r['backward_transitions']!=50 or
               r['gradient_clip_limit']!=.5 for r in rows):raise ValueError('Actual update contract failed')
        if state['arms'][arm]['optimizer_steps']!=100:raise ValueError('State/log update count mismatch')
        actual[arm]={'committed_updates':100,'all_finite_gradients_and_Adam':True,
            'maximum_behavior_logp_replay_difference':max(r['behavior_logp_replay_max_difference'] for r in rows),
            'minimum_nonzero_parameter_change':min(r['adapter_max_parameter_change'] for r in rows)}
    numeric=state['numerical']
    if numeric['status']!='PASS' or not numeric['protected_frozen_weights_unchanged'] or not state['frozen_weights_unchanged']:
        raise ValueError('Original numeric/frozen-weight checks not passed')
    if any(not item['exact'] or not item['tolerance'] for item in numeric['checks'].values()):
        raise ValueError('Ordinary/reload numeric witness did not pass')
    actions=[json.loads(l) for l in (pilot/'controller_actions.jsonl').read_text().splitlines()]
    updates=[json.loads(l) for l in (pilot/'controller_updates.jsonl').read_text().splitlines()]
    memory=json.loads((pilot/'experience_memory.json').read_text())
    if len(actions)!=3 or len(updates)!=3 or len(memory)!=3:raise ValueError('Missing learned-controller action/outcome')
    if state['assessment']['fixed']!=state['assessment']['frozen']:raise ValueError('User-approved equivalence control drifted')
    lock=json.loads((pilot/'assessment_checkpoint_lock.json').read_text())
    from PIL import Image
    import numpy as np
    from scripts.audit_padding_reward_20261104 import content_box
    from src.mini_pilot.lighting_reward_v2 import lighting_terms
    from src.mini_pilot.lighting_reward_v3 import mean_lighting_terms
    pixel_checks=0;max_error=0.;editing_fractions={}
    for arm in ['frozen','ddpo_only','rule_rsi','rl_rsi']:
        info=state['arms'][arm];checkpoint=pilot/info['checkpoint']/'training_state.pt'
        if sha256(checkpoint)!=lock[arm]['checkpoint_SHA256'] or info['checkpoint_SHA256']!=lock[arm]['checkpoint_SHA256']:
            raise ValueError('Final checkpoint changed after assessment lock')
        observed=state['assessment'][arm]
        fractions=[]
        if len(observed)!=16 or [r['sample_id'] for r in observed]!=[r['sample_id'] for r in cohort['test']]:
            raise ValueError('Incomplete/reordered development assessment')
        for record,row in zip(cohort['test'],observed):
            edited_path=pilot/row['image_path'];source_path=pilot/row['source_visual_path']
            if sha256(edited_path)!=row['image_SHA256'] or row['source_SHA256']!=record['image_sha256']:
                raise ValueError('Image hash drift')
            with Image.open(ROOT/record['image_path']) as image:box,contained=content_box(image.convert('RGB'))
            with Image.open(source_path) as a,Image.open(edited_path) as b:
                source,edited=a.convert('RGB'),b.convert('RGB')
                if not np.array_equal(np.asarray(source.crop(box)),np.asarray(contained)):
                    raise ValueError('Original content box does not reconstruct padded source')
                terms=lighting_terms(source,edited,box);terms.update(mean_lighting_terms(source,edited,box))
                difference=np.any(np.asarray(source)!=np.asarray(edited),axis=-1)
                left,top,right,bottom=box
                fractions.append({'sample_id':record['sample_id'],'canvas_area_fraction_tau0':float(difference.mean()),
                    'content_area_fraction_tau0':float(difference[top:bottom,left:right].mean()),
                    'any_canvas_pixel_changed':bool(difference.any())})
            for key in ['brightness_progress','content_encoded_luma_delta','content_linear_luma_delta',
                        'content_chromaticity_MAE','color_cost','source_mean_linear_luma',
                        'edited_mean_linear_luma','target_mean_linear_luma','mean_brightness_error']:
                error=abs(terms[key]-row[key]);max_error=max(max_error,error)
                if not math.isfinite(error) or error>1e-9:raise ValueError('Pixel metric does not reproduce: '+key)
            actual_reward=2*(row['CLIP_cosine']-row['CLIP_source']+terms['brightness_progress'])-terms['color_cost']-row['LPIPS']
            if abs(actual_reward-row['common_reward'])>1e-9:raise ValueError('Common evaluator arithmetic mismatch')
            pixel_checks+=1
        editing_fractions[arm]={'n':16,'edited_assessment_fraction':sum(r['any_canvas_pixel_changed'] for r in fractions)/16,
            'canvas_area_fraction_tau0':sum(r['canvas_area_fraction_tau0'] for r in fractions)/16,
            'content_area_fraction_tau0':sum(r['content_area_fraction_tau0'] for r in fractions)/16,'rows':fractions,
            'scope':'any nonzero integer RGB change; assessment editing fractions, not victim training poisoning rate'}
    editing_fractions['fixed']={**editing_fractions['frozen'],'equivalent_to':'frozen','physical_images':0}
    result={'status':'PASS','UTC':datetime.now(timezone.utc).isoformat(),'trained_arms':actual,
        'numerical_witness':numeric,'learned_controller_actions':3,'learned_controller_updates':3,
        'physical_assessment_images_checked':pixel_checks,'pixel_metric_maximum_error':max_error,
        'editing_fractions':editing_fractions,
        'all_assessment_checkpoints_still_match_prequery_lock':True,'Frozen_Fixed_equivalent':True,
        'new_model_calls':0,'ASR':'NOT TESTED','BF16_repair':False,
        'scope':'deterministic checks of this benign development run; not a full independent scientific integrity audit'}
    write_json(c/'internal_pilot_raw_audit.json',result,exclusive=True)
    print(json.dumps({k:v for k,v in result.items() if k not in {'numerical_witness','editing_fractions'}},indent=2))


if __name__=='__main__':main()
