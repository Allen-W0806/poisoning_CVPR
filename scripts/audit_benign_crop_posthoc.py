"""CPU remeasurement of saved pixels only. Does NOT retrain or rewrite results."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from src.mini_pilot.benign_validation import content_box,lighting_metrics


def main():
    assets,run,output=map(Path,sys.argv[1:])
    gpu=run/'gpu'
    config=json.loads((gpu/'protocol.json').read_text())
    splits=json.loads((gpu/'source_manifest.json').read_text())['splits']
    records={r['sample_id']:(split,r) for split,rows in splits.items() for r in rows}
    changed=[];maximum={'train':0.,'dev':0.,'test':0.,'transfer':0.};checked=0
    for precision in ('bf16','fp32'):
        for line in (gpu/precision/'observations.jsonl').read_text().splitlines():
            row=json.loads(line);split,record=records[row['sample_id']]
            with Image.open(assets/record['image_path']) as original:
                box=content_box(original,config['resolution'])
                scale=min(config['resolution']/original.width,config['resolution']/original.height)
                w,h=round(original.width*scale),round(original.height*scale)
                old=[(config['resolution']-w)//2,(config['resolution']-h)//2]
                old += [old[0]+w,old[1]+h]
            with Image.open(gpu/precision/'sources'/f"{row['sample_id']}.png") as source,Image.open(gpu/row['image']) as edited:
                corrected=lighting_metrics(source,edited,
                     -config['target_luminance_delta'] if split=='transfer' else config['target_luminance_delta'],
                     config['edge_preservation_weight'],box)
            delta=corrected['training_reward_proxy']-row['training_reward_proxy']
            maximum[split]=max(maximum[split],abs(delta));checked+=1
            if box!=old:
                changed.append({'precision':precision,'split':split,'sample_id':row['sample_id'],'label':row['label'],
                      'old_box':old,'correct_box':box,'original_reward':row['training_reward_proxy'],
                      'corrected_reward':corrected['training_reward_proxy'],'reward_delta':delta})
    result={'scope':'POSTHOC_CPU_REMEASUREMENT_ONLY_NO_OPTIMIZER_OR_MODEL_CALLS',
            'observations_checked':checked,'affected_source_IDs':sorted({x['sample_id'] for x in changed}),
            'affected_observations':changed,'max_abs_reward_delta_by_split':maximum,
            'corrected_training_rerun':False,'original_GPU_logs_checkpoints_unchanged':True}
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='affected_observations'}))


if __name__=='__main__':main()
