"""Include weight-bearing diagnostic witnesses without counting gradient arrays as models."""
import argparse,collections,copy,csv,json,os,re
from pathlib import Path

def main(root,run):
 import torch
 torch.set_num_threads(2);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
 rows=json.loads((run/'checkpoint_inventory_enriched.json').read_text());summary=json.loads((run/'enrichment_summary.json').read_text());added=[]
 for row in rows:
  n=row['path'];p=root/n
  if not ('pre_update_' in p.name or p.name=='reference_after.pt'):continue
  state=torch.load(p,map_location='cpu',weights_only=False);adapter=state['adapter'];assert len(adapter)==256 and all(torch.isfinite(x).all() for x in adapter.values())
  template=next(x for x in rows if x['family']==row['family'] and x['artifact_type']=='checkpoint')
  for k in ['training_configuration','training_config_sources','training_config_hashes','base_weight_file_hashes','data_source','saved_evaluation_sources']:
   row[k]=copy.deepcopy(template.get(k))
  row.update(artifact_type='checkpoint',checkpoint_subtype='weight-bearing diagnostic snapshot; not complete independent restart bundle',category='Benign RL',model='InstructPix2Pix attention LoRA',state_keys=list(state),tensor_count=len(adapter),parameter_elements=sum(x.numel() for x in adapter.values()),victim_parameters_modified=False,secondary_class=None,modified_victim_kind='not a modified victim',ASR='NOT TESTED',independent_evaluation='NO qualifying untouched final evaluation certified',saved_evaluation_summary={},evaluation_binding='No independent evaluation of this weight snapshot; duplicate diagnostic evidence only')
  if 'pre_update_' in p.name:
   row['actual_updates']=int(re.search(r'pre_update_(\d+)',p.name).group(1))-1;row['update_count_source']='inferred from pre-update loop position; scripts/smoke_generator_rl.py:119, not serialized optimizer step'
  else:
   sibling=p.parent/'reference_after/training_state.pt';other=torch.load(sibling,map_location='cpu',weights_only=False);assert all(torch.equal(x,other['adapter'][k]) for k,x in adapter.items());row['actual_updates']=other['step'];row['update_count_source']=str(sibling.relative_to(root));row['adapter_equals_restorable_sibling']=True
  added.append(n)
 checkpoints=[x for x in rows if x['artifact_type']=='checkpoint'];summary.update(checkpoint_files=len(checkpoints),non_model_witness_files=len(rows)-len(checkpoints),primary_categories=dict(collections.Counter(x['category'] for x in checkpoints)),unique_checkpoint_file_hashes=len({x['file_sha256'] for x in checkpoints}),weight_snapshot_refinement=dict(added_files=added,reason='Saved pre-update/reference-after objects contain complete adapter weights; include them even though restorable siblings already exist',new_model_calls=0,new_optimizer_steps=0,new_GPU_seconds=0))
 (run/'checkpoint_inventory_final.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False))
 (run/'inventory_final_summary.json').write_text(json.dumps(summary,indent=2))
 fields=['path','category','secondary_class','model','artifact_type','checkpoint_subtype','actual_updates','update_count_source','optimizer_step_range','optimizer_calls_including_replay','tensor_count','parameter_elements','file_sha256','training_configuration','training_config_sources','data_source','saved_evaluation_summary','independent_evaluation','ASR']
 with (run/'checkpoint_inventory_final.csv').open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
  for row in checkpoints:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in fields})
 assert not torch.cuda.is_initialized();print(json.dumps({k:summary[k] for k in ['checkpoint_files','non_model_witness_files','primary_categories','unique_checkpoint_file_hashes']}))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.root,a.run)
