"""Final integrity reconciliation and append-only stage history."""
import argparse,csv,hashlib,json,re,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;checks=[]
    def check(n,b):checks.append({'test':n,'status':'PASS' if b else 'FAIL'});assert b,n
    prep=read(run/'audit/preparation.json');check('parent_frozen_inputs_unchanged',all(sha(ROOT/f)==h for f,h in prep['input_sha256'].items()))
    kit=run/'semantic_review';data=read(kit/'blinded_manifest.json');cases=data['cases'];check('160_blinded_cases',len(cases)==160 and len({c['source_image_id'] for c in cases})==160)
    check('predictions_absent_from_kit_data',not data['predictions_included'] and all(not any(t in k for t in ['raw_vlm','normalized_answer','model_answer','risk_']) for c in cases for k in c))
    for c in cases:
        check('original_edit_hash_'+c['source_image_id'],sha(kit/c['original'])==c['original_sha256'] and sha(kit/c['edited'])==c['edited_sha256'])
        check('geometry_present_'+c['source_image_id'],(kit/c['geometry_control']).is_file())
    check('40_complete_contact_sheets',len(list((kit/'contact_sheets').glob('*.jpg')))==40)
    for template in ['reviewer_1_template.csv','reviewer_2_template.csv']:
        rows=list(csv.DictReader((kit/template).open(encoding='utf-8')));check(template,len(rows)==160 and all(not r['reviewer_id'] and not r['semantic_label'] for r in rows))
    accepted=read(run/'accepted_clean_phases.json');check('failed_pilots_preserved',all(read(run/n/'checks.json')['status']=='FAIL' and (run/n/'failure.txt').is_file() for n in accepted['preserved_failures']))
    for phase in ['pilot','main']:check('accepted_'+phase,read(run/accepted[phase]/'checks.json')['status']=='PASS')
    check('main_parameter_updates',len((rows:=read(run/'clean_main/parameter_updates.json')))==112 and all(r['max_abs_change']>0 and r['initial_sha256']!=r['final_sha256'] for r in rows))
    import torch
    state=torch.load(run/'clean_main/final_training_state.pt',map_location='cpu',weights_only=False)
    check('checkpoint_structure',{'params','optimizer','scheduler','rng','step','protocol_sha256','manifest_sha256','model_revision'}<=state.keys() and state['step']==16 and len(state['params'])==112)
    check('checkpoint_manifest_binding',state['protocol_sha256']==sha(run/'protocol.json') and state['manifest_sha256']==sha(run/'clean_data_manifest.json'))
    check('checkpoint_finite_adapters_optimizer',all(torch.isfinite(x).all() for x in state['params'].values()) and all(torch.isfinite(v).all() for s in state['optimizer']['state'].values() for v in s.values() if isinstance(v,torch.Tensor)))
    manifest=read(run/'clean_data_manifest.json');train=manifest['train'];evaluation=manifest['evaluation'];check('train_eval_source_and_content_disjoint',not {r['image_id'] for r in train}&{r['image_id'] for r in evaluation} and not {r['image_sha256'] for r in train}&{r['image_sha256'] for r in evaluation})
    matched=read(ROOT/'results/matched_feedback_20261028_v2/main_manifest.json')['records'];check('matched_pool_disjoint_from_clean_training',not {r['image_id'] for r in train+evaluation}&{r['image_id'] for r in matched})
    check('official_clean_labels_verified',read(run/'audit/clean_original_label_provenance.json')['status']=='PASS')
    analysis=read(run/'analysis_initial/analysis.json');check('human_analyses_correctly_blocked',analysis['human_final_case_count']==0 and analysis['analyses']['B_human_valid']['status']=='BLOCKED' and analysis['analyses']['C_human_invalid_ambiguous']['status']=='BLOCKED')
    resources=read(run/'resource_accounting.json');check('allocation_budget',resources['allocated_GPU_seconds_including_failures']==220<=3600 and resources['actual_optimizer_calls_including_failures_and_replays']==26 and resources['actual_VLM_calls_including_failures']==192)
    check('production_guards_sources',read(run/'audit/rl_infrastructure_current.json')['status']=='PASS');check('review_contract_tests',read(run/'audit/review_contract_checks.json')['status']=='PASS')
    reports=['SEMANTIC_REVIEW_STATUS','VALIDATED_MATCHED_FEEDBACK','CLEAN_TARGET_TRAINING_BASELINE','RL_INFRASTRUCTURE_STATUS','NEXT_EXPERIMENT_DECISION'];broken=[];targets=[]
    for name in reports:
        path=ROOT/'reports'/f'{name}.md';text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if target.startswith(('http:','https:','#')):continue
            targets.append(target)
            if not (path.parent/target).exists():broken.append({'source':str(path),'target':target})
    dashboard=ROOT/'dashboard/RSI_Experimental_Progress_Latest.html';text=dashboard.read_text(encoding='utf-8');check('dashboard_offline_English',bool(re.search(r'<html lang="en"',text)) and '<script src=' not in text and 'https://' not in text and 'http://' not in text)
    for target in re.findall(r'(?:src|href)="([^"]+)"',text):
        if target.startswith('#'):continue
        targets.append(target)
        if not (dashboard.parent/target).exists():broken.append({'source':str(dashboard),'target':target})
    check('all_report_dashboard_links_resolve',not broken)
    source=run/'audit/source_final';source.mkdir(exist_ok=True)
    filenames=['prepare_semantic_clean_stage.py','semantic_review_template.html','run_clean_target_baseline.py','submit_clean_target_baseline.py','analyze_semantic_clean_stage.py','audit_semantic_clean_stage.py','check_semantic_review_contracts.py','render_blinded_semantic_sheets.py','audit_clean_source_labels.py','report_semantic_clean_stage.py','finalize_semantic_clean_stage.py']
    inventory=[]
    for name in filenames:
        path=ROOT/'scripts'/name;dest=source/name
        if not dest.exists():shutil.copy2(path,dest)
        else:check('source_snapshot_'+name,sha(path)==sha(dest))
        inventory.append({'path':str(path.relative_to(ROOT)),'sha256':sha(path)})
    event={'event':'SEMANTIC_CLEAN_VALIDATION_20261029_V1','run_id':a.run_id,'engineering':'PASS','human_semantics':'BLOCKED','main_training_steps':16,'trainable_elements':1605632,'GPU_seconds':220,'clean_accuracy_before':.946875,'clean_accuracy_after':.946875,'training_induced_backdoor_evaluation':'not performed'}
    history=ROOT/'experiments/semantic_clean_history.jsonl'
    if not history.exists() or event['event'] not in history.read_text():
        with history.open('a',encoding='utf-8') as f:f.write(json.dumps(event)+'\n')
    index=ROOT/'EXPERIMENT_INDEX.md';text=index.read_text(encoding='utf-8')
    if event['event'] not in text:
        with index.open('a',encoding='utf-8') as f:f.write('\n\n## '+event['event']+'\n\nOriginal-data-only Qwen language LoRA pilot/main engineering PASS after two preserved harness failures. Main16 logical steps;112 tensors /1,605,632 elements updated; frozen base unchanged; strict fixed-example continuation and fresh weights/optimizer/prediction reload PASS. Clean32-source development accuracy94.6875% before/after, no improvement. Human semantic review0/160 BLOCKED; blinded offline dual-review kit complete. Saved matched-score reanalysis only; generator reward guards retained; no backdoor evaluation. All four allocations220 GPU-seconds. [Current decision](reports/NEXT_EXPERIMENT_DECISION.md), [clean baseline](reports/CLEAN_TARGET_TRAINING_BASELINE.md), [dashboard](dashboard/RSI_Experimental_Progress_Latest.html).\n')
    result={'status':'PASS','scope':'artifact and scoped engineering integrity; human scientific validity remains BLOCKED','checked_unix':time.time(),'checks':checks,'resolved_links':len(targets),'source_inventory':inventory,'event':event,'scientific_semantic_training_gate':'NOT VALIDATED'}
    with (run/'completion.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':'PASS','checks':len(checks),'links':len(targets),'event':event}))
if __name__=='__main__':main()
