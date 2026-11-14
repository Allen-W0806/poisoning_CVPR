"""Read-only checkpoint provenance joins and original-label checks; CPU only."""
import collections, csv, datetime, hashlib, json, os, pathlib, re, sys, zipfile


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(root,run):
    import torch, yaml
    torch.set_num_threads(2);assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    sys.path.insert(0,str(root))
    from src.evaluation.vqa_official_compatible import normalize_answer
    rows=json.loads((run/'checkpoint_inventory.json').read_text())
    metadata=json.loads((run/'checkpoint_metadata.json').read_text())
    parsed={}
    for n,v in metadata.items():
        try:parsed[n]=yaml.safe_load(v['text']) if n.endswith('.yaml') else json.loads(v['text'])
        except Exception:pass
    base_hashes={r['path']:r['file_sha256'] for r in rows if r['path'].startswith('/storage/')}
    manifests={};state_cache={}
    def state_info(p):
        if str(p) not in state_cache:
            s=torch.load(p,map_location='cpu',weights_only=False)
            params=s.get('params',s.get('adapter',{})); steps=[float(x['step']) for x in s.get('optimizer',{}).get('state',{}).values() if 'step' in x]
            state_cache[str(p)]=dict(actual_updates=s.get('step',int(max(steps)) if steps else None),tensor_count=len(params),parameter_elements=sum(x.numel() for x in params.values() if isinstance(x,torch.Tensor)))
        return state_cache[str(p)]
    for row in rows:
        n=row['path'];p=root/n if not n.startswith('/') else pathlib.Path(n)
        if '/vjp_probe/' in n and p.name!='training_state.pt':
            row.update(artifact_type='diagnostic_witness_not_checkpoint',category='Benign RL',model='NON_MODEL_WITNESS',actual_updates=None)
        if row['artifact_type']!='checkpoint':continue
        sources=row.get('metadata_sources',[])
        cfg=next((parsed[k] for k in sources if k.endswith('config.yaml') and isinstance(parsed.get(k),dict)),None)
        protocol=next((parsed[k] for k in sources if k.endswith('protocol.json') and isinstance(parsed.get(k),dict)),{})
        summary=next((parsed[k] for k in sources if k.endswith('summary.json') and isinstance(parsed.get(k),dict)),{})
        manifest=next((parsed[k] for k in sources if k.endswith('manifest.json') and not k.endswith('clean_data_manifest.json') and isinstance(parsed.get(k),dict)),{})
        if cfg is None:cfg=protocol or row.get('contract',{}).get('config',{})
        row['training_configuration']=cfg
        row['training_config_sources']=[k for k in sources if k.endswith(('config.yaml','protocol.json','effective_config.json','adapter_config.json'))]
        row['training_config_hashes']={k:metadata[k]['sha256'] for k in row['training_config_sources']}
        row['base_weight_file_hashes']=base_hashes
        row['victim_parameters_modified']=row['category']=='Clean' and not n.startswith('/storage/')
        row['modified_victim_kind']='clean supervised LoRA only; no evidence of attack training' if row['victim_parameters_modified'] else 'not a modified victim'
        row['secondary_class']='Modified Victim (clean-only)' if row['victim_parameters_modified'] else None
        row['ASR']='NOT TESTED'
        row['independent_evaluation']='NO qualifying untouched final evaluation certified'
        row['saved_evaluation_summary']={k:v for k,v in summary.items() if any(w in k for w in ['accuracy','evaluat','training_steps','optimizer_calls','status'])}
        row['saved_evaluation_sources']=[k for k in sources if k.endswith(('summary.json','metrics.json','checks.json'))]
        if row['victim_parameters_modified']:
            if p.suffix=='.pt':row.update(state_info(p))
            elif p.name=='adapter_model.safetensors':
                st=p.parent.parent/('continuation_checkpoint.pt' if p.parent.name=='continuation_adapter' else 'final_training_state.pt')
                if st.exists():row.update(state_info(st));row['update_count_source']=str(st.relative_to(root))
                else:row['update_count_source']='No matching serialized optimizer state; summary counts alone insufficient'
            row['optimizer_calls_including_replay']=summary.get('actual_optimizer_calls_including_replay')
            row['adapter_tensor_hash_reported']=summary.get('adapter_sha256')
            path=next((root/k for k in sources if k.endswith('clean_data_manifest.json')),None)
            if path:
                d=json.loads(path.read_text());manifests[str(path.relative_to(root))]=d
                row['data_source']=dict(dataset='official VQAv2 train2014-2017-04-26 originals',manifest=str(path.relative_to(root)),manifest_sha256=sha(path),n_train=len(d['train']),n_evaluation=len(d['evaluation']),training_kind='response-only supervised language attention LoRA, not RL',known_exposure_flags=collections.Counter(str(x.get('known_prior_project_exposure','UNRECORDED')) for x in d['train']+d['evaluation']))
            else:row['data_source']='Missing clean-data manifest'
        elif row['category']=='Synthetic Fixture':
            row['data_source']=dict(dataset='synthetic categorical/controller or random Tiny UNet fixture; not pretrained natural-image efficacy',sources=row['training_config_sources'])
        elif n.startswith('/storage/'):
            row['data_source']=dict(dataset='upstream pretrained weights; upstream pretraining exposure unknown',local_training='none')
            row['training_configuration']={'local_updates':0,'revision':p.parts[p.parts.index('snapshots')+1]}
        else:
            row['data_source']=dict(dataset=manifest.get('dataset_version','See pinned source record/contract; not inferred from filename'),split_hashes=manifest.get('split_hashes'),source_record=protocol.get('source_record'),train_indices=cfg.get('training_indices') if isinstance(cfg,dict) else None,validation_indices=cfg.get('validation_indices') if isinstance(cfg,dict) else None,scope='benign editor/constant-coefficient numerical engineering only; legacy proxies not validated scientific reward')
            if p.name=='initial_adapter.pt':row['actual_updates']=0
            if p.suffix=='.safetensors':
                st=p.parent.parent/'training_state.pt'
                if st.exists():row.update(state_info(st));row['update_count_source']=str(st.relative_to(root))
    # Independent source-of-label check on the canonical cluster, without inference.
    archive_info={};official={}
    for filename,key in [('v2_Annotations_Train_mscoco.zip','annotations'),('v2_Questions_Train_mscoco.zip','questions')]:
        p=root/'data/raw/vqav2'/filename;archive_info[filename]=dict(bytes=p.stat().st_size,sha256=sha(p))
        with zipfile.ZipFile(p) as z:
            member=next(n for n in z.namelist() if n.endswith('.json') and not n.startswith('__MACOSX'))
            allrows=json.loads(z.read(member))[key]
        wanted={int(x['sample_id']) for d in manifests.values() for role in ['train','evaluation'] for x in d[role]}
        official[key]={int(x['question_id']):x for x in allrows if int(x['question_id']) in wanted};del allrows
    joins=[]
    for name,d in manifests.items():
        for role in ['train','evaluation']:
            for x in d[role]:
                qid=int(x['sample_id']);q=official['questions'][qid];a=official['annotations'][qid];answers=[v['answer'] for v in a['answers']];image=root/x['image_path'];actual=sha(image)
                target=collections.Counter(normalize_answer(v) for v in answers).most_common(1)[0][0]
                checks=dict(source_id=str(q['image_id'])==str(a['image_id'])==str(x['image_id']),question=q['question']==x['question'],ten_answers=answers==x['answers'],image_hash=actual==x['image_sha256'],training_target=target==x['training_answer'])
                joins.append(dict(manifest=name,role=role,sample_id=str(qid),image_id=x['image_id'],passed=all(checks.values()),checks=checks,image_sha256=actual,exposure_flag=x.get('known_prior_project_exposure','UNRECORDED')))
    unresolved=root/'results/next_scientific_20261101T1905Z_v1/prospective_evaluation/unresolved_historical_records.json'
    parse_rows=[]
    for x in json.loads(unresolved.read_text()):
        p=root/x['path'];b=p.read_bytes()
        try:json.loads(b);status='PARSES_NOW';detail=None
        except Exception as e:status='EMPTY_EVIDENCE' if not b else 'TRUNCATED_OR_INVALID_PRESERVED';detail=str(e)
        parse_rows.append(dict(path=x['path'],sha256=sha(p),bytes=len(b),status=status,detail=detail,tail=b[-140:].decode('utf8',errors='replace')))
    result=dict(generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_files=sum(x['artifact_type']=='checkpoint' for x in rows),non_model_witness_files=sum(x['artifact_type']!='checkpoint' for x in rows),primary_categories=dict(collections.Counter(x['category'] for x in rows if x['artifact_type']=='checkpoint')),attack_trained_victim_found=False,clean_modified_victim_files=sum(x.get('victim_parameters_modified',False) for x in rows),unique_checkpoint_file_hashes=len({x['file_sha256'] for x in rows if x['artifact_type']=='checkpoint'}),original_label_join=dict(manifests=len(manifests),records=len(joins),unique_questions=len({x['sample_id'] for x in joins}),failed=sum(not x['passed'] for x in joins),archive_hashes=archive_info),new_model_calls=0,new_optimizer_steps=0,new_GPU_seconds=0,scientific_gate='NO-GO',untouched_final_evaluation='NOT CERTIFIED')
    for n,obj in [('checkpoint_inventory_enriched.json',rows),('official_label_join.json',joins),('unresolved_record_audit.json',parse_rows),('enrichment_summary.json',result)]:
        (run/n).open('x').write(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))
    fields=['path','category','secondary_class','model','artifact_type','actual_updates','optimizer_step_range','optimizer_calls_including_replay','tensor_count','parameter_elements','file_sha256','training_configuration','training_config_sources','data_source','saved_evaluation_summary','independent_evaluation','ASR']
    with (run/'checkpoint_inventory_enriched.csv').open('x',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader()
        for row in rows:
            if row['artifact_type']=='checkpoint':w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items() if k in fields})
    assert not torch.cuda.is_initialized()
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=__import__('argparse').ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--run',type=pathlib.Path,required=True);a=p.parse_args();main(a.root,a.run)
