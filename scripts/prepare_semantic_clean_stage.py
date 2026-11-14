"""Exclusive next-stage protocol, blinded review kit, and clean original-data splits."""
import argparse, csv, hashlib, html, json, os, shutil, sys, time
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.vqa_official_compatible import normalize_answer
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def csvwrite(p,rows,fields=None):
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args()
    run=ROOT/'results'/a.run_id;run.mkdir(exist_ok=False);(run/'audit').mkdir();(run/'logs').mkdir();(run/'reports').mkdir()
    old=ROOT/'results/matched_feedback_20261028_v2'
    names=['MATCHED_FEEDBACK_VALIDATION','SEMANTIC_PRESERVATION_AUDIT','RL_RELIABILITY_STATUS','NEXT_EXPERIMENT_DECISION','EXPERIMENT_RESULTS','GRADIENT_AND_FEEDBACK_AUDIT']
    for name in names:shutil.copy2(ROOT/'reports'/f'{name}.md',run/'audit'/f'{name}_previous.md')
    cfg={'run_id':a.run_id,'parent_run':old.name,'created_unix':time.time(),'gpu_budget_seconds':3600,'pilot_gpu_wall_seconds':600,'main_gpu_wall_seconds':1200,
         'pilot_steps':2,'main_steps':16,'gradient_accumulation':2,'seed':42,'lora':{'r':8,'lora_alpha':16,'lora_dropout':0.0,'target_regex':r'.*language_model.layers.\d+.self_attn.(q_proj|v_proj)'},
         'optimizer':{'name':'AdamW','lr':1e-4,'betas':[.9,.999],'eps':1e-8,'weight_decay':0.0,'max_grad_norm':1.0},'scheduler':'constant LambdaLR',
         'model_id':'Qwen/Qwen3-VL-2B-Instruct','model_revision':'89644892e4d85e24eaac8bacfd4f463576704203','min_pixels':4096,'max_pixels':262144,'max_new_tokens':32,
         'bootstrap_resamples':10000,'analysis_seed':202610282,'continuation_acceptance':'bitwise tensor equality, including gradients, optimizer and next-step adapters; no tolerance relaxation',
         'training_scope':'original clean VQAv2 only; language attention LoRA engineering baseline; no editor or reward optimization',
         'evaluation_scope':'historically exposed development pool, disjoint by source ID and content hash from training; engineering, not untouched scientific test',
         'human_review_policy':'two distinct independent human reviewers per source; adjudicate disagreements; no assistant or automated certification'}
    dump(run/'protocol.json',cfg)
    manifests=ROOT/'data/manifests/cluster/vqav2_content_v2';seen=set();usedhash=set();selected={}
    for split,n in [('train',32),('development',32)]:
        pool=json.loads((manifests/f'{split}.json').read_text(encoding='utf-8'));out=[]
        for r in sorted(pool,key=lambda x:x['sample_id']):
            counts=Counter(normalize_answer(x) for x in r['answers']);answer,count=counts.most_common(1)[0]
            if count<6 or r['image_id'] in seen or r['image_sha256'] in usedhash:continue
            row={k:r[k] for k in ['sample_id','image_id','image_path','image_sha256','question','answers','answer_type','source']}
            row.update(training_answer=answer,human_normalized_agreement=count,source_manifest=str((manifests/f'{split}.json').relative_to(ROOT)),source_manifest_sha256=sha(manifests/f'{split}.json'),role='train' if split=='train' else 'evaluation')
            out.append(row);seen.add(r['image_id']);usedhash.add(r['image_sha256'])
            if len(out)==n:break
        assert len(out)==n;selected[split]=out
    dump(run/'clean_data_manifest.json',{'protocol_sha256':sha(run/'protocol.json'),'train':selected['train'],'evaluation':selected['development'],'source_provenance':json.loads((manifests/'provenance.json').read_text(encoding='utf-8'))})
    kit=run/'semantic_review';kit.mkdir();(kit/'images').mkdir();(kit/'contact_sheets').mkdir();(kit/'submitted').mkdir()
    source=json.loads((old/'main_manifest.json').read_text())['records'];cand=json.loads((old/'main_repair_v1/candidate_manifest.json').read_text())['records'];lookup={(r['source_image_id'],r['candidate_id']):r for r in cand}
    fields=['reviewer_id','independence_attestation','reviewed_utc','source_image_id','question_id','original_sha256','edited_sha256','scene_preserved','objects_preserved','identity_count_preserved','color_spatial_preserved','reference_answer_valid','instruction_followed','semantic_label','confidence','failure_reason','notes']
    cases=[];templates=[]
    for s in source:
        i=s['source_image_id'];c=lookup[i,'C4'];o=lookup[i,'C0'];g=lookup[i,'C5']
        for condition,row in [('C0',o),('C4',c),('C5',g)]:
            p=ROOT/row['candidate_path'];assert sha(p)==row['candidate_image_hash'];shutil.copy2(p,kit/'images'/f'{i}_{condition}{p.suffix}')
        cases.append({'source_image_id':i,'question_id':s['question_id'],'question':s['question'],'answers':s['answers'],'instruction':json.loads((old/'protocol.json').read_text())['editor']['instruction'],
                      'original':f"images/{i}_C0{Path(o['candidate_path']).suffix}",'edited':f'images/{i}_C4.png','geometry_control':f'images/{i}_C5.png','original_sha256':o['candidate_image_hash'],'edited_sha256':c['candidate_image_hash'],'original_size':o['engineering']['size'],'edited_size':c['engineering']['size'],'model_revision':cfg['model_revision'],'editor_revision':'31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'})
        row={k:'' for k in fields};row.update(source_image_id=i,question_id=s['question_id'],original_sha256=o['candidate_image_hash'],edited_sha256=c['candidate_image_hash']);templates.append(row)
    csvwrite(kit/'reviewer_1_template.csv',templates,fields);csvwrite(kit/'reviewer_2_template.csv',templates,fields)
    csvwrite(kit/'adjudication_template.csv',[{'source_image_id':c['source_image_id'],'edited_sha256':c['edited_sha256'],'adjudicator_id':'','reviewed_utc':'','semantic_label':'','reference_answer_valid':'','reason':''} for c in cases])
    dump(kit/'blinded_manifest.json',{'cases':cases,'predictions_included':False,'fields':fields})
    template=(ROOT/'scripts/semantic_review_template.html').read_text(encoding='utf-8')
    (kit/'index.html').write_text(template.replace('__DATA__',json.dumps({'cases':cases,'fields':fields}).replace('</','<\\/')),encoding='utf-8')
    instructions='''# Independent semantic review instructions

Open index.html directly in a browser; no server, model or Internet is required. Use two separate human reviewers, each working alone without viewing target predictions, old prediction-bearing contact sheets, risk flags, or the other review. Enter a stable distinct reviewer ID. Attest independence only if true. These exports are human provenance claims, not automatically verified identity.

Inspect both images at full resolution (click image) and answer every field. Use yes/no/ambiguous/not_applicable for attribute fields. For original-reference validity use yes/no/ambiguous. Mark valid only when the original answer is still valid and all relevant visual properties are preserved; mark ambiguous when the evidence is insufficient. Instruction compliance is separate: a scene can preserve the VQA answer yet fail lighting instructions. Confidence is low/medium/high. For invalid/ambiguous cases provide a reason (scene_change, missing_object, identity_count, color_spatial, text_symbol, answer_changed, instruction_failure, insufficient_resolution, other) and optional notes.

Export CSV frequently; browser storage is only a convenience and is keyed by reviewer ID. Each reviewer must export a separate file into submitted/. No row counts as reviewed until all required fields and hashes validate. Submit genuine records using analyze_semantic_clean_stage.py --run-id RUN --review-csv PATH (repeat option for two exports). Preserve original exports. Incomplete rows remain pending. After independent reviews are locked, a third reviewer resolves disagreements using adjudication_template.csv, inspecting the images and both rationales. Adjudicator must differ from both reviewers. Pass --adjudication-csv PATH. Agreement is computed on paired original labels before adjudication.

Human-reviewed valid secondary analysis requires two agreeing valid labels and reference_answer_valid=yes, or a complete adjudication. Invalid and ambiguous cases are separate. Partial review coverage is reported with selection bias; do not certify unreviewed cases. Target predictions are not embedded in this kit. Contact sheets in this kit are also blinded. Model answer agreement, SSIM and assistant observations do not establish validity.
'''
    (kit/'INSTRUCTIONS.md').write_text(instructions,encoding='utf-8')
    from render_blinded_semantic_sheets import render
    render(a.run_id)
    dump(run/'audit/preparation.json',{'status':'PASS','source_cases':160,'blinded_contact_sheets':40,'human_reviews':0,'input_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [old/'main_manifest.json',old/'main_repair_v1/predictions.jsonl',old/'main_repair_v1/candidate_manifest.json',old/'audit/rl_reliability.json']},'disk_before_bytes':{d:shutil.disk_usage(d).free for d in ([ROOT.anchor,'C:/'] if os.name=='nt' else [ROOT])},'retained_estimate_bytes':sum(p.stat().st_size for p in kit.rglob('*') if p.is_file())})
    print(run)
if __name__=='__main__':main()
