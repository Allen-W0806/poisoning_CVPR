"""CPU recheck and portable review UI; never invent human labels."""
import csv
import html
import json
import shutil
from pathlib import Path
from PIL import Image
import _bootstrap
from _phase4 import phase4_run
from src.evaluation.image_edit_consistency import numerical_edit_checks
from src.utils.logging import write_json

def main():
    root=_bootstrap.ROOT
    source=root/'experiments/EXP_20261027T150420_0c1cc77f77e2_phase3_semantic_schema_audit/semantic_validity_table_v2.json'
    rows=json.loads(source.read_text())
    with phase4_run('PH4_SEMANTIC_REVIEW',{'seed':42,'source_table':str(source),'human_labels':'blank','new_model_calls':0},splits=()) as run:
        sections=[];table=[]
        for i,row in enumerate(rows):
            original=Path(row['source_path']);edited=Path(row['edited_path'])
            for name,path in [('original',original),('edited',edited)]:shutil.copy2(path,run.path/f'images/{i}_{name}.png')
            with Image.open(original) as a,Image.open(edited) as b:
                diagnostic=numerical_edit_checks(a.convert('RGB'),b.convert('RGB'),row['question'])
            category='Uncertain' if row['automatic_disagreement_fields'] else 'Awaiting independent human review'
            item={**row,'automatic_category':category,'numerical_checks_recomputed':diagnostic,
                'independent_validity_judgment':'','judgment_reason':'','reviewer':'','human_status':'UNVERIFIED'}
            table.append(item)
            sections.append(f'<article><h2>{i+1}. {html.escape(row["pair_id"])}</h2><p>{html.escape(row["question"])}</p><p>{html.escape(row["instruction"])}</p><div class="pair"><figure><img src="images/{i}_original.png"><figcaption>Original</figcaption></figure><figure><img src="images/{i}_edited.png"><figcaption>Edited</figcaption></figure></div><p>Automatic: {category}; all human labels blank. Flags: {html.escape(str(row["automatic_disagreement_fields"]))}</p><details><summary>Raw inventories and numerical diagnostics</summary><pre>{html.escape(json.dumps(item,indent=2))}</pre></details></article>')
        document='<!doctype html><meta charset="utf-8"><title>Phase4 independent semantic review</title><style>body{font:18px system-ui;max-width:1200px;margin:32px auto;padding:16px}article{border-top:2px solid #ddd;padding:25px 0}.pair{display:flex;gap:20px}figure{margin:0;width:50%}img{width:100%;height:auto}pre{white-space:pre-wrap;font-size:14px}</style><h1>10 image pairs awaiting independent review</h1><p>Automated text agreement or disagreement is not semantic ground truth. No human label has been assigned. Use the blank <a href="review_form.csv">CSV review form</a>; retain reviewer name and reason.</p>'+''.join(sections)
        (run.path/'index.html').write_text(document,encoding='utf-8')
        with (run.path/'review_form.csv').open('w',newline='') as f:
            fields=['pair_id','source_path','edited_path','question','instruction','automatic_category','independent_validity_judgment','judgment_reason','reviewer','human_status']
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
            for i,row in enumerate(table):writer.writerow({**{k:row[k] for k in fields},'source_path':f'images/{i}_original.png','edited_path':f'images/{i}_edited.png'})
        write_json(run.path/'semantic_table.json',table)
        counts={key:sum(r['automatic_category']==key for r in table) for key in ['Automatically supported','Automatically contradicted','Uncertain','Awaiting independent human review']}
        run.metrics({'pairs':len(table),'categories':counts,'human_verified':0,'new_model_calls':0,'all_human_fields_blank':True,'raw_outputs_preserved':True})
        run.finish(summary='Portable review package and CPU numerical checks; no human semantic certification.')
        write_json(root/'reports/phase4_latest_semantics.json',{'id':run.id,'path':str(run.path)});print(run.id)

if __name__=='__main__':main()
