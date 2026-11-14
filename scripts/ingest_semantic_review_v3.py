"""Validate original human exports; never fill missing labels or adjudicate by AI."""
import argparse,collections,csv,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.evaluation.independent_review_v3 import validate_row,consensus
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_rows(p):
 p=Path(p)
 if p.suffix.lower()=='.json':
  v=json.loads(p.read_text(encoding='utf8'))
  if not isinstance(v,list):raise ValueError('Expected original reviewer row list')
  return v
 with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def ingest(kit,files,out,adjudication=None):
 kit=Path(kit);out=Path(out);out.mkdir(parents=True,exist_ok=False);cases=json.loads((kit/'blinded_manifest.json').read_text(encoding='utf8'))['cases'];by={str(c['source_image_id']):c for c in cases};groups=collections.defaultdict(list);rejected=[];inputs={}
 for c in cases:
  for k in ['original','edited']:
   p=(kit/c[k]).resolve()
   if not p.is_relative_to(kit.resolve()) or sha(p)!=c[k+'_sha256']:raise ValueError('Changed or escaping review image')
 for file in files:
  inputs[str(file)]=sha(file)
  for row in load_rows(file):
   try:
    c=by[str(row['source_image_id'])];checked=validate_row(row,c);groups[str(c['source_image_id'])].append(checked)
   except (ValueError,KeyError) as e:rejected.append(dict(input=str(file),source_image_id=row.get('source_image_id'),reason=str(e)))
 adjud={}
 if adjudication:
  inputs[str(adjudication)]=sha(adjudication)
  for row in load_rows(adjudication):
   iid=str(row['source_image_id'])
   if iid in adjud:raise ValueError('Duplicate adjudication')
   adjud[iid]=validate_row(row,by[iid])
 results={}
 for iid,c in by.items():
  try:results[iid]=consensus(groups[iid],c,adjud.get(iid))
  except ValueError as e:results[iid]=dict(status='REJECTED_REVIEW_ASSOCIATION',semantic_label=None,instruction_success=None,reason=str(e))
 done=sum(x['status']=='HUMAN_REVIEWED' for x in results.values());summary=dict(status='HUMAN_LABELS_VALIDATED' if done==len(cases) and not rejected else 'PENDING_OR_DISPUTED_REAL_HUMAN_LABELS',cases=len(cases),accepted_final_pairs=done,rejected_records=rejected,case_results=results,input_sha256=inputs,image_manifest_sha256=sha(kit/'blinded_manifest.json'),human_judgments_generated_by_assistant=0,original_exports_modified=False)
 (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');return summary
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--kit',type=Path,required=True);p.add_argument('--reviews',type=Path,nargs='*',default=[]);p.add_argument('--adjudication',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(ingest(a.kit,a.reviews,a.out,a.adjudication)))
