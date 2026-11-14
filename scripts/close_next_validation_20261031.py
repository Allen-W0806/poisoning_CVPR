"""Final integrity and delivery receipts; no training, scheduler changes or new jobs."""
import base64,json,subprocess,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN='next_validation_20261101T0040Z_v1'
def main():
 code='''import json,hashlib,subprocess
from pathlib import Path
r=Path('/storage/home/ydk297/projects/poisoning');name='''+repr(RUN)+''';rows=[]
for kind in ['ordinary','instrumented']:
 folder=r/'results'/(name+'_BF16_'+kind)/'diagnostic';inventory=json.loads((folder/'artifact_inventory.json').read_text())
 for item in inventory['files']:
  p=folder/item['path'];digest=hashlib.sha256(p.read_bytes()).hexdigest();rows.append(dict(path=str(p.relative_to(r)),matches=digest==item['sha256'],bytes=p.stat().st_size))
assert rows and all(x['matches'] for x in rows)
parent=json.loads((r/'results/matched_feedback_20261028_v2/main_repair_v1/candidate_manifest.json').read_text());geom=[x for x in parent['records'] if x['candidate_id']=='C5'];assert len(geom)==160
for item in geom:assert hashlib.sha256((r/item['candidate_path']).read_bytes()).hexdigest()==item['candidate_image_hash']
status=json.loads((r/'reports/NEXT_VALIDATION_STATUS_20261031.json').read_text());assert all(status['source_hashes_verified'].values()) and status['reward_guard_intact']
assert status['resource_accounting']['consumed']==1955 and status['resource_accounting']['reserved']==0
result=dict(status='PASS',BF16_artifact_inventory_files=len(rows),verified_files=rows,all160_geometry_controls_match_frozen_parent=True,new_gpu_allocation_seconds=279,validation_consumed_seconds=1955,validation_remaining_seconds=1645,monitor_next_job='88008',new_stage_reserved_seconds=0)
(r/'results'/name/'final_integrity.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
'''
 p=subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=code.encode(),capture_output=True,timeout=80);p.check_returncode();data=json.loads(p.stdout);(ROOT/'results'/RUN/'final_integrity.json').write_text(json.dumps(data,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in data.items() if k!='verified_files'}))
 qa=json.loads((ROOT/'reports/next_validation_browser_20261031/QA.json').read_text());cloud=json.loads((ROOT/'reports/deidentified_next_validation_20261031/PUBLICATION_VERIFIED.json').read_text(encoding='utf-8'));assert qa['status']==cloud['status']=='PASS'
 report=ROOT/'reports/NEXT_VALIDATION_DECISION_20261031.md';content=report.read_text(encoding='utf-8')
 appendix='\n## Delivery verification\n\nCPU88013, GPU88016 and one-shot finalizer88019 are COMPLETED0:0. Last live check20:44 EDT October31: no active Poisoning experiment; existing monitor88008 PENDING BeginTime, eligible22:32:35 EDT. Other account GPU jobs belong to unrelated projects and were not modified. Last successful monitor87971 finished20:32:52 EDT. Cluster cron is absent and user systemd bus unavailable; existing native Slurm successor chain remains in use.\n\nNew BF16 checkpoint/output inventories: '+str(data['BF16_artifact_inventory_files'])+' files rehashed successfully, including saved negative-gradient evidence. All160 C5 geometry-control hashes match the frozen candidate manifest. Pinned editor31519b5cb02a7fd89b906d88731cd4d6a7bbf88d; Qwen89644892e4d85e24eaac8bacfd4f463576704203. QA: reports/next_validation_browser_20261031/QA.json; bilingual desktop and mobile payload equality and32-pair image loading PASS;0 human labels entered. New phase retained about303MB of cluster evidence and about13MB of Windows metadata/review kit, plus QA screenshots; no new pretrained weights or multi-GB C: artifacts. QA run-specific scratch was verified and removed.\n\nOnly the new deidentified summary and aggregate CSV were uploaded to the existing RSI-Poisoning reports folder. Remote connector readback matched both byte hashes and folder parents. Receipt: reports/deidentified_next_validation_20261031/PUBLICATION_VERIFIED.json. The older scheduled report-sync package is stage-specific and still describes its prior completed stage; it was not repurposed or overwritten.\n'
 if '## Delivery verification' not in content:report.write_text(content+appendix,encoding='utf-8')
 upload={'reports/NEXT_VALIDATION_DECISION_20261031.md':base64.b64encode(report.read_bytes()).decode()}
 for rel in ['reports/next_validation_browser_20261031/QA.json','reports/deidentified_next_validation_20261031/PUBLICATION_VERIFIED.json']:
  upload[rel]=base64.b64encode((ROOT/rel).read_bytes()).decode()
 transfer='import base64\nfrom pathlib import Path\nr=Path("/storage/home/ydk297/projects/poisoning")\nfor n,v in '+repr(upload)+'.items():\n p=r/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(v))\n'
 subprocess.run(['ssh','-o','BatchMode=yes','utc-login','python3','-'],input=transfer.encode(),check=True,timeout=80)
 (ROOT/'results'/RUN/'delivery_complete.json').write_text(json.dumps(dict(status='COMPLETED_WITH_LIMITATIONS',scientific_gate='NO-GO',cluster_integrity=data['status'],browser_QA=qa['status'],deidentified_cloud_readback=cloud['status'],GPU_seconds=1955,remaining_GPU_seconds=1645),indent=2),encoding='utf-8')
if __name__=='__main__':main()
