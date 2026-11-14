"""Persistent claim-specific dashboard extension; never schedules or trains."""
import html,json,re,uuid,os
from pathlib import Path

def dashboard(root):
 root=Path(root);report=root/'reports/SCIENTIFIC_GATE_STATUS.json'
 if not report.exists():return
 status=json.loads(report.read_text(encoding='utf8'));p=root/'dashboard/dashboard_data.json';data=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
 data['execution_remediation']=status;encoded=json.dumps(data,ensure_ascii=False).replace('</','<\\/')
 rows=''.join('<tr><td>'+html.escape(k)+'</td><td>'+html.escape(v['status'])+'</td><td>'+html.escape(v.get('scope',''))+'</td></tr>' for k,v in status['statuses'].items())
 block='<section id="execution-remediation" style="padding:20px;background:#e9f5ee;color:#183529"><h2>RSI-Poisoning · Execution remediation / 执行修复</h2><p>'+html.escape(status['updated_UTC'])+' · Main-paper NO-GO does not prohibit independent engineering work.</p><table><thead><tr><th>Dimension</th><th>Status</th><th>Scope</th></tr></thead><tbody>'+rows+'</tbody></table><p><a href="../reports/EXECUTION_REMEDIATION_LATEST.md">Executed fixes and results</a> · <a href="../reports/SCIENTIFIC_GATE_STATUS.md">Claim-specific gates</a> · <a href="../reports/NEXT_VALIDATION_DECISION.md">Next validation</a> · <a href="../reports/MAIN_EXPERIMENT_GAP_MATRIX.md">Main experiment gaps</a></p></section>'
 for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html','index_zh.html','index_en.html','RSI_Experimental_Progress_Latest.html']:
  path=root/'dashboard'/n
  if not path.exists():continue
  s=path.read_text(encoding='utf8');s=re.sub(r'<!--EXECUTION_REMEDIATION_START-->.*?<!--EXECUTION_REMEDIATION_END-->','',s,flags=re.S)
  s=re.sub(r'(<body[^>]*>)',lambda m:m[1]+'<!--EXECUTION_REMEDIATION_START-->'+block+'<!--EXECUTION_REMEDIATION_END-->',s,count=1)
  s=re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)',lambda m:m[1]+encoded+m[2],s,flags=re.S);path.write_text(s,encoding='utf8')
 blob=json.dumps(data,indent=2,ensure_ascii=False)
 for n in ['state.json','dashboard_data.json']:
  path=root/'dashboard'/n;q=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial');q.write_text(blob,encoding='utf8');os.replace(q,path)

if __name__=='__main__':dashboard(Path(__file__).resolve().parents[1])
