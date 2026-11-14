import json,re,unittest
from src.utils.scratch import scratch_directory
from scripts.render_execution_remediation_dashboard_20261102 import dashboard
from scripts.overnight_supervisor_20261101 import component_statuses
class StatusDashboard(unittest.TestCase):
 def test_independent_gates_and_embedded_data(self):
  with scratch_directory() as p:
   (p/'reports').mkdir();(p/'dashboard').mkdir();status={'updated_UTC':'unit-fixture','statuses':{'ENGINEERING_STATUS':{'status':'PASS','scope':'CPU only'},'MAIN_PAPER_EVIDENCE_STATUS':{'status':'NO-GO','scope':'paper claims'}}};(p/'reports/SCIENTIFIC_GATE_STATUS.json').write_text(json.dumps(status));(p/'dashboard/dashboard_data.json').write_text('{"historical_evidence_retained":true}')
   for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html']:(p/'dashboard'/n).write_text('<body><script id="boot-data" type="application/json">{}</script></body>')
   dashboard(p);dashboard(p);data=json.loads((p/'dashboard/dashboard_data.json').read_text());self.assertTrue(data['historical_evidence_retained']);self.assertEqual((p/'dashboard/dashboard_data.json').read_bytes(),(p/'dashboard/state.json').read_bytes())
   self.assertEqual(component_statuses(p),status['statuses'])
   for n in ['rsi_poisoning_zh.html','rsi_poisoning_en.html']:
    s=(p/'dashboard'/n).read_text();self.assertEqual(s.count('id="execution-remediation"'),1);embedded=re.search(r'<script id="boot-data" type="application/json">(.*?)</script>',s,re.S).group(1);self.assertEqual(json.loads(embedded),data)
 def test_legacy_monitor_without_new_status_file(self):
  with scratch_directory() as p:self.assertEqual(component_statuses(p),{})
if __name__=='__main__':unittest.main()
