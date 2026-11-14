"""Bind already cloud-verified index bytes to the existing desktop publisher."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import sync_rsi_poisoning_reports as publisher

def main():
 root=publisher.validate_root();out=root/'results/overnight_benign_20261102_v2';index=(root/'reports/LATEST_REPORT_INDEX.md').read_bytes();receipt=json.loads((out/'cloud_index_verification.json').read_text(encoding='utf8'))
 assert receipt['status']=='CLOUD_VERIFIED' and receipt['index_sha256']==publisher.sha(index)
 assert receipt['index_id']=='10YL4tTFprNpd2p_X3XBn2o-02c6h_sQI' and receipt['index_parent_id']=='1MmOBPXEkKGdU03FcvOtiZuHcVYLfO8j9'
 with publisher.publication_lock(root/publisher.CONTROL):
  old_sha=publisher.sha((root/publisher.CONTROL/'APPROVED_REPORTS.json').read_bytes());blobs,base=publisher.validated_package(root)
  publisher.atomic_write(root/publisher.CONTROL/'VERIFIED_EXTERNAL_INDEX.md',index)
  publisher.atomic_write(root/publisher.CONTROL/'VERIFIED_EXTERNAL_INDEX.json',publisher.encoded(receipt))
  effective,approval=publisher.with_verified_external_index(root,blobs,base)
  assert effective['LATEST_REPORT_INDEX.md']==index and approval['version_id']!=base['version_id']
  assert publisher.sha((root/publisher.CONTROL/'APPROVED_REPORTS.json').read_bytes())==old_sha
 result=publisher.run()
 actual=publisher.read_json(root/publisher.CONTROL/'LAST_ATTEMPT.json')
 publisher.atomic_write(out/'desktop_index_activation.json',publisher.encoded(dict(PROJECT_ID=publisher.PROJECT_ID,exit_code=result,base_approved_package_unchanged=True,base_version=base['version_id'],effective_version=approval['version_id'],index_sha256=receipt['index_sha256'],desktop_result=actual,remote_verification_receipt='cloud_index_verification.json',scope='Local mount copy does not independently verify cloud; connector readback is separate')))
 return result
if __name__=='__main__':raise SystemExit(main())
