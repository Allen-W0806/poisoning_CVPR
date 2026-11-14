"""Propose reusable workflow changes only when a recorded diagnosis supports them."""
from .harness_spec import HarnessSpec

def propose(parent,diagnosis,parent_version):
    if diagnosis['diagnosis_status']!='SUPPORTED_SCOPED' or 'dataset_corruption_or_duplication' not in diagnosis['supported']:
        return None
    if 'content_hash' in parent.checks:return None
    checks=tuple(dict.fromkeys(parent.checks+('content_hash',)))
    candidate=HarnessSpec('verify_content_before_validity',checks)
    return {'parent_version':parent_version,'candidate':candidate.to_dict(),
        'change':'Add manifest-content verification for every independent image',
        'supporting_evidence':diagnosis,'expected_effect':'Reject changed file bytes even when geometry matches',
        'affected_modules':['workflow.checks'],
        'regression_tests':['geometry failures remain invalid','unmodified held-out constructed images remain valid'],
        'category':'evidence_acquisition_and_validation_orchestration'}
