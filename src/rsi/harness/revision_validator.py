"""The candidate can revise executable workflow configuration, not the evaluator."""
from .harness_spec import HarnessSpec

def validate_revision(proposal,expected_parent,parent_program):
    required={'parent_version','candidate','change','supporting_evidence','expected_effect','affected_modules','regression_tests','category'}
    if set(proposal)!=required or proposal['parent_version']!=expected_parent:raise ValueError('Malformed or stale harness revision')
    if not proposal['supporting_evidence'] or not proposal['regression_tests']:raise ValueError('Evidence and regression checks required')
    allowed={'evidence_acquisition_and_validation_orchestration':['workflow.checks'],
       'semantic_verification_workflow':['workflow.checks'],
       'resource_management':['workflow.max_diagnostics'],
       'experience_memory':['workflow.memory_mode'],
       'benign_learning_curriculum':['workflow.curriculum']}
    if proposal['category'] not in allowed or proposal['affected_modules']!=allowed[proposal['category']]:
        raise ValueError('Evaluator, labels, targets and objectives are immutable')
    candidate=HarnessSpec.from_dict(proposal['candidate'])
    permitted={'workflow.checks':'checks','workflow.max_diagnostics':'max_diagnostics',
        'workflow.memory_mode':'memory_mode','workflow.curriculum':'curriculum'}
    allowed_fields={permitted[m] for m in proposal['affected_modules']}
    changed={k for k,v in candidate.to_dict().items() if v!=parent_program.to_dict()[k] and k!='name'}
    if not changed or changed-allowed_fields:raise ValueError('Actual executable diff exceeds declared revision scope or is empty')
    return candidate
