"""Pure budget/concurrency/retry rules used by the CPU coordinator."""
import math

TERMINAL={'COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','PREEMPTED','DEADLINE','REVOKED'}
FATAL_LOGS=('out of memory','nonfinite','nan loss','nan gradient','AssertionError','checkpoint save failed','PytorchStreamWriter','No space left on device')


def accounting(records,states):
    actual=committed=0;active=[]
    for record in records:
        if record.get('submission_failed_no_job'):continue
        if not record.get('job_id'):raise RuntimeError('Uncertain submission must be reconciled; never resubmit blindly')
        status=states.get(record['job_id'])
        if status is None:raise RuntimeError('Missing Slurm accounting; keep reservation, block submission')
        elapsed=int(status['elapsed_seconds'])
        if elapsed<0:raise ValueError('Invalid elapsed time')
        actual+=elapsed
        if status['state'] in TERMINAL:committed+=elapsed
        else:committed+=record['reserved_seconds'];active.append(record['job_id'])
    return {'actual_GPU_seconds':actual,'committed_GPU_seconds':committed,'active_GPU_jobs':active}


def can_submit(records,states,stage,attempt,reservation,cap=36000):
    value=accounting(records,states)
    if value['active_GPU_jobs']:raise RuntimeError('One-GPU-at-a-time guard')
    if any(r['stage']==stage and r['attempt']==attempt for r in records):raise RuntimeError('Duplicate stage/config attempt')
    if not 0<reservation<=cap or value['committed_GPU_seconds']+reservation>cap:raise RuntimeError('Night GPU-hour cap would be exceeded')
    return value


def safe_retry(record,status,log,checkpoint_valid,updates):
    if record['attempt']>=1:return False,'retry_limit'
    lower=log.lower()
    if status in {'OUT_OF_MEMORY','TIMEOUT'} or any(term.lower() in lower for term in FATAL_LOGS):return False,'numerical_memory_checkpoint_or_timeout_failure'
    recoverable=status in {'NODE_FAIL','BOOT_FAIL','PREEMPTED'} or (status=='FAILED' and any(term in lower for term in ('stale file handle','input/output error','temporarily unavailable')))
    if not recoverable:return False,'not_predeclared_transient_failure'
    if updates>0 and not checkpoint_valid:return False,'no_exact_completed_checkpoint'
    return True,'one_safe_same_config_resume' if updates else 'one_safe_preupdate_restart'


def finite_update(row):
    keys=['loss','raw_gradient_norm','max_parameter_change']
    return all(k in row and math.isfinite(float(row[k])) for k in keys) and row['max_parameter_change']>0
