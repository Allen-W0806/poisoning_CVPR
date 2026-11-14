import json
from pathlib import Path
import time
import uuid
from collections import Counter
from datetime import datetime, timezone, timedelta
import yaml
from ..utils.logging import write_json
from .common import ACTIVE, TERMINAL, FileLock, append_json, command, now, read_json, slurm_snapshot, json_safe
from .diagnostics import inspect_run, issue, verify_artifacts
from .integrity_v6 import check_integrity
from .recovery import Recovery
from .phase3 import phase3_checks
from .phase4 import phase4_checks


def load_config(root):
    config = yaml.safe_load((Path(root)/'configs/monitoring.yaml').read_text())['monitoring']
    if config['interval_hours'] != 2 or config['max_concurrent_monitor_runs'] != 1:
        raise ValueError('Two-hour, single-instance monitoring is required')
    if any(config[k] for k in ['allow_scientific_objective_changes', 'allow_reward_definition_changes', 'allow_dataset_changes', 'allow_test_protocol_changes', 'allow_unbounded_gpu_jobs']):
        raise ValueError('Scientific and resource invariants cannot be relaxed')
    if not config['preserve_failed_runs'] or not config['preserve_checkpoints']:
        raise ValueError('Evidence retention is mandatory')
    if config['max_auto_retries_per_failure'] > 2 or config['total_phase_gpu_seconds'] != 7200:
        raise ValueError('Retry and GPU budget ceilings cannot be expanded')
    return config


def public_run(run):
    keys = ['samples', 'accuracy_percent', 'official_vqa_accuracy', 'adapter_max_parameter_change',
        'frozen_weights_unchanged', 'resume_next_update_equal', 'optimizer_resume_equal', 'scheduler_resume_equal',
        'generator_training_kind', 'generator_rl_status', 'fixed_rl_validation', 'rsi_selected_validation',
        'mean_step_seconds', 'peak_gpu_memory_bytes', 'generated_count', 'nonidentity_human_verified']
    metrics = {k:run['metrics'][k] for k in keys if k in run['metrics']}
    component = 'generator reward RL (DDPO-SF)' if metrics.get('generator_training_kind') == 'reward_based_generator_RL' else 'frozen generator reference' if metrics.get('generator_training_kind')=='frozen_reference' else 'supervised generator LoRA' if 'generator_training_kind' in metrics else 'controller RL with frozen generator' if 'fixed_rl_validation' in metrics else 'frozen evaluation/editor or infrastructure'
    return {k: run[k] for k in ['id', 'status', 'job_id', 'scheduler_state', 'progress', 'checkpoint_files', 'checkpoint_status']} | {'metrics': metrics, 'component': component}


def is_new_failure(run, previous, first, last_check):
    failed = run['status']=='FAIL' or run['scheduler_state'] in TERMINAL-{'COMPLETED'}
    if first or not failed:
        return False
    if previous.get(run['id'],{}).get('status') in ACTIVE:
        return True
    if run['id'] not in previous and run['scheduler_state'] in TERMINAL-{'COMPLETED'} and last_check:
        try:
            return datetime.fromisoformat(run['manifest']['created_at']) > datetime.fromisoformat(last_check)
        except (KeyError, ValueError):
            pass
    return False


def is_new_completion(run,previous,first,last_check):
    complete=run['status'] not in ACTIVE or run['scheduler_state'] in TERMINAL
    if first or not complete:
        return False
    if previous.get(run['id'],{}).get('status') in ACTIVE:
        return True
    if run['id'] not in previous and last_check:
        try:
            return datetime.fromisoformat(run['manifest']['created_at']) > datetime.fromisoformat(last_check)
        except (KeyError,ValueError):
            pass
    return False


def run_monitor(root, snapshot=None, config=None, scientific_checks=True, apply=True):
    root = Path(root).resolve()
    config = config or load_config(root)
    base = root/'reports/monitoring'
    base.mkdir(parents=True, exist_ok=True)
    with FileLock(base/'monitor.lock') as acquired:
        if not acquired:
            return {'status': 'SKIPPED_LOCKED', 'timestamp': now(), 'submissions': 0}
        started, timestamp = time.perf_counter(), now()
        state_path = base/'state.json'
        state = read_json(state_path, {})
        previous = state.get('observations', {})
        first = not state.get('initialized')
        findings, runs = [], []
        registry = root/'experiments/registry.jsonl'
        rows = []
        for number, line in enumerate(registry.read_text().splitlines() if registry.exists() else []):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                findings.append(issue('invalid_registry_line', severity='error', line=number+1))
        counts = Counter(r.get('experiment_id') for r in rows)
        for run_id, count in counts.items():
            if count>1:
                findings.append(issue('duplicate_experiment_id', run_id, 'error', count=count))
        known = set(counts)
        paths = sorted((root/'experiments').glob('EXP_*'))
        for path in paths:
            try:
                runs.append(inspect_run(path))
            except Exception as error:
                findings.append(issue('unreadable_experiment', path.name, 'error', error=str(error)))
        job_ids = [r['job_id'] for r in runs if r['job_id']]
        snapshot = snapshot if snapshot is not None else slurm_snapshot(job_ids)
        if snapshot.get('queue', {}).get('exit_code', 0) != 0 or snapshot.get('accounting', {}).get('exit_code', 0) != 0:
            findings.append(issue('scheduler_unavailable', severity='error'))
        runs = []
        for path in paths:
            try:
                runs.append(inspect_run(path, snapshot.get('jobs', {})))
            except Exception:
                continue
        passing = {}
        for run in runs:
            kind = run['id'].split('_phase2_')[-1].split('_phase3_')[-1]
            if run['status'] == 'PASS':
                passing[kind] = run['id']
        artifacts, cache = {}, state.setdefault('artifact_cache', {})
        for run in runs:
            if run['id'] not in known:
                findings.append(issue('missing_registry_entry', run['id']))
            for finding in run['findings']:
                kind = run['id'].split('_phase2_')[-1].split('_phase3_')[-1]
                if run['status'] == 'FAIL' and passing.get(kind, '') > run['id']:
                    finding.update(resolved_historically=True, resolution_witness=passing[kind])
                findings.append(finding)
            try:
                artifacts[run['id']] = verify_artifacts(root, run, cache)
                findings.extend(artifacts[run['id']]['findings'])
            except Exception as error:
                findings.append(issue('artifact_inspection_failed', run['id'], 'error', error=str(error)))
        integrity = check_integrity(root, runs, state, config.get('minimum_free_disk_bytes', 2**31)) if scientific_checks else {'findings': [], 'fixture_only': True}
        findings.extend(integrity['findings'])
        phase3=phase3_checks(root,runs,state,snapshot,protect=scientific_checks)
        findings.extend(phase3['findings'])
        phase4=phase4_checks(root,runs,state,snapshot,protect=scientific_checks)
        findings.extend(phase4['findings'])
        active = [r for r in runs if r['status'] in ACTIVE and r['scheduler_state'] not in TERMINAL]
        pending=state.setdefault('pending_failure_jobs',{})
        retryable={'transient_io','node_failure','preempted'}
        for run in runs:
            if run['scheduler_state'] in TERMINAL and not any(f['code'] in retryable for f in run['findings']):
                pending.pop(str(run.get('job_id') or run['id']),None)
        new_failures = [r for r in runs if is_new_failure(r,previous,first,state.get('last_check'))
            or (str(r.get('job_id') or r['id']) in pending and (r['status']=='FAIL' or r['scheduler_state'] in TERMINAL-{'COMPLETED'}))]
        for run in new_failures:
            key=str(run.get('job_id') or run['id'])
            if any(f['code'] in retryable for f in run['findings']):
                pending.setdefault(key,{'first_seen':timestamp,'reason':'Awaiting reliable scheduler/integrity evidence before an attempt'})
        for run in runs:
            if run['status']=='PASS':
                pending.pop(str(run.get('job_id') or run['id']),None)
        # A terminal newly observed campaign counts as activity; old failures never do.
        campaign_ids = {r['id'] for r in active + new_failures}
        permit_compute = bool(campaign_ids) and not any(f['severity']=='error' and not f.get('resolved_historically')
            and (f.get('experiment_id') is None or f.get('experiment_id') in campaign_ids)
            and f['code'] not in {'transient_io','node_failure','preempted'} for f in findings)
        # Phase6 explicitly revokes implicit scientific GPU retries, including old-campaign recovery.
        if (root/'configs/phase6.json').is_file():
            permit_compute=False
        repairs = []
        recovery = Recovery(root, config, state)
        indexed = {r['id']:r for r in runs}
        handled_jobs = set()
        if apply:
            for finding in findings:
                if finding.get('resolved_historically'):
                    continue
                run = indexed.get(finding.get('experiment_id'))
                transient = finding['code'] in {'transient_io','node_failure','preempted'}
                if transient:
                    if not run or run not in new_failures or run.get('job_id') in handled_jobs:
                        continue
                    handled_jobs.add(run.get('job_id'))
                try:
                    repair = recovery.apply(finding, run, permit_compute)
                    if repair:
                        repairs.append(repair)
                        if transient and (repair['action']=='restart_attempt' or (repair['action']=='BLOCKED' and permit_compute)):
                            pending.pop(str(run.get('job_id') or run['id']),None)
                except Exception as error:
                    repairs.append(recovery.record({'action':'BLOCKED','experiment_id':finding.get('experiment_id'),'reason':str(error)}))
        completed = [r['id'] for r in runs if is_new_completion(r,previous,first,state.get('last_check'))]
        schedule = read_json(base/'scheduler.json', {})
        phase_jobs_path=root/'reports/phase2_jobs.jsonl'
        phase_jobs=[json.loads(s) for s in phase_jobs_path.read_text().splitlines()] if phase_jobs_path.exists() else []
        charged=0
        for job in phase_jobs:
            row=snapshot.get('jobs',{}).get(str(job['job_id']))
            charged+=int(row['ElapsedRaw']) if row and row['State'] in TERMINAL else job['reserved_seconds']
        report = {'timestamp':timestamp, 'status':'COMPLETED_WITH_FINDINGS' if findings else 'PASS',
            'initial_inventory':first, 'fixture_only':not scientific_checks, 'active_jobs': [public_run(r) for r in active],
            'completed_since_previous':completed, 'tracked_experiments':len(runs), 'runs':[public_run(r) for r in runs],
            'scheduler':snapshot, 'integrity':integrity, 'artifacts':artifacts, 'detected_problems':findings,
            'recovery_actions':repairs,
            'automatic_fixes':[r for r in repairs if r['action'] in {'ensure_active_output_directory','append_missing_registry_entry','reconcile_active_terminal_status'}],
            'new_experiment_ids':[i for r in repairs for i in r.get('new_experiment_ids',[])],
            'next_scheduled_check':schedule.get('next_due_utc'), 'next_job_id':schedule.get('next_job_id'),
            'elapsed_seconds':time.perf_counter()-started, 'idle_no_research_submission':not bool(active or new_failures),
            'phase2_charged_gpu_seconds':charged,'phase2_maximum_gpu_seconds':7200,
            'pending_failure_jobs':dict(pending),
            'phase3':phase3,
            'phase4':phase4,
            'remaining_blockers':([] if phase3['generator_rl_implementation_ready'] else ['Batch10 generator RL strict continuation not fully validated' if phase3['one_update_witnesses'] else 'Generator reward-trajectory optimizer not yet fully validated'])+['Changed natural-image semantics await independent review','No scientific RSI advantage established'],
            'resource_scope':'CPU-only monitor; no idle model loading/tensor replay; separate Phase2/Phase3 caps7200s each, Phase4 cap14400s including diagnostics2700s; no automatic Phase4 science submissions',
            'follow_up':'Review metrics/sampling/normalization/gradient evidence before proposing one-variable equal-budget studies; no automatic reward tuning'}
        check_id = 'CHECK_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f') + '_' + uuid.uuid4().hex[:6]
        json_path, md_path = base/(check_id+'.json'), base/(check_id+'.md')
        report = json_safe(report)
        write_json(json_path, report)
        lines = [f'# {check_id}', '', f"UTC: {timestamp}; tracked {len(runs)} experiments; active {len(active)}; first inventory={first}.",
            f"Next eligibility: {report['next_scheduled_check']}; job {report['next_job_id']} (queue delay possible).", '',
            f"Completed since previous: {completed}", f"New experiment IDs: {report['new_experiment_ids']}",
            f"CPU check duration: {report['elapsed_seconds']:.3f}s. Idle={report['idle_no_research_submission']}.", '',
            '## Problems and diagnoses', '']
        lines += [f"- {f['code']} — {f.get('experiment_id') or 'project'}; severity={f['severity']}; historical resolution={f.get('resolved_historically',False)}" for f in findings] or ['None.']
        lines += ['', '## Repairs', '', '```json', json.dumps(repairs, indent=2), '```', '',
            '## Training, reward and resource evidence', '', '```json', json.dumps([public_run(r) for r in runs if r['progress'] or 'adapter_max_parameter_change' in r['metrics'] or 'fixed_rl_validation' in r['metrics']], indent=2), '```', '',
            '## Boundaries and follow-up', '', report['resource_scope'], report['follow_up'], *report['remaining_blockers']]
        md_path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
        state.update(initialized=True, last_check=timestamp, observations={r['id']:{'status':r['status'],'scheduler_state':r['scheduler_state']} for r in runs})
        write_json(state_path, state)
        status = {k:report[k] for k in ['timestamp','status','tracked_experiments','completed_since_previous','next_scheduled_check','next_job_id','remaining_blockers','idle_no_research_submission']}
        status.update(latest_json=str(json_path), latest_report=str(md_path), active_experiment_ids=[r['id'] for r in active],
            problems=len(findings), automatic_fixes=len(report['automatic_fixes']),
            blocked_actions=sum(r['action']=='BLOCKED' for r in repairs),
            automatic_recovery_attempts=sum(r['action']=='restart_attempt' for r in repairs),
            new_experiment_ids=report['new_experiment_ids'])
        write_json(root/'reports/MONITORING_STATUS.json', status)
        with (root/'reports/MONITORING_LOG.md').open('a', encoding='utf-8') as out:
            out.write(f"\n- {timestamp}: [{check_id}](monitoring/{check_id}.md); tracked={len(runs)}, active={len(active)}, problems={len(findings)}, repairs={len(repairs)}; next={report['next_scheduled_check']}\n")
        if (root/'configs/phase6.json').is_file():
            from ..progression.monitor import cycle
            status['phase6']=cycle(root,snapshot)
            write_json(root/'reports/MONITORING_STATUS.json',status)
        return status
