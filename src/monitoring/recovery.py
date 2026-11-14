"""Small allowlist of automatic repairs; never mutate a finalized run."""
import json
import math
from pathlib import Path
import re
import uuid
from .common import ACTIVE, TERMINAL, append_json, command, now, read_json, safe_path
from ..utils.logging import Experiment
from ..utils.logging import write_json


class Recovery:
    def __init__(self, root, config, state, submit=None):
        self.root, self.config, self.state = Path(root).resolve(), config, state
        self.submit = submit or command
        self.state.setdefault('retry_counts', {})
        self.state.setdefault('lineage_by_job', {})
        self.state.setdefault('lineage_attempts', {})

    def record(self, action):
        event = {'repair_id': 'REPAIR_' + uuid.uuid4().hex, 'timestamp': now(), **action}
        append_json(self.root/'reports/monitoring/repairs.jsonl', event)
        return event

    def apply(self, finding, run=None, permit_compute=False):
        code = finding['code']
        run_id = finding.get('experiment_id')
        path = safe_path(self.root, 'experiments/' + run_id) if run_id else None
        current = read_json(path/'status.json', {}).get('status') if path else None
        if code == 'missing_output_directory' and current in ACTIVE:
            directory = finding['directory']
            if directory not in {'images', 'checkpoints'}:
                raise ValueError('Output directory is not allowlisted')
            target = safe_path(path, directory)
            existed = target.exists()
            target.mkdir(exist_ok=True)
            return self.record({'action': 'ensure_active_output_directory', 'experiment_id': run_id,
                'before': {'exists': existed}, 'after': {'exists': True}, 'path': str(target), 'new_experiment_ids': []})
        if code == 'missing_registry_entry' and path and (path/'manifest.json').is_file():
            manifest = read_json(path/'manifest.json')
            if manifest.get('experiment_id') != run_id:
                return self.record({'action': 'BLOCKED', 'reason': 'Manifest identity mismatch', 'experiment_id': run_id})
            registry = self.root/'experiments/registry.jsonl'
            existing = [json.loads(s) for s in registry.read_text().splitlines()] if registry.exists() else []
            if not any(r['experiment_id'] == run_id for r in existing):
                append_json(registry, {'experiment_id': run_id, 'path': str(path), 'created_at': manifest.get('created_at'),
                    'repaired_at': now(), 'repair_source': 'verified existing immutable manifest'})
                return self.record({'action': 'append_missing_registry_entry', 'experiment_id': run_id,
                    'before': {'registered': False}, 'after': {'registered': True}, 'measurements_unchanged': True})
            return None
        if code == 'stale_status' and current in ACTIVE and finding.get('scheduler_state') in TERMINAL:
            if finding['scheduler_state'] == 'COMPLETED' and finding.get('exit_code') == '0:0':
                target = 'COMPLETED_UNVERIFIED'
                for line in (path/'run.log').read_text().splitlines() if (path/'run.log').exists() else []:
                    event = json.loads(line)
                    if event.get('event') == 'finished' and event.get('value', {}).get('status') in {'PASS', 'FAIL', 'BLOCKED'}:
                        target = event['value']['status']
            else:
                target = 'FAIL'
            attached = Experiment.attach_active(self.root, path)
            attached.finish(target, 'Monitor reconciled an unfinalized run from terminal scheduler evidence. COMPLETED_UNVERIFIED is not scientific PASS.',
                scheduler_state=finding['scheduler_state'], scheduler_exit=finding.get('exit_code'))
            append_json(self.root/'experiments/state_observations.jsonl', {'timestamp': now(), 'experiment_id': run_id,
                'before': current, 'after': target, 'scheduler_state': finding['scheduler_state']})
            return self.record({'action': 'reconcile_active_terminal_status', 'experiment_id': run_id,
                'before': {'status': current}, 'after': {'status': target}, 'measurements_unchanged': True})
        if code in {'transient_io', 'node_failure', 'preempted'} and run:
            return self.retry(run, finding, permit_compute)
        if code in {'invalid_checkpoint_path', 'duplicate_experiment_id', 'nonfinite_metrics', 'oom', 'cuda', 'dependency', 'timeout', 'no_parameter_updates', 'diffusion_probability', 'conditioning_mismatch', 'invalid_generator_gradient', 'restart_reproducibility'}:
            return self.record({'action': 'BLOCKED', 'experiment_id': run_id, 'reason': code,
                'follow_up': 'Preserve evidence; require validated compatible repair/resume and regression proof. No fabricated checkpoint or scientific changes.'})
        return None

    def retry(self, run, finding, permit_compute):
        job_id = str(run.get('job_id') or '')
        lineage = self.state['lineage_by_job'].get(job_id, job_id or run['id'])
        signature = lineage + ':' + finding['code']
        counts = self.state['retry_counts']
        count = counts.get(signature, 0)
        lineage_count = self.state['lineage_attempts'].get(lineage,0)
        if count >= min(2, self.config['max_auto_retries_per_failure']) or lineage_count >= 2:
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'signature': signature, 'reason': 'retry_limit_reached'})
        if not permit_compute:
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'signature': signature, 'reason': 'historical, integrity gate, or no active campaign'})
        if self.state.get('submission_uncertain'):
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'reason': 'Previous submission outcome uncertain; reconcile scheduler first'})
        if run['manifest'].get('source_tree_clean_at_creation') is not True:
            return self.record({'action':'BLOCKED','experiment_id':run['id'],'reason':'Original clean-source provenance is unavailable'})
        current_commit = command(['git', 'rev-parse', 'HEAD'], cwd=self.root)['stdout'].strip()
        if current_commit != run['manifest'].get('git_commit'):
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'reason': 'Source differs from original; no approved tested repair rule'})
        dirty = command(['git','status','--porcelain','--untracked-files=normal'],cwd=self.root)
        if dirty['exit_code'] != 0 or dirty['stdout'].strip():
            return self.record({'action':'BLOCKED','experiment_id':run['id'],'reason':'Source tree is dirty or cannot be verified'})
        phase3 = '_phase3_' in run['id']
        ledger = self.root/('reports/phase3_jobs.jsonl' if phase3 else 'reports/phase2_jobs.jsonl')
        records = [json.loads(s) for s in ledger.read_text().splitlines()] if ledger.exists() else []
        original = next((r for r in records if str(r['job_id']) == job_id), None)
        allowed = {'one', 'smoke', 'semantics'} if phase3 else {'editor', 'training'}
        if not original or original['task'] not in allowed:
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'reason': 'No supported identical-config editor/training restart; held-out never retried'})
        minutes = original['reserved_seconds'] // 60
        if not 1 <= minutes <= self.config['max_recovery_job_minutes']:
            return self.record({'action': 'BLOCKED', 'experiment_id': run['id'], 'reason': 'Original reservation exceeds recovery ceiling'})
        # The existing wrapper checks the cumulative GPU ledger and all active jobs.
        counts[signature] = count + 1  # Count unsuccessful/uncertain submissions too.
        self.state['lineage_attempts'][lineage]=lineage_count+1
        write_json(self.root/'reports/monitoring/state.json', self.state)
        result = self.submit([str(self.root/'.venv/bin/python'), str(self.root/('scripts/submit_phase3.py' if phase3 else 'scripts/submit_phase2.py')),
                              '--task', original['task'], '--minutes', str(minutes)], cwd=self.root)
        new_ids = []
        if result['exit_code'] == 0:
            try:
                submission = json.loads(result['stdout'].splitlines()[-1])
                self.state['lineage_by_job'][str(submission['job_id'])] = lineage
                new_ids.append(submission['parent_run'])
            except (KeyError, json.JSONDecodeError, IndexError):
                self.state['submission_uncertain'] = True
        elif 'budget' not in result['stderr'].lower() and not any(message in result['stderr'] for message in [
                'One GPU at a time', 'Previous Phase3 allocation active/unconfirmed',
                'Another user GPU allocation is active/pending']):
            self.state['submission_uncertain'] = True
        write_json(self.root/'reports/monitoring/state.json', self.state)
        return self.record({'action': 'restart_attempt', 'experiment_id': run['id'], 'signature': signature,
            'attempt': count+1, 'before': {'task': original['task'], 'minutes': minutes},
            'after': {'task': original['task'], 'minutes': minutes, 'initialization': 'fresh; NOT checkpoint resume'},
            'new_experiment_ids': new_ids, 'submission': result})
