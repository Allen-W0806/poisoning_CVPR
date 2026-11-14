"""Evidence-bounded diagnosis. Scientific disappointments never trigger tuning."""
import json
import math
from pathlib import Path
import re
import time
import zipfile
from PIL import Image
from .common import ACTIVE, TERMINAL, digest, read_json, safe_path

PATTERNS = [
    ('restart_reproducibility', r'AssertionError:.*adapter_max_difference'),
    ('diffusion_probability', r'Nonfinite transition likelihood|Behavior likelihood replay mismatch|Zero/nonfinite DDIM variance|Stale or incompatible behavior trajectory|Incomplete/mismatched diffusion timestep'),
    ('conditioning_mismatch', r'Conditioning association mismatch|Source conditioning differs from official path'),
    ('invalid_generator_gradient', r'Missing/nonfinite actual generator gradients|Zero/nonfinite generator gradient'),
    ('oom', r'CUDA out of memory|OutOfMemoryError|oom-kill|out.of.memory'),
    ('dependency', r'ModuleNotFoundError|ImportError|No module named'),
    ('missing_file', r'FileNotFoundError|No such file or directory'),
    ('cuda', r'CUDA error|device-side assert|CUBLAS_STATUS|illegal memory access'),
    ('transient_io', r'Stale file handle|Input/output error|Temporary failure in name resolution|Connection reset by peer'),
    ('timeout', r'TIME LIMIT|time.limit|DUE TO TIME LIMIT'),
]


def issue(code, run_id=None, severity='warning', **details):
    return {'code': code, 'experiment_id': run_id, 'severity': severity, **details}


def nonfinite(value, prefix=''):
    found = []
    if isinstance(value, float) and not math.isfinite(value):
        found.append(prefix)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(nonfinite(item, f'{prefix}.{key}'))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(nonfinite(item, f'{prefix}[{index}]'))
    return found


def tail(path, limit=50000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(max(0, Path(path).stat().st_size - limit))
            return stream.read().decode('utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def inspect_run(path, scheduler=None):
    run_id = path.name
    findings = []
    manifest = read_json(path / 'manifest.json', {})
    status = read_json(path / 'status.json', {}).get('status', 'UNKNOWN')
    metrics = read_json(path / 'metrics.json', {})
    submission = read_json(path / 'job_submission.json', {})
    # A submission wrapper may itself run inside a CPU monitor allocation.
    job_id = str(submission.get('job_id') or manifest.get('slurm_job_id') or '')
    observed = (scheduler or {}).get(job_id, {})
    scheduler_state = observed.get('State')
    for filename in ['manifest.json', 'status.json', 'config.yaml', 'metrics.json', 'environment.txt', 'git_commit.txt', 'run.log']:
        if not (path / filename).is_file():
            findings.append(issue('missing_metadata', run_id, filename=filename))
    if status in ACTIVE:
        for name in ['images', 'checkpoints']:
            if not (path / name).is_dir():
                findings.append(issue('missing_output_directory', run_id, directory=name))
        if scheduler_state in TERMINAL:
            findings.append(issue('stale_status', run_id, scheduler_state=scheduler_state, exit_code=observed.get('ExitCode'), job_id=job_id))
        elif scheduler_state == 'RUNNING':
            activity=[path/'run.log',path/'metrics.json']
            if submission.get('job_id'):
                for child in path.parent.glob('EXP_*'):
                    if read_json(child/'manifest.json',{}).get('parent_experiment')==run_id:
                        activity.extend([child/'run.log',child/'metrics.json'])
            progress_time = max((p.stat().st_mtime for p in activity if p.exists()), default=time.time())
            if time.time() - progress_time > 1800:
                findings.append(issue('stale_progress', run_id, seconds_since_event=time.time()-progress_time))
    bad = nonfinite(metrics)
    if bad:
        findings.append(issue('nonfinite_metrics', run_id, 'error', fields=bad))
    log = tail(path / 'run.log')
    log += '\n'.join(tail(p) for p in path.glob('slurm-*.err'))
    # Child logs often live in the parent Slurm run.
    parent = manifest.get('parent_experiment')
    if parent and re.fullmatch(r'EXP_[A-Za-z0-9_]+', parent):
        log += '\n'.join(tail(p) for p in (path.parent / parent).glob('slurm-*.err'))
    failed = status == 'FAIL' or scheduler_state in TERMINAL - {'COMPLETED'}
    if failed:
        category = next((name for name, pattern in PATTERNS if re.search(pattern, log, re.I)), None)
        category = category or {'OUT_OF_MEMORY': 'oom', 'TIMEOUT': 'timeout', 'NODE_FAIL': 'node_failure', 'PREEMPTED': 'preempted'}.get(scheduler_state, 'unexpected_termination')
        findings.append(issue(category, run_id, 'error', job_id=job_id, signature=category, log_excerpt=log[-1600:]))
    if status == 'BLOCKED':
        findings.append(issue('declared_blocker', run_id, reason=tail(path / 'summary.md', 3000)))
    progress = []
    for line in tail(path / 'run.log', 100000).splitlines():
        try:
            event = json.loads(line)
            value = event.get('value', {})
            if isinstance(value, dict) and any(k in value for k in ['step', 'loss', 'gradient_norm', 'reward_mean', 'trainable_max_change']):
                progress.append(value)
        except json.JSONDecodeError:
            pass
    updates = metrics.get('updates', []) or metrics.get('fixed_rl_updates', []) or progress
    if isinstance(updates, list):
        for update in updates:
            if not isinstance(update, dict):
                continue
            delta = update.get('trainable_max_change', update.get('parameter_change'))
            if delta is not None and delta <= 0:
                findings.append(issue('no_parameter_updates', run_id, 'error', step=update.get('step')))
    if metrics.get('adapter_max_parameter_change') is not None and metrics['adapter_max_parameter_change'] <= 0:
        findings.append(issue('no_parameter_updates', run_id, 'error'))
    if 'generator_training_kind' in metrics and metrics['generator_training_kind'] not in {'supervised_NOT_RL','reward_based_generator_RL','frozen_reference'}:
        findings.append(issue('unrecognized_generator_objective', run_id, 'error'))
    if metrics.get('generator_training_kind')=='frozen_reference' and (metrics.get('updates') or metrics.get('frozen_weights_unchanged') is not True):
        findings.append(issue('invalid_frozen_reference',run_id,'error'))
    if metrics.get('rsi_candidate_validation') is not None and metrics['rsi_candidate_validation'] <= metrics['rsi_base_validation']:
        findings.append(issue('scientific_no_rsi_gain', run_id, 'info', action='diagnose only; preserve rejected revision and reward'))
    if metrics.get('generator_rl_status', '').startswith('BLOCKED'):
        findings.append(issue('generator_rl_unsupported', run_id, 'info'))
    checkpoint_paths=[p for p in (path/'checkpoints').rglob('*') if p.is_file()]
    checkpoint_status={'file_count':len(checkpoint_paths),'bytes':sum(p.stat().st_size for p in checkpoint_paths),
        'latest_age_seconds':max(0,time.time()-max(p.stat().st_mtime for p in checkpoint_paths)) if checkpoint_paths else None,
        'scheduler_optimizer_state':'Prior execution resume assertions and checkpoint structure; not a new tensor-state replay'}
    return {'id': run_id, 'status': status, 'job_id': job_id or None, 'scheduler_state': scheduler_state,
        'manifest': manifest, 'metrics': metrics, 'progress': progress[-6:], 'findings': findings,
        'checkpoint_status':checkpoint_status,
        'checkpoint_files': [str(p.relative_to(path)) for p in (path/'checkpoints').rglob('*') if p.is_file()]}


def verify_artifacts(root, run, cache):
    path = root / 'experiments' / run['id']
    findings, hashes, duplicates = [], {}, {}
    candidates = [p for base in [path/'images',path/'checkpoints'] for p in base.rglob('*') if p.is_file()]
    if run['status'] not in ACTIVE:
        candidates += [p for p in path.iterdir() if p.is_file()]
    for artifact in candidates:
            if artifact.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.pt', '.safetensors', '.json', '.jsonl', '.yaml', '.csv', '.log', '.txt', '.md', '.out', '.err', '.sbatch'}:
                continue
            safe_path(root, artifact)
            key = artifact.relative_to(root).as_posix()
            stat = artifact.stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            previous = cache.get(key, {})
            if previous.get('signature') == signature:
                record = dict(previous)
                if run['status'] not in ACTIVE and not record.get('immutable'):
                    record['immutable'] = True
                    cache[key] = record
            else:
                record = {'signature': signature, 'sha256': digest(artifact), 'format_valid': True}
                try:
                    if artifact.suffix.lower() in {'.png', '.jpg', '.jpeg'}:
                        with Image.open(artifact) as im:
                            record['dimensions'] = list(im.size)
                            if min(im.size) < 1:
                                raise ValueError('Invalid dimensions')
                            im.verify()
                    elif artifact.suffix == '.pt' and zipfile.is_zipfile(artifact):
                        with zipfile.ZipFile(artifact) as archive:
                            if not any(name.endswith('data.pkl') for name in archive.namelist()):
                                raise ValueError('Missing Torch state metadata')
                            if archive.testzip() is not None:
                                raise ValueError('Checkpoint CRC failed')
                    elif artifact.suffix == '.pt':
                        record['format_note'] = 'Legacy Torch serialization; hashed, not unpickled'
                    elif artifact.suffix == '.safetensors':
                        with artifact.open('rb') as handle:
                            size = int.from_bytes(handle.read(8), 'little')
                            if not 1 <= size < stat.st_size:
                                raise ValueError('Invalid safetensors header size')
                            json.loads(handle.read(size))
                except Exception as error:
                    record.update(format_valid=False, error=str(error))
                if previous.get('immutable') and previous.get('sha256') != record['sha256']:
                    findings.append(issue('finalized_artifact_changed', run['id'], 'error', path=key))
                    # Preserve the original expected hash, so later checks cannot silently accept a mutation.
                else:
                    cache[key] = {**record, 'immutable': run['status'] not in ACTIVE}
            if not record['format_valid']:
                findings.append(issue('corrupt_artifact', run['id'], 'error', path=key, error=record.get('error')))
            hashes[key] = record['sha256']
            if artifact.suffix.lower() == '.png':
                duplicates.setdefault(record['sha256'], []).append(key)
    prefix = path.relative_to(root).as_posix()+'/'
    for key, previous in cache.items():
        if key.startswith(prefix) and previous.get('immutable') and not (root/key).is_file():
            findings.append(issue('finalized_artifact_missing',run['id'],'error',path=key))
    return {'checked': len(hashes), 'findings': findings, 'duplicate_groups': [p for p in duplicates.values() if len(p)>1],
        'duplicate_note': 'Identity controls and exact reload checks intentionally repeat pixels; never delete duplicates',
        'tensor_finiteness': 'Stored runtime gradient/parameter checks inspected; checkpoint tensor values not unpickled by monitor'}
