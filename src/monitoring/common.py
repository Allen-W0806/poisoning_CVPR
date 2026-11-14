import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import time
from datetime import datetime, timezone

ACTIVE = {'CREATED', 'RUNNING', 'SUBMITTED'}
TERMINAL = {'COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY', 'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE'}


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path, default=None):
    for attempt in range(3):
        try:
            return json.loads(Path(path).read_text(encoding='utf-8-sig'))
        except FileNotFoundError:
            return default
        except OSError as error:
            if error.errno not in {5, 11, 116} or attempt == 2:
                raise
            time.sleep(.05)


def append_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as out:
        out.write(json.dumps(value, allow_nan=False) + '\n')


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def command(args, timeout=40, cwd=None, env=None):
    try:
        proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)
        return {'command': list(map(str, args)), 'exit_code': proc.returncode, 'stdout': proc.stdout, 'stderr': proc.stderr}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'command': list(map(str, args)), 'exit_code': -1, 'stdout': '', 'stderr': str(error)}


def safe_path(root, path):
    root = Path(root).resolve()
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f'Path escapes project: {path}')
    return result


def json_safe(value):
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return 'NONFINITE:' + str(value)
    if isinstance(value, dict):
        return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


class FileLock:
    def __init__(self, path):
        self.path, self.handle, self.acquired = Path(path), None, False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.handle.write(b'0')
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.acquired = True
        except (OSError, BlockingIOError):
            self.handle.close()
            self.handle = None
        return self.acquired

    def __exit__(self, *args):
        if self.handle:
            if os.name == 'nt':
                import msvcrt
                self.handle.seek(0)
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()


def table(text):
    return list(csv.DictReader(io.StringIO(text), delimiter='|'))


def slurm_snapshot(job_ids):
    queue = command(['squeue', '-h', '-u', os.environ.get('USER', ''), '-o', '%i|%T|%j|%b|%M|%l|%R'])
    result = {'queue': queue, 'accounting': {'exit_code': 0, 'stdout': ''}, 'jobs': {}, 'live_statistics': None}
    if job_ids:
        result['accounting'] = command(['sacct', '-X', '-j', ','.join(sorted(set(job_ids))), '-P',
            '--format=JobIDRaw,State,ExitCode,ElapsedRaw,ReqTRES,AllocTRES,Timelimit'])
        for row in table(result['accounting']['stdout']):
            if row.get('JobIDRaw', '').isdigit():
                row['State'] = row['State'].split()[0].rstrip('+')
                result['jobs'][row['JobIDRaw']] = row
        running = [j for j, row in result['jobs'].items() if row['State'] == 'RUNNING']
        if running:
            result['live_statistics'] = command(['sstat', '--allsteps', '-j', ','.join(running), '-P',
                '--format=JobID,AveRSS,MaxRSS,AveCPU,TRESUsageInAve,TRESUsageInMax'])
    return result
