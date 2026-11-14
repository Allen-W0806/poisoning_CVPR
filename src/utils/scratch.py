"""Run-specific scratch with checked cleanup, avoiding Windows redirected temp ACLs."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile


@contextmanager
def scratch_directory(root=None):
    base = Path(root or Path(__file__).resolve().parents[2]) / "tmp"
    base.mkdir(parents=True, exist_ok=True)
    base = base.resolve()
    path = Path(tempfile.mkdtemp(prefix="run_", dir=base)).resolve()
    try:
        yield path
    finally:
        if path.parent != base or not path.name.startswith("run_"):
            raise RuntimeError(f"Unsafe scratch cleanup target: {path}")
        shutil.rmtree(path)
