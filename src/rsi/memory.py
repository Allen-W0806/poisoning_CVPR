import json
from pathlib import Path
from ..utils.logging import write_json


class RevisionMemory:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=False)

    def append(self, revision):
        path = self.root / f"revision_{revision['revision_number']:04d}.json"
        with path.open("x", encoding="utf-8") as f:
            json.dump(revision, f, indent=2, allow_nan=False)

    def snapshot(self, state):
        write_json(self.root / "state.json", state)
