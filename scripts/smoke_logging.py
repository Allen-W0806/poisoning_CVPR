import json
import _bootstrap
from _runs import new_run

if __name__ == "__main__":
    paths = []
    first_config = None
    for seed, lr in [(42, 1e-5), (123, 1e-4)]:
        with new_run("logging_smoke", {"seed": seed, "learning_rate": lr}) as run:
            run.metrics({"engineering_fixture_value": seed})
            run.finish(summary="Tiny logging-only run; fixture value is not a scientific result.")
            paths.append(run.path)
            if first_config is None:
                first_config = (run.path / "config.yaml").read_bytes()
    assert paths[0] != paths[1]
    assert first_config == (paths[0] / "config.yaml").read_bytes()
    assert [json.loads((p / "metrics.json").read_text())["engineering_fixture_value"] for p in paths] == [42, 123]
    index = (_bootstrap.ROOT / "EXPERIMENT_INDEX.md").read_text()
    assert all(p.name in index for p in paths)
    print(json.dumps({"runs": [str(p) for p in paths], "independent": True}))
