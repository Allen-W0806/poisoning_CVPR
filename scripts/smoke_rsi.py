import _bootstrap
from _runs import new_run
from src.rsi.controller import RSIController


def main():
    with new_run("rsi_smoke", {"scores": "toy_only", "budget_gpu_hours": .1}) as run:
        controller = RSIController(run.path / "rsi", {"batch_size": 10, "learning_rate": 1e-5}, .5, "toy-validation-v1", .1)
        first = controller.revise({"learning_rate": 5e-5}, .6, .01, "toy-validation-v1", policy_version="toy_v1")
        second = controller.revise({"batch_size": 20}, .55, .01, "toy-validation-v1")
        assert first["accepted"] and not second["accepted"]
        assert controller.state["training_config"] == {"batch_size": 10, "learning_rate": 5e-5}
        assert controller.state["policy_version"] == "toy_v1"
        run.metrics({"accepted": 1, "rejected": 1, "cost_gpu_hours_toy": .02,
                     "final_config": controller.state["training_config"], "source": "toy_scores_not_empirical"})
        run.finish(summary="Toy validation accepts one revision and rejects the second without overwriting accepted configuration.")
        print(run.path)


if __name__ == "__main__":
    main()
