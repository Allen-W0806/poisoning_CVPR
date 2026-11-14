import _bootstrap
from _phase2 import phase2_run, mark_latest
from src.training.generator_optimizer_interface import GeneratorOptimizerContract

if __name__ == "__main__":
    contract = GeneratorOptimizerContract()
    with phase2_run("generator_rl_contract", {"requested": "reward-based actual generator update", "contract": contract.__dict__}) as run:
        try:
            contract.reward_update(rewards=[0.0] * 10)
        except NotImplementedError as exc:
            run.metrics({"reward_based_generator_update_verified": False, "status": "BLOCKED", "reason": str(exc),
                "separate_supervised_lora_test_is_not_rl": True, "security_objectives_used": False})
            run.finish("BLOCKED", str(exc))
        else:
            raise AssertionError("Unsupported optimizer contract must not pretend to train")
        mark_latest("generator_rl_contract", run)
        print(run.path)
