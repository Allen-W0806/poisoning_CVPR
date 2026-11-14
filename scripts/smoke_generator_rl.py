"""Real IP2Pix LoRA reward update, immutable trajectories and exact restart witness."""
import argparse
import copy
import gc
import json
import random
import time
import numpy as np
import torch
import yaml
import _bootstrap
from _phase3 import phase3_run,mark_latest
from _runs import model_options
from src.data.dataset import VQADataset,sha256
from src.models.pretrained_editor_loader import load_pretrained_editor
from src.models.qwen_loader import QwenVQA
from src.training.generator_adapter import adapter_state,frozen_hashes
from src.rl.diffusion_policy import DiffusionPolicy,INSTRUCTION
from src.rl.diffusion_rl_trainer import DiffusionRLTrainer
from src.rl.diffusion_trajectory import transition
from src.rl.optimizer_validation import state_equal,tensor_changes
from src.rl.benign_reward_v3 import caller_feedback
from src.rl.phase5_backend import require_training_reward
from src.editing.semantic_validator import SemanticResult
from src.evaluation.semantic_validation import review_pair,semantic_decision
from src.evaluation.answer_consistency import answer_consistency
from src.evaluation.image_quality import image_quality
from src.evaluation.vqa_official_compatible import normalize_answer
from src.utils.logging import write_json

def rng_state():
    return {"torch":torch.get_rng_state(),"cuda":torch.cuda.get_rng_state_all(),"numpy":np.random.get_state(),"python":random.getstate()}

@torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
def collect(policy,conditions,records,qwen,original_answers,run,label):
    episodes=[]; observations=[]
    for i,(condition,record) in enumerate(zip(conditions,records)):
        seed=int(torch.randint(0,2**30,(1,)).item())
        image,trajectory=policy.sample(condition,seed,f"{label}_{i}")
        source_path=run.path/f"images/source_{i}.png"
        if not source_path.exists():condition["source"].save(source_path)
        path=run.path/f"images/{label}_{i}.png"; image.save(path)
        started=time.perf_counter()
        answer=qwen.answer([image],[record["question"]])[0]
        consistency=answer_consistency(original_answers[i],answer,record)
        judgment=review_pair(qwen,condition["source"],image,record)
        semantic=semantic_decision(condition["source"],image,record,consistency,judgment)
        quality=image_quality(condition["source"],image)
        reward=caller_feedback(policy.config['reward'],semantic,consistency,quality,
            policy.config['num_inference_steps'],trajectory['generation_seconds'])
        trajectory["reward"]=reward
        torch.save(trajectory,run.path/f"checkpoints/{label}_trajectory_{i}.pt")
        item={"sample_id":record["sample_id"],"source_image_id":str(record["image_id"]),"question":record["question"],
            "answers":record["answers"],"instruction":INSTRUCTION,"seed":seed,"source_path":str(source_path),"output_path":str(path),
            "trajectory_id":trajectory["trajectory_id"],"policy_version":policy.version,"generation_seconds":trajectory["generation_seconds"],
            "reward_seconds":time.perf_counter()-started,"reward":reward,"semantic":semantic,"answer_consistency":consistency,
            "structured_judgment":judgment,"quality":quality,"settings":{"input_dimensions":list(condition['source'].size),
                "output_dimensions":list(image.size),"num_inference_steps":policy.config['num_inference_steps']},
            "vqa_diagnostic":{"raw_response":answer,"parsed_answer":answer.strip(),"normalized_answer":normalize_answer(answer),
                "original_raw_response":original_answers[i],"original_normalized_answer":normalize_answer(original_answers[i]),
                "reference_answers":record["answers"],"score":consistency["edited_reference_score"],"eligible_for_accuracy":False},
            "qwen_checkpoint":qwen.revision,"qwen_inference":qwen.metadata(),"human_status":"UNVERIFIED"}
        observations.append(item);episodes.append((trajectory,condition,reward["reward"]))
        run.event("trajectory_reward",item)
    write_json(run.path/f"{label}_observations.json",observations)
    return episodes,observations

@torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
def conditioning_and_reward_test(policy,condition,record,trajectory):
    with torch.no_grad():
        pixels=policy.pipe.image_processor.preprocess(condition["source"]).to(policy.editor.device,policy.editor.dtype)
        official=policy.pipe.prepare_image_latents(pixels,1,1,policy.editor.dtype,policy.editor.device,True)
        expected=torch.cat([condition["image"],condition["image"],torch.zeros_like(condition["image"])])
        assert torch.equal(official,expected),"Source conditioning differs from official path"
        alternate=policy.condition(record,"Slightly soften the lighting. Keep all objects unchanged.")
        assert not torch.equal(condition["hidden"],alternate["hidden"])
        step=trajectory["steps"][0];state=step["state"].to(policy.editor.device)
        normal=policy.predict(state,step["t"],condition)
        zero={**condition,"image":torch.zeros_like(condition["image"])}
        image_effect=float((normal-policy.predict(state,step["t"],zero)).abs().max())
        text_effect=float((normal-policy.predict(state,step["t"],alternate)).abs().max())
        assert image_effect>0 and text_effect>0
        # Repeating explicit seed must reproduce complete FP32 trajectory and final PNG.
        repeated,replay=policy.sample(condition,trajectory["seed"],"seed_control")
        assert all(torch.equal(a["action"],b["action"]) for a,b in zip(trajectory["steps"],replay["steps"]))
    epsilon=policy.predict(state,step["t"],condition)
    _,lp,_,_=transition(policy.pipe.scheduler,epsilon,step["t"],state,action=step["action"].to(state.device))
    params=[p for p in policy.pipe.unet.parameters() if p.requires_grad]
    plus=torch.autograd.grad(-.5*lp.sum(),params,retain_graph=True,allow_unused=True)
    minus=torch.autograd.grad(.5*lp.sum(),params,allow_unused=True)
    valid=[(a,b) for a,b in zip(plus,minus) if a is not None]
    assert valid and all(torch.equal(a,-b) for a,b in valid)
    norm=float(torch.sqrt(sum(a.double().square().sum() for a,b in valid)))
    assert norm>0
    return {"official_source_latent_exact":True,"image_ablation_max_prediction_difference":image_effect,
        "instruction_ablation_max_prediction_difference":text_effect,"explicit_seed_exact_trajectory_replay":True,
        "actual_generator_reward_sign_reversal_exact":True,"reward_test_gradient_norm":norm,
        "scope":"Wiring and conditioning, not image quality improvement"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--updates",type=int,choices=[1,3],required=True);p.add_argument("--batch-size",type=int,choices=[1,10],required=True)
    a=p.parse_args();root=_bootstrap.ROOT
    if (a.updates,a.batch_size) not in [(1,1),(3,10)]:raise ValueError("Only staged feasibility configurations enabled")
    config=yaml.safe_load((root/"configs/phase3_generator_rl.yaml").read_text())
    config.update(logical_batch_size=a.batch_size,gradient_accumulation_steps=a.batch_size,initial_optimizer_updates=a.updates)
    data=VQADataset(root/"data/processed/vqav2_seed42_v1/train.json",root,"train")
    indices=list(range(len(data)));random.Random(42).shuffle(indices); records=[data[i] for i in indices[:a.batch_size]]
    with phase3_run("generator_rl_one" if a.updates==1 else "generator_rl_smoke",config) as run:
        editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,config)
        policy.contract["training_data"]={"manifest_sha256":sha256(root/"data/processed/vqav2_seed42_v1/train.json"),"sample_ids":[r["sample_id"] for r in records]}
        contract=copy.deepcopy(policy.contract)
        trainer=DiffusionRLTrainer(policy)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):conditions=[policy.condition(r) for r in records]
        initial=adapter_state(policy.pipe.unet); frozen=frozen_hashes(policy.pipe);write_json(run.path/"frozen_before.json",frozen)
        qwen=QwenVQA(**model_options("cuda"))
        original_answers=qwen.answer([c["source"] for c in conditions],[r["question"] for r in records])
        trainer.save(run.path/"checkpoints/step_0000")
        history=[];all_observations=[];tests=None
        for step in range(1,a.updates+1):
            before_collect=adapter_state(policy.pipe.unet)
            torch.save({"adapter":before_collect,"conditions":[{"image":c["image"].cpu(),"hidden":c["hidden"].cpu()} for c in conditions]},run.path/f"checkpoints/pre_update_{step}.pt")
            episodes,observations=collect(policy,conditions,records,qwen,original_answers,run,f"step{step}")
            assert state_equal(before_collect,adapter_state(policy.pipe.unet)),"No-update collection changed generator"
            if step==1:tests=conditioning_and_reward_test(policy,conditions[0],records[0],episodes[0][0])
            # Test uses explicit local seeds and must not affect rollout seed RNG.
            # Diagnostic bookkeeping does not release the scientific reward path.
            for trajectory,_,_ in episodes:require_training_reward(trajectory['reward'])
            result=trainer.update(episodes);history.append(result);all_observations.extend(observations)
            torch.save(trainer.last_gradients,run.path/f"checkpoints/gradients_{step}.pt")
            trainer.save(run.path/f"checkpoints/step_{step:04d}")
            run.event("generator_rl_update",result)
            run.metrics({"updates":history,"observations":all_observations,"implementation_tests":tests,"in_progress":step<a.updates})
        after=adapter_state(policy.pipe.unet); final_optimizer=copy.deepcopy(trainer.optimizer.state_dict())
        final_scheduler=copy.deepcopy(trainer.scheduler.state_dict()); final_rng=rng_state()
        assert frozen==frozen_hashes(policy.pipe),"Frozen generator/VAE/text tensors changed"
        write_json(run.path/"frozen_after.json",frozen)
        unet_calls=policy.unet_calls
        # Fresh model and trainer, not merely reattaching the same Python objects.
        del trainer,policy,editor,conditions,episodes;gc.collect();torch.cuda.empty_cache()
        editor=load_pretrained_editor(root);policy=DiffusionPolicy(editor,config);policy.contract=contract
        trainer=DiffusionRLTrainer(policy)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):conditions=[policy.condition(r) for r in records]
        trainer.restore(run.path/f"checkpoints/step_{a.updates-1:04d}")
        saved_inputs=torch.load(run.path/f"checkpoints/pre_update_{a.updates}.pt",weights_only=False)
        input_diagnostic={"adapter_exact":state_equal(saved_inputs["adapter"],adapter_state(policy.pipe.unet)),
            "conditions_exact":all(torch.equal(c["image"].cpu(),s["image"]) and torch.equal(c["hidden"].cpu(),s["hidden"]) for c,s in zip(conditions,saved_inputs["conditions"]))}
        write_json(run.path/"restart_inputs.json",input_diagnostic)
        assert all(input_diagnostic.values()),input_diagnostic
        replay_episodes,replay_observations=collect(policy,conditions,records,qwen,original_answers,run,"restart")
        if a.updates==1:conditioning_and_reward_test(policy,conditions[0],records[0],replay_episodes[0][0])
        replay_result=trainer.update(replay_episodes)
        torch.save(trainer.last_gradients,run.path/"checkpoints/restarted_gradients.pt")
        diagnostic={"fresh_model":True,"adapter_exact":state_equal(after,adapter_state(policy.pipe.unet)),
            "optimizer_exact":state_equal(final_optimizer,trainer.optimizer.state_dict()),
            "scheduler_exact":state_equal(final_scheduler,trainer.scheduler.state_dict()),"rng_exact":state_equal(final_rng,rng_state()),
            "next_step_loss_exact":history[-1]["loss"]==replay_result["loss"],
            "rollout_seeds_exact":[o["seed"] for o in all_observations[-a.batch_size:]]==[o["seed"] for o in replay_observations],
            "reward_components_exact":[o["reward"] for o in all_observations[-a.batch_size:]]==[o["reward"] for o in replay_observations],
            "adapter_max_difference":tensor_changes(after,adapter_state(policy.pipe.unet))["max_abs_change"]}
        trainer.save(run.path/"checkpoints/restarted_final");write_json(run.path/"restart_diagnostic.json",diagnostic)
        run.event("restart_diagnostic",diagnostic)
        assert all(diagnostic[k] for k in ["adapter_exact","optimizer_exact","scheduler_exact","rng_exact","next_step_loss_exact","rollout_seeds_exact","reward_components_exact"]),diagnostic
        assert frozen==frozen_hashes(policy.pipe)
        run.metrics({"algorithm":"DDPO_SF_v1","generator_training_kind":"reward_based_generator_RL",
            "updates":history,"observations":all_observations,"implementation_tests":tests,"restart":diagnostic,
            "parameters":policy.parameters,"adapter_changes":tensor_changes(initial,after),"frozen_tensor_count":len(frozen),
            "frozen_weights_unchanged":True,"no_update_control_unchanged":True,"qwen_frozen":all(not p.requires_grad for p in qwen.model.parameters()),
            "source_images":len(records),"training_trajectories":a.updates*a.batch_size,"restart_diagnostic_trajectories":a.batch_size,
            "seed_control_extra_generations":2 if a.updates==1 else 1,"transitions_per_trajectory":10,
            "training_optimizer_updates":a.updates,"restart_diagnostic_optimizer_updates":1,
            "unet_calls_including_diagnostics":unet_calls+policy.unet_calls,"qwen_calls":qwen.calls,
            "generation_seconds":sum(o["generation_seconds"] for o in all_observations+replay_observations),
            "reward_seconds":sum(o["reward_seconds"] for o in all_observations+replay_observations),
            "human_verified_changed_edits":0,"independent_quality_gain_established":False})
        run.finish(summary="Actual pretrained image-conditioned generator LoRA updated by terminal benign reward score-function gradients; exact restart witnessed. No semantic/performance advantage claim.")
        mark_latest("one" if a.updates==1 else "smoke",run);print(run.path)

if __name__=="__main__":main()
