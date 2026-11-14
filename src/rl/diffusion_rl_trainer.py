"""Single-pass on-policy DDPO score-function gradient on the actual UNet LoRA.

Loss = -mean_i R_i * mean_t mean_coordinate log N(z_next; mu_psi, sigma_t^2).
All transitions are stochastic. Fixed coordinate/time scaling preserves gradient
direction for expected terminal reward. No PPO ratios, KL or stale replay.
"""
import random
import time
import numpy as np
import torch
from ..training.generator_adapter import adapter_state,load_adapter_state
from .diffusion_trajectory import transition,assert_fresh_batch
from .reward_normalization import fixed_advantage
from .optimizer_validation import tensor_changes

class DiffusionRLTrainer:
    def __init__(self,policy):
        self.policy=policy; self.config=policy.config
        if self.config['physical_micro_batch_size']!=1 or self.config['gradient_accumulation_steps']!=self.config['logical_batch_size']:
            raise ValueError("Only one source trajectory per physical microbatch with explicit logical accumulation is validated")
        self.params=[p for p in policy.pipe.unet.parameters() if p.requires_grad]
        self.optimizer=torch.optim.AdamW(self.params,lr=self.config["learning_rate"],weight_decay=0.)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda step:1.)
        self.step=0; self.consumed=set()

    @torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
    def update(self,episodes):
        if len(episodes)!=self.config["logical_batch_size"]:
            raise ValueError("Logical source/trajectory batch count mismatch")
        assert_fresh_batch([t for t,c,r in episodes],self.policy.version,self.policy.contract,self.consumed)
        before=adapter_state(self.policy.pipe.unet)
        for trajectory,condition,reward in episodes:
            if trajectory["sample_id"]!=condition["sample_id"] or trajectory["instruction"]!=condition["instruction"] or trajectory["image_id"]!=condition["image_id"]:
                raise ValueError("Conditioning association mismatch")
            if [s['t'] for s in trajectory['steps']] != self.policy.pipe.scheduler.timesteps.tolist():
                raise ValueError("Incomplete/mismatched diffusion timestep sequence")
        started=time.perf_counter(); self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        losses=[]; replays=[]; reward_values=[]
        for trajectory,condition,reward in episodes:
            advantage=fixed_advantage(reward); reward_values.append(reward)
            for item in trajectory["steps"]:
                state=item["state"].to(self.policy.editor.device); action=item["action"].to(state.device)
                _,logp,_,_=transition(self.policy.pipe.scheduler,self.policy.predict(state,item["t"],condition),item["t"],state,
                    self.config["eta"],action=action)
                difference=float((logp.detach().cpu()-item["logp"]).abs().max()); replays.append(difference)
                if difference > 1e-6: raise ValueError(f"Behavior likelihood replay mismatch {difference}")
                loss=-advantage*logp.mean()/(len(episodes)*len(trajectory["steps"]))
                loss.backward(); losses.append(float(loss.detach()))
        gradients=[p.grad for p in self.params if p.grad is not None]
        if not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise FloatingPointError("Missing/nonfinite actual generator gradients")
        norm=torch.nn.utils.clip_grad_norm_(self.params,self.config["max_grad_norm"])
        self.last_gradients=[p.grad.detach().cpu().clone() if p.grad is not None else None for p in self.params]
        if not torch.isfinite(norm) or norm <= 0: raise FloatingPointError("Zero/nonfinite generator gradient")
        self.optimizer.step(); self.scheduler.step(); self.step+=1; self.policy.version+=1
        self.consumed.update(t["trajectory_id"] for t,c,r in episodes)
        torch.cuda.synchronize()
        changes=tensor_changes(before,adapter_state(self.policy.pipe.unet))
        if changes["max_abs_change"]<=0: raise AssertionError("Generator adapter did not update")
        return {"step":self.step,"loss":sum(losses),"reward_mean":float(np.mean(reward_values)),
            "reward_variance":float(np.var(reward_values)),"individual_rewards":reward_values,
            "gradient_norm":float(norm),"finite_gradients":True,"probabilities_finite":True,
            "behavior_logp_replay_max_difference":max(replays),"conditioning_verified":True,
            "logical_batch_size":len(episodes),"physical_micro_batch_size":1,"gradient_accumulation_steps":len(episodes),
            "backward_transition_calls":len(losses),"optimizer_updates":1,"changes":changes,
            "trainable_max_change":changes["max_abs_change"],"step_seconds":time.perf_counter()-started,
            "peak_gpu_memory_bytes":torch.cuda.max_memory_allocated(),"algorithm":"DDPO_SF_v1"}

    def save(self,directory):
        directory.mkdir(parents=True,exist_ok=False)
        torch.save({"adapter":adapter_state(self.policy.pipe.unet),"optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict(),"step":self.step,"version":self.policy.version,
            "contract":self.policy.contract,"consumed":self.consumed,"rng_torch":torch.get_rng_state(),
            "rng_cuda":torch.cuda.get_rng_state_all(),"rng_numpy":np.random.get_state(),"rng_python":random.getstate()},
            directory/"training_state.pt")

    def restore(self,directory):
        state=torch.load(directory/"training_state.pt",map_location="cpu",weights_only=False)
        if state["contract"]!=self.policy.contract: raise ValueError("Checkpoint contract mismatch")
        load_adapter_state(self.policy.pipe.unet,state["adapter"])
        self.optimizer.load_state_dict(state["optimizer"]); self.scheduler.load_state_dict(state["scheduler"])
        self.step=state["step"]; self.policy.version=state["version"]; self.consumed=state["consumed"]
        torch.set_rng_state(state["rng_torch"]); torch.cuda.set_rng_state_all(state["rng_cuda"])
        np.random.set_state(state["rng_numpy"]); random.setstate(state["rng_python"])
