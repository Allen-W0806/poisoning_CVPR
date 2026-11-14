"""Finite benign policy-gradient checks. No image attack or model fine-tuning.

python -B scripts/validate_policy_gradients.py --run-id <unique ID>
Acceptance criteria are read and copied BEFORE any sampling. A failed assertion
is recorded, never silently dropped. Uses existing EditPolicy/PolicyTrainer and
existing DDIM transition, plus an isolated one-step GRPO reference.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from src.rl.policy import EditPolicy
from src.rl.trainer import PolicyTrainer
from src.editing.semantic_validator import SemanticResult


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)


def csv_write(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def tensor_hash(p):
    return hashlib.sha256(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def gradients(model):
    return [p for p in model.parameters() if p.requires_grad]


def flat_grad(value, params):
    return torch.cat([g.flatten() for g in torch.autograd.grad(value, params)])


def group_advantages(rewards):
    """Groups are last axis, SAME context per group; zero std -> exact zero."""
    r = rewards.detach()
    std = r.std(-1, correction=0, keepdim=True)
    return torch.where(std > 1e-12, (r-r.mean(-1, keepdim=True))/std.clamp_min(1e-12), torch.zeros_like(r))


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', required=True)
    parser.add_argument('--protocol',default='configs/gradient_feedback_validation_v1.json')
    args = parser.parse_args(); run = ROOT/'results'/args.run_id
    run.mkdir(parents=True, exist_ok=False)
    scratch = ROOT/'tmp'/args.run_id; scratch.mkdir(parents=True, exist_ok=False)
    os.environ['MPLCONFIGDIR'] = str(scratch/'matplotlib')
    protocol_path = ROOT/args.protocol
    cfg = json.loads(protocol_path.read_text()); dump(run/'protocol.json', cfg)
    dump(run/'protocol_freeze.json', {'time_unix': time.time(), 'sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
                                     'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    checks=[]; rows=[]; updates=[]; learning=[]; started=time.perf_counter()
    def check(name, ok, **details):
        checks.append({'test':name, 'passed':bool(ok), **details})
        print(json.dumps(checks[-1]), flush=True)
    torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
    try:
        for kind in ['categorical_logits', 'existing_EditPolicy']:
            torch.manual_seed(1729)
            if kind == 'categorical_logits':
                model=torch.nn.Linear(1,3,bias=False); model.weight.data.copy_(torch.tensor([[.2],[-.3],[.1]]))
                features=torch.ones(1,1)
            else:
                model=EditPolicy().double(); features=torch.tensor([[.2,.5,.8]])
            params=gradients(model); logits=model(features)[0]; p=logits.softmax(0)
            rewards=torch.tensor([0.,.35,1.]); expected=(p*rewards).sum()
            exact=flat_grad(expected, params).detach()
            # Independent analytic score: gradient log p(a), then categorical enumeration.
            score=torch.stack([flat_grad(model(features)[0].log_softmax(0)[i],params).detach() for i in range(3)])
            contrib=score*rewards[:,None]; enum=(p.detach()[:,None]*contrib).sum(0)
            check(kind+'_enumeration_matches_autodiff', torch.allclose(exact,enum,atol=1e-12,rtol=1e-12), max_difference=float((exact-enum).abs().max()))
            var=(p.detach()[:,None]*(contrib-exact).square()).sum(0)
            for seed in range(cfg['gradient_seeds']):
                gen=torch.Generator().manual_seed(10000+seed)
                for n in cfg['sample_counts']:
                    t=time.perf_counter(); actions=torch.multinomial(p.detach(),n,replacement=True,generator=gen)
                    # Autodiff through actual sampled REINFORCE loss, rewards detached.
                    objective=(model(features)[0].log_softmax(0)[actions]*rewards[actions].detach()).mean()
                    estimate=flat_grad(objective,params).detach()
                    analytic_sample=contrib[actions].mean(0)
                    check_name = kind+'_sample_loss_gradient_identity'
                    if seed==0 and n==cfg['sample_counts'][0]:
                        check(check_name,torch.allclose(estimate,analytic_sample,atol=1e-12,rtol=1e-12))
                    rel=float((estimate-exact).norm()/exact.norm())
                    cosine=float(torch.nn.functional.cosine_similarity(estimate,exact,dim=0))
                    for j,(e,g,v) in enumerate(zip(exact.tolist(),estimate.tolist(),var.tolist())):
                        rows.append(dict(estimator=kind,seed=seed,n=n,component=j,exact=e,estimate=g,
                                         sampling_variance=v/n,relative_error=rel,cosine=cosine,seconds=time.perf_counter()-t))
            # Fixed simultaneous four-SE rule across coordinates, over independent seed means.
            trend=[]
            for n in cfg['sample_counts']:
                subset=[r for r in rows if r['estimator']==kind and r['n']==n]
                est=np.array([r['estimate'] for r in subset]).reshape(cfg['gradient_seeds'],-1)
                rmse=float(np.sqrt(np.mean(np.sum((est-exact.numpy())**2,axis=1)))/exact.norm())
                trend.append(rmse)
            n=cfg['sample_counts'][-1]; subset=[r for r in rows if r['estimator']==kind and r['n']==n]
            est=np.array([r['estimate'] for r in subset]).reshape(cfg['gradient_seeds'],-1)
            se=np.sqrt(var.numpy()/n/cfg['gradient_seeds']); z=np.abs(est.mean(0)-exact.numpy())/np.maximum(se,1e-30)
            slope=float(np.polyfit(np.log(cfg['sample_counts']),np.log(trend),1)[0])
            a=cfg['gradient_acceptance']; lo,hi=a['rmse_log_log_slope_range']
            check(kind+'_statistical_agreement',float(z.max())<a['simultaneous_z_limit'] and trend[-1]<a['largest_n_relative_rmse_max'],
                  max_z=float(z.max()),relative_rmse=trend[-1],rmse_by_n=trend)
            check(kind+'_convergence',lo<slope<hi and trend[-1]<trend[0],slope=slope)
            check(kind+'_largest_n_direction', min(r['cosine'] for r in subset)>a['largest_n_cosine_min'],min_cosine=min(r['cosine'] for r in subset))

        # Exercise the EXISTING production trainer on known action rewards, harmless identity editor.
        torch.set_default_dtype(torch.float32); torch.manual_seed(42)
        policy=EditPolicy(); reward_values=[0.,.35,1.]
        class MarkEditor:
            def __call__(self, images, conditions, seed):
                self.actions=[c['action'] for c in conditions]
                self.outputs=[]
                for image,action in zip(images,self.actions):
                    out=image.copy(); out.info['action']=action; self.outputs.append(out)
                return self.outputs
        def evaluator(original, output, record):
            return {'semantics':SemanticResult(True,1.,'known_synthetic','VERIFIED'),
                    'answer_consistency':reward_values[output.info['action']], 'quality':{'distortion':0.},'cost':0.}
        from src.rl.reward import RewardWeights
        trainer=PolicyTrainer(policy,MarkEditor(),'cpu',{'batch_size':64,'learning_rate':.01,'mixed_precision':'no','weight_decay':0.,'total_updates':3},
                              evaluator,RewardWeights(semantic_weight=0.,accuracy_weight=1.,distortion_weight=0.,cost_weight=0.,invalid_semantics_penalty=0.))
        ownership={id(p) for g in trainer.optimizer.param_groups for p in g['params']}
        check('production_optimizer_exact_ownership',ownership=={id(p) for p in policy.parameters() if p.requires_grad})
        dump(run/'optimizer_groups.json',[{'group':i,'lr':g['lr'],'weight_decay':g['weight_decay'],
            'parameters':[{'name':n,'shape':list(p.shape),'numel':p.numel()} for n,p in policy.named_parameters() if id(p) in {id(q) for q in g['params']}]} for i,g in enumerate(trainer.optimizer.param_groups)])
        before={n:p.detach().clone() for n,p in policy.named_parameters()}
        probe=policy(torch.tensor([[.2,.5,.8]])).log_softmax(-1).sum(); check('loss_requires_grad',probe.requires_grad)
        batch=[{'sample_id':str(i),'split':'train','image':Image.new('RGB',(8,8),(50+i,100,150))} for i in range(64)]
        receipt=trainer.update([batch]); dump(run/'production_trainer_update.json',receipt)
        for name,p in policy.named_parameters():
            delta=float((p.detach()-before[name]).abs().max()); finite=p.grad is None or bool(torch.isfinite(p.grad).all())
            row={'model':'existing_EditPolicy','parameter':name,'trainable':p.requires_grad,'optimizer_owned':id(p) in ownership,
                 'max_abs_change':delta,'grad_norm':float(p.grad.norm()) if p.grad is not None else 0.,'gradient_finite':finite,
                 'before_sha256':tensor_hash(before[name]),'after_sha256':tensor_hash(p)}; updates.append(row)
            check('parameter_'+name,finite and ((delta>0 and p.grad is not None) if p.requires_grad else delta==0),**row)
        # Exact checkpoint state and exact next-step continuation, including RNG and scheduler.
        state={'policy':policy.state_dict(),'optimizer':trainer.optimizer.state_dict(),'scheduler':trainer.scheduler.state_dict(),
               'step':trainer.step,'rng':torch.get_rng_state()}
        torch.save(state,run/'categorical_training_state.pt')
        saved=copy.deepcopy(state); expected_receipt=trainer.update([batch]); expected_params=copy.deepcopy(policy.state_dict())
        loaded=torch.load(run/'categorical_training_state.pt',weights_only=True)
        policy.load_state_dict(loaded['policy']); trainer.optimizer.load_state_dict(loaded['optimizer'])
        trainer.scheduler.load_state_dict(loaded['scheduler']); trainer.step=loaded['step']; torch.set_rng_state(loaded['rng'])
        check('checkpoint_save_reload_exact',all(torch.equal(v,saved['policy'][k]) for k,v in policy.state_dict().items()))
        resumed=trainer.update([batch])
        check('checkpoint_next_step_exact',expected_receipt==resumed and all(torch.equal(v,expected_params[k]) for k,v in policy.state_dict().items()))
        for invalid in [float('nan'),float('inf')]:
            from src.rl.reward_normalization import fixed_advantage
            rejected=False
            try: fixed_advantage(invalid)
            except ValueError: rejected=True
            check('reject_nonfinite_reward_'+str(invalid),rejected)

        # DDPO transition: an exact Gaussian reward expectation, NOT VLM/editor training.
        try:
            from diffusers import DDIMScheduler
            from src.rl.diffusion_trajectory import transition
            scheduler=DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear',clip_sample=False,steps_offset=1,set_alpha_to_one=False)
            scheduler.set_timesteps(10)
            x=torch.zeros(1,4,2,2); theta=torch.tensor(.1,requires_grad=True); t=scheduler.timesteps[0]
            _,_,mean,std=transition(scheduler,theta.expand_as(x),t,x,noise=torch.zeros_like(x))
            # R=cos(first action coordinate), bounded and analytically integrable.
            exact_reward=torch.cos(mean[0,0,0,0])*torch.exp(-.5*std.square())
            exact=float(torch.autograd.grad(exact_reward,theta)[0])/16
            ddpo_errors=[]
            ddpo_counts=cfg.get('ddpo_sample_counts',cfg['sample_counts'])
            for n in ddpo_counts:
                estimates=[]
                for seed in range(cfg['gradient_seeds']):
                    g=torch.Generator().manual_seed(90000+seed+n); theta=torch.tensor(.1,requires_grad=True)
                    estimated=0.
                    for start in range(0,n,32768):
                        count=min(32768,n-start);x=torch.zeros(count,4,2,2); noise=torch.randn(x.shape,generator=g)
                        action,lp,_,_=transition(scheduler,theta.expand_as(x),t,x,noise=noise)
                        reward=action[:,0,0,0].detach().cos(); objective=(lp*reward).sum()/n
                        estimated+=float(torch.autograd.grad(objective,theta)[0])
                    estimates.append(estimated)
                    rows.append(dict(estimator='existing_DDPO_transition_scaled_1_over_16',seed=seed,n=n,component=0,exact=exact,estimate=estimated,
                                     relative_error=abs(estimated-exact)/abs(exact)))
                rmse=float(np.sqrt(np.mean((np.array(estimates)-exact)**2))/abs(exact)); ddpo_errors.append(rmse)
            z=abs(np.mean(estimates)-exact)/(np.std(estimates,ddof=1)/math.sqrt(len(estimates)))
            slope=float(np.polyfit(np.log(ddpo_counts),np.log(ddpo_errors),1)[0])
            check('existing_DDPO_transition_statistical_agreement',z<4 and ddpo_errors[-1]<.1,z=float(z),relative_rmse=ddpo_errors[-1],rmse_by_n=ddpo_errors)
            check('existing_DDPO_transition_convergence',-.75<slope<-.25,slope=slope)
        except ImportError as e:
            check('existing_DDPO_transition_execution',False,error=str(e),status='BLOCKED_DEPENDENCY')

        # Isolated one-action GRPO reference: exact categorical KL replaces noisy sampled KL.
        torch.set_default_dtype(torch.float64); gc=cfg['group_task']; reward_table=torch.tensor(gc['reward_tables'])
        probe=torch.tensor([[1.,1.,1.],[0.,.3,1.]],requires_grad=True); adv=group_advantages(probe)
        check('group_zero_variance_finite_zero',torch.isfinite(adv).all() and torch.equal(adv[0],torch.zeros(3)))
        check('group_advantage_detached',not adv.requires_grad)
        check('group_advantage_center_scale',abs(float(adv[1].mean()))<1e-12 and abs(float(adv[1].std(correction=0))-1)<1e-12)
        # Explicit clipping unit witness, including both advantage signs.
        ratio=torch.tensor([2.,.5,2.,.5]); aa=torch.tensor([1.,1.,-1.,-1.])
        surrogate=torch.minimum(ratio*aa,ratio.clamp(.8,1.2)*aa)
        check('group_ratio_clipping_signs',torch.allclose(surrogate,torch.tensor([1.2,.5,-2.,-.8])))
        for seed in cfg['seeds']:
            for method in ['REINFORCE','one_step_GRPO_reference']:
                torch.manual_seed(seed); logits=torch.nn.Parameter(torch.zeros(2,3))
                opt=torch.optim.Adam([logits],lr=gc['learning_rate']); reference=torch.full((2,3),1/3)
                initial=float((logits.detach().softmax(-1)*reward_table).sum(-1).mean())
                frozen_reference=reference.clone()
                gen=torch.Generator().manual_seed(seed)
                max_norm=0.; max_ratio_error=0.; max_kl=0.; clip_seen=False
                for step in range(gc['updates']):
                    old=logits.detach().log_softmax(-1).clone()
                    # [context, independent group, candidate]; no mixed difficulty groups.
                    action=torch.multinomial(old.exp(),gc['groups_per_context']*gc['group_size'],replacement=True,generator=gen).reshape(2,gc['groups_per_context'],gc['group_size'])
                    rr=reward_table[torch.arange(2)[:,None,None],action].detach()
                    oldlp=old[torch.arange(2)[:,None,None],action].clone().detach()
                    advantages=group_advantages(rr)
                    epochs=gc['epochs'] if method!='REINFORCE' else 1
                    for epoch in range(epochs):
                        lp=logits.log_softmax(-1); sampled=lp[torch.arange(2)[:,None,None],action]
                        ratios=(sampled-oldlp).exp()
                        if epoch==0: max_ratio_error=max(max_ratio_error,float((ratios.detach()-1).abs().max()))
                        p=lp.exp(); kl=(p*(lp-reference.log())).sum(-1).mean()
                        if method=='REINFORCE': loss=-(sampled*rr).mean()
                        else: loss=-torch.minimum(ratios*advantages,ratios.clamp(1-gc['clip_epsilon'],1+gc['clip_epsilon'])*advantages).mean()+gc['kl_beta']*kl
                        opt.zero_grad(); loss.backward(); norm=torch.nn.utils.clip_grad_norm_([logits],gc['max_grad_norm'])
                        if not torch.isfinite(norm): raise FloatingPointError('nonfinite group gradient')
                        max_norm=max(max_norm,float(norm)); max_kl=max(max_kl,float(kl.detach()))
                        clip_seen=clip_seen or bool(((ratios.detach()-1).abs()>gc['clip_epsilon']).any())
                        opt.step()
                    if step%10==0 or step==gc['updates']-1:
                        learning.append({'seed':seed,'method':method,'step':step+1,'expected_reward':float((logits.detach().softmax(-1)*reward_table).sum(-1).mean()),'kl_to_initial':float(kl.detach()),'preclip_grad_norm':float(norm)})
                final=float((logits.detach().softmax(-1)*reward_table).sum(-1).mean())
                check(f'{method}_learning_seed_{seed}',final-initial>gc['minimum_expected_reward_gain'] and max_ratio_error<1e-12 and torch.equal(reference,frozen_reference),
                      initial_reward=initial,final_reward=final,max_initial_ratio_error=max_ratio_error,max_preclip_gradient_norm=max_norm,max_kl=max_kl,ratio_clip_observed=clip_seen)
        # Exact KL is positive off reference and minimized at it.
        q=torch.tensor([.2,.3,.5],requires_grad=True); kl=(q*(q.log()-torch.full((3,),1/3).log())).sum()
        check('group_exact_reference_KL_positive',float(kl.detach())>0 and torch.isfinite(torch.autograd.grad(kl,q)[0]).all())
    except Exception as e:
        import traceback
        check('unhandled_execution_error',False,error=repr(e),traceback=traceback.format_exc())
    finally:
        csv_write(run/'gradient_validation.csv',rows); csv_write(run/'parameter_update_audit.csv',updates)
        csv_write(run/'group_learning.csv',learning)
        dump(run/'synthetic_checks.json',{'passed':all(c['passed'] for c in checks),'checks':checks,'seconds':time.perf_counter()-started,
                                         'torch':torch.__version__,'device':'cpu','protocol_sha256':hashlib.sha256(protocol_path.read_bytes()).hexdigest()})
        resolved=scratch.resolve(); parent=(ROOT/'tmp').resolve()
        if resolved.parent!=parent or resolved.name!=args.run_id: raise RuntimeError('Unsafe cleanup path')
        shutil.rmtree(resolved)
    print('RESULT_DIRECTORY',run)
    return 0 if all(c['passed'] for c in checks) else 1


if __name__=='__main__': raise SystemExit(main())
