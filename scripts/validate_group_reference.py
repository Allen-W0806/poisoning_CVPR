"""Exact enumeration of a one-step GRPO surrogate, independent of raw reward PG.
Finite action space 3, group size 3; all 27 ordered groups enumerated. Stored
behavior probabilities and normalized group advantages are constants during the
surrogate derivative. This is not an unbiased gradient of raw expected reward.
"""
import itertools
import argparse
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from validate_policy_gradients import group_advantages


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',default='group_reference_20261028_v1')
    args=parser.parse_args()
    run=ROOT/'results'/args.run_id;run.mkdir(parents=True,exist_ok=False)
    cfg={'actions':3,'group_size':3,'logits':[.2,-.3,.1],'rewards':[0.,.35,1.],
         'seeds':32,'groups_per_seed':65536,'z_limit':4.,'finite_difference_atol':1e-8,'gradient_clip_atol':1e-10}
    (run/'protocol.json').write_text(json.dumps(cfg,indent=2))
    started=time.perf_counter();torch.set_num_threads(2);torch.set_default_dtype(torch.float64)
    theta=torch.tensor(cfg['logits'],requires_grad=True);rewards=torch.tensor(cfg['rewards'])
    old=theta.detach().log_softmax(0);oldp=old.exp();groups=torch.tensor(list(itertools.product(range(3),repeat=3)))
    adv=group_advantages(rewards[groups]);weights=oldp[groups].prod(-1).detach()
    def objective(value):
        ratio=(value.log_softmax(0)[groups]-old[groups]).exp()
        return (weights*(ratio*adv).mean(-1)).sum()
    exact=torch.autograd.grad(objective(theta),theta)[0]
    numerical=[];eps=1e-5
    for j in range(3):
        delta=torch.zeros(3);delta[j]=eps
        numerical.append(float((objective(theta.detach()+delta)-objective(theta.detach()-delta))/(2*eps)))
    estimates=[]
    for seed in range(32):
        gen=torch.Generator().manual_seed(80000+seed);sample=torch.multinomial(oldp,65536*3,replacement=True,generator=gen).reshape(-1,3)
        advantage=group_advantages(rewards[sample]);ratios=(theta.log_softmax(0)[sample]-old[sample]).exp()
        estimate=torch.autograd.grad((ratios*advantage).mean(),theta)[0];estimates.append(estimate.detach().numpy())
    est=np.array(estimates);z=np.abs(est.mean(0)-exact.detach().numpy())/(est.std(0,ddof=1)/np.sqrt(32))
    rawtheta=theta.detach().clone().requires_grad_();rawgrad=torch.autograd.grad((rawtheta.softmax(0)*rewards).sum(),rawtheta)[0]
    # Clipping is actually activated, not only called with already-small gradients.
    parameter=torch.nn.Parameter(torch.ones(3));parameter.grad=torch.tensor([3.,4.,0.])*100
    norm=torch.nn.utils.clip_grad_norm_([parameter],1.)
    # Zero-variance normalized group at reference -> zero surrogate gradient.
    zero_adv=group_advantages(torch.ones(2,3));zero_loss=(theta.log_softmax(0)[groups[:2]]*zero_adv).mean()
    zerograd=torch.autograd.grad(zero_loss,theta)[0]
    kltheta=torch.tensor([.8,-.2,0.],requires_grad=True);ref=torch.full((3,),1/3)
    def kl(value):p=value.softmax(0);return (p*(value.log_softmax(0)-ref.log())).sum()
    oldkl=kl(kltheta);kg=torch.autograd.grad(oldkl,kltheta)[0];newkl=kl(kltheta.detach()-.1*kg)
    checks=[{'test':'enumerated_surrogate_gradient_vs_finite_difference','passed':bool(np.max(np.abs(exact.detach().numpy()-numerical))<1e-8)},
        {'test':'MC_surrogate_gradient_agreement','passed':bool(z.max()<4.),'max_z':float(z.max())},
        {'test':'gradient_clip_activated','passed':bool(abs(float(parameter.grad.norm())-float(norm)/(float(norm)+1e-6))<cfg['gradient_clip_atol'] and float(parameter.grad.norm())<=1 and float(norm)>1),
         'expected_norm_with_PyTorch_epsilon':float(norm)/(float(norm)+1e-6),'observed_norm':float(parameter.grad.norm()),'atol':cfg['gradient_clip_atol']},
        {'test':'zero_variance_group_zero_gradient','passed':bool(torch.equal(zerograd,torch.zeros_like(zerograd)))},
        {'test':'KL_gradient_restores_reference','passed':bool(float(newkl)<float(oldkl.detach()))},
        {'test':'old_logprobs_and_advantages_detached','passed':not old.requires_grad and not adv.requires_grad}]
    result={'passed':all(c['passed'] for c in checks),'checks':checks,'exact_surrogate_gradient':exact.tolist(),
        'raw_expected_reward_gradient':rawgrad.tolist(),'gradient_difference_is_expected':True,
        'interpretation':'Group standardization defines a different surrogate; do not claim unbiased raw reward gradient. Enumeration holds old distribution and reward advantages fixed.',
        'mean_estimated_surrogate_gradient':est.mean(0).tolist(),'finite_difference_gradient':numerical,'seconds':time.perf_counter()-started}
    (run/'checks.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
