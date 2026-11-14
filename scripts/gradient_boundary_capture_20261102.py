"""Minimal end-of-backward witness. No forward hooks or activation retention.

Only snapshot() promises no mutation. capture_clip() intentionally invokes the
unchanged clipping primitive once, then observes its results. FP64 references
are computed from CPU copies, not inserted into the scientific update.
"""
import copy,math
import torch

def snapshot(named_parameters):
 return [(n,None if p.grad is None else p.grad.detach().cpu().clone()) for n,p in named_parameters]

def norms(raw):
 per=[]
 for name,g in raw:
  if g is None:per.append(dict(name=name,missing=True));continue
  v=g.double();sq=float(v.square().sum());per.append(dict(name=name,dtype=str(g.dtype),elements=g.numel(),L2_FP64=math.sqrt(sq),sum_squares_FP64=sq,L2_native_CPU=float(g.norm(2))))
 return dict(per_tensor=per,total_L2_FP64=math.sqrt(math.fsum(x['sum_squares_FP64'] for x in per if not x.get('missing'))),reference='FP64 CPU sum of squared copied raw gradients, math.fsum across tensors; not runtime GPU reduction')

class CaptureClip:
 def __init__(self,named_parameters,original):
  self.named=list(named_parameters);self.original=original;self.records=[]
 def __call__(self,parameters,max_norm,*args,**kwargs):
  ps=list(parameters);assert [id(p) for _,p in self.named]==[id(p) for p in ps]
  raw=snapshot(self.named)
  total=self.original(ps,max_norm,*args,**kwargs)
  # Recompute coefficient after the real clipping call; no scientific state is
  # altered. Formula/source are pinned to the executed Torch2.6 implementation.
  coefficient=(float(max_norm)/(total.detach()+1e-6)).clamp(max=1.)
  self.records.append(dict(raw=raw,postclip=snapshot(self.named),runtime_total_norm=total.detach().cpu().clone(),runtime_coefficient_formula=coefficient.cpu().clone(),max_norm=float(max_norm),norm_type=kwargs.get('norm_type',args[0] if args else 2.),foreach=kwargs.get('foreach',None)))
  return total
