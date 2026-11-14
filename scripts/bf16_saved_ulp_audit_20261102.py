"""New CPU-only ULP/sign/relative-error diagnosis of preserved raw gradients.

No model construction, generation, backward, optimization, clipping replay or
GPU call. Bitwise acceptance stays unchanged. Distributional evidence cannot
identify an unobserved upstream adjoint or a CUDA primitive.
"""
import argparse, csv, hashlib, json, os
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main(root,out):
    import numpy as np
    import torch
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    torch.set_num_threads(2)
    parent=root/'results/overnight_supervisor_20261101_v1_P1_ABBA_v1_a0'
    inputs=[parent/f'{i}_instrumented/gradients.pt' for i in [1,2]]
    names=json.loads((root/'results/method_code_audit_20261101T0510Z_v1/generator_ownership.json').read_text())['parameter_order']
    out.mkdir(exist_ok=False)
    protocol=dict(scope='CPU ULP/sign diagnostic of saved 88099 raw gradients',inputs={str(p.relative_to(root)):sha(p) for p in inputs},acceptance='bitwise exact for numerical reproducibility; never relaxed',quantiles=[0.,.5,.9,.99,1.],scientific_release=False,new_model_calls=0,new_optimizer_steps=0,GPU_seconds=0,questions=['Are mismatches one FP32 ULP or larger?', 'Do they create sign changes or finite/nonfinite changes?', 'What is the L2/direction perturbation before clipping?'],limitations='Cannot distinguish upstream adjoint/recompute/local reduction/leaf accumulation without paired captures; not a numerical fix')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    gs=[torch.load(p,map_location='cpu',weights_only=False)['raw_gradients'][0] for p in inputs]
    rows=[];dot=aa=bb=dd=0.; total_changed=0;all_ulp=[];all_relative=[]
    for name,a,b in zip(names,*gs):
        assert a.dtype==b.dtype==torch.float32 and a.shape==b.shape
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        av=a.numpy().reshape(-1);bv=b.numpy().reshape(-1);ad=av.astype(np.float64);bd=bv.astype(np.float64);diff=ad-bd;changed=av!=bv;n=int(changed.sum());total_changed+=n
        # Monotone float32 integer ordering, including negative values and +/-0.
        def ordered(v):
            u=v.view(np.uint32).astype(np.int64)
            return np.where((u & 0x80000000)!=0,0x80000000-(u & 0x7fffffff),0x80000000+u)
        ulp=np.abs(ordered(av)-ordered(bv))[changed]
        rel=np.abs(diff[changed])/np.maximum(np.abs(ad[changed]),np.abs(bd[changed])) if n else np.array([])
        if n:all_ulp.append(ulp);all_relative.append(rel)
        dot+=float(ad@bd);aa+=float(ad@ad);bb+=float(bd@bd);dd+=float(diff@diff)
        rows.append(dict(parameter=name,elements=a.numel(),changed_elements=n,bitwise_equal=torch.equal(a,b),sign_flips=int(((av*bv<0)&changed).sum()),one_zero=int(((av==0)^(bv==0)).sum()),max_abs=float(np.abs(diff).max()),max_FP32_ULP=int(ulp.max()) if n else 0,ULP_gt1=int((ulp>1).sum()),relative_error_quantiles=np.quantile(rel,protocol['quantiles']).tolist() if n else [],FP32_ULP_quantiles=np.quantile(ulp,protocol['quantiles']).tolist() if n else []))
    u=np.concatenate(all_ulp);rel=np.concatenate(all_relative)
    result=dict(status='NUMERICAL_MISMATCH_PRESERVED',tensors=len(rows),different_tensors=sum(not x['bitwise_equal'] for x in rows),elements=sum(x['elements'] for x in rows),changed_elements=total_changed,sign_flips=sum(x['sign_flips'] for x in rows),one_zero=sum(x['one_zero'] for x in rows),FP32_ULP_quantiles=np.quantile(u,protocol['quantiles']).tolist(),relative_error_quantiles=np.quantile(rel,protocol['quantiles']).tolist(),ULP_gt1=int((u>1).sum()),relative_L2=(dd/max(aa,bb))**.5,raw_gradient_cosine=dot/(aa*bb)**.5,new_model_calls=0,new_optimizer_steps=0,new_GPU_seconds=0,numerical_fix=False,root_CUDA_primitive='UNRESOLVED',ordinary_training_release=False,limitation=protocol['limitations'])
    assert all(sha(p)==protocol['inputs'][str(p.relative_to(root))] for p in inputs)
    assert not torch.cuda.is_initialized()
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    with (out/'per_tensor.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.root,a.out)
