import torch

def tensor_changes(before, after):
    differences = [(after[k]-v).abs() for k,v in before.items()]
    return {"max_abs_change": max(float(d.max()) for d in differences),
        "mean_abs_change": sum(float(d.double().sum()) for d in differences)/sum(d.numel() for d in differences),
        "updated_tensors": sum(bool(torch.any(d > 0)) for d in differences), "adapter_tensor_count": len(differences)}

def state_equal(a,b):
    import numpy as np
    if isinstance(a,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,torch.Tensor): return torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict): return a.keys()==b.keys() and all(state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(state_equal(x,y) for x,y in zip(a,b))
    return a==b
