import json
import torch
import _bootstrap

if __name__=="__main__":
    import sys
    root=_bootstrap.ROOT/"experiments"/sys.argv[1]
    a=torch.load(root/"checkpoints/step_0001/training_state.pt",weights_only=False,map_location="cpu")
    b=torch.load(root/"checkpoints/restarted_final/training_state.pt",weights_only=False,map_location="cpu")
    zero=torch.load(root/"checkpoints/step_0000/training_state.pt",weights_only=False,map_location="cpu")
    rows=[]
    for i,(key,value) in enumerate(a["adapter"].items()):
        d=float((value-b["adapter"][key]).abs().max())
        oa=a["optimizer"]["state"][i]; ob=b["optimizer"]["state"][i]
        rows.append({"key":key,"parameter_difference":d,"dtype":str(value.dtype),
            "original_change":float((value-zero["adapter"][key]).abs().max()),
            "moment_diff":float((oa["exp_avg"]-ob["exp_avg"]).abs().max()),
            "moment_absmax":float(oa["exp_avg"].abs().max())})
    result={"largest":sorted(rows,key=lambda r:r["parameter_difference"],reverse=True)[:12],
        "total_different":sum(r["parameter_difference"]>0 for r in rows)}
    gp=root/"checkpoints/gradients_1.pt"
    if gp.is_file():
        ga=torch.load(gp,weights_only=True); gb=torch.load(root/"checkpoints/restarted_gradients.pt",weights_only=True)
        result["gradients"]=[{"index":i,"max_difference":float((x-y).abs().max()),"original_max":float(x.abs().max()),"restart_max":float(y.abs().max())} for i,(x,y) in enumerate(zip(ga,gb)) if x is not None and not torch.equal(x,y)]
        result["gradients"]=sorted(result["gradients"],key=lambda r:r["max_difference"],reverse=True)[:10]
    print(json.dumps(result,indent=2))
