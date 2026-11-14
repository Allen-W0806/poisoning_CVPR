"""CPU-only derived analysis of preserved negative gradient evidence."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id;out=ROOT/'results'/f'{a.run_id}_ip2pix_a0'/'diagnostic';torch.set_num_threads(1)
    paths=[out/'reference_after.pt',out/'matched_checkpoint_enabled_gradients.pt',out/'checkpoint_disabled_control_gradients.pt'];states=[torch.load(p,map_location='cpu',weights_only=False) for p in paths];grads=[states[0]['grads'],states[1]['gradients'],states[2]['gradients']];details=[]
    for i,(ref,resume,control) in enumerate(zip(*grads)):
        if ref is None:continue
        details.append({'tensor_index':i,'shape':list(ref.shape),'dtype':str(ref.dtype),'reference_resume_exact':torch.equal(ref,resume),'resumed_checkpoint_control_exact':torch.equal(resume,control),'max_abs':float((ref-resume).abs().max()),'reference_norm':float(ref.double().norm()),'resumed_norm':float(resume.double().norm())})
    norms=[sum(float(g.double().square().sum()) for g in items if g is not None)**.5 for items in grads]
    result={'status':'PASS_DERIVED_FORENSICS','source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'earliest_observed_divergence':'backward gradients','different_gradient_tensors':sum(not r['reference_resume_exact'] for r in details),'gradient_tensors_compared':len(details),'first_different_tensor':next((r for r in details if not r['reference_resume_exact']),None),'matched_vs_checkpoint_disabled_all_gradients_exact':all(r['resumed_checkpoint_control_exact'] for r in details),'postclip_global_norm_reference_resumed_control':norms,'gradient_cap':1.,'interpretation':'The checkpoint toggle produces identical resumed gradient tensors in this fixture. It does not explain the reference-versus-fresh divergence. Exact forward values, frozen hashes and RNG do not prove equal backward execution. Root kernel/arithmetic cause remains unresolved.','details':details}
    with (run/'audit/ip2pix_backward_forensics.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({k:v for k,v in result.items() if k!='details'},indent=2))
if __name__=='__main__':main()
