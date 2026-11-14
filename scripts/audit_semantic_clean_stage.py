"""Read-only production guard/source audit; reuse prior costly gradient evidence."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);a=p.parse_args();run=ROOT/'results'/a.run_id
    prior=ROOT/'results/matched_feedback_20261028_v2/audit/rl_reliability.json';old=json.loads(prior.read_text());files=[]
    for r in old['source_files']:
        actual=hashlib.sha256((ROOT/r['path']).read_bytes().replace(b'\r\n',b'\n')).hexdigest();files.append({'path':r['path'],'prior_sha256':r['current_hash'],'current_sha256':actual,'unchanged':actual==r['current_hash']})
    from src.rl.phase5_backend import require_training_reward
    guards=[]
    for reward in [{},{'training_eligible':False},{'training_eligible':True}]:
        rejected=False
        try:require_training_reward(reward)
        except RuntimeError:rejected=True
        guards.append({'receipt':reward,'rejected':rejected})
    # A minimal independent witness of the checkpoint-harness bug and repair.
    import torch,copy
    p=torch.nn.Parameter(torch.ones(2));opt=torch.optim.AdamW([p]);p.grad=torch.ones(2);opt.step();checkpoint=copy.deepcopy(opt.state_dict());before=checkpoint['state'][0]['step'].item();opt.load_state_dict(checkpoint);p.grad=torch.ones(2);opt.step();alias=checkpoint['state'][0]['step'].item()!=before
    fixed=copy.deepcopy(opt.state_dict());fixed_before=fixed['state'][0]['step'].clone();opt.load_state_dict(copy.deepcopy(fixed));p.grad=torch.ones(2);opt.step();unchanged=torch.equal(fixed_before,fixed['state'][0]['step'])
    result={'status':'PASS' if all(f['unchanged'] for f in files) and all(g['rejected'] for g in guards) and alias and unchanged else 'FAIL','checked_unix':time.time(),'source_files':files,'guard_checks':guards,'prior_evidence_path':str(prior.relative_to(ROOT)),'prior_evidence_sha256':sha(prior),'prior_scoped_status':old['status'],'large_MC_rerun':False,'algorithms':old['algorithms'],'checkpoint_aliasing_witness':{'bug_reproduced':alias,'deepcopy_reference_unchanged':unchanged,'torch':torch.__version__},'full_pretrained_diffusion_continuation':'NOT VALIDATED; historical negative evidence unchanged','clean_target_training_scope':'Separate supervised original-VQA language LoRA; does not release generator rewards.'}
    target=run/'audit/rl_infrastructure_current.json'
    with target.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'production_files':len(files),'guards':guards,'aliasing_witness':result['checkpoint_aliasing_witness']}))
if __name__=='__main__':main()
