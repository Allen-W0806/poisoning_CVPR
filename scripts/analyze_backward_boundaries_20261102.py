"""CPU operation-boundary, local-VJP and accumulation-order comparisons."""
import itertools,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.numerical_boundaries_20261102 import load,compare,put
from scripts.backward_boundary_capture_20261102 import decode

def main(out):
 import torch
 torch.set_num_threads(2);a=load(out/'worker0/witness.pt');b=load(out/'worker1/witness.pt');x=a['boundaries'];y=b['boundaries'];pairs={};events=[];order_audit=[]
 for k in ['before','conditioning']:pairs[k]=compare(a[k],b[k])
 rx={(r['parameter'],r['transition'],r['call']):r for r in x['records']};ry={(r['parameter'],r['transition'],r['call']):r for r in y['records']};assert set(rx)==set(ry)
 local_rows=[];forward_rows=[]
 for key in sorted(rx,key=lambda k:(k[1],k[2],k[0])):
  ra,rb=rx[key],ry[key];name,tr,call=key
  if name==x['target']:
   for field in ['input','output','grad_output']:
    if field not in ra or field not in rb:continue
    v=compare(decode(x,ra[field]),decode(y,rb[field]));forward_rows.append(dict(parameter=name,transition=tr,call=call,boundary=field,comparison=v,layout0=ra[field]['layout'],layout1=rb[field]['layout']))
    if not v['exact']:events.append(dict(transition=tr,call=call,parameter=name,boundary=field,max_abs=v['max_abs'],affected_elements=v['affected_elements'],relative_L2=v['relative_L2']))
  v=compare(decode(x,ra['local_weight_VJP']),decode(y,rb['local_weight_VJP']));local_rows.append(dict(parameter=name,transition=tr,call=call,node=ra['node'],comparison=v))
  if not v['exact']:events.append(dict(transition=tr,call=call,parameter=name,boundary='local_weight_VJP',max_abs=v['max_abs'],affected_elements=v['affected_elements'],relative_L2=v['relative_L2']))
 for name,vals in x['parameter_contributions'].items():
  other=y['parameter_contributions'][name];assert len(vals)==len(other)==20
  for va,vb in zip(vals,other):
   assert va['transition']==vb['transition'];tr=va['transition'];d=compare(decode(x,va['gradient']),decode(y,vb['gradient']))
   if not d['exact']:events.append(dict(transition=tr,parameter=name,boundary='parameter_contribution',max_abs=d['max_abs'],affected_elements=d['affected_elements'],relative_L2=d['relative_L2']))
   for payload,lookup,current,side in [(x,rx,va,0),(y,ry,vb,1)]:
    locals_=[decode(payload,r['local_weight_VJP']) for k,r in sorted(lookup.items()) if k[0]==name and k[1]==tr];assert len(locals_)==3;actual=decode(payload,current['gradient']);matches=[]
    for perm in itertools.permutations(range(3)):
     z=(locals_[perm[0]]+locals_[perm[1]])+locals_[perm[2]]
     if torch.equal(z,actual):matches.append(perm)
    order_audit.append(dict(parameter=name,transition=tr,worker=side,matching_FP32_three_branch_sum_orders=matches))
 accumulated=[]
 for va,vb in zip(x['accumulated'],y['accumulated']):
  assert va['transition']==vb['transition'];tr=va['transition']
  for name in va['gradients']:
   d=compare(decode(x,va['gradients'][name]),decode(y,vb['gradients'][name]));accumulated.append(dict(parameter=name,transition=tr,comparison=d))
   if not d['exact']:events.append(dict(transition=tr,parameter=name,boundary='accumulated_parameter_gradient',max_abs=d['max_abs'],affected_elements=d['affected_elements'],relative_L2=d['relative_L2']))
 pairs['raw']=compare(a['clip']['raw'],b['clip']['raw']);pairs['postclip']=compare(a['clip']['postclip'],b['clip']['postclip']);pairs['after']=compare(a['after'],b['after'])
 refs=[]
 # Independent local FP64 reference for the first target call and first target
 # unequal local VJP if one exists. It is not an exact FP32/CUDA oracle.
 chosen=[k for k in rx if k[0]==x['target'] and k[2]==0];chosen += [ (r['parameter'],r['transition'],r['call']) for r in local_rows if r['parameter']==x['target'] and not r['comparison']['exact']][:1]
 for key in dict.fromkeys(chosen):
  for payload,lookup,side in [(x,rx,0),(y,ry,1)]:
   r=lookup[key]
   if 'input' not in r:refs.append(dict(parameter=key[0],transition=key[1],call=key[2],worker=side,status='NOT CAPTURED outside first source'));continue
   inp=decode(payload,r['input']);go=decode(payload,r['grad_output']);fp64=go.reshape(-1,go.shape[-1]).double().T@inp.reshape(-1,inp.shape[-1]).double();actual=decode(payload,r['local_weight_VJP']);d=compare(fp64,actual.double());refs.append(dict(parameter=key[0],transition=key[1],call=key[2],worker=side,FP64_reference_error=d,scope='Reference rounding error; no requirement of FP32 equality to FP64'))
 rank={'input':0,'output':1,'grad_output':2,'local_weight_VJP':3,'parameter_contribution':4,'accumulated_parameter_gradient':5};events.sort(key=lambda r:(r['transition'],rank[r['boundary']],r.get('call',-1),r['parameter']))
 first=events[0] if events else None
 # Boundary ordering is causal only within the selected module/transition;
 # forward-order call IDs do not identify a CUDA timestamp or full network.
 result=dict(first_observed=first,events=events,comparisons=pairs,forward_target=forward_rows,local_VJP=local_rows,accumulated=accumulated,three_branch_sum_order_audit=order_audit,FP64_local_reference=refs,all_initial_exact=pairs['before']['exact'] and pairs['conditioning']['exact'],ordinary_uninstrumented=False,root_cause_established=False,source_fix='NONE',scope='Only selected to_q LoRA_A has full forward/input/upstream tensors. All32 have local weight VJP,per-transition contribution and accumulated gradients; uncaptured upstreams are NOT TESTED.')
 put(out/'boundary_analysis.json',result);summary=dict(first_observed=first,forward_input_unequal=sum(not r['comparison']['exact'] for r in forward_rows if r['boundary']=='input'),forward_output_unequal=sum(not r['comparison']['exact'] for r in forward_rows if r['boundary']=='output'),upstream_unequal=sum(not r['comparison']['exact'] for r in forward_rows if r['boundary']=='grad_output'),local_VJP_unequal=sum(not r['comparison']['exact'] for r in local_rows),raw_exact=pairs['raw']['exact'],raw_max_abs=pairs['raw']['max_abs'],initial_exact=result['all_initial_exact'],contribution_unequal=sum(r['boundary']=='parameter_contribution' for r in events),sum_order_cases_with_no_match=sum(not r['matching_FP32_three_branch_sum_orders'] for r in order_audit));put(out/'boundary_summary.json',summary);print(json.dumps(summary))
if __name__=='__main__':main(ROOT/'results/numerical_fix_20261102_v1'/sys.argv[1])
