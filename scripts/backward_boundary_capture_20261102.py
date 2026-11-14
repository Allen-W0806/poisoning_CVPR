"""Targeted autograd-node witnesses, with exact lossless CPU tensor storage.

No tensor/gradient is returned from a hook and no graph is replaced. Hooks and
CPU copies remain an intervention; they cannot release ordinary training.
"""
import hashlib,json
import torch

def layout(x):
 return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),contiguous=x.is_contiguous(),storage_offset=x.storage_offset(),alignment_mod256=x.data_ptr()%256)

class TensorStore:
 def __init__(self):self.blobs={}
 def add(self,x):
  y=x.detach().cpu().contiguous();h=hashlib.sha256(str((y.dtype,tuple(y.shape))).encode()+y.view(torch.uint8).numpy().tobytes()).hexdigest()
  if h not in self.blobs:
   z=y
   if y.dtype==torch.float32:
    small=y.to(torch.bfloat16)
    if torch.equal(y,small.float()):z=small
   self.blobs[h]=dict(value=z.clone(),original_dtype=str(y.dtype),shape=list(y.shape))
  return dict(ref=h,layout=layout(x))
 def restore(self,entry):
  r=self.blobs[entry['ref']];return r['value'].to(getattr(torch,r['original_dtype'].removeprefix('torch.')))

def linear_mm_node(output):
 pending=[(output.grad_fn,0)];visited=set()
 while pending:
  node,depth=pending.pop(0)
  if node is None or node in visited:continue
  visited.add(node);name=node.name()
  if name in {'MmBackward0','AddmmBackward0'}:return node,1 if name=='MmBackward0' else 2
  if depth<3:pending.extend((n,depth+1) for n,_ in node.next_functions)
 raise RuntimeError('Pinned local linear MmBackward node not found: '+str(output.grad_fn))

def tap_sdpa(store,record,q,k,v,output):
 for key,tensor in [('q',q),('k',k),('v',v),('output',output)]:record[key]=store.add(tensor)
 for key,tensor in [('dq',q),('dk',k),('dv',v),('grad_output',output)]:
  def observe(g,n=key):record[n]=store.add(g);return None
  tensor.register_hook(observe)

class BoundaryCapture:
 def __init__(self,unet,target,phase):
  self.store=TensorStore();self.records=[];self.parameter_contributions={};self.accumulated=[];self.handles=[];self.phase=phase;self.target=target;self.target_B=target.replace('.lora_A.','.lora_B.');self.full_targets={target,self.target_B};self.sdpa=[];self.in_attention=False;self.named=[(n,p) for n,p in unet.named_parameters() if p.requires_grad and n.startswith('down_blocks.0.attentions.')];assert len(self.named)==32
  attention=unet.get_submodule(target.split('.to_q.')[0]);self.original_sdpa=torch.nn.functional.scaled_dot_product_attention
  def enter(*_):self.in_attention=True
  def leave(*_):self.in_attention=False
  self.handles.append(attention.register_forward_pre_hook(enter));self.handles.append(attention.register_forward_hook(leave))
  def sdpa(q,k,v,*args,**kwargs):
   result=self.original_sdpa(q,k,v,*args,**kwargs)
   if self.in_attention and self.phase['transition']<2:
    row=dict(transition=self.phase['transition'],call=len(self.sdpa));tap_sdpa(self.store,row,q,k,v,result);self.sdpa.append(row)
   return result
  torch.nn.functional.scaled_dot_product_attention=sdpa
  for name,param in self.named:
   module=unet.get_submodule(name.removesuffix('.weight'));counter={'n':0}
   def forward(m,args,value,n=name,c=counter):
    call=c['n'];c['n']+=1;tr=self.phase['transition'];node,slot=linear_mm_node(value);entry=dict(parameter=n,call=call,transition=tr,CFG_branch=call%3,node=node.name(),weight=layout(m.weight),input_layout=layout(args[0]),output_layout=layout(value));self.records.append(entry)
    self.capture_forward(entry,args[0],value)
    def node_hook(grad_inputs,grad_outputs,e=entry,s=slot):
     if grad_inputs[s] is None:raise RuntimeError('Missing actual local weight VJP')
     e['local_weight_VJP']=self.store.add(grad_inputs[s].transpose(0,1));e['grad_output_layout']=layout(grad_outputs[0])
     if e['parameter']==self.target:e['grad_output']=self.store.add(grad_outputs[0])
     if e['parameter']==self.target_B and e['transition']<2:
      e['grad_output']=self.store.add(grad_outputs[0]);e['local_input_VJP']=self.store.add(grad_inputs[0])
     return None
    self.handles.append(node.register_hook(node_hook))
   self.handles.append(module.register_forward_hook(forward))
   def parameter_hook(g,n=name):
    self.parameter_contributions.setdefault(n,[]).append(dict(transition=self.phase['transition'],gradient=self.store.add(g)))
    return None
   self.handles.append(param.register_hook(parameter_hook))
 def after_backward(self):
  self.accumulated.append(dict(transition=self.phase['transition'],gradients={n:self.store.add(p.grad) for n,p in self.named}))
 def capture_forward(self,entry,input_tensor,output_tensor):
  # Full values only for the first source's two transitions. Later upstream
  # adjoints/local weight VJPs remain captured; missing forwards stay explicit.
  if entry['parameter'] in self.full_targets and entry['transition']<2:
   entry['input']=self.store.add(input_tensor);entry['output']=self.store.add(output_tensor)
 def export(self):
  return dict(target=self.target,target_B=self.target_B,records=self.records,sdpa=self.sdpa,parameter_contributions=self.parameter_contributions,accumulated=self.accumulated,blobs=self.store.blobs,storage_bytes=sum(v['value'].numel()*v['value'].element_size() for v in self.store.blobs.values()))
 def close(self):
  torch.nn.functional.scaled_dot_product_attention=self.original_sdpa
  for h in self.handles:h.remove()

def decode(payload,entry):
 r=payload['blobs'][entry['ref']];return r['value'].to(getattr(torch,r['original_dtype'].removeprefix('torch.')))
