"""New CPU unit checks of node-hook math and lossless storage, no model smoke."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from scripts.backward_boundary_capture_20261102 import TensorStore,linear_mm_node,BoundaryCapture,tap_sdpa

class Capture(unittest.TestCase):
 def test_lossless_BF16_pack(self):
  x=torch.tensor([.5,1.25,-2.],dtype=torch.float32);s=TensorStore();r=s.add(x);self.assertEqual(s.blobs[r['ref']]['value'].dtype,torch.bfloat16);self.assertTrue(torch.equal(x,s.restore(r)))
 def test_general_FP32_never_quantized(self):
  x=torch.tensor([.123456789]);s=TensorStore();r=s.add(x);self.assertEqual(s.blobs[r['ref']]['value'].dtype,torch.float32);self.assertTrue(torch.equal(x,s.restore(r)))
 def test_dedup_and_no_alias(self):
  x=torch.tensor([1.,2.]);s=TensorStore();a=s.add(x);b=s.add(x);x.zero_();self.assertEqual(a['ref'],b['ref']);self.assertEqual(len(s.blobs),1);self.assertEqual(float(s.restore(a).sum()),3.)
 def test_node_hook_exposes_local_VJP(self):
  x=torch.arange(24.).reshape(1,6,4);w=torch.nn.Parameter(torch.tensor([[.1,.2,.3,.4],[.5,.6,.7,.8]]));rng=torch.get_rng_state();y=torch.nn.functional.linear(x,w);node,slot=linear_mm_node(y);seen={}
  def hook(gi,go):seen['g']=gi[slot].t().detach().clone();seen['go']=go[0].detach().clone()
  h=node.register_hook(hook);y.sum().backward();h.remove();expected=torch.ones(6,2).t()@x.reshape(6,4);self.assertTrue(torch.equal(seen['g'],expected));self.assertTrue(torch.equal(w.grad,expected));self.assertTrue(torch.equal(rng,torch.get_rng_state()))
 def test_stride_metadata_not_lost(self):
  x=torch.arange(12.).reshape(3,4).t();s=TensorStore();r=s.add(x);self.assertEqual(r['layout']['stride'],list(x.stride()));self.assertTrue(torch.equal(s.restore(r),x))
 def test_forward_retention_stops_after_first_source(self):
  c=BoundaryCapture.__new__(BoundaryCapture);c.store=TensorStore();c.target='target';c.full_targets={'target'};first=dict(parameter='target',transition=0);later=dict(parameter='target',transition=2);x=torch.ones(1,4,3);y=torch.ones(1,4,2);c.capture_forward(first,x,y);count=len(c.store.blobs);c.capture_forward(later,x*2,y*2);self.assertIn('input',first);self.assertNotIn('input',later);self.assertEqual(len(c.store.blobs),count)
 def test_SDPA_hooks_do_not_change_gradients(self):
  torch.manual_seed(7);q=torch.randn(1,2,3,4,requires_grad=True);k=torch.randn(1,2,3,4,requires_grad=True);v=torch.randn(1,2,3,4,requires_grad=True);out=torch.nn.functional.scaled_dot_product_attention(q,k,v);store=TensorStore();r={};tap_sdpa(store,r,q,k,v,out);out.sum().backward();self.assertTrue(torch.equal(store.restore(r['dq']),q.grad));self.assertTrue(torch.equal(store.restore(r['dk']),k.grad));self.assertTrue(torch.equal(store.restore(r['dv']),v.grad))
if __name__=='__main__':unittest.main()
