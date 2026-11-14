import json,types,unittest
import torch
from src.utils.scratch import scratch_directory
from src.rl.runtime_checkpoint_metadata import encode,decode,write_runtime_metadata,restore_runtime_metadata
class RuntimeMetadata(unittest.TestCase):
 def setup_trainer(self):
  x=torch.nn.Parameter(torch.tensor(1.));opt=torch.optim.AdamW([x],lr=2e-5);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
  return types.SimpleNamespace(optimizer=opt,scheduler=sched,policy=types.SimpleNamespace(config={'learning_rate':1e-5}))
 def test_infinite_comparison_sentinel_roundtrips(self):
  x={'mode_worse':float('-inf'),'best':.1};self.assertEqual(decode(encode(x)),x)
 def test_active_LR_and_ordinary_scheduler_restore(self):
  with scratch_directory() as p:
   (p/'training_state.pt').write_bytes(b'UNIT_FIXTURE_NOT_A_RESEARCH_MODEL');t=self.setup_trainer();outer=torch.optim.lr_scheduler.ReduceLROnPlateau(t.optimizer,mode='max');outer.step(.2)
   state=dict(optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),contract={'config':{'learning_rate':1e-5}})
   write_runtime_metadata(p,state,outer.state_dict());fresh=torch.optim.lr_scheduler.ReduceLROnPlateau(t.optimizer,mode='max');obj=restore_runtime_metadata(t,p,fresh)
   self.assertEqual(t.policy.config['learning_rate'],2e-5);self.assertEqual(fresh.state_dict(),outer.state_dict());self.assertFalse(obj['initial_contract_is_active_revision'])
 def test_wrong_bytes_rejected(self):
  with scratch_directory() as p:
   (p/'training_state.pt').write_bytes(b'fixture');t=self.setup_trainer();write_runtime_metadata(p,dict(optimizer=t.optimizer.state_dict(),scheduler=t.scheduler.state_dict(),contract={}))
   (p/'training_state.pt').write_bytes(b'changed')
   with self.assertRaises(ValueError):restore_runtime_metadata(t,p)
 def test_wrong_scheduler_base_rejected(self):
  with scratch_directory() as p:
   (p/'training_state.pt').write_bytes(b'fixture');t=self.setup_trainer();s=t.scheduler.state_dict();s['base_lrs']=[1e-5]
   with self.assertRaises(ValueError):write_runtime_metadata(p,dict(optimizer=t.optimizer.state_dict(),scheduler=s,contract={}))
if __name__=='__main__':unittest.main()
