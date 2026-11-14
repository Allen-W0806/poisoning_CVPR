import copy
import io
import unittest
import numpy as np
import torch
from src.evaluation.phase4_statistics import paired_summary,rsi_plateau

class Phase4Tests(unittest.TestCase):
    def test_active_wrapper_uses_child_progress(self):
        import json,os,time
        from tests.test_monitoring import fixture
        from src.utils.scratch import scratch_directory
        from src.monitoring.diagnostics import inspect_run
        with scratch_directory() as root:
            parent=fixture(root,'EXP_wrapper')
            (parent/'job_submission.json').write_text(json.dumps({'job_id':'11'}))
            child=fixture(root,'EXP_child')
            manifest=json.loads((child/'manifest.json').read_text());manifest['parent_experiment']=parent.name
            (child/'manifest.json').write_text(json.dumps(manifest))
            old=time.time()-1900
            for name in ['run.log','metrics.json']:os.utime(parent/name,(old,old))
            self.assertNotIn('stale_progress',[f['code'] for f in inspect_run(parent,{'11':{'State':'RUNNING'}})['findings']])
            for name in ['run.log','metrics.json']:os.utime(child/name,(old,old))
            self.assertIn('stale_progress',[f['code'] for f in inspect_run(parent,{'11':{'State':'RUNNING'}})['findings']])

    def test_monitor_recognizes_frozen_reference_without_training(self):
        from tests.test_monitoring import fixture
        from src.utils.scratch import scratch_directory
        from src.monitoring.diagnostics import inspect_run
        with scratch_directory() as root:
            path=fixture(root,metrics={'generator_training_kind':'frozen_reference','updates':[],'frozen_weights_unchanged':True})
            self.assertNotIn('unrecognized_generator_objective',[x['code'] for x in inspect_run(path)['findings']])
            import json
            (path/'metrics.json').write_text(json.dumps({'generator_training_kind':'frozen_reference','updates':[{'step':1}],'frozen_weights_unchanged':True}))
            self.assertIn('invalid_frozen_reference',[x['code'] for x in inspect_run(path)['findings']])

    def test_adamw_accumulation_exact_restart(self):
        torch.manual_seed(9);layer=torch.nn.Linear(8,4)
        opt=torch.optim.AdamW(layer.parameters(),lr=1e-5,weight_decay=0)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
        batches=[(torch.randn(1,8),torch.randn(1,4)) for _ in range(10)]
        def update(model,optimizer,sched):
            optimizer.zero_grad()
            for x,y in batches:((model(x)-y).square().mean()/10).backward()
            gradients=[p.grad.clone() for p in model.parameters()]
            optimizer.step();sched.step();return gradients
        update(layer,opt,scheduler)
        buf=io.BytesIO();torch.save({'model':layer.state_dict(),'opt':opt.state_dict(),'scheduler':scheduler.state_dict(),'rng':torch.get_rng_state()},buf)
        grads=update(layer,opt,scheduler);expected=copy.deepcopy(layer.state_dict())
        buf.seek(0);state=torch.load(buf,weights_only=True)
        fresh=torch.nn.Linear(8,4);fresh.load_state_dict(state['model'])
        freshopt=torch.optim.AdamW(fresh.parameters(),lr=1e-5,weight_decay=0);freshopt.load_state_dict(state['opt'])
        freshscheduler=torch.optim.lr_scheduler.LambdaLR(freshopt,lambda _:1);freshscheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['rng']);replay=update(fresh,freshopt,freshscheduler)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(grads,replay)))
        self.assertTrue(all(torch.equal(expected[k],v) for k,v in fresh.state_dict().items()))
        self.assertEqual(scheduler.state_dict(),freshscheduler.state_dict())

    def test_accumulation_normalization(self):
        p=torch.tensor(.3,requires_grad=True)
        for i in range(10):((p-(i/10))**2/10).backward()
        self.assertAlmostEqual(float(p.grad),2*(.3-.45),places=6)

    def test_paired_bootstrap_and_mismatch(self):
        base=[{'sample_id':i,'seed':i+1,'reward':{'reward':0}} for i in range(10)]
        current=copy.deepcopy(base)
        for x in current:x['reward']['reward']=.1
        result=paired_summary(current,base)
        np.testing.assert_allclose(result['ci95'],[.1,.1]);self.assertFalse(rsi_plateau([result,result]))
        current[0]['seed']=9
        with self.assertRaises(ValueError):paired_summary(current,base)

    def test_rsi_requires_two_unresolved_looks(self):
        x={'mean_delta':0,'ci_half_width':.01}
        self.assertFalse(rsi_plateau([x]));self.assertTrue(rsi_plateau([x,x]))

    def test_existing_controller_schedule_extension(self):
        from src.rsi.phase4_schedule import Phase4ScheduleController
        from src.utils.scratch import scratch_directory
        with scratch_directory() as root:
            controller=Phase4ScheduleController(root/'rsi',{'batch_size':10,'learning_rate':1e-5},-.5,'fixed',1.)
            look={'mean_delta':0,'ci_half_width':.01}
            first=controller.revise_schedule([look],-.5,.1,'fixed');self.assertFalse(first['accepted_schedule'])
            second=controller.revise_schedule([look,look],-.5,.2,'fixed');self.assertTrue(second['accepted_schedule'])
            self.assertFalse(second['quality_improvement_accepted']);self.assertEqual(controller.state['training_config']['learning_rate'],5e-6)
            self.assertFalse(controller.revise_schedule([look,look],-.5,.3,'fixed')['accepted_schedule'])
            with self.assertRaises(ValueError):controller.revise_schedule([look,look],-.5,.4,'changed')

if __name__=='__main__':unittest.main()
