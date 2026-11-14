import json
from pathlib import Path
import unittest
from unittest.mock import patch
from datetime import datetime, timezone, timedelta
import yaml
from src.utils.scratch import scratch_directory
from src.monitoring.common import FileLock, read_json
from src.monitoring.core import run_monitor, load_config, is_new_failure, is_new_completion
from src.monitoring.diagnostics import inspect_run, verify_artifacts
from src.monitoring.integrity import check_integrity, check_associations
from src.monitoring.recovery import Recovery
from src.monitoring.scheduler import Scheduler

ROOT=Path(__file__).resolve().parents[1]
CONFIG=load_config(ROOT)
SNAPSHOT={'queue':{'exit_code':0,'stdout':''},'accounting':{'exit_code':0,'stdout':''},'jobs':{}}


def fixture(root, name='EXP_fixture', status='RUNNING', job='11', metrics=None, register=True):
    path=root/'experiments'/name
    path.mkdir(parents=True)
    for directory in ['images','checkpoints']:
        (path/directory).mkdir()
    files={'manifest.json':{'experiment_id':name,'created_at':'fixture','slurm_job_id':job,'git_commit':'fixture','source_tree_clean_at_creation':True},
        'status.json':{'status':status},'metrics.json':{'loss':1.0} if metrics is None else metrics}
    for filename,value in files.items():
        (path/filename).write_text(json.dumps(value))
    (path/'config.yaml').write_text(yaml.safe_dump({'effective_batch_size':10,'learning_rate':1e-5,'objective':'fixed benign fixture'}))
    for filename in ['environment.txt','git_commit.txt','run.log','train.log','summary.md']:
        (path/filename).write_text('')
    (root/'reports').mkdir(exist_ok=True)
    if register:
        with (root/'experiments/registry.jsonl').open('a') as out:
            out.write(json.dumps({'experiment_id':name,'path':str(path)})+'\n')
    return path


class FakeSlurm:
    def __init__(self):
        self.jobs=[]
        self.submissions=0
        self.cancelled=[]

    def __call__(self,args,**kwargs):
        result={'exit_code':0,'stdout':'','stderr':''}
        if args[0]=='squeue':
            result['stdout']='\n'.join('|'.join(j) for j in self.jobs)
        elif args[0]=='sbatch':
            self.submissions+=1
            job=str(1000+self.submissions)
            self.jobs.append([job,args[args.index('--job-name')+1],args[args.index('--comment')+1],'PENDING',args[args.index('--begin')+1]])
            result['stdout']=job+'\n'
            assert '--gres' not in args
        elif args[0]=='scancel':
            self.cancelled.append(args[1])
            self.jobs=[j for j in self.jobs if j[0]!=args[1]]
        else:
            raise AssertionError('Unexpected command '+str(args))
        return result


class MonitoringTest(unittest.TestCase):
    def test_missing_output_directory_repaired_without_config_change(self):
        with scratch_directory() as root:
            path=fixture(root)
            (path/'images').rmdir()
            before=(path/'config.yaml').read_bytes()
            status=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
            self.assertTrue((path/'images').is_dir())
            self.assertEqual(before,(path/'config.yaml').read_bytes())
            self.assertEqual(status['new_experiment_ids'],[])
            self.assertEqual(status['automatic_fixes'],1)

    def test_invalid_checkpoint_escalates_without_fake_file(self):
        with scratch_directory() as root:
            path=fixture(root,'EXP_fixture_rl_rsi_validation',status='PASS',metrics={
                'revision':{'accepted':False,'previous_config':{'learning_rate':1e-5},'proposed_config':{'learning_rate':5e-5},
                'rollback':{'checkpoint':'checkpoints/missing.pt'},'proposal_reason':'fixture','evidence':{'fixture':True}}})
            (path/'rsi').mkdir()
            (path/'rsi/state.json').write_text(json.dumps({'training_config':{'learning_rate':1e-5}}))
            findings=check_integrity(root,[inspect_run(path)],{},0)['findings']
            problem=next(f for f in findings if f['code']=='invalid_checkpoint_path')
            event=Recovery(root,CONFIG,{}).apply(problem)
            self.assertEqual(event['action'],'BLOCKED')
            self.assertFalse((path/'checkpoints/missing.pt').exists())

    def test_terminal_running_reconciled_but_not_promoted_to_pass(self):
        with scratch_directory() as root:
            path=fixture(root)
            metrics=(path/'metrics.json').read_bytes()
            row={'11':{'State':'COMPLETED','ExitCode':'0:0'}}
            problem=next(f for f in inspect_run(path,row)['findings'] if f['code']=='stale_status')
            Recovery(root,CONFIG,{}).apply(problem)
            self.assertEqual(read_json(path/'status.json')['status'],'COMPLETED_UNVERIFIED')
            self.assertEqual(metrics,(path/'metrics.json').read_bytes())
            self.assertTrue((root/'experiments/state_observations.jsonl').exists())

    def test_duplicate_ids_are_not_deleted_or_rewritten(self):
        with scratch_directory() as root:
            fixture(root,status='PASS')
            registry=root/'experiments/registry.jsonl'
            registry.write_bytes(registry.read_bytes()*2)
            before=registry.read_bytes()
            status=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
            report=read_json(status['latest_json'])
            self.assertIn('duplicate_experiment_id',[f['code'] for f in report['detected_problems']])
            self.assertEqual(before,registry.read_bytes())

    def test_oom_is_diagnosed_without_resource_or_objective_change(self):
        with scratch_directory() as root:
            path=fixture(root,status='FAIL')
            (path/'slurm-11.err').write_text('torch.OutOfMemoryError: CUDA out of memory')
            before=(path/'config.yaml').read_bytes()
            status=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
            report=read_json(status['latest_json'])
            self.assertIn('oom',[f['code'] for f in report['detected_problems']])
            self.assertEqual(before,(path/'config.yaml').read_bytes())
            self.assertEqual(read_json(path/'status.json')['status'],'FAIL')
            self.assertEqual(status['new_experiment_ids'],[])

    def test_nonfinite_metrics_report_is_valid_json(self):
        with scratch_directory() as root:
            fixture(root,status='PASS',metrics={'accuracy_percent':float('nan')})
            status=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
            report=json.loads(Path(status['latest_json']).read_text(),parse_constant=lambda s:(_ for _ in ()).throw(ValueError(s)))
            self.assertIn('nonfinite_metrics',[f['code'] for f in report['detected_problems']])
            self.assertEqual(report['runs'][0]['metrics']['accuracy_percent'],'NONFINITE:nan')

    def test_concurrent_monitor_is_skipped(self):
        with scratch_directory() as root:
            with FileLock(root/'reports/monitoring/monitor.lock') as acquired:
                self.assertTrue(acquired)
                result=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
                self.assertEqual(result['status'],'SKIPPED_LOCKED')
            self.assertFalse((root/'reports/MONITORING_STATUS.json').exists())

    def test_pending_is_normal_no_retry(self):
        with scratch_directory() as root:
            fixture(root,status='SUBMITTED')
            snapshot={**SNAPSHOT,'jobs':{'11':{'State':'PENDING'}}}
            result=run_monitor(root,snapshot=snapshot,config=CONFIG,scientific_checks=False)
            report=read_json(result['latest_json'])
            self.assertEqual(len(report['active_jobs']),1)
            self.assertEqual(result['automatic_fixes'],0)
            self.assertEqual(result['new_experiment_ids'],[])

    def test_missing_registry_is_append_only_repair(self):
        with scratch_directory() as root:
            path=fixture(root,status='PASS',register=False)
            before=(path/'metrics.json').read_bytes()
            result=run_monitor(root,snapshot=SNAPSHOT,config=CONFIG,scientific_checks=False)
            self.assertEqual(result['automatic_fixes'],1)
            self.assertEqual(before,(path/'metrics.json').read_bytes())
            self.assertEqual(len((root/'experiments/registry.jsonl').read_text().splitlines()),1)

    def test_finalized_changes_remain_visible_on_every_check(self):
        with scratch_directory() as root:
            path=fixture(root,status='PASS')
            run=inspect_run(path)
            cache={}
            verify_artifacts(root,run,cache)
            (path/'metrics.json').write_text('{"loss":2}')
            for _ in range(2):
                check=verify_artifacts(root,run,cache)
                self.assertIn('finalized_artifact_changed',[f['code'] for f in check['findings']])

    def test_retry_limit_follows_lineage_and_is_durable(self):
        with scratch_directory() as root:
            path=fixture(root,status='FAIL')
            records=[{'job_id':'11','task':'training','reserved_seconds':600}]
            (root/'reports/phase2_jobs.jsonl').write_text(json.dumps(records[0])+'\n')
            state={}
            calls=[]
            def submit(args,**kwargs):
                calls.append(args)
                return {'exit_code':0,'stdout':json.dumps({'job_id':str(11+len(calls)),'parent_run':'EXP_retry'+str(len(calls))}), 'stderr':''}
            recovery=Recovery(root,CONFIG,state,submit)
            run=inspect_run(path)
            def source_command(args,**kwargs):
                return {'stdout':'fixture' if 'rev-parse' in args else '', 'exit_code':0,'stderr':''}
            with patch('src.monitoring.recovery.command',side_effect=source_command):
                first=recovery.apply({'code':'preempted','experiment_id':path.name},run,True)
                self.assertEqual(first['attempt'],1)
                run['job_id']='12'
                with (root/'reports/phase2_jobs.jsonl').open('a') as out:
                    out.write(json.dumps({'job_id':'12','task':'training','reserved_seconds':600})+'\n')
                second=recovery.apply({'code':'preempted','experiment_id':path.name},run,True)
                self.assertEqual(second['attempt'],2)
                run['job_id']='13'
                last=recovery.apply({'code':'node_failure','experiment_id':path.name},run,True)
            self.assertEqual(last['reason'],'retry_limit_reached')
            self.assertEqual(len(calls),2)
            self.assertEqual(read_json(root/'reports/monitoring/state.json')['retry_counts']['11:preempted'],2)

    def test_scheduler_idempotence_tick_and_foreign_job_protection(self):
        with scratch_directory() as root:
            fake=FakeSlurm()
            fake.jobs.append(['99','unrelated','other-project','PENDING','Unknown'])
            scheduler=Scheduler(root,CONFIG,fake)
            first=scheduler.install()
            self.assertEqual(scheduler.install()['next_job_id'],first['next_job_id'])
            self.assertEqual(fake.submissions,1)
            due=first['next_due_utc']
            with patch.dict('os.environ',{'SLURM_JOB_ID':first['next_job_id']}):
                next(j for j in fake.jobs if j[0]==first['next_job_id'])[3]='RUNNING'
                result=scheduler.tick(due,lambda:{'status':'PASS'})
            self.assertEqual(result['status'],'PASS')
            self.assertEqual(fake.submissions,2)
            self.assertEqual(scheduler.install()['next_job_id'],'1002')
            self.assertEqual(fake.submissions,2)
            future=read_json(scheduler.path)
            self.assertEqual(datetime.fromisoformat(future['next_due_utc'])-datetime.fromisoformat(due),timedelta(hours=2))
            scheduler.stop()
            self.assertEqual([j[0] for j in fake.jobs],['99'])
            self.assertNotIn('99',fake.cancelled)

    def test_uncertain_submission_does_not_loop(self):
        with scratch_directory() as root:
            fake=FakeSlurm()
            def runner(args,**kwargs):
                if args[0]=='sbatch':
                    return {'exit_code':-1,'stdout':'','stderr':'timeout'}
                return fake(args,**kwargs)
            scheduler=Scheduler(root,CONFIG,runner)
            with self.assertRaises(RuntimeError):scheduler.install()
            with self.assertRaises(RuntimeError):scheduler.install()
            self.assertTrue(read_json(scheduler.path)['submission_uncertain'])

    def test_active_artifact_promotes_to_immutable_on_finalization(self):
        with scratch_directory() as root:
            path=fixture(root)
            artifact=path/'images/source.png'
            from PIL import Image
            Image.new('RGB',(8,8),'red').save(artifact)
            cache={}
            run=inspect_run(path)
            verify_artifacts(root,run,cache)
            run['status']='PASS'
            verify_artifacts(root,run,cache)
            Image.new('RGB',(8,8),'blue').save(artifact)
            self.assertIn('finalized_artifact_changed',[f['code'] for f in verify_artifacts(root,run,cache)['findings']])

    def test_submitted_job_overrides_monitor_launcher_allocation(self):
        with scratch_directory() as root:
            path=fixture(root,job='900')
            (path/'job_submission.json').write_text(json.dumps({'job_id':'901'}))
            run=inspect_run(path,{'900':{'State':'COMPLETED','ExitCode':'0:0'},'901':{'State':'PENDING'}})
            self.assertEqual(run['job_id'],'901')
            self.assertNotIn('stale_status',[f['code'] for f in run['findings']])

    def test_failure_starting_and_finishing_between_checks_is_new(self):
        run={'id':'EXP_new','status':'FAIL','scheduler_state':'PREEMPTED','manifest':{'created_at':'2026-10-27T06:30:00+00:00'}}
        self.assertTrue(is_new_failure(run,{},False,'2026-10-27T06:00:00+00:00'))
        self.assertTrue(is_new_completion(run,{},False,'2026-10-27T06:00:00+00:00'))
        self.assertFalse(is_new_failure(run,{},True,None))
        self.assertFalse(is_new_failure(run,{'EXP_new':{'status':'FAIL'}},False,'2026-10-27T06:00:00+00:00'))

    def test_dirty_same_commit_source_cannot_restart(self):
        with scratch_directory() as root:
            path=fixture(root,status='FAIL')
            calls=[]
            recovery=Recovery(root,CONFIG,{},lambda *args,**kwargs:calls.append(args))
            def dirty_command(args,**kwargs):
                return {'stdout':'fixture' if 'rev-parse' in args else ' M scripts/smoke_generator_training.py', 'exit_code':0,'stderr':''}
            with patch('src.monitoring.recovery.command',side_effect=dirty_command):
                result=recovery.apply({'code':'preempted','experiment_id':path.name},inspect_run(path),True)
            self.assertIn('dirty',result['reason'])
            self.assertEqual(calls,[])

    def test_editor_source_pixels_and_output_paths_are_associated(self):
        with scratch_directory() as root:
            from PIL import Image
            raw=root/'raw.png'; source=root/'source.png'; output=root/'edited.png'
            Image.new('RGB',(8,8),'red').save(raw)
            Image.new('RGB',(8,8),'blue').save(source)
            Image.new('RGB',(8,8),'green').save(output)
            row={'image_id':'1','image_path':str(raw),'question':'What color?','answers':['red']}
            item={'sample_id':'2','source_image_id':'1','source_path':str(source),'output_path':str(output),
                  'question':row['question'],'answers':row['answers']}
            result=check_associations(root,'EXP_fixture',[item],{'2':row},expected_resolution=8)
            self.assertIn('editor_source_output_association',[f['code'] for f in result['findings']])
            Image.new('RGB',(8,8),'red').save(source)
            self.assertEqual(check_associations(root,'EXP_fixture',[item],{'2':row},expected_resolution=8)['checked_pairs'],1)
            output.unlink()
            self.assertEqual(check_associations(root,'EXP_fixture',[item],{'2':row},expected_resolution=8)['checked_pairs'],0)

    def test_recorded_resolution_cannot_be_silently_reduced(self):
        with scratch_directory() as root:
            from PIL import Image
            for name in ['raw','source','edited']:
                Image.new('RGB',(256,256),'red').save(root/(name+'.png'))
            row={'image_id':'1','image_path':str(root/'raw.png'),'question':'Color?','answers':['red']}
            item={'sample_id':'2','source_image_id':'1','source_path':str(root/'source.png'),'output_path':str(root/'edited.png'),
                'question':'Color?','answers':['red'],'settings':{'image_resolution':512}}
            result=check_associations(root,'EXP_fixture',[item],{'2':row})
            self.assertEqual(result['checked_pairs'],0)
            self.assertIn('resolution',result['findings'][0]['reason'])

    def test_accounting_outage_preserves_unattempted_failure_eligibility(self):
        with scratch_directory() as root:
            path=fixture(root)
            run_monitor(root,snapshot={**SNAPSHOT,'jobs':{'11':{'State':'RUNNING'}}},config=CONFIG,scientific_checks=False)
            (path/'status.json').write_text('{"status":"FAIL"}')
            (path/'slurm-11.err').write_text('Stale file handle')
            outage={'queue':{'exit_code':-1},'accounting':{'exit_code':-1},'jobs':{}}
            calls=[]
            def retry(instance,run,finding,permit_compute):
                calls.append(permit_compute)
                return {'action':'restart_attempt' if permit_compute else 'BLOCKED','experiment_id':path.name,'new_experiment_ids':[]}
            with patch('src.monitoring.recovery.Recovery.retry',autospec=True,side_effect=retry):
                run_monitor(root,snapshot=outage,config=CONFIG,scientific_checks=False)
                self.assertIn('11',read_json(root/'reports/monitoring/state.json')['pending_failure_jobs'])
                recovered={**SNAPSHOT,'jobs':{'11':{'State':'NODE_FAIL','ExitCode':'1:0'}}}
                run_monitor(root,snapshot=recovered,config=CONFIG,scientific_checks=False)
            self.assertEqual(calls,[False,True])
            self.assertNotIn('11',read_json(root/'reports/monitoring/state.json')['pending_failure_jobs'])

    def test_skipped_tick_is_not_success_and_failed_tick_is_reported(self):
        with scratch_directory() as root:
            fake=FakeSlurm(); scheduler=Scheduler(root,CONFIG,fake)
            first=scheduler.install()
            next(j for j in fake.jobs if j[0]==first['next_job_id'])[3]='RUNNING'
            with patch.dict('os.environ',{'SLURM_JOB_ID':first['next_job_id']}):
                scheduler.tick(first['next_due_utc'],lambda:{'status':'SKIPPED_LOCKED'})
            self.assertNotIn('last_success_utc',read_json(scheduler.path))
            state=read_json(scheduler.path)
            second=state['next_job_id']
            fake.jobs=[j for j in fake.jobs if j[0]!=first['next_job_id']]
            next(j for j in fake.jobs if j[0]==second)[3]='RUNNING'
            def fail():raise ValueError('synthetic check failure')
            with patch.dict('os.environ',{'SLURM_JOB_ID':second}):
                with self.assertRaises(ValueError):scheduler.tick(state['next_due_utc'],fail)
            self.assertEqual(read_json(root/'reports/MONITORING_STATUS.json')['status'],'MONITOR_FAILURE')
            self.assertEqual(read_json(scheduler.path)['consecutive_failures'],1)
