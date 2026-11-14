"""Persistent Slurm CPU successor chain. No sleeping process or GPU allocation."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import uuid
from ..utils.logging import write_json
from .common import FileLock, append_json, command, now, read_json


class Scheduler:
    def __init__(self, root, config, runner=command):
        self.root, self.config, self.run = Path(root).resolve(), config, runner
        self.base = self.root/'reports/monitoring'
        self.base.mkdir(parents=True, exist_ok=True)
        self.path = self.base/'scheduler.json'
        self.tag = config['scheduler']['comment'] + ':' + hashlib.sha256(str(self.root).encode()).hexdigest()[:12]

    def control_report(self,status,message,state,update_status=False):
        check='CHECK_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')+'_'+uuid.uuid4().hex[:6]
        report={'timestamp':now(),'status':status,'message':message,'live_inspection_performed':False,
            'active_jobs':'N/A: inspection failed/skipped','completed_since_previous':'N/A','automatic_fixes':[],
            'new_experiment_ids':[],'next_scheduled_check':state.get('next_due_utc'),'next_job_id':state.get('next_job_id')}
        path=self.base/(check+'.json')
        write_json(path,report)
        (self.base/(check+'.md')).write_text('# '+check+'\n\n'+json.dumps(report,indent=2)+'\n')
        with (self.root/'reports/MONITORING_LOG.md').open('a') as stream:
            stream.write('\n- '+report['timestamp']+': '+status+'; '+message+'; '+str(path)+'\n')
        if update_status:
            previous=read_json(self.root/'reports/MONITORING_STATUS.json',{})
            previous.update(report,latest_json=str(path),latest_report=str(self.base/(check+'.md')))
            write_json(self.root/'reports/MONITORING_STATUS.json',previous)

    def owned(self):
        result = self.run(['squeue', '-h', '-u', os.environ.get('USER',''), '-o', '%i|%j|%k|%T|%S'])
        if result['exit_code'] != 0:
            raise RuntimeError('Cannot inspect scheduler; refusing duplicate submission/cancellation: '+result['stderr'])
        jobs=[]
        for line in result['stdout'].splitlines():
            fields = line.strip().split('|')
            if len(fields)==5 and fields[1]==self.config['scheduler']['name'] and fields[2].startswith(self.tag+':'):
                jobs.append(dict(zip(['job_id','name','comment','state','start_time'], fields)))
        return jobs

    def _submit(self, state, slot):
        spec = self.config['scheduler']
        if spec['partition'] != 'epyc-cpu' or spec['cpus'] != 1 or spec['minutes'] > 5 or spec['memory'] != '2G':
            raise ValueError('Monitor resources must remain bounded CPU-only')
        token = uuid.uuid4().hex
        script = self.base/'monitor_job.sbatch'
        script.write_text('#!/bin/bash\nset -Eeuo pipefail\ncd '+shlex.quote(str(self.root))+
            '\nexport PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'+
            'exec '+shlex.quote(str(self.root/'.venv/bin/python'))+' scripts/manage_monitor.py tick --slot "$1"\n')
        state.update(submission_uncertain=True, pending_submission_token=token, pending_due_utc=slot.isoformat())
        write_json(self.path, state)
        args = ['sbatch','--parsable','--partition',spec['partition'],'--account',spec['account'],
            '--cpus-per-task','1','--mem',spec['memory'],'--time','00:05:00','--job-name',spec['name'],
            '--comment',self.tag+':'+token,'--begin',slot.strftime('%Y-%m-%dT%H:%M:%S'),
            '--output',str(self.base/'monitor-%j.out'),'--error',str(self.base/'monitor-%j.err'),
            str(script),slot.isoformat()]
        result = self.run(args, cwd=self.root, env={**os.environ,'TZ':'UTC'})
        append_json(self.base/'scheduler_events.jsonl', {'timestamp':now(),'action':'submit_cpu_successor','slot':slot.isoformat(),'result':result})
        job_id = result['stdout'].strip().split(';')[0]
        if result['exit_code'] != 0 or not job_id.isdigit():
            state['last_error'] = result
            write_json(self.path, state)
            raise RuntimeError('Monitor submission failed/uncertain; inspect scheduler_events.jsonl before retry')
        state.update(next_job_id=job_id,next_due_utc=slot.isoformat(),submission_uncertain=False,installed=True)
        write_json(self.path, state)
        return state

    def install(self):
        with FileLock(self.base/'scheduler.lock') as acquired:
            if not acquired:
                raise RuntimeError('Another scheduler operation is active')
            state = read_json(self.path,{})
            jobs = self.owned()
            pending=[j for j in jobs if j['state']=='PENDING']
            running=[j for j in jobs if j['state'] in {'RUNNING','CONFIGURING'}]
            if len(jobs)>2 or len(pending)>1 or len(running)>1 or len(pending)+len(running)!=len(jobs):
                raise RuntimeError('Duplicate owned monitor jobs require explicit inspection')
            if jobs:
                successor=(pending or running)[0]
                state.update(enabled=True,installed=True,next_job_id=successor['job_id'],submission_uncertain=False)
                state.setdefault('next_due_utc',state.get('pending_due_utc'))
                write_json(self.path,state)
                return state
            if state.get('submission_uncertain'):
                raise RuntimeError('Prior submission uncertain; reconcile before installing')
            state.update(enabled=True,installed=True,interval_hours=2,method='slurm_cpu_successor',consecutive_failures=0)
            return self._submit(state,datetime.now(timezone.utc).replace(microsecond=0))

    def stop(self, remove=False):
        with FileLock(self.base/'scheduler.lock') as acquired:
            if not acquired:
                raise RuntimeError('Another scheduler operation is active')
            state=read_json(self.path,{})
            state.update(enabled=False,installed=not remove,paused_at=now())
            write_json(self.path,state)
            jobs=self.owned()
            for job in jobs:
                result=self.run(['scancel',job['job_id']])
                if result['exit_code'] != 0:
                    raise RuntimeError('Could not cancel owned monitor job '+job['job_id'])
            state.update(next_job_id=None,next_due_utc=None,cancelled_monitor_job_ids=[j['job_id'] for j in jobs])
            write_json(self.path,state)
            append_json(self.base/'scheduler_events.jsonl',{'timestamp':now(),'action':'remove' if remove else 'pause','jobs':jobs})
            return state

    def status(self):
        return {'state':read_json(self.path,{}),'owned_jobs':self.owned(),'job_tag':self.tag,
            'timing':'7200-second eligibility intervals; actual starts depend on Slurm queue'}

    def tick(self, slot, monitor):
        with FileLock(self.base/'scheduler.lock') as acquired:
            if not acquired:
                return {'status':'SKIPPED_SCHEDULER_LOCKED'}
            state=read_json(self.path,{})
            if not state.get('enabled') or not self.config['enabled']:
                return {'status':'PAUSED'}
            current=os.environ.get('SLURM_JOB_ID')
            owned=self.owned()
            if not current or not any(j['job_id']==current for j in owned):
                raise RuntimeError('tick must execute inside an owned monitor Slurm allocation')
            if state.get('last_check_outcome')=='RUNNING':
                state['consecutive_failures']=state.get('consecutive_failures',0)+1
            if state.get('consecutive_failures',0)>=3:
                state.update(enabled=False,last_error='Three interrupted/failed checks; circuit breaker')
                write_json(self.path,state)
                self.control_report('BLOCKED_MONITOR_FAILURE_LIMIT',state['last_error'],state,True)
                return {'status':'BLOCKED_MONITOR_FAILURE_LIMIT'}
            jobs=[j for j in owned if j['job_id'] != current]
            if len(jobs)>1:
                raise RuntimeError('Multiple successor jobs detected; refusing further submission')
            due=datetime.fromisoformat(slot)+timedelta(hours=2)
            while due <= datetime.now(timezone.utc):
                due+=timedelta(hours=2)
            if jobs:
                state['next_job_id']=jobs[0]['job_id']
            else:
                state=self._submit(state,due)
            state.update(last_tick_job_id=current,last_check_outcome='RUNNING',last_tick_utc=now())
            write_json(self.path,state)
        try:
            result=monitor()
        except BaseException as error:
            with FileLock(self.base/'scheduler.lock') as acquired:
                if acquired:
                    state=read_json(self.path,{})
                    state.update(last_check_outcome='FAIL',last_error=str(error),consecutive_failures=state.get('consecutive_failures',0)+1)
                    if state['consecutive_failures']>=3:
                        state['enabled']=False
                        for job in self.owned():
                            if job['job_id']!=current:
                                self.run(['scancel',job['job_id']])
                        state.update(next_job_id=None,next_due_utc=None)
                    write_json(self.path,state)
                    self.control_report('MONITOR_FAILURE',str(error),state,True)
            raise
        with FileLock(self.base/'scheduler.lock') as acquired:
            if acquired:
                state=read_json(self.path,{})
                state['last_check_outcome']=result.get('status')
                if result.get('status','').startswith('SKIPPED'):
                    state['last_skipped_utc']=now()
                    self.control_report(result['status'],'Concurrent inspection owns the monitor lock; no success claimed',state)
                else:
                    state.update(consecutive_failures=0,last_success_utc=now())
                write_json(self.path,state)
        return result
