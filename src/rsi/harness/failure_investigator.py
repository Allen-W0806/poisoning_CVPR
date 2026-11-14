"""Acquire additional training-side evidence before proposing a revision."""
import hashlib
import copy
import time
from PIL import Image
from .trace_collector import Resources

FAILURES=('instruction_following_failure','image_quality_measurement_error','question_answer_validity_error',
 'semantic_verifier_disagreement','reward_or_configuration_error','gradient_or_optimizer_instability',
 'checkpoint_continuation_discrepancy','dataset_corruption_or_duplication','evaluation_leakage','resource_inefficiency')

class FailureInvestigator:
    def __init__(self,executor,trace,training_cases,max_queries=4):
        self.executor=executor;self.trace=trace;self.max_queries=max_queries;self.queries=[];self.resources=Resources()
        if any(c.get('split')!='train' for c in training_cases):raise PermissionError('Training registry cannot contain held-out cases')
        self._cases={c['case_id']:copy.deepcopy(c) for c in training_cases}
        if len(self._cases)!=len(training_cases):raise ValueError('Duplicate training capability')

    def _authorize(self,case):
        if case.get('split')!='train' or self._cases.get(case.get('case_id'))!=case:
            raise PermissionError('Investigation requires an exact registered training capability')

    def observe(self,case,query):
        self._authorize(case)
        if len(self.queries)>=self.max_queries:raise RuntimeError('Investigation budget exhausted')
        start=time.perf_counter();cpu=time.process_time();bytes_read=0
        if query=='image_headers':
            p=self.executor.path(case['image_path']);data=p.read_bytes();bytes_read=len(data)
            import io
            with Image.open(io.BytesIO(data)) as image:observation={'actual_size':list(image.size),'expected_size':list(case['expected_size'])}
        elif query=='content_hash':
            p=self.executor.path(case['image_path']);data=p.read_bytes();bytes_read=len(data)
            observation={'actual_sha256':hashlib.sha256(data).hexdigest(),'expected_sha256':case['expected_sha256']}
        elif query=='pair_quality':
            from ...evaluation.image_quality import image_quality
            a=self.executor.path(case['source_path']);b=self.executor.path(case['edited_path'])
            bytes_read=a.stat().st_size+b.stat().st_size
            with Image.open(a) as original,Image.open(b) as edited:observation=image_quality(original,edited)
        elif query=='trajectory_summary':
            import torch
            p=self.executor.path(case['trajectory_path']);bytes_read=p.stat().st_size
            trajectory=torch.load(p,map_location='cpu',weights_only=False)
            observation={'sample_id':trajectory['sample_id'],'policy_version':trajectory['policy_version'],
                'transitions':len(trajectory['steps']),'seed':trajectory['seed'],
                'finite_states':all(bool(torch.isfinite(s['state']).all() and torch.isfinite(s['action']).all()) for s in trajectory['steps'])}
        else:raise ValueError('Unpermitted diagnostic query')
        cost=Resources(time.perf_counter()-start,time.process_time()-cpu,1,bytes_read)
        self.resources.add(cost)
        record={'query':query,'case_id':case['case_id'],'observation':observation,'resources':cost.__dict__}
        self.trace.append('additional_observation',record);self.queries.append(record)
        return record

    def investigate_integrity_failure(self,case):
        self._authorize(case)
        hypotheses=[{'cause':'dataset_corruption_or_duplication','prediction':'actual file digest differs from the recorded manifest'},
                    {'cause':'image_quality_measurement_error','prediction':'reported geometry disagrees with an independent header read'}]
        self.trace.append('competing_hypotheses',{'case_id':case['case_id'],'hypotheses':hypotheses})
        # Both queries add file evidence absent from the failed incumbent trace.
        self.observe(case,'image_headers');self.observe(case,'content_hash')
        return self.diagnose(case,hypotheses)

    def diagnose(self,case,hypotheses):
        self._authorize(case)
        observations={r['query']:r['observation'] for r in self.queries if r['case_id']==case['case_id']}
        supported=[];contradicted=[]
        h=observations.get('image_headers');s=observations.get('content_hash')
        if s and s['actual_sha256']!=s['expected_sha256']:supported.append('dataset_corruption_or_duplication')
        if h and h['actual_size']==h['expected_size']:contradicted.append('geometry_checker_reported_wrong_size')
        result={'case_id':case['case_id'],'hypotheses':hypotheses,'supported':supported,'contradicted':contradicted,
            'unresolved':[f for f in FAILURES if f not in supported],
            'contradiction_scope':'Header agreement contradicts only the proposed wrong-geometry explanation, not every quality measurement error',
            'evidence_queries':[r['query'] for r in self.queries if r['case_id']==case['case_id']],
            'diagnosis_status':'SUPPORTED_SCOPED' if supported else 'UNRESOLVED',
            'scope':'digest discrepancy; not a claim of intentional corruption or semantic failure root cause'}
        self.trace.append('diagnosis',result)
        return result
