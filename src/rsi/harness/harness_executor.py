"""Read-only interpreter for reusable validation programs; no eval/exec or shell."""
from pathlib import Path
import hashlib
import time
from PIL import Image
from .trace_collector import Resources

class HarnessExecutor:
    def __init__(self,data_root):self.root=Path(data_root).resolve()

    def path(self,value):
        p=Path(value);p=(p if p.is_absolute() else self.root/p).resolve()
        if not p.is_relative_to(self.root):raise PermissionError('Tool file outside permitted case data root')
        return p

    def run(self,spec,case):
        started=time.perf_counter();cpu=time.process_time();resources=Resources();observations=[]
        path=self.path(case['image_path']);valid=True
        if len(spec.checks)>spec.max_diagnostics:raise RuntimeError('Executable workflow exceeds its diagnostic budget')
        for check in spec.checks:
            t=time.perf_counter();c=time.process_time();resources.diagnostic_operations+=1
            if check=='content_hash':
                data=path.read_bytes();resources.bytes_read+=len(data)
                actual=hashlib.sha256(data).hexdigest();passed=actual==case['expected_sha256']
                evidence={'actual_sha256':actual,'matches_manifest':passed}
            elif check in {'geometry','decode'}:
                # Count actual supplied bytes as well as wall/CPU cost.
                data=path.read_bytes();resources.bytes_read+=len(data)
                import io
                try:
                    with Image.open(io.BytesIO(data)) as image:
                        image.load();size=list(image.size);mode=image.mode
                    passed=(size==list(case['expected_size'])) if check=='geometry' else True
                    evidence={'size':size,'mode':mode,'decoded':True}
                except (OSError,ValueError):passed=False;evidence={'decoded':False}
            elif check=='independent_review':
                review=case.get('independent_review')
                passed=isinstance(review,dict) and review.get('status')=='VERIFIED' and review.get('valid') is True and bool(review.get('source'))
                evidence={'independent_review_available':bool(review),'verified_valid':passed}
            else:raise ValueError('Unknown executable tool')
            observations.append({'tool':check,'passed':passed,'evidence':evidence,
                'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-c})
            valid=valid and passed
        resources.wall_seconds=time.perf_counter()-started;resources.cpu_seconds=time.process_time()-cpu
        if spec.memory_mode=='summary':observations=[{'tool':o['tool'],'passed':o['passed']} for o in observations]
        return {'case_id':case['case_id'],'predicted_valid':bool(valid),'observations':observations,'resources':resources.validate().__dict__}
