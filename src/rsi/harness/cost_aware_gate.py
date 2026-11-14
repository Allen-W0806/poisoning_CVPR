"""Trusted evaluator returns aggregate paired outcomes only to the Editor."""
from dataclasses import dataclass,asdict
import copy
import math
from .trace_collector import Resources

@dataclass(frozen=True)
class GatePolicy:
    min_sources: int=24
    minimum_accuracy_gain: float=.05
    maximum_operation_growth: float=1.
    minimum_operation_reduction: float=.20
    schema_version: str='fixed_integrity_oracle_gate_v1'

    def __post_init__(self):
        if type(self.min_sources) is not int or self.min_sources<24:raise ValueError('At least24 independent engineering sources')
        if not 0<self.minimum_accuracy_gain<=1 or not 0<self.minimum_operation_reduction<1 or not math.isfinite(self.maximum_operation_growth) or self.maximum_operation_growth<0:
            raise ValueError('Invalid fixed Gate thresholds')
        if self.schema_version!='fixed_integrity_oracle_gate_v1':raise ValueError('Unknown gate definition')

class IndependentGate:
    def __init__(self,executor,cases,oracle,policy=GatePolicy(),kind='constructed_integrity_oracle'):
        if kind!='constructed_integrity_oracle':raise ValueError('Natural-image Gate requires independent semantic/task judgments; currently unavailable')
        self._executor=executor;self._cases=copy.deepcopy(cases);self._oracle=copy.deepcopy(oracle);self.policy=policy
        ids=[c['case_id'] for c in cases]
        if any(c.get('split')!='gate' for c in cases):raise ValueError('Selection must use the fixed Gate capability')
        if len(ids)!=len(set(ids)) or set(ids)!=set(oracle) or len(ids)<policy.min_sources:raise ValueError('Insufficient independent paired Gate coverage')
        if any(type(v) is not bool for v in oracle.values()):raise ValueError('Exact oracle booleans required')
        if len({c['expected_sha256'] for c in cases})!=len(cases):raise ValueError('Repeated content is not an independent Gate source')
        self.last_private_artifact=None

    def compare(self,parent,candidate):
        rows=[];costs=[Resources(),Resources()];correct=[0,0];false_valid=[0,0];new_false_valid=0;new_errors=0
        for case in self._cases:
            pair=[self._executor.run(parent,case),self._executor.run(candidate,case)]
            expected=self._oracle[case['case_id']]
            for i,row in enumerate(pair):
                costs[i].add(Resources(**row['resources']));correct[i]+=row['predicted_valid']==expected
                false_valid[i]+=row['predicted_valid'] and not expected
            new_false_valid+=not expected and pair[1]['predicted_valid'] and not pair[0]['predicted_valid']
            new_errors+=(pair[0]['predicted_valid']==expected and pair[1]['predicted_valid']!=expected)
            rows.append({'case_id':case['case_id'],'expected':expected,'parent':pair[0],'candidate':pair[1]})
        n=len(rows);gain=(correct[1]-correct[0])/n
        p,c=costs[0].diagnostic_operations,costs[1].diagnostic_operations
        quality=gain>=self.policy.minimum_accuracy_gain and c<=p*(1+self.policy.maximum_operation_growth)
        efficient=gain==0 and p>0 and c<=p*(1-self.policy.minimum_operation_reduction)
        no_regression=false_valid[1]<=false_valid[0] and new_errors==0
        accepted=bool(no_regression and (quality or efficient))
        public={'accepted':accepted,'n_sources':n,'parent_accuracy':correct[0]/n,'candidate_accuracy':correct[1]/n,
            'accuracy_delta':gain,'parent_false_valid':int(false_valid[0]),'candidate_false_valid':int(false_valid[1]),
            'new_false_valid':int(new_false_valid),
            'new_errors_on_previously_correct_sources':int(new_errors),
            'parent_resources':asdict(costs[0]),'candidate_resources':asdict(costs[1]),'policy':asdict(self.policy),
            'evaluator_definitions_changed':False,'matched_exposure':True,
            'reason':'verified engineering gain within fixed operation budget' if accepted else 'no meaningful gain, cost bound or integrity regression',
            'scope':'deterministic harmless task; not a natural-image semantic or security result'}
        self.last_private_artifact={'paired_rows':rows,'aggregate':copy.deepcopy(public)}
        return public
