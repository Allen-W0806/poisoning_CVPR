"""Executable declarative workflow; trusted tool implementations are not editable."""
from dataclasses import dataclass,asdict

ALLOWED_CHECKS={'geometry','decode','content_hash','independent_review'}

@dataclass(frozen=True)
class HarnessSpec:
    name: str
    checks: tuple=('geometry',)
    schema_version: str='workflow_program_v1'
    max_diagnostics: int=8
    memory_mode: str='full_evidence'
    curriculum: str='input_order'

    def __post_init__(self):
        if not self.name or self.schema_version!='workflow_program_v1':raise ValueError('Invalid harness program')
        if not isinstance(self.checks,tuple) or not self.checks or len(set(self.checks))!=len(self.checks) or set(self.checks)-ALLOWED_CHECKS:
            raise ValueError('Unpermitted or duplicated workflow tool')
        if type(self.max_diagnostics) is not int or not 1<=self.max_diagnostics<=64:raise ValueError('Invalid diagnostic operation budget')
        if self.memory_mode not in {'full_evidence','summary'} or self.curriculum not in {'input_order','uncertain_first'}:
            raise ValueError('Invalid memory/curriculum program')

    def to_dict(self):
        return {**asdict(self),'checks':list(self.checks)}

    @classmethod
    def from_dict(cls,value):
        if set(value)!=set(cls.__dataclass_fields__):raise ValueError('Revision may not alter evaluator/labels/objectives')
        value=dict(value);value['checks']=tuple(value['checks'])
        return cls(**value)

    def training_order(self,cases):
        if any(c.get('split')!='train' for c in cases):raise PermissionError('Curriculum cannot organize held-out cases')
        return sorted(cases,key=lambda c:(-float(c.get('uncertainty',0)),str(c['case_id']))) if self.curriculum=='uncertain_first' else list(cases)
