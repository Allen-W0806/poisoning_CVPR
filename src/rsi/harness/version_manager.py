"""Immutable accepted/rejected programs with an explicit incumbent pointer."""
from pathlib import Path
import json
from ..memory import RevisionMemory
from ...utils.logging import write_json
from .harness_spec import HarnessSpec
from .revision_validator import validate_revision

class VersionManager:
    def __init__(self,root,initial):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=False)
        self.memory=RevisionMemory(self.root/'history');self.incumbent='H0000';self.attempt=0
        self._save_program(self.incumbent,initial);self.initial=self.incumbent;self._pointer()

    def _save_program(self,version,spec):
        with (self.root/(version+'.json')).open('x',encoding='utf8') as f:json.dump(spec.to_dict(),f,indent=2)

    def _pointer(self):write_json(self.root/'incumbent.json',{'version':self.incumbent})

    def program(self,version=None):
        return HarnessSpec.from_dict(json.loads((self.root/((version or self.incumbent)+'.json')).read_text()))

    def evaluate(self,proposal,gate):
        parent=self.incumbent;candidate=validate_revision(proposal,parent,self.program())
        self.attempt+=1;version=f'H{self.attempt:04d}';self._save_program(version,candidate)
        result=gate.compare(self.program(),candidate)
        record={'revision_number':self.attempt,'parent_version':parent,'candidate_version':version,
            'proposal':proposal,'measured_validation':result,'accepted':result['accepted'],
            'actual_evaluation_resources':{'parent':result['parent_resources'],'candidate':result['candidate_resources']}}
        if result['accepted']:self.incumbent=version
        self.memory.append(record);self._pointer()
        private=self.root/'evaluation_private';private.mkdir(exist_ok=True)
        with (private/(version+'.json')).open('x',encoding='utf8') as f:json.dump(gate.last_private_artifact,f,indent=2)
        return record

    def rollback(self,version,reason):
        previous=self.incumbent;self.program(version);self.incumbent=version;self._pointer()
        with (self.root/f'rollback_{self.attempt:04d}_{previous}_to_{version}.json').open('x',encoding='utf8') as f:
            json.dump({'from':previous,'to':version,'reason':reason},f)
