"""Hash-checked v2 manifests; old loaders and v1 evidence remain unchanged."""
import hashlib
import json
from pathlib import Path
from .dataset import VQADataset,sha256
from .semantic_contract_v2 import validate

class VerifiedVQADataset(VQADataset):
    def __init__(self,manifest,root,expected_split,expected_sha256,access_role='operator'):
        if access_role=='editor' and expected_split!='train':raise PermissionError('Editor cannot load Gate/test/development labels')
        if access_role not in {'operator','editor','evaluator'}:raise ValueError('Unknown data capability')
        if sha256(manifest)!=expected_sha256:raise ValueError('Manifest digest mismatch')
        super().__init__(manifest,root,expected_split)
        self.expected_sha256=expected_sha256
        for row in self.records:self.verify_record(row)

    def verify_record(self,row):
        if len(row['answers'])!=10 or any(not isinstance(a,str) or not a for a in row['answers']):raise ValueError('Malformed annotation answers')
        if row.get('source_question_image_id')!=str(row['image_id']):raise ValueError('Question/image annotation join mismatch')
        path=self.image_path(row).resolve()
        if not path.is_relative_to(self.root):raise ValueError('Image outside declared data root')
        if path.name!=f"COCO_train2014_{int(row['image_id']):012d}.jpg":raise ValueError('Filename/image association mismatch')
        if sha256(path)!=row['image_sha256']:raise ValueError('Image content checksum mismatch')
        contract=validate(row['semantic_contract'])
        o=contract['original']
        if contract['sample_id']!=row['sample_id'] or contract['split']!=row['split'] or o['image_id']!=row['image_id'] or o['sha256']!=row['image_sha256'] or o['question']!=row['question'] or o['dataset_answers']!=row['answers']:
            raise ValueError('Semantic/loader record association mismatch')

    def __getitem__(self,index):
        if sha256(self.manifest)!=self.expected_sha256:raise ValueError('Manifest changed after initialization')
        self.verify_record(self.records[index])
        return super().__getitem__(index)

def validate_content_disjoint(manifests):
    seen_ids={};seen_hashes={}
    for split,rows in manifests.items():
        for row in rows:
            for key,mapping in [('image_id',seen_ids),('image_sha256',seen_hashes)]:
                identity=str(row[key]);old=mapping.setdefault(identity,split)
                if old!=split:raise ValueError('Cross-split '+key+' overlap')
    return True
