import hashlib
import json
import os
from pathlib import Path
import yaml
import _bootstrap
from src.data.dataset import sha256
from src.utils.logging import Experiment

def phase4_run(name,config,splits=('train','validation')):
    root=_bootstrap.ROOT
    m=json.loads((root/'reports/phase2_model_metadata.json').read_text())['timbrooks/instruct-pix2pix']
    q=json.loads((root/'reports/model_ready.json').read_text())
    parent=Path(os.environ['POISONING_JOB_RUN']).name if os.environ.get('POISONING_JOB_RUN') else None
    return Experiment(root,'phase4_'+name,{'phase':4,**config},{
        'config_sha256':hashlib.sha256(yaml.safe_dump(config,sort_keys=True).encode()).hexdigest(),
        'model_identifiers':[m['id'],q['id']], 'model_revisions':{m['id']:m['revision'],q['id']:q['revision']},
        'dataset_version':'VQAv2-train2014-2017-04-26',
        'split_hashes':{s:sha256(root/f'data/processed/vqav2_seed42_v1/{s}.json') for s in splits}},parent)
