"""Immutable source/content splits and statistics for benign P0 validation."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def validate_splits(splits, sizes, root=None):
    if set(splits)!=set(sizes):raise ValueError('Unexpected or missing split')
    ids,hashes,samples=set(),set(),set()
    for split,count in sizes.items():
        if len(splits[split])!=count:raise ValueError('Wrong split size '+split)
        for row in splits[split]:
            image_id,sample_id=str(row['image_id']),str(row['sample_id'])
            if image_id in ids or row['image_sha256'] in hashes or sample_id in samples:
                raise ValueError('Source/content/question leakage or duplicate')
            ids.add(image_id);hashes.add(row['image_sha256']);samples.add(sample_id)
            if len(row['answers'])!=10 or not row['training_answer']:
                raise ValueError('Original ten-answer QA record required')
            if root is not None:
                base=Path(root).resolve();path=(base/row['image_path']).resolve()
                if not path.is_relative_to(base) or not path.is_file() or sha256(path)!=row['image_sha256']:
                    raise ValueError('Image path/content mismatch')
    return {'source_IDs':len(ids),'content_hashes':len(hashes),'question_IDs':len(samples),
            'split_sizes':sizes,'exact_ID_content_disjoint':True,
            'perceptual_near_duplicate_check':'NOT_RUN','pretraining_exposure':'UNKNOWN'}


def metric_summary(values, seed=20261111, resamples=10000):
    import numpy as np
    data=np.asarray(values,dtype=np.float64)
    if data.ndim!=1 or not len(data) or not np.isfinite(data).all():raise ValueError('Finite source-level observations required')
    generator=np.random.default_rng(seed)
    boot=data[generator.integers(0,len(data),(resamples,len(data)))].mean(1)
    return {'n_sources':len(data),'mean':float(data.mean()),
            'source_SD':float(data.std(ddof=1)) if len(data)>1 else None,
            'source_bootstrap_95CI':np.quantile(boot,[.025,.975]).tolist(),
            'resamples':resamples,'analysis_seed':seed,'training_seed_uncertainty':'NOT_ESTIMATED'}


def require_manifest(run, assets, protocol):
    path=Path(run)/'manifest.json'
    receipt=json.loads((Path(run)/'manifest_lock.json').read_text())
    if sha256(path)!=receipt['manifest_SHA256']:raise ValueError('Frozen manifest changed')
    config=Path(run)/'code/configs/p0_foundations_20261111.json'
    if sha256(config)!=receipt['protocol_SHA256']:raise ValueError('Frozen protocol changed')
    manifest=json.loads(path.read_text())
    validate_splits(manifest['splits'],protocol['split_sizes'],assets)
    return manifest
