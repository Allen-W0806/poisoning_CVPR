import itertools
import json
import hashlib
from pathlib import Path
import shutil
from PIL import Image, ImageOps
from .common import command, digest, read_json, safe_path
from .diagnostics import issue

PROTOCOL_SHA = '8874954829b2f2584f82e7eec6ad73f6158a0648aa095191ea584eaeb450d927'
SCORER_SHA = 'afbf0df6385e6ec431a205fd76f7ae9fbccca09b0bd692fd97136a076c5a6063'
QWEN_REV = '89644892e4d85e24eaac8bacfd4f463576704203'
EDITOR_REV = '31519b5cb02a7fd89b906d88731cd4d6a7bbf88d'


def check_associations(root, run_id, items, indexed, source_cache=None, expected_resolution=512):
    findings, checked = [], 0
    source_cache = {} if source_cache is None else source_cache
    for item in items:
        sample_id=str(item.get('sample_id') or item.get('pair_id','').split('_')[0])
        row=indexed.get(sample_id)
        if not row or item.get('question')!=row['question'] or item.get('answers')!=row['answers']:
            findings.append(issue('editor_annotation_association_mismatch',run_id,'error',sample_id=sample_id))
            continue
        if 'source_image_id' in item and str(item['source_image_id'])!=str(row['image_id']):
            findings.append(issue('editor_source_id_mismatch',run_id,'error',sample_id=sample_id))
        try:
            source=safe_path(root,item['source_path'])
            output=safe_path(root,item.get('output_path') or item['edited_path'])
            if not source.is_file() or not output.is_file():
                raise ValueError('Referenced source or generated image is missing')
            with Image.open(output) as edited, Image.open(source) as saved:
                recorded_resolution=item.get('settings',{}).get('image_resolution',expected_resolution)
                if recorded_resolution!=expected_resolution or saved.size!=(expected_resolution,expected_resolution) or edited.size!=saved.size:
                    raise ValueError('Generated/source dimensions differ from the fixed recorded resolution')
                key=(sample_id,str(source))
                if key not in source_cache:
                    raw=Path(row['image_path'])
                    with Image.open(raw if raw.is_absolute() else root/raw) as original:
                        expected=ImageOps.pad(original.convert('RGB'),saved.size,color='white',method=3)
                        source_cache[key]=hashlib.sha256(expected.tobytes()).digest()==hashlib.sha256(saved.convert('RGB').tobytes()).digest()
                if not source_cache[key]:
                    raise ValueError('Saved source pixels do not match the declared padded dataset image')
            checked+=1
        except (OSError, ValueError, KeyError) as error:
            findings.append(issue('editor_source_output_association',run_id,'error',sample_id=sample_id,reason=str(error)))
    return {'checked_pairs':checked,'findings':findings,'scope':'Metadata and padded source pixels; no semantic-truth claim'}


def check_integrity(root, runs, state, minimum_free_bytes):
    findings, data, models = [], {}, {}
    protected = ['configs/phase2_vqa_protocol.json', 'src/evaluation/_vendor/official_vqa.py',
        'src/evaluation/vqa_official_compatible.py', 'src/evaluation/semantic_validation.py',
        'src/rl/reward.py', 'src/training/generator_trainer.py', 'scripts/compare_rl_rsi.py',
        'scripts/run_vqa_baseline_full.py', 'configs/generator_lora_smoke.yaml', 'configs/pretrained_editor.yaml',
        'configs/generator_rl_smoke.yaml', 'configs/vqa_baseline_full.yaml', 'reports/phase2_test_opened.json',
        'reports/model_ready.json', 'reports/phase2_model_metadata.json']
    current = {name: digest(root/name) for name in protected if (root/name).exists()}
    old = state.get('protected_hashes', {})
    for name, expected in [('configs/phase2_vqa_protocol.json', PROTOCOL_SHA), ('src/evaluation/_vendor/official_vqa.py', SCORER_SHA)]:
        if current.get(name) != expected:
            findings.append(issue('frozen_protocol_mismatch', severity='error', path=name))
    for name in old:
        if current.get(name) != old[name]:
            findings.append(issue('scientific_source_changed', severity='error', path=name))
    if not old:
        state['protected_hashes'] = current
    split_sets, all_records = {}, {}
    protocol = read_json(root/'configs/phase2_vqa_protocol.json', {})
    for split in ['train', 'validation', 'test']:
        path = root/f'data/processed/vqav2_seed42_v1/{split}.json'
        if not path.exists():
            findings.append(issue('missing_dataset_manifest', severity='error', split=split))
            continue
        checksum = digest(path)
        rows = read_json(path, [])
        all_records.update({str(r['sample_id']):r for r in rows})
        split_sets[split] = {str(r['image_id']) for r in rows}
        data[split] = {'count': len(rows), 'sha256': checksum}
        if checksum != protocol.get('split_hashes', {}).get(split):
            findings.append(issue('split_hash_mismatch', severity='error', split=split))
        if len({r['sample_id'] for r in rows}) != len(rows):
            findings.append(issue('duplicate_sample_id', severity='error', split=split))
        missing = []
        for row in rows:
            path = Path(row['image_path'])
            if row['split'] != split or not row.get('question') or not row.get('answers'):
                findings.append(issue('invalid_dataset_record', severity='error', sample_id=row.get('sample_id')))
            if not (path if path.is_absolute() else root/path).is_file():
                missing.append(row['sample_id'])
        if missing:
            findings.append(issue('missing_source_images', severity='error', split=split, count=len(missing), examples=missing[:5]))
    for left, right in itertools.combinations(split_sets, 2):
        if split_sets[left] & split_sets[right]:
            findings.append(issue('image_split_leakage', severity='error', splits=[left, right]))
    for name, expected in [('model_ready.json', QWEN_REV)]:
        metadata = read_json(root/'reports'/name, {})
        models[name] = {'revision': metadata.get('revision'), 'expected': expected}
        if metadata.get('revision') != expected:
            findings.append(issue('model_revision_mismatch', severity='error', metadata=name))
        local_path = metadata.get('path') or metadata.get('local_path')
        if local_path:
            models[name]['cache_path_exists'] = Path(local_path).exists()
    editor = read_json(root/'reports/phase2_model_metadata.json', {}).get('timbrooks/instruct-pix2pix', {})
    models['editor'] = {'revision': editor.get('revision'), 'expected': EDITOR_REV, 'native_image_conditioning': 'instructpix2pix'}
    if editor.get('revision') != EDITOR_REV:
        findings.append(issue('model_revision_mismatch', severity='error', metadata='editor'))
    qwen_cache = read_json(root/'reports/model_ready.json', {}).get('path')
    if qwen_cache:
        editor_cache=Path(qwen_cache).parents[2]/'models--timbrooks--instruct-pix2pix/snapshots'/EDITOR_REV
        missing_components=[name for name in editor.get('files',[]) if not (editor_cache/name).is_file()]
        models['editor'].update(cache_path=str(editor_cache),missing_components=missing_components)
        if missing_components:
            findings.append(issue('missing_editor_cache_components',severity='error',files=missing_components))
        unet=read_json(editor_cache/'unet/config.json',{})
        models['editor']['unet_in_channels']=unet.get('in_channels')
        if unet and unet.get('in_channels') != 8:
            findings.append(issue('editor_not_native_image_conditioned',severity='error'))
    for env in ['.venv','.venv-editor']:
        models[env]={'python_exists':(root/env/'bin/python').exists()}
        if not models[env]['python_exists']:
            findings.append(issue('missing_documented_environment',severity='error',environment=env))
    associations, source_cache = {}, {}
    heldout_passes=[]
    for run in runs:
        manifest, metrics = run['manifest'], run['metrics']
        for model_id, revision in manifest.get('model_revisions', {}).items():
            expected = QWEN_REV if 'Qwen3-VL-2B' in model_id else EDITOR_REV if model_id == 'timbrooks/instruct-pix2pix' else None
            if expected and revision != expected:
                findings.append(issue('run_model_revision_mismatch', run['id'], 'error', model_id=model_id))
        if 'phase2_qwen_baseline_' in run['id'] and run['status'] == 'PASS':
            if 'phase2_qwen_baseline_test' in run['id']:
                heldout_passes.append(run['id'])
            path = root/'experiments'/run['id']/'predictions.jsonl'
            if not path.exists():
                findings.append(issue('missing_predictions', run['id'], 'error'))
                continue
            predictions = [json.loads(s) for s in path.read_text().splitlines()]
            if len(predictions) != metrics.get('samples'):
                findings.append(issue('prediction_count_mismatch', run['id'], 'error', count=len(predictions)))
            split = metrics.get('split')
            rows = read_json(root/f'data/processed/vqav2_seed42_v1/{split}.json', [])
            indexed = {str(r['sample_id']): r for r in rows}
            for prediction in predictions:
                row = indexed.get(str(prediction.get('sample_id')))
                if not row or prediction.get('question') != row['question'] or prediction.get('human_reference_answers') != row['answers']:
                    findings.append(issue('prediction_annotation_mismatch', run['id'], 'error', sample_id=prediction.get('sample_id')))
                    break
            suspicious = [p.get('sample_id') for p in predictions if not p.get('raw_answer', '').strip() or len(p.get('raw_answer', '').split()) >= 64]
            if suspicious:
                findings.append(issue('potential_answer_truncation', run['id'], examples=suspicious[:8], note='Heuristic only; stored token stop reasons are unavailable'))
        items=metrics.get('outputs',[]) or metrics.get('all_decisions',[]) or metrics.get('observations',[])
        feedback=root/'experiments'/run['id']/'feedback.json'
        if feedback.is_file():
            items=read_json(feedback,[])
        if items:
            associations[run['id']]=check_associations(root,run['id'],items,all_records,source_cache)
            findings.extend(associations[run['id']]['findings'])
        if 'rl_rsi_validation' in run['id']:
            revision = metrics.get('revision', {})
            if revision:
                memory = read_json(root/'experiments'/run['id']/'rsi/state.json', {})
                selected = revision['proposed_config' if revision.get('accepted') else 'previous_config']
                if memory.get('training_config') != selected or not revision.get('proposal_reason') or not revision.get('evidence'):
                    findings.append(issue('rsi_selection_integrity', run['id'], 'error'))
                if not revision.get('accepted') and not (root/'experiments'/run['id']/revision.get('rollback', {}).get('checkpoint', 'MISSING')).is_file():
                    findings.append(issue('invalid_checkpoint_path', run['id'], 'error'))
    if len(heldout_passes)>1:
        findings.append(issue('heldout_evaluated_more_than_once',severity='error',experiment_ids=heldout_passes))
    disk = shutil.disk_usage(root)
    if disk.free < minimum_free_bytes:
        findings.append(issue('low_disk_capacity', severity='error', free_bytes=disk.free))
    git = command(['git', 'status', '--porcelain'], cwd=root)
    if git['exit_code'] != 0:
        findings.append(issue('git_status_unavailable', error=git['stderr']))
    elif git['stdout'].strip():
        findings.append(issue('uncommitted_source', paths=git['stdout'].splitlines()[:30]))
    return {'findings': findings, 'dataset': data, 'models': models, 'git_status': git, 'image_associations':associations,
        'disk': {'free_bytes': disk.free, 'total_bytes': disk.total},
        'live_model_load': 'N/A: no idle GPU probing; pinned cache/config and previous actual run witnesses only',
        'live_gpu_utilization': 'N/A unless experiment supplies telemetry; sstat RAM/CPU queried only for active project jobs',
        'baseline_scope': 'First activation baseline is not evidence of integrity before activation'}
