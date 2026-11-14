"""CPU-only inventory of existing RSI-Poisoning model artifacts; never trains.

Hashes are byte SHA256 (not tensor hashes). Every weight/state and excluded
trajectory witness remains listed. No checkpoint is copied or overwritten.
"""
import argparse, collections, csv, datetime, hashlib, json, os, pathlib, re


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return None


def main(root, out):
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    out.mkdir(parents=True, exist_ok=False)
    rows, metadata, groups, errors = [], {}, {}, []
    roots = [root]
    cache = pathlib.Path('/storage/home/ydk297/.cache/huggingface/hub')
    for name in ['models--Qwen--Qwen3-VL-2B-Instruct', 'models--timbrooks--instruct-pix2pix']:
        if (cache / name / 'snapshots').exists():
            roots.append(cache / name / 'snapshots')
    hashes = {}
    for scan in roots:
        for base, dirs, files in os.walk(scan, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in ['.git', '.venv', '.venv-editor', '__pycache__', 'node_modules', 'tmp'])
            for name in sorted(files):
                p = pathlib.Path(base) / name
                if p.suffix not in ['.pt', '.pth', '.safetensors', '.bin', '.ckpt']:
                    continue
                rel = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
                resolved = p.resolve()
                row = dict(path=rel, bytes=p.stat().st_size, symlink=p.is_symlink(), resolved=str(resolved))
                is_witness = any(x in name for x in ['trajectory', 'gradients', 'rollout', 'forward.pt', 'observations', 'reference_after.pt', 'pre_update'])
                row['artifact_type'] = 'diagnostic_witness_not_checkpoint' if is_witness else 'checkpoint'
                family = '/'.join(p.relative_to(root).parts[:2]) if p.is_relative_to(root) else str(p.parents[0])
                row['family'] = family
                try:
                    stat = resolved.stat()
                    key = (str(resolved), stat.st_size, stat.st_mtime_ns)
                    if key not in hashes:
                        hashes[key] = sha(resolved)
                    row['file_sha256'] = hashes[key]
                    row['stable_during_read'] = (stat.st_size, stat.st_mtime_ns) == (resolved.stat().st_size, resolved.stat().st_mtime_ns)
                    if scan != root:
                        row.update(category='Clean' if 'Qwen' in str(scan) else 'Frozen', model='Qwen3-VL-2B-Instruct' if 'Qwen' in str(scan) else 'InstructPix2Pix', actual_updates=0, independent_evaluation='NOT CERTIFIED')
                    else:
                        row.update(category='UNRESOLVED', model='UNRESOLVED', actual_updates=None, independent_evaluation='NOT CERTIFIED')
                        if not is_witness and p.suffix == '.pt':
                            state = torch.load(p, map_location='cpu', weights_only=False)
                            if isinstance(state, dict):
                                row['state_keys'] = list(state)[:30]
                                for k in ['step', 'version', 'training_steps', 'global_step']:
                                    if isinstance(state.get(k), (int, float, str)):
                                        row[k] = state[k]
                                params = state.get('adapter', state.get('policy', state.get('model', state.get('state_dict', {}))))
                                if not isinstance(params, dict):
                                    params = {}
                                keys = list(params)
                                if not keys and all(isinstance(v, torch.Tensor) for v in state.values()):
                                    params = state; keys = list(state)
                                shapes = {k: list(v.shape) for k, v in params.items() if isinstance(v, torch.Tensor)}
                                row['tensor_count'] = len(shapes)
                                row['parameter_elements'] = sum(v.numel() for v in params.values() if isinstance(v, torch.Tensor))
                                row['tensor_name_examples'] = keys[:3]
                                optimizer = state.get('optimizer', {})
                                steps = []
                                if isinstance(optimizer, dict):
                                    for v in optimizer.get('state', {}).values():
                                        if isinstance(v, dict) and 'step' in v:
                                            steps.append(float(v['step']))
                                    row['optimizer_groups'] = [{k:v for k,v in g.items() if k != 'params'} for g in optimizer.get('param_groups', [])]
                                row['optimizer_step_range'] = [min(steps), max(steps)] if steps else None
                                row['actual_updates'] = row.get('step', int(max(steps)) if steps else None)
                                if 'contract' in state:
                                    row['contract'] = state['contract']
                                if any('language_model' in k for k in keys) or 'qwen' in rel.lower() or '/clean_' in rel:
                                    row.update(category='Clean', model='Qwen3-VL-2B-Instruct language-attention LoRA')
                                elif row['parameter_elements'] == 1594368 or 'phase4_PH4_' in rel or 'BF16' in rel or 'ip2pix' in rel or 'P1_ABBA' in rel or 'phase2_generator_lora' in rel or 'phase3_generator' in rel:
                                    row['model'] = 'InstructPix2Pix attention LoRA'
                                    row['category'] = 'RSI' if 'GENERATOR_RSI' in rel else ('Frozen' if 'FROZEN_GENERATOR' in rel else 'Benign RL')
                                else:
                                    row.update(category='Synthetic Fixture', model='Tiny UNet or categorical/controller fixture')
                                del state
                        elif not is_witness and p.suffix == '.safetensors':
                            from safetensors import safe_open
                            with safe_open(str(p), framework='pt', device='cpu') as f:
                                keys = list(f.keys()); row['tensor_count'] = len(keys); row['tensor_name_examples'] = keys[:3]
                            row.update(category='Clean' if 'qwen' in rel or '/clean_' in rel else 'Benign RL', model='Qwen3-VL-2B-Instruct language-attention LoRA' if 'qwen' in rel or '/clean_' in rel else 'InstructPix2Pix attention LoRA')
                        if is_witness:
                            row.update(category='Synthetic Fixture' if any(k in rel for k in ['threeway', 'tiny_', '/integrated/']) else 'Benign RL', model='NON_MODEL_WITNESS', actual_updates=None)
                        # Preserve ancestor configuration/manifest/summary provenance verbatim.
                        ancestors = [p.parent]
                        if p.is_relative_to(root):
                            ancestors += [a for a in p.parents if a != p.parent and a != root and a.is_relative_to(root)]
                        sources = []
                        for a in ancestors:
                            for n in ['config.yaml', 'protocol.json', 'effective_config.json', 'manifest.json', 'summary.json', 'clean_data_manifest.json', 'checks.json', 'metrics.json', 'status.json', 'adapter_config.json']:
                                q = a / n
                                if q.is_file() and q.stat().st_size < 5_000_000:
                                    qrel = str(q.relative_to(root))
                                    sources.append(qrel)
                                    if qrel not in metadata:
                                        metadata[qrel] = dict(sha256=sha(q), text=q.read_text(encoding='utf-8'))
                        row['metadata_sources'] = sources
                    rows.append(row)
                except Exception as e:
                    row['error'] = repr(e); errors.append(dict(path=rel, error=repr(e))); rows.append(row)
    for row in rows:
        groups.setdefault(row['family'], []).append(row)
    summary = dict(generated_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), scope=[str(p) for p in roots], exclusions=['virtual environments', '.git', 'tmp', 'unrelated projects and unrelated HF caches'], files=len(rows), bytes=sum(x['bytes'] for x in rows), checkpoint_files=sum(x['artifact_type']=='checkpoint' for x in rows), category_counts=dict(collections.Counter(x.get('category') for x in rows if x['artifact_type']=='checkpoint')), errors=errors, cuda_initialized=torch.cuda.is_initialized(), modified_victim='No attack-trained victim artifact found; clean-only Qwen LoRA changes victim parameters and is separately marked Clean, not attack-trained', independent_evaluation='Per-family evidence must still be checked for exposure; split separation alone is not independence')
    for name, obj in [('checkpoint_inventory.json', rows), ('checkpoint_metadata.json', metadata), ('inventory_summary.json', summary)]:
        (out/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    fields = ['path','category','model','artifact_type','actual_updates','step','version','optimizer_step_range','tensor_count','parameter_elements','bytes','file_sha256','independent_evaluation','family','error']
    with (out/'checkpoint_inventory.csv').open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    (out/'family_index.json').write_text(json.dumps({k:dict(files=len(v),checkpoints=sum(x['artifact_type']=='checkpoint' for x in v),categories=sorted({x.get('category','ERROR') for x in v}),max_step=max((x.get('actual_updates') or 0 for x in v),default=0)) for k,v in groups.items()},indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--out',type=pathlib.Path,required=True);a=p.parse_args();main(a.root.resolve(),a.out.resolve())
