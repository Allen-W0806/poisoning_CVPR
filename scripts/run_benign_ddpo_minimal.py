"""Bounded real IP2Pix lighting validation. Run ONLY in a one-GPU allocation.

Stages: native frozen inference -> BF16/FP32 numerical gates -> paired Inner
score-function control -> fixed/rule/UCB recipe search -> untouched transfer.
All task scores are benign proxies. This entrypoint never imports a victim,
security evaluator, poisoning controller, or learned reward.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import checked_image, sha256, validate_cohort, write_json
from src.mini_pilot.benign_validation import RecipeSelector, content_box, lighting_metrics, paired_summary
from scripts.test_benign_ddpo_audit import source_hashes


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--assets-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--cpu-receipt', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    defaults = json.loads((ROOT/'configs/benign_ddpo_minimal.json').read_text())
    if config != defaults or config['allocation_cap_seconds'] > 1200:
        raise ValueError('This minimal entrypoint accepts only the audited bounded benign protocol')
    cpu = json.loads(Path(args.cpu_receipt).read_text())
    if not cpu['passed'] or cpu['source_sha256'] != source_hashes():
        raise ValueError('Passing CPU tests for this exact source snapshot are required')
    output, assets = Path(args.output).resolve(), Path(args.assets_root).resolve()
    if not output.is_relative_to(assets/'experiments') or output == assets/'experiments':
        raise ValueError('Output must be an isolated run under canonical experiments/')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'protocol.json', config, exclusive=True)
    state = {'status': 'STARTED', 'started_UTC': utc(), 'PID': os.getpid(),
             'SLURM_JOB_ID': os.getenv('SLURM_JOB_ID'), 'config_SHA256': sha256(args.config),
             'source_sha256': source_hashes(), 'seed': config['seed'], 'stages': {},
             'statistical_scope': config['scope'], 'efficacy_claim': 'NOT_ESTABLISHED',
             'BF16_repair_claim': False, 'RSI_value_claim': 'NOT_ESTABLISHED'}
    started = time.perf_counter()
    def persist():
        state['elapsed_seconds'] = time.perf_counter()-started
        write_json(output/'status.json', state)
    def event(kind, value):
        row = {'UTC': utc(), 'kind': kind, **value}
        with (output/'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row, allow_nan=False), flush=True)
    def check_time():
        if time.perf_counter()-started > config['wall_seconds']:
            raise TimeoutError('Bounded minimal wall-time cap; no automatic continuation')
    persist()
    try:
        import numpy as np
        import torch
        from PIL import Image
        from src.editing.pretrained_editor import PretrainedEditor
        from src.models.pretrained_editor_loader import pretrained_config
        from src.mini_pilot.engine import PilotPolicy, PilotTrainer, compare_trees, frozen_digest
        if not os.getenv('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('One explicit Slurm CUDA allocation is mandatory; login-node execution prohibited')
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        random.seed(config['seed']); np.random.seed(config['seed']); torch.manual_seed(config['seed'])
        torch.cuda.manual_seed_all(config['seed'])
        kernel = torch.randn(16, 16, device='cuda', generator=torch.Generator(device='cuda').manual_seed(42))
        witness = kernel@kernel
        if not torch.isfinite(witness).all(): raise RuntimeError('Seeded CUDA kernel failed')
        state['GPU'] = {'name': torch.cuda.get_device_name(), 'visible_devices': os.getenv('CUDA_VISIBLE_DEVICES'),
            'total_memory': torch.cuda.get_device_properties(0).total_memory,
            'initial_free_memory': torch.cuda.mem_get_info()[0],
            'nvidia_smi': subprocess.check_output(['nvidia-smi','--query-gpu=index,name,memory.used,memory.total',
                                                  '--format=csv,noheader'], text=True),
            'kernel_sha256': hashlib.sha256(witness.cpu().numpy().tobytes()).hexdigest()}
        event('kernel_witness', state['GPU'])
        # Only previously exposed development TRAIN is opened. The historical
        # test and prospective reservation remain closed.
        manifest = assets/'data/processed/vqav2_content_v2/train.json'
        records = sorted(json.loads(manifest.read_text()), key=lambda r: (str(r['image_id']),str(r['sample_id'])))
        random.Random(config['seed']).shuffle(records)
        selected, seen_ids, seen_hashes = [], set(), set()
        for record in records:
            if str(record['image_id']) in seen_ids or record['image_sha256'] in seen_hashes: continue
            checked_image(assets, record)
            selected.append({key: record[key] for key in ('sample_id','image_id','image_path','image_sha256')})
            seen_ids.add(str(record['image_id'])); seen_hashes.add(record['image_sha256'])
            if len(selected) == sum(config['split_sizes'].values()): break
        cohort, offset = {}, 0
        for split, size in config['split_sizes'].items():
            cohort[split] = selected[offset:offset+size]; offset += size
        audit = validate_cohort(assets, cohort, config['split_sizes'])
        write_json(output/'source_manifest.json', {'splits':cohort, 'audit':audit,
                    'parent_manifest':str(manifest), 'parent_sha256':sha256(manifest),
                    'exposure': config['scope'], 'historical_test_opened':False}, exclusive=True)
        event('cohort_locked', audit)
        metric_test = Image.fromarray(np.tile(np.arange(64,dtype=np.uint8)[None,:,None],(64,1,3))+64)
        brighter = Image.fromarray(np.asarray(metric_test)+10)
        sanity = {'identity_reward':lighting_metrics(metric_test,metric_test)['training_reward_proxy'],
                  'brightened_reward':lighting_metrics(metric_test,brighter)['training_reward_proxy']}
        if abs(sanity['identity_reward']) > 1e-12 or sanity['brightened_reward'] <= 0:
            raise RuntimeError('Benign reward sanity check failed')
        state['reward_sanity'] = sanity
        native_done = False
        active_precision = None
        for precision in ('bf16','fp32'):
            check_time()
            stage = output/precision
            stage.mkdir()
            editor = policy = trainer = None
            precision_start = time.perf_counter()
            try:
                editor = PretrainedEditor(pretrained_config(assets,precision=precision,seed=config['seed'],
                    image_resolution=config['resolution'],num_inference_steps=config['num_inference_steps'],guidance_scale=5.),device='cuda')
                if not native_done:
                    record = cohort['train'][0]
                    native = editor.edit(assets/record['image_path'],config['instruction'],record['image_id'],
                                         config['seed'],stage/'native_frozen.png')
                    write_json(stage/'native_frozen.json', {**native.metadata,'output_SHA256':sha256(stage/'native_frozen.png')},exclusive=True)
                    event('native_frozen_inference',{'precision':precision,'output_SHA256':sha256(stage/'native_frozen.png')})
                    native_done = True
                diffusion = {'lora_rank':config['lora_rank'],'lora_alpha':2*config['lora_rank'],
                    'adapter_compute_precision':'fp32','gradient_checkpointing':True,
                    'learning_rate':config['learning_rate'],'num_inference_steps':config['num_inference_steps'],
                    'eta':config['eta'],'logical_batch_size':config['logical_batch_size'],
                    'physical_micro_batch_size':1,'gradient_accumulation_steps':config['logical_batch_size'],
                    'max_grad_norm':config['max_grad_norm']}
                policy = PilotPolicy(editor,diffusion)
                policy.contract['benign_protocol_SHA256'] = state['config_SHA256']
                policy.contract['source_manifest_SHA256'] = sha256(output/'source_manifest.json')
                trainer = PilotTrainer(policy,config['learning_rate'],config['max_grad_norm'])
                conditions = {}
                for split, rows in cohort.items():
                    for record in rows:
                        with Image.open(assets/record['image_path']) as image:
                            image = image.convert('RGB')
                            condition = policy.condition({**record,'image':image},
                                config['transfer_instruction'] if split=='transfer' else config['instruction'])
                            condition['content_box'] = content_box(image,config['resolution'])
                        conditions[record['sample_id']] = condition
                        path = stage/'sources'/f"{record['sample_id']}.png"
                        path.parent.mkdir(exist_ok=True)
                        condition['source'].save(path)
                frozen = frozen_digest(policy.pipe)
                initial = stage/'checkpoints/initial'
                trainer.save(initial,{'purpose':'shared zero-LoRA initialization'})
                def collect(record,label,seed,trajectory_id=None,retain=False):
                    check_time()
                    condition = conditions[record['sample_id']]
                    nfe = policy.unet_calls
                    with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                        image, trajectory = policy.sample(condition,seed,trajectory_id or label+'_'+record['sample_id'])
                    path = stage/'images'/label/f"{record['sample_id']}.png"
                    path.parent.mkdir(parents=True,exist_ok=True)
                    image.save(path)
                    metrics = lighting_metrics(condition['source'],image,
                        -config['target_luminance_delta'] if record in cohort['transfer'] else config['target_luminance_delta'],
                        config['edge_preservation_weight'],condition['content_box'])
                    row = {'label':label,'sample_id':record['sample_id'],'image_id':record['image_id'],
                           'source_SHA256':record['image_sha256'],'seed':seed,'policy_version':policy.version,
                           'output_SHA256':sha256(path),'image':str(path.relative_to(output)),
                           'precision':precision,'sampling_UNet_NFE':policy.unet_calls-nfe, **metrics}
                    with (stage/'observations.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
                    if retain:
                        dest = stage/'trajectories'/label/f"{record['sample_id']}.pt"
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        torch.save(trajectory,dest)
                    return (trajectory,condition,metrics['training_reward_proxy']),row
                def training_step(index,label,numeric=False):
                    batch, rows = [], []
                    for j in range(config['logical_batch_size']):
                        record = cohort['train'][(index*config['logical_batch_size']+j)%len(cohort['train'])]
                        episode,row = collect(record,label,config['seed']*1000000+index*100+j,
                                              f"numeric_{index}_{j}" if numeric else None,True)
                        batch.append(episode); rows.append(row)
                    receipt = trainer.update(batch,keep_gradients=numeric)
                    event('optimizer_update',{'precision':precision,'label':label,
                                             'mean_terminal_reward':float(np.mean([r['training_reward_proxy'] for r in rows])),**receipt})
                    return rows
                def numerical_step(index,label):
                    rows = training_step(index,label,True)
                    result = {**trainer.witness(),'output_hashes':[r['output_SHA256'] for r in rows],
                              'state':trainer.state()}
                    path = stage/'numerical'/f'{label}.pt'
                    path.parent.mkdir(exist_ok=True)
                    torch.save(result,path)
                    return result
                a1 = numerical_step(0,'ordinary_A1')
                after_one = stage/'checkpoints/after_one'
                trainer.save(after_one)
                a2 = numerical_step(1,'ordinary_A2')
                trainer.restore(initial)
                b1 = numerical_step(0,'ordinary_B1')
                b2 = numerical_step(1,'ordinary_B2')
                trainer.restore(after_one)
                resumed = numerical_step(1,'resumed_next_update')
                checks = {'first_update':compare_trees(a1,b1),'second_update':compare_trees(a2,b2),
                          'completed_next_update_restore':compare_trees(a2,resumed)}
                protected = frozen_digest(policy.pipe)==frozen
                passed = all(x['exact'] and x['tolerance'] for x in checks.values()) and protected
                gate = {'passed':passed,'checks':checks,'frozen_weights_unchanged':protected,
                        'optimizer_updates_executed':5,'base_precision':str(editor.dtype),
                        'LoRA_precision':sorted({str(p.dtype) for p in trainer.params}),
                        'adapter_linear_compute':'FP32 outside autocast','BF16_repair_claim':False,
                        'seconds':time.perf_counter()-precision_start}
                write_json(stage/'numerical_gate.json',gate,exclusive=True)
                state['stages'][precision] = gate
                event('numerical_gate',{'precision':precision,**gate})
                del a1,a2,b1,b2,resumed
                persist()
                if not passed:
                    continue
                active_precision = precision
                # Prefer the audited FP32 route for the controlled pilot even if
                # BF16 repeats pass at this small shape. This is preregistered,
                # not evidence that the historical BF16 problem is fixed.
                if precision == 'bf16': continue
                trainer.restore(initial)
                def evaluate(split,label):
                    offset = {'dev':200000,'test':300000,'transfer':400000}[split]
                    return [collect(r,label,config['seed']*1000000+offset+j)[1]
                            for j,r in enumerate(cohort[split])]
                baseline = {split:evaluate(split,'frozen_'+split) for split in ('dev','test','transfer')}
                for index in range(config['inner_updates']): training_step(index,'inner_'+str(index))
                inner_checkpoint = stage/'checkpoints/inner_final'
                trainer.save(inner_checkpoint,{'active_lr':config['learning_rate'],'optimizer_steps':trainer.step})
                saved_inner = trainer.state()
                trainer.restore(inner_checkpoint)
                if not compare_trees(saved_inner,trainer.state())['exact']: raise RuntimeError('Inner checkpoint reload failed')
                inner = {split:evaluate(split,'inner_'+split) for split in ('dev','test','transfer')}
                state['inner'] = {split:paired_summary(baseline[split],inner[split]) for split in ('test','transfer')}
                state['inner']['checkpoint'] = str(inner_checkpoint/'training_state.pt')
                state['inner']['checkpoint_SHA256'] = sha256(inner_checkpoint/'training_state.pt')
                state['inner']['optimizer_updates'] = config['inner_updates']
                write_json(stage/'inner_comparison.json',{'frozen':baseline,'inner':inner,'paired':state['inner']},exclusive=True)
                event('inner_complete',state['inner'])
                persist()
                # Fixed recipe, deterministic rule, and UCB consume identical
                # rollout/update/dev query counts. Test/transfer never guide search.
                outer = {}
                for method in ('fixed_recipe','fixed_rule','ucb_search'):
                    trainer.restore(inner_checkpoint)
                    selector = RecipeSelector(method,config['outer_learning_rates'],config['learning_rate'])
                    incumbent, incumbent_dev = inner_checkpoint, float(np.mean([r['training_reward_proxy'] for r in inner['dev']]))
                    history, method_start, nfe_start = [],time.perf_counter(),policy.unet_calls
                    for round_index in range(config['outer_rounds']):
                        check_time()
                        trainer.restore(incumbent)
                        rate = selector.propose(round_index)
                        for group in trainer.optimizer.param_groups: group['lr'] = rate
                        round_start, round_nfe = time.perf_counter(),policy.unet_calls
                        for update_index in range(config['outer_candidate_updates']):
                            training_step(10+round_index*config['outer_candidate_updates']+update_index,
                                          f'{method}_r{round_index}_u{update_index}')
                        candidate = stage/f'checkpoints/{method}_candidate_{round_index}'
                        trainer.save(candidate,{'active_lr':rate,'selection_split':'dev_only'})
                        rows = evaluate('dev',f'{method}_r{round_index}_dev')
                        score = float(np.mean([r['training_reward_proxy'] for r in rows]))
                        delta = score-incumbent_dev
                        accepted = delta>config['outer_accept_min_delta']
                        selector.observe(rate,delta)
                        entry = {'method':method,'round':round_index,'learning_rate':rate,'incumbent_dev':incumbent_dev,
                            'candidate_dev':score,'dev_delta':delta,'accepted':accepted,
                            'reason':'dev_proxy_above_fixed_threshold' if accepted else 'dev_proxy_below_fixed_threshold',
                            'candidate_checkpoint':str(candidate/'training_state.pt'),
                            'candidate_SHA256':sha256(candidate/'training_state.pt'),
                            'optimizer_updates_attempted':config['outer_candidate_updates'],
                            'training_rollouts':config['outer_candidate_updates']*config['logical_batch_size'],
                            'dev_queries':len(rows),'total_UNet_NFE':policy.unet_calls-round_nfe,
                            'seconds':time.perf_counter()-round_start,'selector_state':json.loads(json.dumps(selector.state()))}
                        history.append(entry); event('outer_candidate',entry)
                        if accepted: incumbent,incumbent_dev = candidate,score
                    trainer.restore(incumbent)
                    selected = stage/f'checkpoints/{method}_selected'
                    trainer.save(selected,{'selector':selector.state(),'active_lr':trainer.optimizer.param_groups[0]['lr']})
                    results = {split:evaluate(split,method+'_'+split) for split in ('test','transfer')}
                    outer[method] = {'history':history,'final_evaluation':results,
                        'paired_against_inner':{split:paired_summary(inner[split],results[split]) for split in results},
                        'optimizer_updates_attempted':config['outer_rounds']*config['outer_candidate_updates'],
                        'total_UNet_NFE':policy.unet_calls-nfe_start,'seconds':time.perf_counter()-method_start,
                        'selected_checkpoint':str(selected/'training_state.pt'),
                        'selected_checkpoint_SHA256':sha256(selected/'training_state.pt'),
                        'classification':'recipe_adaptive_search_HPO; RSI_not_established'}
                    write_json(stage/'outer_comparison_partial.json',outer)
                    persist()
                if len({x['total_UNet_NFE'] for x in outer.values()}) != 1:
                    raise RuntimeError('Outer methods consumed unmatched model evaluation budgets')
                state['outer'] = outer
                write_json(stage/'outer_comparison.json',outer,exclusive=True)
                state['frozen_weights_unchanged_at_finish'] = frozen_digest(policy.pipe)==frozen
                if not state['frozen_weights_unchanged_at_finish']: raise RuntimeError('Frozen base weights changed')
                state['precision_used_for_comparison'] = precision
                state['total_UNet_NFE_FP32'] = policy.unet_calls
                state['peak_GPU_allocated_bytes'] = torch.cuda.max_memory_allocated()
                state['status'] = 'COMPLETED_BOUNDED_DEVELOPMENT_SMOKE'
            except Exception:
                error = traceback.format_exc()
                (stage/'failure.txt').write_text(error,encoding='utf-8')
                state['stages'][precision] = {'passed':False,'failure':error,'seconds':time.perf_counter()-precision_start}
                event('stage_failed',{'precision':precision,'error':error})
                if precision == 'fp32': raise
            finally:
                # No base weights are copied or deleted. Release only this process's tensors.
                if trainer is not None: del trainer
                if policy is not None: del policy
                if editor is not None: del editor
                gc.collect(); torch.cuda.empty_cache()
                persist()
        if state['status'] != 'COMPLETED_BOUNDED_DEVELOPMENT_SMOKE':
            raise RuntimeError('No precision passed the complete numerical and control path')
    except Exception:
        state['status'] = 'FAILED_STOPPED_NO_EXPANSION'
        state['failure'] = traceback.format_exc()
        (output/'failure.txt').write_text(state['failure'],encoding='utf-8')
        raise
    finally:
        state['finished_UTC'] = utc()
        state['retained_bytes'] = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        persist()


if __name__ == '__main__':
    main()
