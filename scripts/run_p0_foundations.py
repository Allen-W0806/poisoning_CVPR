"""Single bounded P0: Frozen quality calibration and clean VLM technical loop.

No diffusion/Outer learning experiment is run here. Clean VQA SFT is the
explicit P0.3 engineering exception, not a P1 method result.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.data.p0_protocol import sha256,require_manifest,metric_summary
from scripts.test_p0_foundations import source_hashes


def utc():return datetime.now(timezone.utc).isoformat()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--assets-root',required=True);ap.add_argument('--run',required=True);a=ap.parse_args()
    assets,run=Path(a.assets_root).resolve(),Path(a.run).resolve()
    protocol=json.loads((ROOT/'configs/p0_foundations_20261111.json').read_text())
    cpu=json.loads((run/'cpu_tests.json').read_text())
    if not cpu['passed'] or cpu['source_SHA256']!=source_hashes():raise ValueError('Passing CPU gate for exact source required')
    manifest=require_manifest(run,assets,protocol)
    if (run/'status.json').exists():raise ValueError('Refuse to reopen existing P0 execution')
    started=time.perf_counter();state={'started_UTC':utc(),'PID':os.getpid(),'SLURM_JOB_ID':os.environ.get('SLURM_JOB_ID'),
        'status':'P0_RUNNING','stages':{},'P1_launched':False,'protocol_SHA256':sha256(ROOT/'configs/p0_foundations_20261111.json'),
        'manifest_SHA256':sha256(run/'manifest.json'),'source_SHA256':source_hashes()}
    def persist():
        state['elapsed_seconds']=time.perf_counter()-started
        (run/'status.json').write_text(json.dumps(state,indent=2))
    def event(kind,data):
        row={'UTC':utc(),'kind':kind,**data}
        with (run/'events.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    def check_time():
        if time.perf_counter()-started>protocol['application_wall_seconds']:raise TimeoutError('P0 bounded wall clock cap')
    import numpy as np
    import torch
    if not os.environ.get('SLURM_JOB_ID') or not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('P0 model work requires an explicit single-GPU Slurm allocation')
    torch.set_num_threads(4);random.seed(protocol['seed']);np.random.seed(protocol['seed'])
    torch.manual_seed(protocol['seed']);torch.cuda.manual_seed_all(protocol['seed'])
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    kernel=torch.randn(16,16,device='cuda',generator=torch.Generator(device='cuda').manual_seed(42))
    if not torch.isfinite(kernel@kernel).all():raise RuntimeError('CUDA witness failed')
    state['GPU']={'name':torch.cuda.get_device_name(),'initial_free_bytes':torch.cuda.mem_get_info()[0],
                  'total_bytes':torch.cuda.get_device_properties(0).total_memory,'visible':os.environ.get('CUDA_VISIBLE_DEVICES')}
    event('kernel_witness',state['GPU']);persist()
    try:
        state['stages']['frozen_quality']={'status':'RUNNING'};persist()
        try:
            quality=frozen_quality(assets,run,protocol,manifest,check_time,event)
            state['stages']['frozen_quality']=quality
        except Exception:
            trace=traceback.format_exc();(run/'frozen_failure.txt').write_text(trace)
            state['stages']['frozen_quality']={'status':'FAILED','trace':trace};event('frozen_failed',{'trace':trace})
        gc.collect();torch.cuda.empty_cache();persist()
        # Independent P0.3 proceeds even if editor quality/metrics were blocked.
        state['stages']['clean_vlm']={'status':'RUNNING'};persist()
        try:
            clean=clean_vlm(assets,run,protocol,manifest,check_time,event)
            state['stages']['clean_vlm']=clean
        except Exception:
            trace=traceback.format_exc();(run/'clean_vlm_failure.txt').write_text(trace)
            state['stages']['clean_vlm']={'status':'FAILED','trace':trace};event('clean_vlm_failed',{'trace':trace})
        state['status']='P0_EXECUTED_VISUAL_REVIEW_REQUIRED' if all(v['status']!='FAILED' for v in state['stages'].values()) else 'P0_PARTIAL_FAILURE'
    finally:
        state['finished_UTC']=utc();state['peak_GPU_allocated_bytes']=torch.cuda.max_memory_allocated();persist()
    if state['status']=='P0_PARTIAL_FAILURE':raise SystemExit(1)


def frozen_quality(assets,run,protocol,manifest,check_time,event):
    import numpy as np
    import torch
    from PIL import Image
    from src.editing.pretrained_editor import PretrainedEditor
    from src.models.pretrained_editor_loader import pretrained_config
    from src.evaluation.benign_editing_quality import EditingQualityMetrics
    from src.mini_pilot.engine import PilotPolicy,frozen_digest
    output=run/'frozen';output.mkdir(exist_ok=False)
    editor=PretrainedEditor(pretrained_config(assets,precision=protocol['editor_precision'],seed=protocol['seed'],
        image_resolution=protocol['editor_resolution'],num_inference_steps=30,
        guidance_scale=protocol['editor_text_guidance'],image_guidance_scale=protocol['editor_image_guidance']),device='cuda')
    if editor.config.revision!=protocol['editor_revision']:raise ValueError('Unexpected editor revision')
    metadata=json.loads((assets/'reports/phase2_model_metadata.json').read_text())['timbrooks/instruct-pix2pix']
    model_files={name:sha256(Path(metadata['path'])/name) for name in metadata['files'] if name.endswith('.safetensors')}
    (output/'model_weight_hashes.json').write_text(json.dumps(model_files,indent=2))
    metrics=EditingQualityMetrics(assets)
    (output/'metric_resources.json').write_text(json.dumps(metrics.receipt,indent=2))
    originals={};observations=[]
    for row in manifest['splits']['dev']:
        with Image.open(assets/row['image_path']) as image:image=image.convert('RGB')
        originals[row['sample_id']]=image
        path=output/'sources'/f"{row['sample_id']}.png";path.parent.mkdir(exist_ok=True)
        editor.prepare_image(image).save(path)
    # Encoder/decoder roundtrip separates reconstruction from diffusion quality.
    with torch.no_grad():
        for row in manifest['splits']['dev']:
            check_time();image=originals[row['sample_id']];source=editor.prepare_image(image)
            tensor=editor.pipe.image_processor.preprocess(source).to(editor.device,editor.dtype)
            latent=editor.pipe.vae.encode(tensor).latent_dist.mode()
            decoded=editor.pipe.vae.decode(latent,return_dict=False)[0]
            reconstructed=editor.pipe.image_processor.postprocess(decoded,output_type='pil')[0]
            path=output/'vae_roundtrip'/f"{row['sample_id']}.png";path.parent.mkdir(exist_ok=True);reconstructed.save(path)
            record={'config':'vae_roundtrip','sample_id':row['sample_id'],'image_id':row['image_id'],
                    'instruction':'reconstruct source; no diffusion edit','path':path.relative_to(run).as_posix(),
                    'image_SHA256':sha256(path),**metrics.measure(image,source,reconstructed,protocol['editor_resolution'])}
            observations.append(record);event('frozen_observation',record)
    for setting in protocol['editor_configs']:
        check_time();policy=None
        if setting['scheduler']=='ddim_stochastic':
            policy=PilotPolicy(editor,{'lora_rank':2,'lora_alpha':4,'adapter_compute_precision':'fp32',
                'gradient_checkpointing':False,'num_inference_steps':setting['steps'],'eta':1.,
                'learning_rate':1e-5,'logical_batch_size':1,'physical_micro_batch_size':1,
                'gradient_accumulation_steps':1,'max_grad_norm':.5})
            protected=frozen_digest(editor.pipe)
        else:protected=frozen_digest(editor.pipe)
        for index,row in enumerate(manifest['splits']['dev']):
            check_time();image=originals[row['sample_id']];source=editor.prepare_image(image)
            path=output/setting['name']/f"{row['sample_id']}.png";path.parent.mkdir(exist_ok=True)
            seed=protocol['seed']*1000000+index;started=time.perf_counter()
            if policy is None:
                generated=editor.edit(image,protocol['editor_instruction'],row['image_id'],seed,path,setting['steps']).image
            else:
                condition=policy.condition({**row,'image':image},protocol['editor_instruction'])
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    generated,trajectory=policy.sample(condition,seed,f"quality_{row['sample_id']}")
                generated.save(path)
            record={'config':setting['name'],'sample_id':row['sample_id'],'image_id':row['image_id'],
                    'instruction':protocol['editor_instruction'],'steps':setting['steps'],'seed':seed,
                    'resolution':protocol['editor_resolution'],'scheduler':type(editor.pipe.scheduler).__name__,
                    'path':path.relative_to(run).as_posix(),'image_SHA256':sha256(path),
                    'seconds':time.perf_counter()-started,'optimizer_updates':0,
                    **metrics.measure(image,source,generated,protocol['editor_resolution'])}
            observations.append(record);event('frozen_observation',record)
        if frozen_digest(editor.pipe)!=protected:raise AssertionError('Frozen weights changed during inference')
    summary={name:{key:float(np.mean([row[key] for row in observations if row['config']==name]))
                         for key in ('SSIM','LPIPS','luminance_delta','clipped_channel_fraction')}
             for name in ['vae_roundtrip']+[s['name'] for s in protocol['editor_configs']]}
    (output/'observations.json').write_text(json.dumps(observations,indent=2));(output/'summary.json').write_text(json.dumps(summary,indent=2))
    result={'status':'MEASURED_VISUAL_REVIEW_REQUIRED','dev_sources':len(manifest['splits']['dev']),
            'configs':summary,'optimizer_updates':0,'instruction_success_ground_truth':'NOT_AVAILABLE',
            'P1_gate':'NOT_OPEN','base_variant':'fp16 weights, BF16 compute'}
    event('frozen_complete',result)
    return result


def clean_vlm(assets,run,protocol,manifest,check_time,event):
    import torch
    from peft import PeftModel
    from src.models.qwen_loader import QwenVQA
    from src.training.clean_vlm_pipeline import CleanVLMTrainer,attach_language_lora,parameter_hash
    from src.mini_pilot.engine import compare_trees,cpu_copy
    output=run/'clean_vlm';output.mkdir(exist_ok=False)
    manifest_sha=sha256(run/'manifest.json')
    def load():
        check_time()
        return QwenVQA(model_id=protocol['vlm_model_id'],revision=protocol['vlm_revision'],device='cuda',
                       max_new_tokens=protocol['vlm_max_new_tokens'],min_pixels=4096,max_pixels=protocol['vlm_max_pixels'],seed=protocol['seed'])
    qwen=attach_language_lora(load(),protocol['vlm_rank'],protocol['vlm_alpha'])
    trainer=CleanVLMTrainer(qwen,assets,protocol,manifest_sha)
    before_hash=parameter_hash((n,p) for n,p in qwen.model.named_parameters() if not p.requires_grad)
    (output/'optimizer_inventory.json').write_text(json.dumps({'trainable_tensors':len(trainer.params),
        'trainable_elements':sum(p.numel() for p in trainer.params.values()),
        'names_shapes_dtypes':{n:{'shape':list(p.shape),'dtype':str(p.dtype)} for n,p in trainer.params.items()},
        'vision_frozen':all(not p.requires_grad for n,p in qwen.model.named_parameters() if '.visual.' in n)},indent=2))
    predictions=[];training=[]
    def evaluate(split,label,overrides=None):
        check_time();rows,stats=trainer.evaluate(manifest['splits'][split],label,overrides)
        predictions.extend(rows)
        with (output/'predictions.jsonl').open('a') as stream:
            for row in rows:stream.write(json.dumps(row)+'\n')
        event('clean_evaluation',{'label':label,**stats});return rows,stats
    def update(rows,label):
        check_time();receipt=trainer.update(rows);receipt['execution_label']=label;training.append(receipt)
        with (output/'training.jsonl').open('a') as stream:stream.write(json.dumps(receipt)+'\n')
        event('clean_optimizer_update',receipt)
    before,stats_before=evaluate('dev','dev_before')
    train=manifest['splits']['train'];update(train[:2],'retained_step1')
    checkpoint=output/'checkpoints/step1';trainer.save(checkpoint)
    update(train[2:4],'reference_step2')
    expected={'state':trainer.state(),'raw_gradients':cpu_copy(trainer.raw_gradients)}
    torch.save(expected,output/'reference_next_update.pt')
    # Restore a new pinned model + public adapter + new optimizer, then take
    # the same actual next step. This is stronger than weight-only restoration.
    fresh=load();fresh.model=PeftModel.from_pretrained(fresh.model,checkpoint/'adapter',is_trainable=True)
    qwen=fresh;trainer=CleanVLMTrainer(qwen,assets,protocol,manifest_sha);trainer.restore(checkpoint)
    update(train[2:4],'fresh_object_replayed_step2')
    actual={'state':trainer.state(),'raw_gradients':cpu_copy(trainer.raw_gradients)}
    torch.save(actual,output/'fresh_next_update.pt')
    check=compare_trees(expected,actual)
    (output/'resume_comparison.json').write_text(json.dumps(check,indent=2))
    if not check['exact']:raise AssertionError('Fresh-object clean VLM next-step equality failed')
    after,stats_after=evaluate('dev','dev_after')
    final=output/'checkpoints/final';trainer.save(final)
    frozen_after=parameter_hash((n,p) for n,p in qwen.model.named_parameters() if not p.requires_grad)
    if frozen_after!=before_hash:raise AssertionError('Clean VLM frozen base changed')
    test,stats_test=evaluate('test','independent_source_test')
    transfer,stats_transfer=evaluate('transfer','source_transfer_clean_QA')
    # Edited dev answers are reference consistency, NOT independently relabeled
    # accuracy. The config was prespecified before editing; never chosen by QA.
    overrides={row['sample_id']:run/'frozen'/protocol['prespecified_QA_edited_dev_config']/f"{row['sample_id']}.png" for row in manifest['splits']['dev']}
    edited=None
    if all(path.is_file() for path in overrides.values()):
        _,edited=evaluate('dev','edited_dev_original_reference_consistency',overrides)
    del expected,actual
    final_state=trainer.state();saved_params=parameter_hash(trainer.params.items())
    fresh=load();fresh.model=PeftModel.from_pretrained(fresh.model,final/'adapter',is_trainable=True)
    qwen=fresh;trainer=CleanVLMTrainer(qwen,assets,protocol,manifest_sha);trainer.restore(final)
    repeat,repeat_stats=evaluate('dev','fresh_reload_dev')
    if not compare_trees(final_state,trainer.state())['exact']:
        # Greedy evaluation should not consume training RNG; check explicitly.
        raise AssertionError('Fresh final state/RNG changed during greedy dev evaluation')
    if [(r['prediction'],r['score']) for r in after]!=[(r['prediction'],r['score']) for r in repeat]:
        raise AssertionError('Fresh final reloaded predictions differ')
    if parameter_hash(trainer.params.items())!=saved_params:raise AssertionError('Fresh adapter checksum differs')
    result={'status':'PASS_ENGINEERING_ONLY','model_id':protocol['vlm_model_id'],'revision':protocol['vlm_revision'],
            'seed':protocol['seed'],'retained_optimizer_steps':trainer.step,'actual_optimizer_calls':len(training),
            'train_sources_used':4,'dev':stats_after,'dev_before':stats_before,'test':stats_test,'transfer_clean_QA':stats_transfer,
            'edited_dev_original_reference_consistency_proxy':edited,'fresh_object_next_step_exact':check['exact'],
            'fresh_final_predictions_exact':True,'frozen_base_unchanged':True,'frozen_base_SHA256':before_hash,
            'final_checkpoint_SHA256':sha256(final/'training_state.pt'),'final_adapter_parameter_SHA256':saved_params,
            'peak_GPU_bytes':torch.cuda.max_memory_allocated(),'full_model_training':False,
            'scientific_learning_benefit':'NOT_ESTABLISHED','independent_task_family_RSI':'NOT_RUN'}
    (output/'metrics.json').write_text(json.dumps(result,indent=2));event('clean_complete',result)
    return result


if __name__=='__main__':main()
