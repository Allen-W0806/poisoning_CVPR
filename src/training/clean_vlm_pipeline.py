"""Clean response-only VQA LoRA, strict checkpoints, and normal evaluation."""
import copy
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.data.p0_protocol import sha256,metric_summary
from src.evaluation.vqa_official_compatible import score_answer
from src.mini_pilot.engine import cpu_copy,compare_trees


def rng_state():
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            'numpy':np.random.get_state(),'python':random.getstate()}


def restore_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None:torch.cuda.set_rng_state_all(state['cuda'])


def parameter_hash(items):
    import hashlib
    h=hashlib.sha256()
    for name,p in items:
        h.update(name.encode());h.update(str(p.dtype).encode());h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()


def response_labels(prompt_ids,full_ids):
    if prompt_ids.ndim!=2 or full_ids.ndim!=2 or len(prompt_ids)!=1 or len(full_ids)!=1:
        raise ValueError('Physical clean QA batch must contain one image/question')
    n=prompt_ids.shape[1]
    if full_ids.shape[1]<=n or not torch.equal(prompt_ids,full_ids[:,:n]):
        raise ValueError('Assistant-response prefix/label mismatch')
    labels=full_ids.clone();labels[:,:n]=-100
    return labels


def attach_language_lora(qwen,rank,alpha):
    from peft import get_peft_model,LoraConfig
    regex=r'.*language_model\.layers\.\d+\.self_attn\.(q_proj|v_proj)'
    qwen.model=get_peft_model(qwen.model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.,target_modules=regex,bias='none'))
    for name,p in qwen.model.named_parameters():
        if p.requires_grad:p.data=p.data.float()
    qwen.model.config.use_cache=False
    return qwen


class CleanVLMTrainer:
    def __init__(self,qwen,assets_root,protocol,manifest_sha):
        self.qwen,self.model,self.assets=qwen,qwen.model,Path(assets_root)
        self.protocol=protocol;self.step=0
        self.params={n:p for n,p in self.model.named_parameters() if p.requires_grad}
        if not self.params or any('language_model.layers.' not in n or 'lora_' not in n or p.dtype!=torch.float32 for n,p in self.params.items()):
            raise ValueError('Only FP32 language attention LoRA may train')
        self.optimizer=torch.optim.AdamW(list(self.params.values()),lr=protocol['vlm_learning_rate'],betas=(.9,.999),eps=1e-8,weight_decay=0.)
        self.scheduler=torch.optim.lr_scheduler.LambdaLR(self.optimizer,lambda _:1.)
        self.contract={'schema':'clean_vlm_p0_v1','model_revision':qwen.revision,'protocol':copy.deepcopy(protocol),
                       'manifest_SHA256':manifest_sha,'trainable_names':list(self.params)}
        self.raw_gradients={}

    def example(self,row):
        from PIL import Image
        with Image.open(self.assets/row['image_path']) as image:image=image.convert('RGB')
        messages=[{'role':'system','content':[{'type':'text','text':self.qwen.system_prompt}]},
                  {'role':'user','content':[{'type':'image','image':image},{'type':'text','text':row['question']}]}]
        processor=self.qwen.processor
        prompt=processor.apply_chat_template([messages],tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',padding=True)
        full=processor.apply_chat_template([messages+[{'role':'assistant','content':[{'type':'text','text':row['training_answer']}]}]],
              tokenize=True,add_generation_prompt=False,return_dict=True,return_tensors='pt',padding=True)
        full['labels']=response_labels(prompt['input_ids'],full['input_ids'])
        return full.to(self.qwen.device)

    def update(self,rows):
        if len(rows)!=self.protocol['vlm_accumulation']:raise ValueError('Wrong clean logical batch')
        self.model.train();self.model.config.use_cache=False;self.optimizer.zero_grad(set_to_none=True)
        before={n:p.detach().cpu().clone() for n,p in self.params.items()};losses=[];started=time.perf_counter()
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for row in rows:
                output=self.model(**self.example(row));loss=output.loss/len(rows)
                if not loss.requires_grad or not torch.isfinite(loss):raise FloatingPointError('Disconnected/nonfinite supervised loss')
                loss.backward();losses.append(float(output.loss.detach()))
        self.raw_gradients={n:p.grad.detach().cpu().clone() for n,p in self.params.items() if p.grad is not None}
        if len(self.raw_gradients)!=len(self.params) or not all(torch.isfinite(v).all() for v in self.raw_gradients.values()):
            raise FloatingPointError('Missing/nonfinite clean LoRA gradients')
        norm=torch.nn.utils.clip_grad_norm_(list(self.params.values()),self.protocol['vlm_max_grad_norm'])
        if not torch.isfinite(norm) or norm<=0:raise FloatingPointError('Zero/nonfinite clean gradient')
        self.optimizer.step();self.scheduler.step();self.step+=1
        if not all(torch.isfinite(v).all() for state in self.optimizer.state.values() for v in state.values() if isinstance(v,torch.Tensor)):
            raise FloatingPointError('Nonfinite Adam state')
        changes=[float((p.detach().cpu()-before[n]).abs().max()) for n,p in self.params.items()]
        if max(changes)<=0:raise FloatingPointError('No actual clean LoRA update')
        return {'retained_step':self.step,'sample_ids':[r['sample_id'] for r in rows],'loss':float(np.mean(losses)),
                'raw_gradient_norm':float(norm),'max_parameter_change':max(changes),'changed_tensors':sum(x>0 for x in changes),
                'finite_FP32_gradients':True,'actual_optimizer_calls':1,'seconds':time.perf_counter()-started,
                'objective':'response_only_clean_QA_cross_entropy_NOT_RL'}

    def state(self):
        return {'params':{n:p.detach().cpu().clone() for n,p in self.params.items()},
                'optimizer':cpu_copy(self.optimizer.state_dict()),'scheduler':cpu_copy(self.scheduler.state_dict()),
                'rng':rng_state(),'step':self.step,'contract':copy.deepcopy(self.contract)}

    def save(self,directory):
        path=Path(directory);path.mkdir(parents=True,exist_ok=False)
        self.model.save_pretrained(path/'adapter',safe_serialization=True)
        torch.save(self.state(),path/'training_state.pt')
        (path/'integrity.json').write_text(json.dumps({'training_state_SHA256':sha256(path/'training_state.pt'),
              'adapter_files':{p.name:sha256(p) for p in (path/'adapter').iterdir() if p.is_file()}},indent=2))

    def restore(self,directory):
        path=Path(directory);receipt=json.loads((path/'integrity.json').read_text())
        if sha256(path/'training_state.pt')!=receipt['training_state_SHA256']:raise ValueError('Clean checkpoint byte mismatch')
        state=torch.load(path/'training_state.pt',map_location='cpu',weights_only=False)
        if state['contract']!=self.contract or set(state['params'])!=set(self.params):raise ValueError('Clean checkpoint contract mismatch')
        for n,p in self.params.items():
            if p.shape!=state['params'][n].shape or p.dtype!=state['params'][n].dtype or not torch.isfinite(state['params'][n]).all():
                raise ValueError('Invalid checkpoint adapter tensor')
        for n,p in self.params.items():p.data.copy_(state['params'][n].to(p.device))
        self.optimizer.load_state_dict(copy.deepcopy(state['optimizer']));self.scheduler.load_state_dict(copy.deepcopy(state['scheduler']))
        self.step=state['step'];restore_rng(state['rng'])
        if not compare_trees(state,self.state())['exact']:raise ValueError('Clean full state did not restore exactly')

    def evaluate(self,rows,label,image_overrides=None):
        from PIL import Image
        predictions=[];self.model.eval();self.model.config.use_cache=True
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            for row in rows:
                path=(image_overrides or {}).get(row['sample_id'],self.assets/row['image_path'])
                with Image.open(path) as image:image=image.convert('RGB')
                prediction=self.qwen.answer([image],[row['question']])[0]
                result=score_answer(prediction,row)
                predictions.append({'split':label,'sample_id':row['sample_id'],'image_id':row['image_id'],
                       'source_SHA256':row['image_sha256'],'input_SHA256':sha256(path),'question':row['question'],
                       'prediction':prediction,'normalized_prediction':result['normalized_answer'],'score':result['score'],
                       'reference_scope':'original ten human QA answers; edited inputs are reference consistency only' if image_overrides else 'original ten human QA answers'})
        self.model.config.use_cache=False
        return predictions,metric_summary([r['score'] for r in predictions],self.protocol['analysis_seed'],self.protocol['bootstrap_resamples'])
