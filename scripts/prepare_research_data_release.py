"""Curate small research records and explicitly allowlisted clean-VQA adapters."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import csv
import io
import struct
import subprocess
from export_github_snapshot import PRIVATE_PATTERNS

ROOT=Path(__file__).resolve().parents[1]
TEXT={'.json','.jsonl','.csv','.md','.txt','.log'}
MAX_TEXT=5_000_000
MAX_GIT=100_000_000


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worktree',required=True);a=parser.parse_args();target=Path(a.worktree).resolve()
    if not (target/'.git').is_dir() or target==ROOT:raise ValueError('Use the separate GitHub worktree, not the original source Git history')
    copied=[];skipped=[];findings=[]
    cmp_path=ROOT/'outputs/overnight_benign_20261112T051015Z/comparison.json';comparison=json.loads(cmp_path.read_text(encoding='utf-8'))
    if comparison['status']!='COMPLETE_3_SEED_READ_ONLY_ANALYSIS':raise ValueError('Cannot publish incomplete results as complete')
    rows=[]
    for split,value in comparison['splits'].items():
        rows.extend([{'split':split,'arm':'Frozen','n_sources':value['n_sources'],'training_seeds':0,'score':value['frozen'],'training_seed_SD':None,'paired_delta':None,'paired_source_CI_low':None,'paired_source_CI_high':None},
          {'split':split,'arm':'clean_VQA_SFT','n_sources':value['n_sources'],'training_seeds':3,'score':value['SFT_mean'],'training_seed_SD':value['SFT_training_seed_SD'],'paired_delta':value['paired_delta_mean'],'paired_source_CI_low':value['paired_delta_source_bootstrap_95CI'][0],'paired_source_CI_high':value['paired_delta_source_bootstrap_95CI'][1]}])
    payload={'source':cmp_path.relative_to(ROOT).as_posix(),'source_SHA256':digest(cmp_path),'status':'ACTUALLY_RUN_AND_READ_ONLY_AUDITED','rows':rows,'uncertainty':'Source CI conditional on the three training seeds; training-seed SD separately reported. Transfer is source-held-out same VQA family.','GPU_hours':1.2158333333333333}
    (ROOT/'reports/PAPER_RESULTS_20261114.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (ROOT/'reports/PAPER_RESULTS_20261114.csv').write_text(buffer.getvalue(),encoding='utf-8',newline='')
    for folder in ['outputs','reports']:
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or path.suffix.lower() not in TEXT:continue
            relative=path.relative_to(ROOT).as_posix();lower=relative.lower();parts={p.lower() for p in path.parts}
            if any(p.startswith('github_') for p in parts):continue
            if any(p in parts for p in ['.git','__pycache__','source_snapshot','reproduction_code','posthoc_source','source_code','code','caches','cache','images','checkpoints','tmp']):
                skipped.append({'path':relative,'reason':'source/cache/image/checkpoint subtree'});continue
            if any(p in parts for p in ['monitor','monitoring','overnight_supervisor','rsi_stage_monitor','report_sync']):
                skipped.append({'path':relative,'reason':'account-specific operations metadata, not paper metrics'});continue
            if any(word in lower for word in ['shadowcast','drive','credential','oauth','client_secret','reauthoriz','/sync','report_sync','before.py.txt','before_fix.py.txt','before_no_requeue.py.txt']):
                skipped.append({'path':relative,'reason':'excluded control/authentication/history utility'});continue
            size=path.stat().st_size
            if size>MAX_TEXT:skipped.append({'path':relative,'reason':'larger than small-text release threshold','bytes':size});continue
            data=path.read_bytes()
            try:data.decode('utf-8-sig')
            except UnicodeDecodeError:
                skipped.append({'path':relative,'reason':'non-UTF8 text; preserve original locally'});continue
            local_findings=[kind for kind,pattern in PRIVATE_PATTERNS if pattern.search(data)]
            if local_findings:
                findings.append({'path':relative,'patterns':local_findings});skipped.append({'path':relative,'reason':'credential pattern; original retained locally'});continue
            dest=target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            changed=not dest.exists() or digest(dest)!=hashlib.sha256(data).hexdigest()
            if changed:dest.write_bytes(data)
            copied.append({'path':relative,'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest(),'changed_or_new':changed})
    inventory=json.loads((ROOT/'reports/ASSET_LOCATIONS_20261114.json').read_text(encoding='utf-8'))
    adapters=[]
    for adapter in inventory['selected_clean_VQA_adapters']:
        base=target/'artifacts/clean_vqa_lora'/('seed'+str(adapter['seed']))
        for name,meta in adapter['files'].items():
            p=base/name
            if p.stat().st_size>=MAX_GIT or digest(p)!=meta['SHA256']:raise ValueError('Adapter size/hash gate failed')
        tensor=base/'adapter_model.safetensors'
        with tensor.open('rb') as stream:header=json.loads(stream.read(struct.unpack('<Q',stream.read(8))[0]))
        tensors={k:v for k,v in header.items() if k!='__metadata__'}
        if not tensors or any(v['dtype']!='F32' for v in tensors.values()):raise ValueError('Unexpected adapter tensor dtype')
        adapters.append({'seed':adapter['seed'],'path':tensor.relative_to(target).as_posix(),'bytes':tensor.stat().st_size,'SHA256':digest(tensor),'tensor_count':len(tensors),'dtype':'F32','source_directory':adapter['directory']})
    artifact_manifest={'status':'VERIFIED_BYTE_IDENTICAL_SELECTED_CLEAN_VQA_LORA','base_model_id':'Qwen/Qwen3-VL-2B-Instruct','base_model_revision':'89644892e4d85e24eaac8bacfd4f463576704203','base_model_card':'https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md','source_scope':'Normal VQAv2 response-only SFT, no edited/poison data; selected on dev only.','adapters':adapters,'full_optimizer_RNG_states':'Retained at original cluster checkpoint paths; not uploaded.'}
    (target/'artifacts/clean_vqa_lora/ADAPTER_MANIFEST.json').write_text(json.dumps(artifact_manifest,indent=2),encoding='utf-8')
    skipped_assets=[]
    for model in inventory['models']:
        for file in model['files']:
            if file['bytes']>=MAX_GIT:skipped_assets.append({'path':model['absolute_path']+'/'+file['relative'],'bytes':file['bytes'],'reason':'base model weight; >=100MB; reproduce from pinned HuggingFace ID instead'})
    for item in inventory['metric_assets']:
        if item['bytes']>=MAX_GIT:skipped_assets.append({'path':item['absolute_path'],'bytes':item['bytes'],'reason':'metric backbone weight; >=100MB; download source recorded instead'})
    for item in inventory['dataset_archives']:skipped_assets.append({'path':item['absolute_path'],'bytes':item['bytes'],'reason':'raw dataset archive excluded irrespective of individual size'})
    skipped_assets.append({'path':inventory['clean_cohort']['directory'],'bytes':inventory['clean_cohort']['image_bytes'],'files':inventory['clean_cohort']['images'],'reason':'raw source images; manifest and source URLs published instead'})
    manifest={'prepared_UTC':datetime.now(timezone.utc).isoformat(),'scope':'Small existing benign research records plus three selected normal-VQA LoRA; original evidence retained.','text_threshold_bytes':MAX_TEXT,'normal_Git_file_limit_bytes':MAX_GIT,'copied_result_files':copied,'copied_result_count':len(copied),'new_or_updated_result_count':sum(r['changed_or_new'] for r in copied),'credential_findings_excluded':findings,'skipped_text_files':skipped,'skipped_large_or_raw_assets':skipped_assets,'LoRA':adapters,'LFS_required_for_this_release':False,'new_GPU_jobs':0}
    (target/'reports/DATA_RELEASE_MANIFEST_20261114.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (target/'reports/SKIPPED_ASSETS_20261114.json').write_text(json.dumps({'large_or_raw_assets':skipped_assets,'text_files_not_selected':skipped},indent=2),encoding='utf-8')
    table='\n## 论文结果表与资源复现（2026-11-14）\n\n'
    table+='本节由已核验的`comparison.json`自动生成，对应`reports/PAPER_RESULTS_20261114.json`和CSV。VQA分数范围0–1，越高越好。\n\n|split|source数|Frozen|clean SFT均值 ± seed SD|配对Δ|source bootstrap 95% CI|\n|---|---:|---:|---:|---:|---|\n'
    for split,value in comparison['splits'].items():
        ci=value['paired_delta_source_bootstrap_95CI'];table+=f"|{split}|{value['n_sources']}|{value['frozen']:.6f}|{value['SFT_mean']:.6f} ± {value['SFT_training_seed_SD']:.6f}|{value['paired_delta_mean']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|\n"
    table+='\n每seed实际512次optimizer更新；dev-only选择的最终step分别为128/512/512。transfer CI包含0，不能确认迁移收益。该实验是正常VQA SFT，不能解释为图像编辑RL/Outer/RSI提升。P1未运行单元格仍为空。\n\n'
    table+='三个选定normal-VQA LoRA在`artifacts/clean_vqa_lora/seed42`、`seed123`、`seed3407`，每个tensor文件3,227,648字节。它们是FP32 LoRA，不包含基础模型或optimizer/RNG状态；加载base时必须显式固定上述Qwen revision，不能默认为最新版本。基模型声明见[固定revision模型卡](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct/blob/89644892e4d85e24eaac8bacfd4f463576704203/README.md)。\n\n'
    table+='模型和数据的已验证绝对路径、来源、revision、大小/哈希在`reports/ASSET_LOCATIONS_20261114.json`。原Windows代码位置为`H:/Project/Poisoning`；真实模型和数据位于该清单所列Linux集群路径。\n\n```bash\n# 默认只检查已有资产，不下载或运行训练\nbash scripts/download_and_setup.sh --check\n# 查看具体路径与来源\nbash scripts/download_and_setup.sh --paths-json\n# 未来确需重新下载时显式选择此模式（本轮未执行）\n# bash scripts/download_and_setup.sh --download-missing\n```\n\n'
    table+='--check已在原集群通过：3个model snapshots、1792张固定source、1796个archive/image/metric文件哈希；本轮未重新下载完整模型/原始数据、未运行GPU训练。下载分支未实际执行，不宣称下载重建已验证。\n\n'
    table+='普通Git提交中每个文件均低于100MB；本轮不需要LFS。全球忽略模型/原始数据/outputs缓存，只对三个小adapter显式放行。跳过资产和小文件范围见`reports/SKIPPED_ASSETS_20261114.json`、`DATA_RELEASE_MANIFEST_20261114.json`。`SOURCE_RELEASE_MANIFEST.json`是首次源码快照的历史清单；本次新数据以新manifest为准。\n'
    readme=target/'README.md';before=readme.read_text(encoding='utf-8').replace('私有论文研究代码快照','论文研究代码与审计数据快照').replace('原始图片和tensor checkpoint保留在原资源中，不上传到GitHub。','原始图片和完整tensor checkpoint保留在原资源中；本次只另附三个选定的小LoRA。').replace('凭据、模型权重、大型图像数据','凭据、完整基础模型权重、大型图像数据')
    readme.write_text(before+table,encoding='utf-8',newline='\n')
    (target/'artifacts/clean_vqa_lora/README.md').write_text('# Selected clean-VQA LoRA\n\nThree selected adapters only: seeds42/123/3407, each 3,227,648 bytes, FP32 language attention LoRA. Original files are byte-identical; SHA256 and base model revision are in ADAPTER_MANIFEST.json. Base Qwen3-VL-2B-Instruct revision must be 89644892e4d85e24eaac8bacfd4f463576704203. The upstream model card declares apache-2.0; no new license for the overall repository is granted by this export.\n\nThese are normal original-image QA SFT adapters, with no ASR reward or poisoning data. They are not generator/Inner/Outer/RSI checkpoints. Full optimizer/RNG states remain at the original checkpoint locations. Selection used dev data only.\n',encoding='utf-8')
    (target/'scripts/prepare_research_data_release.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({'status':'DATA_RELEASE_PREPARED','result_files':len(copied),'new_or_updated_result_files':sum(r['changed_or_new'] for r in copied),'excluded_text_files':len(skipped),'credential_findings_excluded':len(findings),'LoRA_tensor_files':len(adapters),'LoRA_bytes_each':adapters[0]['bytes'],'large_assets_omitted':len(skipped_assets),'LFS_required':False}))


if __name__=='__main__':main()
