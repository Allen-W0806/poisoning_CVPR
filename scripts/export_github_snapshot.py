"""Prepare a byte-preserving source/evidence export without touching original Git history."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]
EXTENSIONS={'.py','.json','.yaml','.yml','.md','.txt','.sh','.sbatch','.ps1','.js','.mjs','.cjs','.toml'}
PRIVATE_PATTERNS=[('private_key',re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')),
 ('github_token',re.compile(rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})')),
 ('aws_access_key',re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b')),
 ('google_client_secret',re.compile(rb'GOCSPX-[A-Za-z0-9_-]{20,}'))]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--destination',required=True);a=parser.parse_args()
    destination=Path(a.destination).resolve()
    if not destination.is_relative_to(ROOT/'outputs') or destination.exists():raise ValueError('Require new export under project outputs')
    destination.mkdir(parents=True);included={};excluded=[];violations=[]
    def copy(path):
        path=Path(path);relative=path.relative_to(ROOT).as_posix()
        if relative in included:return
        if path.is_symlink():raise ValueError('Do not export symlinks: '+relative)
        if path.stat().st_size>5_000_000:raise ValueError('Unexpected large file: '+relative)
        data=path.read_bytes()
        for kind,pattern in PRIVATE_PATTERNS:
            if pattern.search(data):violations.append({'path':relative,'kind':kind})
        target=destination/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
        included[relative]={'bytes':len(data),'SHA256':hashlib.sha256(data).hexdigest(),'source':relative}
    for folder in ['src','scripts','configs','tests','slurm']:
        for path in sorted((ROOT/folder).rglob('*')):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix.lower() not in EXTENSIONS:continue
            name=path.name.lower()
            if 'shadowcast' in name:
                excluded.append({'path':path.relative_to(ROOT).as_posix(),'reason':'separate poisoning-control implementation and its tightly coupled tests; not part of this benign export'});continue
            if 'drive' in name or 'sync' in name:
                excluded.append({'path':path.relative_to(ROOT).as_posix(),'reason':'account-specific report transfer/authentication utility'});continue
            if name in ['install-rsiexperimentmonitor.ps1','invoke-rsiexperimentmonitor.ps1','install_rsi_stage_monitor.py']:
                excluded.append({'path':path.relative_to(ROOT).as_posix(),'reason':'retired scheduler installer; user selected Codex scheduling'});continue
            copy(path)
    for name in ['pyproject.toml','requirements.txt','requirements-editor.txt','environment.yml','ARCHITECTURE_DECISION.md','CODEX_START_HERE.md','CODEX_TASKS.md','README_FOR_CODEX_CN.md','train_and_evaluate_pilot.py','train_lighting_v2.py','train_lighting_v3.py','run_mini_pilot.sh']:
        copy(ROOT/name)
    for path in (ROOT/'docs').rglob('*'):
        if path.is_file() and path.suffix.lower() in {'.md','.json','.txt'} and 'benign_starter_original' not in path.parts:copy(path)
    for name in ['IMPLEMENTATION_STATUS.md','code_inventory.md','code_inventory_files.csv','code_inventory_functions.csv','clean_vlm_pipeline.md','frozen_quality_gate.md','frozen_quality_metrics.csv','experiment_protocol_p0.md','outer_controller_audit.md','missing_components.md','OVERNIGHT_METHOD_AUDIT.md','OVERNIGHT_EXPERIMENT_STATUS.md','METHOD_READINESS_20261113.md','P0_EXPERIMENT_AUDIT.md','P0_EXPERIMENT_AUDIT.json','P1_results_unrun.csv','BENIGN_VALIDATION_REPORT_CN.md','IMPLEMENTATION_AUDIT.md','MINIMAL_EXPERIMENT_STATUS.md','NEXT_EXPERIMENT_DECISION.md']:
        if (ROOT/'reports'/name).exists():copy(ROOT/'reports'/name)
    run=ROOT/'outputs/overnight_benign_20261112T051015Z'
    for name in ['protocol.json','source_hashes.json','comparison.json','final_read_only_audit.json','final_read_only_audit.log','COMMANDS.md','new_cpu_tests.log','disk_final.json']:
        if (run/name).exists():copy(run/name)
    remote=run/'remote'
    for pattern in ['*.json','*.log','logs/*.log','slurm/*.sbatch','stages/*/attempt_*/*.json','stages/*/attempt_*/*.jsonl','p1_preparation/*.json','analysis_code/*.json','analysis_code/*.log','controlplane_revisions/*.json']:
        for path in remote.glob(pattern):
            if path.is_file():copy(path)
    # Keep both actual P0 evidence and source-level quality validation reports.
    p0=ROOT/'experiments/p0_foundations_20261111_v1'
    if p0.exists():
        for pattern in ['*.json','*.md','*.log','remote/clean_vlm/*.json','remote/clean_vlm/*.jsonl','remote/frozen/*.json','remote/quality_followup/*.json','remote/quality_followup/observations/*.json']:
            for path in p0.glob(pattern):
                if path.is_file():copy(path)
    syntax=[]
    for path in destination.rglob('*.py'):
        try:ast.parse(path.read_text(encoding='utf-8-sig'),filename=path.relative_to(destination).as_posix())
        except SyntaxError as error:syntax.append({'path':path.relative_to(destination).as_posix(),'error':str(error)})
    if violations or syntax:raise ValueError(json.dumps({'credential_findings':violations,'syntax_errors':syntax}))
    (destination/'.gitattributes').write_bytes(b'* -text\n')
    (destination/'.gitignore').write_bytes(b'__pycache__/\n*.py[cod]\n.venv*/\n.env\n.env.*\n*.pem\n*.key\ncredentials*.json\n*.pt\n*.pth\n*.ckpt\n*.safetensors\n*.bin\n/data/raw/\n/data/processed/\n/checkpoints/\n/logs/\n/tmp/\n')
    (destination/'README.md').write_text('''# poisoning_CVPR

私有论文研究代码快照：正常图像编辑、clean VQA、扩散LoRA数值检查、良性Inner/Controller基础设施及真实实验记录。

## 已运行与验证范围

正常VQA使用Qwen3-VL-2B-Instruct固定revision，train1024/dev128/test512/source-transfer128，seeds42/123/3407，每seed512次实际AdamW更新。test Frozen=0.722070，SFT=0.740625 ± 0.009334（训练seed SD）。配对source bootstrap95%CI=[0.001107,0.036330]。transfer差值CI=[-0.008854,0.062760]包含0；不宣称跨任务RSI或预训练无污染。

GPU实际作业91023/91029/91043/91057/91078均COMPLETED，总4377秒=1.215833 GPU-hours。数值恢复、公开adapter与checkpoint哈希检查记录随附。以上是正常VQA结果，不能替代完整生成器RL/Outer方法验证。

Frozen正常图像编辑质量门槛仍未通过；P1的完整三seed编辑对照保持NOT_RUN。Inner为score-function，Learned Outer为REINFORCE，均不是PPO。旧4-step和低样本smoke不能作为完整方法效能结果。

## 目录

- `src/`：模型/数据/编辑/训练/评估、扩散trajectory、controller与checkpoint实现。
- `scripts/`、`configs/`、`tests/`、`slurm/`：良性实验与诊断入口、固定协议和测试。
- `reports/`：实现清单、质量门槛、方法审计和夜间结果。
- `outputs/overnight_benign_20261112T051015Z/`：固定manifest、逐步训练、逐条预测、计费与完整性证据。原始图片和tensor checkpoint保留在原资源中，不上传到GitHub。
- `experiments/p0_foundations_20261111_v1/`：选取的小型P0证据，不等于再次运行P0。
- `SOURCE_RELEASE_MANIFEST.json`：原文件相对路径、字节数和SHA256。
- `EXPORT_SCOPE.json`：打包范围和明确排除项。

## 环境与检查

Python3.11；固定依赖见`requirements.txt`与`requirements-editor.txt`。CUDA版本应匹配实际资源；本仓库没有附模型权重或大型数据。现有训练/Slurm脚本保留真实运行版本和机器路径，复用时需要自己的路径、新run ID、数据/权重及预算检查，不能直接重开已完成的历史run。

无需模型/GPU的轻量检查：

```bash
python -B -m unittest tests.test_night_budget tests.test_night_analysis tests.test_native_preprocessing tests.test_vqa_official tests.test_vizwiz_official -v
```

依赖torch/diffusers或历史artifact的其他测试需要相应环境；本次上传不宣称全部测试都重新执行。GPU工作需先通过对应数值、质量与预算门槛。

## 快照范围

上传的是现有良性实验实现及审计证据。独立ASR后门优化附件、Shadowcast投毒控制、账户认证/同步工具、凭据、模型权重、大型图像数据、缓存和原Git历史未打包。原工作区全部保留；本快照不是原目录的无差别镜像，也不把缺失的方法标为已实现。

原始机器上的精确日志/checkpoint位置和哈希保留在证据JSON中。早期文档是历史记录；结论以最新`reports/OVERNIGHT_EXPERIMENT_STATUS.md`、`reports/METHOD_READINESS_20261113.md`和原始运行证据为准。

第三方VQA评分代码的原license、出处和固定revision保留在`src/evaluation/_vendor/`。本次导出没有给原作者代码新增许可证。
''',encoding='utf-8')
    (destination/'SOURCE_RELEASE_MANIFEST.json').write_text(json.dumps({'schema':'byte_preserving_research_snapshot_v1','created_UTC':datetime.now(timezone.utc).isoformat(),'source_files':included,'file_count':len(included),'source_bytes':sum(x['bytes'] for x in included.values()),'credential_pattern_findings':[],'Python_AST_errors':[]},indent=2),encoding='utf-8')
    (destination/'EXPORT_SCOPE.json').write_text(json.dumps({'scope':'Benign experiment source plus compact original evidence; no algorithm or training-setting changes.','excluded_files':excluded,'external_attachments_not_uploaded':['E:/asr_poison_rl.py','E:/README_RUN.md','E:/run.slurm'],'bulk_exclusions':['model/checkpoint tensor files','raw image/data archives','caches and virtual environments','credentials and account-specific transfer tools','original Git history'],'original_workspace_preserved':True},indent=2),encoding='utf-8')
    print(json.dumps({'status':'EXPORT_PREPARED','destination':str(destination),'source_files':len(included),'source_bytes':sum(x['bytes'] for x in included.values()),'excluded_selected_source_files':len(excluded),'Python_files_parsed':len(list(destination.rglob('*.py'))),'credential_findings':0}))


if __name__=='__main__':main()
