"""Function-level source census plus evidence-qualified P0 capability audit."""
import ast
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'experiments/p0_foundations_20261111_v1'
REPORTS=ROOT/'reports'


def locate(path,function):
    tree=ast.parse((ROOT/path).read_text(encoding='utf-8-sig'))
    found=[]
    def visit(nodes,prefix=''):
        for node in nodes:
            if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                name=prefix+node.name
                if name==function:found.append(node.lineno)
                visit(node.body,name+'.')
    visit(tree.body)
    return f'`{path}:{found[0]}` `{function}`' if found else f'`{path}` `{function}`（不存在）'


def main():
    REPORTS.mkdir(exist_ok=True)
    cpu=json.loads((RUN/'remote/cpu_tests.json').read_text())
    if not cpu['passed']:raise ValueError('No passing executed CPU receipt')
    config_run=RUN/'remote/status.json'
    state=json.loads(config_run.read_text()) if config_run.exists() else {}
    quality=state.get('stages',{}).get('frozen_quality',{})
    clean=state.get('stages',{}).get('clean_vlm',{})
    C='python scripts/p0_cluster_io.py cpu → isolated code/scripts/test_p0_foundations.py --assets-root /storage/home/ydk297/projects/poisoning --run /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1'
    G='python scripts/p0_cluster_io.py submit → isolated code/scripts/run_p0_foundations.py --assets-root /storage/home/ydk297/projects/poisoning --run /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1'
    E='experiments/p0_foundations_20261111_v1/remote/'
    rows=[]
    def add(cap,path,fn,implementation,run_status,evidence,command,limit):
        rows.append({'能力':cap,'实现分类':implementation,'执行状态':run_status,'定位':locate(path,fn),
                     '实际命令或明确未运行':command,'证据':evidence,'边界':limit})
    add('Frozen加载/预处理/原生推理','src/editing/pretrained_editor.py','PretrainedEditor.edit','已实现并通过测试' if quality else '已实现但尚未验证','RUN_THIS_RUN' if quality else 'NOT_RUN',E+'frozen/observations.json' if quality else '本轮模型调用尚未开始；不引用旧4步为质量PASS',G if quality else 'NOT_RUN；待执行：'+G,'质量门槛另由frozen_quality_gate.md判断')
    add('精度/CFG实际预测','src/rl/diffusion_policy.py','DiffusionPolicy.predict','已实现并通过测试','CPU_RUN',E+'cpu.log; tests.test_benign_ddpo_audit.SamplerRegression.test_fp32_policy_does_not_enable_bf16_autocast',C,'CPU断言+历史短模型数值检查；不推广历史BF16全面修复')
    add('随机DDIM和action-detach likelihood','src/rl/diffusion_trajectory.py','transition','已实现并通过测试','CPU_RUN',E+'cpu.log; tests.test_diffusion_rl; tests.test_benign_ddpo_audit.SamplerRegression',C,'Gaussian回放/解析梯度；无PPO ratio')
    add('LoRA挂接与严格加载','src/training/generator_adapter.py','load_adapter_state','已实现并通过测试','CPU_RUN',E+'cpu.log; CheckpointRegression',C,'完整adapter键集/shape/dtype/有限性/精确恢复')
    add('Inner实际score-function更新','src/mini_pilot/engine.py','PilotTrainer.update','已实现并通过测试','CPU_RUN + HISTORICAL_GPU_RUN','旧 experiments/benign_ddpo_audit_20261111_v1/remote/gpu/events.jsonl; 本轮CPU小UNet',C,'终局R×平均logp；旧4次更新是smoke，正常编辑P1 NOT_RUN')
    add('Inner checkpoint及下一步','src/mini_pilot/engine.py','PilotTrainer.restore','已实现并通过测试','CPU_RUN',E+'cpu.log; CheckpointRegression.test_cpu_update_optimizer_and_completed_next_step_restore_exact',C,'旧GPU步数/shape限定；新512px训练NOT_RUN')
    add('固定baseline advantage','src/rl/reward_normalization.py','fixed_advantage','已实现并通过测试','CPU_RUN',E+'cpu.log; DiffusionRLTests.test_fixed_scale_not_batch_normalized',C,'返回R，baseline0；不是GAE/learned value advantage')
    add('主要Inner替代DDPO trainer','src/rl/diffusion_rl_trainer.py','DiffusionRLTrainer.update','已实现并通过测试','CPU_GUARD_RUN + HISTORICAL_RUN',E+'cpu.log; test_fresh_batch_remediation; 旧DDPO wiring记录',C,'本轮只测freshness拒绝分支；full-update不借用别的trainer测试')
    add('原生Euler reward_update接口','src/training/generator_optimizer_interface.py','GeneratorOptimizerContract.reward_update','部分实现','CPU_RUN',E+'cpu.log; test_generator_contract.test_unsupported_rl_fails_explicitly',C,'接口明确抛NotImplementedError；不能把独立DDIM实现误写成原生Euler已支持')
    add('Learned Outer REINFORCE参数学习','src/mini_pilot/controller_v3.py','LearnedController.learn','已实现并通过测试','CPU_RUN + HISTORICAL_GPU_RUN',E+'cpu.log; LearnedOuterRestoreTests; outputs/reward_debug_20261104T054605Z_s42/internal_pilot/controller_updates.jsonl',C,'历史3个实际policy更新，非PPO；独立任务价值未证明')
    add('Learned Outer状态恢复（本轮补全）','src/mini_pilot/controller_v3.py','LearnedController.load_state_dict','已实现并通过测试','CPU_RUN',E+'cpu.log; LearnedOuterRestoreTests.test_real_parameter_update_and_next_action_update_restore',C,'下一动作/参数/Adam一致；不是长过程Outer实证')
    add('fixed/rule/UCB recipe选择','src/mini_pilot/benign_validation.py','RecipeSelector.propose','已实现并通过测试','CPU_RUN + HISTORICAL_GPU_RUN',E+'cpu.log; 旧 outer_comparison.json',C,'旧9候选全拒绝；不是learned policy')
    add('独立dev Gate/不可变memory','src/rsi/controller.py','RSIController.revise','已实现并通过测试','CPU_RUN',E+'cpu.log; tests.test_rsi_logging.RSILoggingTest',C,'测试含accept/reject、预算、test拒绝；不是实际RSI迁移')
    add('完整普通编辑SFT对照','src/training/generator_trainer.py','GeneratorTrainer.update','部分实现','NOT_RUN','代码为source图epsilon-denoising MSE；有旧训练smoke，缺配对编辑target benchmark','NOT_RUN（本轮不启动P1）','不等于普通编辑SFT效能baseline')
    add('DRaFT/reward-gradient comparator','src/training/generator_trainer.py','DRaFTTrainer','未实现','NOT_RUN','无该类/独立可微终局reward训练闭环','NOT_RUN','现有epsilon-MSE不能冒充DRaFT')
    add('Frozen/SFT/reward-finetuning/RL >=3seeds公平对照','scripts/run_p0_foundations.py','main','部分实现','NOT_RUN','P0协议P1_authorized_by_this_protocol=false','NOT_RUN','未运行论文单元格保持空白')
    add('random及普通HPO正常图像等预算结果','scripts/compare_rl_baselines.py','main','部分实现','NOT_RUN','存在synthetic比较脚本；未有本轮相同良性任务/预算完整结果','NOT_RUN','不混用旧合成或不同任务结果')
    for name,fn in [('response-only监督标签','response_labels'),('clean VLM更新','CleanVLMTrainer.update'),
                    ('clean VLM保存/恢复','CleanVLMTrainer.restore'),('clean VLM独立评估','CleanVLMTrainer.evaluate')]:
        ran=fn=='response_labels' or clean.get('status')=='PASS_ENGINEERING_ONLY'
        add(name,'src/training/clean_vlm_pipeline.py',fn,'已实现并通过测试' if ran else '已实现但尚未验证',
            ('CPU_RUN' if fn=='response_labels' else 'GPU_RUN') if ran else 'NOT_RUN',
            E+('cpu.log' if fn=='response_labels' else 'clean_vlm/metrics.json') if ran else '等待本轮P0.3模型运行',
            C if fn=='response_labels' else (G if ran else 'NOT_RUN；待执行：'+G),'仅clean QA；工程闭环不等于学习提升')
    screening='精确ID/content去重已做；后续16源图120-pair dHash筛查已运行(min19，0flag)，全面/历史感知去重仍NOT_RUN' if (RUN/'split_perceptual_audit.json').exists() else '精确ID/content去重已做；感知近重复NOT_RUN，pretraining UNKNOWN'
    add('固定split/哈希/源图bootstrap','src/data/p0_protocol.py','validate_splits','已实现并通过测试','CPU_RUN + PREPARE_RUN',E+'manifest.json; manifest_lock.json; cpu.log; split_perceptual_audit.json',C+'；python scripts/p0_cluster_io.py prepare',screening+'；pretraining UNKNOWN')
    add('SSIM/校准LPIPS编辑诊断','src/evaluation/benign_editing_quality.py','EditingQualityMetrics.measure','已实现并通过测试' if quality else '已实现但尚未验证','GPU_RUN' if quality else 'NOT_RUN',E+'frozen/observations.json' if quality else '等待本轮P0.2模型运行',G if quality else 'NOT_RUN；待执行：'+G,'仅评估；不进入生成/Controller优化')
    add('独立任务族经验迁移/RSI','src/rsi/controller.py','RSIController.revise','部分实现','NOT_RUN','仅配置/memory机制；当前transfer是独立source split而非独立任务族','NOT_RUN','RSI证据NO，不把HPO改名RSI')
    columns=list(rows[0])
    table='|'+'|'.join(columns)+'|\n|'+'|'.join('---' for _ in columns)+'|\n'
    table+='\n'.join('|'+'|'.join(str(row[c]).replace('|','/') for c in columns)+'|' for row in rows)
    ppo='''## PPO逐项核对

|组件|实际情况|定位|
|---|---|---|
|old_policy参数快照/独立policy对象|四个指定核心文件均没有。policy_version、trajectory.logp及contract里的behavior字符串不是old-policy模型。|DiffusionPolicy.__init__/sample；PilotTrainer.update|
|old/new log_prob ratio|没有exp(new-old)或importance ratio。保存的logp只做未更新策略的1e-6回放一致性校验。|PilotTrainer.update；DiffusionRLTrainer.update|
|advantage|Inner是终局R、零baseline；旧DiffusionRLTrainer调用fixed_advantage(R)=R。Learned Outer有gain−EMA baseline，并真实进入REINFORCE loss。|reward_normalization.fixed_advantage；controller_v3.LearnedController.learn|
|PPO surrogate clipping|没有min(ratio*A, clipped_ratio*A)。现有clip_grad_norm_只裁剪梯度，scheduler.clip_sample=False也不是PPO clipping。|mini_pilot/engine.py；diffusion_trajectory.py|
|PPO日志|无old-policy/ratio/clip-fraction/KL数据。实际日志是loss、terminal reward、gradient norm、参数变化和replay差。|旧events.jsonl/controller_updates.jsonl|

因此不能称Inner PPO或Outer PPO。旧learned Outer是categorical REINFORCE；新UCB是非学习的recipe search。这两个实现必须分开描述。
'''
    text=f'''# P0.1 代码事实与证据清单

本轮已实际执行 {cpu['tests_run']} 项CPU测试，failure/error/skip均为0。证据：`{E}cpu_tests.json`、`{E}cpu.log`。只把具体调用过的能力列为已测试；无本轮或核验历史痕迹的入口明确标NOT_RUN，绝不以IMPLEMENTED_UNTESTED代替。

分类分为已实现并通过测试／已实现但尚未验证（执行列仍写NOT_RUN）／部分实现／未实现。CPU合成验证、历史GPU smoke、当前真实模型验证、方法效能是不同层级。

{table}

{ppo}

## 全仓库函数与配置普查

`reports/code_inventory_functions.csv`逐一记录src、scripts、tests及根runner中的函数/类方法行号；`reports/code_inventory_files.csv`记录所有源文件、配置和Slurm脚本的SHA-256。其默认NOT_RUN表示本轮尚未逐入口执行，不断言历史绝无运行；不把某模块被import视为其全部函数通过测试。上表列出核查过的能力和证据，不用函数计数代替审计。

旧报告与原始目录保留。P0.2/P0.3结果将从本轮单独run回读更新本表，P1没有被此脚本启动。
'''
    (REPORTS/'code_inventory.md').write_text(text,encoding='utf-8')
    files=sorted({p for folder in ('src','scripts','tests','configs','slurm') for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.yml','.sbatch')}
                 |set(ROOT.glob('train*.py')))
    function_rows=[];file_rows=[]
    for path in files:
        relative=path.relative_to(ROOT).as_posix();file_rows.append({'path':relative,'SHA256':hashlib.sha256(path.read_bytes()).hexdigest(),'kind':path.suffix,'entry_execution':'NOT_RUN_UNLESS_CAPABILITY_TABLE_HAS_EVIDENCE'})
        if path.suffix!='.py':continue
        try:tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        except SyntaxError as error:function_rows.append({'path':relative,'function':'PARSE_ERROR','line':error.lineno,'execution':'NOT_RUN'});continue
        def visit(nodes,prefix=''):
            for node in nodes:
                if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    name=prefix+node.name
                    if not isinstance(node,ast.ClassDef):function_rows.append({'path':relative,'function':name,'line':node.lineno,'execution':'NOT_RUN_UNLESS_EXPLICITLY_EVIDENCED'})
                    visit(node.body,name+'.')
        visit(tree.body)
    for name,data in [('code_inventory_functions.csv',function_rows),('code_inventory_files.csv',file_rows)]:
        with (REPORTS/name).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    (RUN/'code_capabilities.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'capabilities':len(rows),'functions':len(function_rows),'files':len(file_rows),'CPU_tests':cpu['tests_run']}))


if __name__=='__main__':main()
