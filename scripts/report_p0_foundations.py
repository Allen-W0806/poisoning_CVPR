"""Verify raw P0 artifacts, build pixel-linked previews and Chinese reports."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
RUN=ROOT/'experiments/p0_foundations_20261111_v1'
REMOTE=RUN/'remote';OUT=ROOT/'outputs/p0_foundations_20261111_v1'
REPORTS=ROOT/'reports'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,text):Path(path).write_text(text.strip()+'\n',encoding='utf-8')
def localtime(value):return datetime.fromisoformat(value).astimezone(ZoneInfo('America/New_York')).isoformat()


def verify_and_preview():
    from PIL import Image,ImageDraw,ImageFont
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=read(REMOTE/'manifest.json');lock=read(REMOTE/'manifest_lock.json');cpu=read(REMOTE/'cpu_tests.json')
    assert sha(REMOTE/'manifest.json')==lock['manifest_SHA256']
    for name,value in cpu['source_SHA256'].items():assert sha(ROOT/name)==value,name
    records=[]
    for split,rows in manifest['splits'].items():
        for row in rows:
            path=REMOTE/'source_images'/Path(row['image_path']).name
            assert sha(path)==row['image_sha256']
            records.append((split,row,path))
    observations=read(REMOTE/'frozen/observations.json')
    for row in observations:assert sha(REMOTE/row['path'])==row['image_SHA256']
    followup=read(REMOTE/'quality_followup/observations.json') if (REMOTE/'quality_followup/observations.json').exists() else []
    for row in followup:
        assert sha(REMOTE/row['path'])==row['image_SHA256']
        assert sha(REMOTE/row['source_path'])==row['source_SHA256']
    for path in (REMOTE/'clean_vlm/checkpoints').glob('*/integrity.json'):
        integrity=read(path);assert sha(path.parent/'training_state.pt')==integrity['training_state_SHA256']
        for name,value in integrity['adapter_files'].items():assert sha(path.parent/'adapter'/name)==value
    clean=read(REMOTE/'clean_vlm/metrics.json')
    assert sha(REMOTE/'clean_vlm/checkpoints/final/training_state.pt')==clean['final_checkpoint_SHA256']
    inventory=[{'path':p.relative_to(REMOTE).as_posix(),'bytes':p.stat().st_size,'SHA256':sha(p)} for p in sorted(REMOTE.rglob('*')) if p.is_file()]
    (RUN/'LOCAL_VERIFICATION.json').write_text(json.dumps({'files':inventory,'all_hash_checks_passed':True,
       'original_sources_verified':len(records),'generated_observations_verified':len(observations)+len(followup),
       'clean_checkpoint_directories_verified':2,'source_matches_CPU_receipt':True},indent=2))
    # 64-bit dHash is a documented screening heuristic, never proof of no
    # perceptual near duplicates or model-pretraining exposure.
    hashes={}
    for split,row,path in records:
        with Image.open(path) as im:
            thumb=im.convert('L').resize((9,8));pixels=list(thumb.get_flattened_data() if hasattr(thumb,'get_flattened_data') else thumb.getdata())
        bits=[pixels[y*9+x]>pixels[y*9+x+1] for y in range(8) for x in range(8)]
        h=sum(int(value)<<i for i,value in enumerate(bits));hashes[row['sample_id']]=(split,h)
    pairs=[];names=list(hashes)
    for i,first in enumerate(names):
        for second in names[i+1:]:
            distance=(hashes[first][1]^hashes[second][1]).bit_count()
            pairs.append({'first':first,'second':second,'cross_split':hashes[first][0]!=hashes[second][0],'dHash_hamming':distance})
    audit={'method':'64-bit horizontal dHash screen only','threshold_flag_leq':5,'pairs':len(pairs),
           'minimum_hamming':min(p['dHash_hamming'] for p in pairs),'flagged_pairs':[p for p in pairs if p['dHash_hamming']<=5],
           'scope':'16 current sources only; not a perceptual uniqueness guarantee or historical exposure audit'}
    (RUN/'split_perceptual_audit.json').write_text(json.dumps(audit,indent=2))
    columns=['sources','vae_roundtrip','native_20','native_30','native_50','ddim_30']
    titles=['Source','VAE reconstruction','Native Euler 20','Native Euler 30','Native Euler 50','Stochastic DDIM 30']
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',13)
    tile=256;gap=12;row_height=302
    sheet=Image.new('RGB',(len(columns)*(tile+gap)+gap,80+4*row_height),'white');draw=ImageDraw.Draw(sheet)
    draw.text((gap,8),'Actual Frozen dev calibration | 512px | fixed source/seed/instruction | no generator updates',font=font,fill='#20334b')
    for j,title in enumerate(titles):draw.text((gap+j*(tile+gap),43),title,font=font,fill='#20334b')
    for i,row in enumerate(manifest['splits']['dev']):
        for j,column in enumerate(columns):
            path=REMOTE/'frozen'/column/f"{row['sample_id']}.png"
            with Image.open(path) as im:sheet.paste(im.convert('RGB').resize((tile,tile)),(gap+j*(tile+gap),80+i*row_height))
            label=row['sample_id']
            if column!='sources':
                value=next(x for x in observations if x['sample_id']==row['sample_id'] and x['config']==column)
                label=f"SSIM {value['SSIM']:.3f}  LPIPS {value['LPIPS']:.3f}"
            draw.text((gap+j*(tile+gap),80+i*row_height+tile+4),label,font=small,fill='#20334b')
    sheet.save(OUT/'paired_examples.png')
    for index in range(2):sheet.crop((0,0 if index==0 else 80+2*row_height,sheet.width,80+2*row_height if index==0 else sheet.height)).save(OUT/f'paired_examples_part{index+1}.png')
    if followup:
        from PIL import ImageOps
        comparison=Image.new('RGB',(4*(tile+gap)+gap,80+4*row_height),'white');draw=ImageDraw.Draw(comparison)
        titles=['Source','Square padded, 30','Native aspect, 30','Native aspect, guidance 5/3']
        for j,title in enumerate(titles):draw.text((gap+j*(tile+gap),35),title,font=font,fill='#20334b')
        for i,row in enumerate(manifest['splits']['dev']):
            values=[None,next(x for x in observations if x['config']=='native_30' and x['sample_id']==row['sample_id'])]
            values += [next(x for x in followup if x['config']==name and x['sample_id']==row['sample_id']) for name in ('native_aspect_30','native_aspect_preserve_30')]
            paths=[REMOTE/'frozen/sources'/f"{row['sample_id']}.png"]+[REMOTE/value['path'] for value in values[1:]]
            for j,path in enumerate(paths):
                with Image.open(path) as im:thumb=ImageOps.pad(im.convert('RGB'),(tile,tile),color='white')
                comparison.paste(thumb,(gap+j*(tile+gap),80+i*row_height))
                label=row['sample_id'] if j==0 else f"SSIM {values[j]['SSIM']:.3f} LPIPS {values[j]['LPIPS']:.3f}"
                draw.text((gap+j*(tile+gap),80+i*row_height+tile+4),label,font=small,fill='#20334b')
        comparison.save(OUT/'paired_preprocessing.png')
    print(json.dumps({'verified_files':len(inventory),'sources':len(records),'editing_observations':len(observations)+len(followup),
                      'perceptual_screen':audit,'preview':str(OUT/'paired_examples.png')},indent=2))


def reports():
    state=read(REMOTE/'status.json');cpu=read(REMOTE/'cpu_tests.json');protocol=read(ROOT/'configs/p0_foundations_20261111.json')
    manifest=read(REMOTE/'manifest.json');summary=read(REMOTE/'frozen/summary.json');obs=read(REMOTE/'frozen/observations.json')
    clean=read(REMOTE/'clean_vlm/metrics.json');resume=read(REMOTE/'clean_vlm/resume_comparison.json')
    submission=read(REMOTE/'submission.json');visual=read(RUN/'visual_review.json')
    predictions=[json.loads(line) for line in (REMOTE/'clean_vlm/predictions.jsonl').read_text().splitlines()]
    for label,key in [('dev_before','dev_before'),('dev_after','dev'),('independent_source_test','test'),('source_transfer_clean_QA','transfer_clean_QA')]:
        values=[r['score'] for r in predictions if r['split']==label]
        assert len(values)==clean[key]['n_sources'] and abs(statistics.mean(values)-clean[key]['mean'])<1e-14
    assert resume['exact'] and clean['fresh_final_predictions_exact']
    metrics_path=ROOT/'outputs/clean_vlm_metrics.json'
    text=json.dumps({'run_id':protocol['run_id'],'raw_metrics_path':str(REMOTE/'clean_vlm/metrics.json'),**clean},indent=2)+'\n'
    if metrics_path.exists() and metrics_path.read_text()!=text:raise ValueError('Refusing to overwrite a prior metrics result')
    metrics_path.write_text(text)
    with (ROOT/'outputs/clean_vlm_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['split','n_sources','consensus_score','source_SD','source_bootstrap_95CI']);writer.writeheader()
        for key in ('dev_before','dev','test','transfer_clean_QA'):
            row=clean[key];writer.writerow({'split':key,'n_sources':row['n_sources'],'consensus_score':row['mean'],'source_SD':row['source_SD'],'source_bootstrap_95CI':row['source_bootstrap_95CI']})
    ft='|配置|n|SSIM ↑|LPIPS ↓|平均亮度变化|方向变亮的源图|\n|---|---:|---:|---:|---:|---:|\n'
    for setting,values in summary.items():
        rows=[r for r in obs if r['config']==setting]
        ft+=f"|{setting}|{len(rows)}|{values['SSIM']:.6f}|{values['LPIPS']:.6f}|{values['luminance_delta']:+.6f}|{sum(r['luminance_delta']>0 for r in rows)}/{len(rows)}|\n"
    followup=read(REMOTE/'quality_followup/observations.json')
    followup_summary=read(REMOTE/'quality_followup/status.json')['summary']
    for setting,values in followup_summary.items():
        rows=[r for r in followup if r['config']==setting]
        ft+=f"|{setting}|{len(rows)}|{values['SSIM']:.6f}|{values['LPIPS']:.6f}|{values['luminance_delta']:+.6f}|{sum(r['luminance_delta']>0 for r in rows)}/{len(rows)}|\n"
    with (REPORTS/'frozen_quality_metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        fields=list(dict.fromkeys(key for row in obs+followup for key in row))
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(obs+followup)
    command=' '.join(submission['command'])
    write(REPORTS/'frozen_quality_gate.md',f'''# P0.2 Frozen 推理质量门槛

状态：**{visual['gate']}**。真实运行，不是旧4步smoke的重命名。固定4张新dev源图，512px，原生Euler20/30/50步和随机DDIM30步，seed/source/指令固定；所有配置0个optimizer更新。原生和DDIM分别记录，不把两种sampler混为一个policy。test没有用于配置选择。

另运行有界预处理诊断（job90978，57 GPU秒）：同一4-dev、指令/seed、30步，去掉人工白色方形padding，按长边512/8倍数网格保留全景；再记录预先固定的text/image guidance5/3。后续dev诊断不更改原manifest或原对照，不调reward。新函数`src/editing/native_preprocessing.py:native_aspect_image`和3项CPU测试保留；旧square-pad默认及历史checkpoint合约未覆盖。不同输入网格改变噪声几何；guidance5/3同时改变两项设置，不能定位唯一原因。

{ft}

指令：`{protocol['editor_instruction']}`。LPIPS为校准AlexNet v0.1，输入[-1,1]，trunk/calibration SHA已检查。只在与Pillow一致的content box上评估，SSIM/LPIPS是源参考诊断，不是任务成功或语义认证。变亮比例只是全局亮度方向代理。

VAE encode(mode)→decode(mode)独立重建检查已运行。其均值SSIM={summary['vae_roundtrip']['SSIM']:.6f}、LPIPS={summary['vae_roundtrip']['LPIPS']:.6f}，不能以重建不完美自动断言代码错误。原生官方pipeline已直接运行，模型、预处理、scheduler、解码路径及权重SHA均记录；没有调整RL reward。

技术视觉检查：{visual['summary']}。检查者为Codex技术审阅，不冒充独立人工质量标签。图为`outputs/p0_foundations_20261111_v1/paired_examples.png`；源图、原尺寸生成PNG、编辑指令、seed、逐图指标在`experiments/p0_foundations_20261111_v1/remote/frozen/`。

预处理配对图为`outputs/p0_foundations_20261111_v1/paired_preprocessing.png`，原始PNG/新source grid在`remote/quality_followup/`。去padding的LPIPS均值0.135883低于方形30步0.167602，但亮度均值仍负；guidance5/3虽平均亮度略正0.001357，LPIPS变差且有颗粒纹理。没有挑一个单项正值作为整体PASS，未将不同网格的指标当作严格配对效能检验。

P1决定：**{visual['P1_decision']}**。未将不同源图/分辨率的旧4步实验与本轮做成配对因果结论；没有用test重新挑选seed或奖励权重。
''')
    write(REPORTS/'clean_vlm_pipeline.md',f'''# P0.3 干净 VLM 技术闭环

**Can run clean VLM end-to-end? YES（有界LoRA工程验证）。** 本轮重新运行，不仅引用旧报告。

- 模型：{clean['model_id']}，revision `{clean['revision']}`；BF16冻结base，FP32语言attention q/v LoRA rank4/alpha8，视觉编码器冻结。
- 数据：4张正常train QA，dev/test/transfer各4张独立source；原始公开COCO图与VQAv2十个人类答案，>=6/10规范化共识作为训练回答。无编辑/后门/投毒数据参与SFT。只监督assistant回答，prompt/image token为-100，前缀必须精确一致。
- AdamW lr1e-4，accum2，clip norm1，constant scheduler；保留2步，实际调用3次（包含一次fresh-object replay）；112个adapter张量在第二步有变化。
- checkpoint后重新加载固定base、公开adapter、新optimizer/scheduler，执行相同下一步。比较{resume['tensor_count']}个张量，差异元素{resume['different_elements']}，最大绝对误差{resume['max_absolute_error']}。原始梯度、参数、Adam、scheduler及全部保存RNG精确一致。
- 最终adapter/optimizer再次由新对象经完整training_state恢复，dev预测逐条相同。该路径会覆盖public adapter张量，因此不单独认证仅adapter文件的推理加载；其文件内容与state一致性另作核验。base聚合SHA `{clean['frozen_base_SHA256']}` 前后一致，最终checkpoint SHA `{clean['final_checkpoint_SHA256']}`。

|划分|n源图|VQA consensus score|源图SD|探索性源图bootstrap95%CI|
|---|---:|---:|---:|---|
'''+ '\n'.join(f"|{key}|{clean[key]['n_sources']}|{clean[key]['mean']:.4f}|{clean[key]['source_SD']:.4f}|{clean[key]['source_bootstrap_95CI']}|" for key in ('dev_before','dev','test','transfer_clean_QA'))+f'''

每划分只有4张图、1训练seed，不能作为完整模型方法验证或泛化改善。dev训练前后分数未变。transfer是另一组正常QA源图，不是独立编辑任务族。source-bootstrap不估计训练seed变异，n=4全对时退化区间不证明总体无不确定性。

预先固定native_30编辑的dev图也做了QA：原始参考答案一致性={clean['edited_dev_original_reference_consistency_proxy']['mean'] if clean['edited_dev_original_reference_consistency_proxy'] else 'NOT_RUN'}，仅为原标签proxy；编辑后的正确答案未独立重新标注，不称“编辑图真实accuracy”。

实际Slurm命令：

```bash
{command}
```

本地提交/查看命令为 `python scripts/p0_cluster_io.py submit` / `status` / `fetch`。单run禁止二次提交，重现实验须创建新的run标识；现有结果不覆盖。

训练日志：`experiments/p0_foundations_20261111_v1/remote/clean_vlm/training.jsonl`；预测：`predictions.jsonl`；resume二进制witness：`reference_next_update.pt`、`fresh_next_update.pt`；checkpoint：`checkpoints/step1`、`checkpoints/final`。规范输出为`outputs/clean_vlm_metrics.json`和`.csv`，原始metrics保留在run内。
''')
    write(REPORTS/'experiment_protocol_p0.md',f'''# P0.4 固定协议与预算

配置：`configs/p0_foundations_20261111.json`；manifest：`experiments/p0_foundations_20261111_v1/remote/manifest.json`。protocol SHA `{state['protocol_SHA256']}`；manifest SHA `{state['manifest_SHA256']}`。全部在模型调用前锁定。

train/dev/test/transfer=4/4/4/4，16个源图ID、内容SHA、question ID全互斥。新原始图片约2.39MB，排除先前缓存COCO ID和已有exposure census；没有打开历史test或prospective annotation。新独立source并不保证模型pretraining未见过。数据在独立run目录，不修改原始官方archive。

指标：clean VQA使用官方兼容ten-answer consensus score（越高越好）；SSIM越高表示源图结构相似；LPIPS越低表示源图感知距离小；亮度变化只表示方向，不能替代指令成功。没有ASR/隐蔽性优化。源图bootstrap10000次、analysis_seed20261111，报告n、SD、区间；单seed不能给跨seedCI。

16源图的额外64bit dHash筛查见`split_perceptual_audit.json`，仅为启发式，不宣称排除所有感知近重复。

预算：主P0 job90965预约1200秒、实际170秒（应用{state['elapsed_seconds']:.3f}秒）；结束后预处理job90978预约300秒、实际57秒。第二次提交前按前一作业终态实际170+新预约300=470秒检查，仍小于1200秒累计承诺上限；单个申请分别1200/300，不能说只有一次GPU提交。合计227 GPU秒（0.063056 GPUh）。包含模型加载/哈希、VAE/Frozen、clean SFT、重复/恢复、评估和全部后续dev诊断。90965记录的峰值allocated GPU为{state['peak_GPU_allocated_bytes']} bytes；90978没有记录峰值，因此整个两作业最大显存未知。沒有P1学习任务。账本见`reports/reproduction_ledger.md`，不得挪用其它项目历史预算。

P1论文表的未运行单元格保持空白（`reports/P1_results_unrun.csv`），不填0或PASS。当前尚缺>=3seeds、普通编辑SFT/可微reward对照、random/HPO完整控制组和独立任务族经验迁移。
''')
    write(REPORTS/'reproduction_ledger.md',f'''# 本轮实际执行账本

|阶段|实际命令|结果|证据|
|---|---|---|---|
|源码语法/普查|python scripts/inventory_p0_code.py|已执行，{sum(1 for _ in csv.DictReader((REPORTS/'code_inventory_files.csv').open(encoding='utf-8-sig')))}个源文件/配置；完整函数计数见CSV|code_inventory.md/CSV|
|CPU gate|python scripts/p0_cluster_io.py cpu|{cpu['tests_run']} passed，0 skip|remote/cpu.log、cpu_tests.json|
|冻结数据|python scripts/p0_cluster_io.py prepare|16源图4/4/4/4互斥；新公开原图2392145 bytes|remote/manifest.json、manifest_lock.json|
|P0 GPU|python scripts/p0_cluster_io.py submit|job90965 COMPLETED；170 GPU秒|submission.json、slurm-90965.log|
|预处理CPU测试|quality_followup内部python -m unittest tests.test_native_preprocessing -v|3项通过；与64项suite分开|remote/quality_followup/cpu.log|
|有界预处理推理|python scripts/p0_cluster_io.py quality_followup|job90978 COMPLETED；57 GPU秒；8个dev输出、0更新|remote/quality_followup/|
|结果回读|python scripts/p0_cluster_io.py fetch|模型日志、图片、checkpoint回读并哈希核验|LOCAL_VERIFICATION.json|
|P1/P2|NOT_RUN|门槛未开|P1_results_unrun.csv空值|

纽约时间：{localtime(state['started_UTC'])} → {localtime(state['finished_UTC'])}。CPU与GPU源码SHA由receipt绑定，配置、数据、editor权重文件、Qwen实际base tensor哈希都保留；依赖版本见remote_inventory.json。旧结果没有覆盖。本轮完整命令：`{command}`。
''')
    write(REPORTS/'missing_components.md','''# 当前缺口（不是新的执行计划）

- PPO未实现：无old-policy模型、ratio、PPO surrogate clipping；Inner为score-function，learned Outer为REINFORCE。
- Frozen有>=20步运行证据，但亮化指令成功与独立视觉质量门槛尚缺；SSIM/LPIPS不能替代。
- 普通编辑的配对SFT benchmark与DRaFT/reward-gradient comparator未形成完整可比实现；当前GeneratorTrainer是epsilon-denoising，不可冒充这些结果。
- >=3训练seed的Inner完整对照NOT_RUN；fixed/random/rule/UCB/HPO/learned Outer的当前任务同预算对照NOT_RUN。
- Learned Outer有真实CPU参数学习和历史3次开发更新，不是未实现；但独立任务族的经验改善和RSI没有证据。
- VLM闭环是LoRA工程smoke，未做全模型训练、多VLM、真实编辑后重标注、独立总体效能验证。
''')
    write(REPORTS/'outer_controller_audit.md','''# Outer事实核查

`src/mini_pilot/controller_v3.py:LearnedController.choose/learn` 为categorical REINFORCE，含`validation_gain-baseline`、log_prob和AdamW；确实能修改policy参数。历史`outputs/reward_debug_20261104T054605Z_s42/internal_pilot/controller_updates.jsonl`有3次更新，max change约0.0010000/0.0010014/0.0010024。该历史小样本不是RSI收益证据。

本轮补全`LearnedController.load_state_dict`，CPU测试了policy/Adam/RNG/baseline/memory恢复及下一动作、下一更新精确一致。本轮没有运行learned Outer的生成器训练对比。v1/v2旧副本仍没有这一新增恢复接口；不将v3测试外推给旧副本。

`src/mini_pilot/benign_validation.py:RecipeSelector` 的fixed/rule/UCB与learned controller是不同实现；旧短作业9候选全拒绝，仅证明fallback。当前不存在Outer PPO的old-policy/ratio/surrogate clipping。现有memory保存和下一轮配置使用不等于改进程序在独立任务族上的改进。

fixed/random/rule/UCB/HPO/learned需要同数据、初始化、候选数、optimizer尝试数、dev查询和总GPU预算。当前完整组合为NOT_RUN，论文数值空置。独立transfer任务族尚无已验证数据/结果，因此只能称recipe controller，不称已验证RSI。
''')
    with (REPORTS/'P1_results_unrun.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['method','seed','task_score','SSIM','LPIPS','GPUh','execution_status'])
        for method in ('Frozen','SFT','reward_finetuning','Inner_RL','fixed','random','rule','UCB','HPO','learned_outer'):
            for seed in (42,123,3407):writer.writerow([method,seed,'','','','','NOT_RUN'])
    write(REPORTS/'IMPLEMENTATION_STATUS.md',f'''# 实际实施状态：P0 foundations

**P0工程工作已实际运行；P1未启动。** 主CPU suite64项通过，另3项预处理测试通过。A100作业90965/90978分别170/57秒，合计227 GPU秒。Clean VLM闭环成功；Frozen已校准到512px/20–50步并做有界预处理对照，但当前任务质量门槛为{visual['gate']}。

|任务|状态|实际证据与限制|
|---|---|---|
|P0.1代码事实核查|已完成|24能力，函数/文件逐项定位，PPO不存在；64项CPU回归。不是所有入口都跑过，无运行证据标NOT_RUN|
|P0.2 Frozen校准|已执行，门槛未完全通过|4张固定dev、VAE重建+20/30/50原生步+30随机DDIM；完整图像/指令/SSIM/LPIPS保存。{visual['summary']}|
|P0.3 clean VLM|PASS_ENGINEERING_ONLY|train4，2 retained updates/3 actual calls，fresh-object next-step及最终预测精确恢复；test consensus72.5%（n4）|
|P0.4协议|固定并验证（有限范围）|4/4/4/4 ID与content互斥、配置/manifest/hash、源图bootstrap及预算账本；独立任务族、pretraining与完整感知去重未验证|
|P1/P2方法价值|NOT_RUN|没有>=3seeds完整对照，未将smoke当成效能或RSI证据|

## 修改文件

- `src/mini_pilot/controller_v3.py`：补全learned controller恢复与严格字段/张量检查。
- 新增`src/training/clean_vlm_pipeline.py`：response-only正常QA、FP32语言LoRA/Adam、严格checkpoint、fresh-object恢复、真实解码评估。
- 新增`src/data/p0_protocol.py`：固定split校验、内容SHA、源图统计；`src/evaluation/benign_editing_quality.py`：仅评估SSIM/校准LPIPS/content crop。
- 新增`src/editing/native_preprocessing.py`及`tests/test_native_preprocessing.py`：无白色方形padding的等比例网格resize；另3项测试。原有预处理默认不改写；在相同4-dev上真实测量了2个30步配置，效果仍有限。
- 新增`configs/p0_foundations_20261111.json`、`CODEX_TASKS.md`、`tests/test_p0_foundations.py`。
- 新增执行/复现脚本：`scripts/p0_cluster_io.py`、`prepare_p0_manifest.py`、`run_p0_foundations.py`、`test_p0_foundations.py`、`inventory_p0_code.py`、`report_p0_foundations.py`。
- 新增`scripts/run_p0_quality_followup.py`：最多5分钟的同dev推理诊断，拒绝覆盖和重复提交，保留全部输出，不使用test或reward优化。
- 未改写旧checkpoint、旧论文、原始数据或上一轮结论；本轮在独立run目录保存来源与结果。

## 关键实测结果

{ft}

clean VLM：dev前后97.5%→97.5%（n4）；独立source test72.5%（n4，源图bootstrap95%CI[25%,100%]）；clean-QA transfer100%（n4，退化经验区间不能证明总体确定）。没有训练收益结论。编辑dev使用原始QA答案的一致性为97.5%，不是编辑后正确答案认证。

## 当前能力回答

- Can run clean VLM end-to-end? **YES，bounded LoRA engineering**。
- Can run reliable benign Inner RL? **PARTIAL**：更新/数值技术路径存在且有smoke，正常编辑学习优势未验证。
- Is learned Outer RL implemented and trained? **YES，REINFORCE的CPU与历史3次开发更新；不是PPO，当前任务效能未验证**。
- Do we have cross-task RSI evidence? **NO**；当前transfer仅独立source的clean QA。
- Do we have fair baseline results? **NO（完整当前任务组合）**；只有Frozen校准结果与旧有限smoke，P1表格未运行值空置。

## 实际命令、证据与下一步

执行：`python scripts/p0_cluster_io.py stage`、`cpu`、`prepare`、`submit`、`status`、`fetch`，另有`quality_followup`和`quality_status`。Slurm内真实命令在`reproduction_ledger.md`，主进程PID {state['PID']}。90965峰值CUDA allocation约{state['peak_GPU_allocated_bytes']/1024**3:.3f}GiB；90978峰值未记录，不声称两作业总体峰值。源图/指令/seed/输出PNG/metrics、训练、预测、checkpoint及resume比较在`experiments/p0_foundations_20261111_v1/remote/`，规范指标在`outputs/clean_vlm_metrics.json`。

下一条可立即执行的无GPU复核命令：

```powershell
python scripts/inventory_p0_code.py
```

CPU回读/可视化命令`python scripts/report_p0_foundations.py --preview-only`已在本机默认Python实际运行；模型工作仍使用现有集群环境，因为本机Anaconda缺Diffusers且实际import torch有已知OpenMP冲突。新的GPU重现必须使用新run标识和固定协议，不可覆盖本轮目录。当前不追加P1训练；先完成当前普通编辑任务的指令符合/质量审阅，并具备完整SFT与reward-finetuning对照、独立任务族和>=3seeds协议。

详细材料：`code_inventory.md`、`frozen_quality_gate.md`、`clean_vlm_pipeline.md`、`experiment_protocol_p0.md`、`outer_controller_audit.md`、`missing_components.md`。所有结论以本轮日志和checkpoint为准。
''')
    if (RUN/'original_annotation_extract.json').exists():
        extract=read(RUN/'original_annotation_extract.json')
        assert extract['all_16_manifest_labels_match_raw_archives']
        note='\n\n## 原始标注复核\n\n已从缓存的官方questions/annotations ZIP直接提取所选16个原始记录，核对全部问题、image/question IDs、160个人类答案及其顺序、共识数和训练回答。原始提取与archive SHA在`experiments/p0_foundations_20261111_v1/original_annotation_extract.json`；没有复制完整大archive或打开历史封存test。\n'
        for name in ('clean_vlm_pipeline.md','experiment_protocol_p0.md','IMPLEMENTATION_STATUS.md'):
            with (REPORTS/name).open('a',encoding='utf-8') as stream:stream.write(note)
    if (RUN/'disk_after.json').exists():
        before={x['Name']:x for x in read(RUN/'disk_before.json')};after={x['Name']:x for x in read(RUN/'disk_after.json')}
        retained=sum(p.stat().st_size for base in (RUN,OUT) for p in base.rglob('*') if p.is_file())
        note=f'\n\n## 保留与清理\n\n本轮源码、日志、checkpoint、原始/生成图与最终预览在H盘保留约{retained/1024**2:.2f}MiB。模型权重与数据未下载到C盘，未安装新运行时。C盘可用{before["C"]["Free"]/1024**3:.2f}→{after["C"]["Free"]/1024**3:.2f}GiB（差额未归因）；H盘可用{after["H"]["Free"]/1024**3:.2f}GiB。本轮空scratch已验证路径后移除；会话历史、共享运行时、模型和旧证据未删除。final_resource_receipt.json确认两作业已COMPLETED且本轮active jobs为空。\n'
        for name in ('IMPLEMENTATION_STATUS.md','reproduction_ledger.md'):
            with (REPORTS/name).open('a',encoding='utf-8') as stream:stream.write(note)
    if (REPORTS/'P0_EXPERIMENT_AUDIT.md').exists():
        files=[ROOT/'src/training/clean_vlm_pipeline.py',ROOT/'src/evaluation/benign_editing_quality.py',
          ROOT/'src/mini_pilot/controller_v3.py',ROOT/'src/data/p0_protocol.py',ROOT/'src/editing/native_preprocessing.py',
          ROOT/'scripts/run_p0_foundations.py',ROOT/'scripts/run_p0_quality_followup.py',
          REMOTE/'manifest.json',REMOTE/'cpu_tests.json',REMOTE/'status.json',REMOTE/'clean_vlm/metrics.json',
          REMOTE/'clean_vlm/resume_comparison.json',REMOTE/'frozen/observations.json',
          REMOTE/'quality_followup/status.json',RUN/'original_annotation_extract.json',RUN/'LOCAL_VERIFICATION.json']
        result={'audit_skill':'experiment-audit','verdict':'WARN','integrity_status':'warn','overall_verdict':'warn',
          'reason_code':'small_single_seed_engineering_scope_and_Frozen_quality_not_cleared',
          'reviewer_model':'gpt-6-astra','reviewer_reasoning':'ultra','reviewer_family':'openai',
          'executor_model':'codex','executor_family':'openai','agent_id':'/root/p0_integrity_review',
          'review_independence':'same-family','acceptance_status':'provisional',
          'generated_at':datetime.now(timezone.utc).isoformat(),
          'trace_path':str((RUN/'REVIEW_TRACE.md').relative_to(ROOT)),
          'audited_input_hashes':{p.relative_to(ROOT).as_posix():'sha256:'+sha(p) for p in files},
          'checks':{'A_ground_truth':'PASS','B_normalization':'PASS','C_artifacts':'PASS','D_execution':'PASS','E_scope':'WARN','F_evaluation_type':'PASS'},
          'clean_QA_evaluation_type':'real_gt_dataset_consensus',
          'edited_QA_evaluation_type':'original_reference_consistency_proxy',
          'editing_quality_evaluation_type':'self_supervised_proxy',
          'controller_CPU_fixture_evaluation_type':'simulation_only',
          'P1_results':'NOT_RUN','full_method_validation':'NOT_ESTABLISHED',
          'residual_report_corrections':'resolved_by_fresh_reviewer'}
        (REPORTS/'P0_EXPERIMENT_AUDIT.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        with (REPORTS/'IMPLEMENTATION_STATUS.md').open('a',encoding='utf-8') as stream:
            stream.write('\n\n## 新只读完整性复核\n\n`experiment-audit`的全新gpt-6-astra/ultra审阅得到A/B/C/D/F PASS、E WARN，总体WARN（same-family/provisional）。实测数值/文件和恢复一致性通过，科学范围与Frozen任务质量仍有限。详见`P0_EXPERIMENT_AUDIT.md`及JSON；不外推为完整论文方法验证。\n')
    print('Chinese P0 reports and real clean VLM metrics written.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preview-only',action='store_true');args=parser.parse_args()
    verify_and_preview()
    if not args.preview_only:reports()
