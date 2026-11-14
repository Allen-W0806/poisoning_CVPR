"""Build Chinese reports from finalized logs; verify bytes before reporting."""
import csv
import hashlib
import json
from pathlib import Path
import statistics
import tarfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'experiments/benign_ddpo_audit_20261111_v1'
REMOTE = RUN/'remote'
GPU = REMOTE/'gpu'
REPORTS = ROOT/'reports'


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name, text): (REPORTS/name).write_text(text.strip()+'\n',encoding='utf-8')
def local_time(value): return datetime.fromisoformat(value).astimezone(ZoneInfo('America/New_York')).isoformat()
def mean(rows,key): return statistics.mean(row[key] for row in rows)


def main():
    REPORTS.mkdir(exist_ok=True)
    status, cpu = read(GPU/'status.json'),read(REMOTE/'cpu/cpu_tests.json')
    inner, outer = read(GPU/'fp32/inner_comparison.json'),read(GPU/'fp32/outer_comparison.json')
    cohort, protocol = read(GPU/'source_manifest.json'),read(GPU/'protocol.json')
    events = [json.loads(line) for line in (GPU/'events.jsonl').read_text().splitlines()]
    assert status['status']=='COMPLETED_BOUNDED_DEVELOPMENT_SMOKE' and cpu['passed']
    observations = [json.loads(line) for line in (GPU/'fp32/observations.jsonl').read_text().splitlines()]
    bf16 = [json.loads(line) for line in (GPU/'bf16/observations.jsonl').read_text().splitlines()]
    checks, inventory = [], []
    for row in observations+bf16:
        path = GPU/row['image']
        assert digest(path)==row['output_SHA256'],path
    for path in GPU.rglob('checkpoint_integrity.json'):
        assert digest(path.parent/'training_state.pt')==read(path)['sha256'],path
        checks.append(str(path.parent.relative_to(GPU)))
    archive_hash=read(REMOTE/'source_upload.json')['archive_sha256']
    archive_path=REMOTE/'source_archives'/(archive_hash+'.tar.gz')
    assert digest(archive_path)==archive_hash
    executed_source=RUN/'executed_source'
    executed_source.mkdir(exist_ok=True)
    with tarfile.open(archive_path,'r:gz') as archive:
        for member in archive.getmembers():
            target=(executed_source/member.name).resolve()
            if not member.isfile() or not target.is_relative_to(executed_source.resolve()):
                raise ValueError('Invalid executed source archive member')
            target.parent.mkdir(parents=True,exist_ok=True)
            incoming=archive.extractfile(member).read()
            if target.exists() and target.read_bytes()!=incoming: raise ValueError('Executed source changed')
            target.write_bytes(incoming)
    for name,value in cpu['source_sha256'].items():
        assert digest(executed_source/name)==value,name
    current_changes=[path.relative_to(executed_source).as_posix() for path in executed_source.rglob('*')
                     if path.is_file() and (ROOT/path.relative_to(executed_source)).is_file()
                     and digest(ROOT/path.relative_to(executed_source))!=digest(path)]
    paired = status['inner']
    for split in ('test','transfer'):
        differences = [b['training_reward_proxy']-a['training_reward_proxy']
                       for a,b in zip(inner['frozen'][split],inner['inner'][split])]
        assert abs(statistics.mean(differences)-paired[split]['mean_reward_delta'])<1e-14
    all_candidate_entries = [row for value in outer.values() for row in value['history']]
    all_updates = [row for row in events if row['kind']=='optimizer_update']
    for path in sorted(REMOTE.rglob('*')):
        if path.is_file(): inventory.append({'path':str(path.relative_to(REMOTE)),'bytes':path.stat().st_size,'sha256':digest(path)})
    (RUN/'LOCAL_EVIDENCE_VERIFICATION.json').write_text(json.dumps({
        'executed_source_snapshot_matches_CPU_receipt':True,
        'current_source_differs_after_review':current_changes,
        'image_hashes_verified':len(observations)+len(bf16),
        'checkpoint_sidecars_verified':len(checks),'checkpoint_directories':checks,
        'inner_means_recomputed_from_observations':True,'files':inventory,
        'retained_remote_copy_bytes':sum(row['bytes'] for row in inventory)},indent=2),encoding='utf-8')
    rows=[]
    for split in ('test','transfer'):
        rows.append({'split':split,'method':'Frozen','n':len(inner['frozen'][split]),
            'reward':mean(inner['frozen'][split],'training_reward_proxy'),
            'luminance_error':mean(inner['frozen'][split],'absolute_target_error'),
            'ssim_source':mean(inner['frozen'][split],'evaluation_ssim_source'),'delta_Frozen':0.})
        rows.append({'split':split,'method':'Inner','n':len(inner['inner'][split]),
            'reward':mean(inner['inner'][split],'training_reward_proxy'),
            'luminance_error':mean(inner['inner'][split],'absolute_target_error'),
            'ssim_source':mean(inner['inner'][split],'evaluation_ssim_source'),
            'delta_Frozen':paired[split]['mean_reward_delta']})
        for method,value in outer.items():
            data=value['final_evaluation'][split]
            rows.append({'split':split,'method':method,'n':len(data),'reward':mean(data,'training_reward_proxy'),
                         'luminance_error':mean(data,'absolute_target_error'),
                         'ssim_source':mean(data,'evaluation_ssim_source'),
                         'delta_Frozen':mean(data,'training_reward_proxy')-mean(inner['frozen'][split],'training_reward_proxy')})
    with (REPORTS/'BENIGN_MINIMAL_RESULTS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    table='|任务|方法|源图数|平均 reward proxy|平均亮度目标误差|SSIM（对源图）|相对 Frozen|\n|---|---|---:|---:|---:|---:|---:|\n'
    table+='\n'.join(f"|{r['split']}|{r['method']}|{r['n']}|{r['reward']:.9f}|{r['luminance_error']:.9f}|{r['ssim_source']:.6f}|{r['delta_Frozen']:+.9f}|" for r in rows)
    outer_table='|方法|候选数|尝试 optimizer 更新|训练 rollout|dev 查询|UNet predict 调用|接受|相对 Inner 的 test / transfer 差|\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for method,value in outer.items():
        hist=value['history']
        outer_table+=f"|{method}|{len(hist)}|{sum(x['optimizer_updates_attempted'] for x in hist)}|{sum(x['training_rollouts'] for x in hist)}|{sum(x['dev_queries'] for x in hist)}|{value['total_UNet_NFE']}|{sum(x['accepted'] for x in hist)}|{value['paired_against_inner']['test']['mean_reward_delta']:+.9f} / {value['paired_against_inner']['transfer']['mean_reward_delta']:+.9f}|\n"
    split_table='|划分|sample_id|image_id|源文件 SHA-256|\n|---|---|---|---|\n'
    for split,records in cohort['splits'].items():
        split_table+='\n'.join(f"|{split}|{r['sample_id']}|{r['image_id']}|`{r['image_sha256']}`|" for r in records)+'\n'
    submission=read(REMOTE/'submission.json')
    command=' '.join(submission['command'])
    audit_text=f'''# 良性 IP2Pix / LoRA 实现审计

本轮为实际修复与运行，不是计划。固定证据目录：`experiments/benign_ddpo_audit_20261111_v1/remote/`。入口包原文在 `docs/benign_starter_original/`；其架构判断按待验证提议处理。

## 文件审查与修复

|文件|观察到的问题 / 本轮改动|验证依据|
|---|---|---|
|`src/rl/diffusion_policy.py`|父类无条件 CUDA/BF16 autocast，可能令声明 FP32 的调用实际执行 BF16。现在按 editor 精度选择 autocast，输入 dtype 与 base 一致；CPU 路径不强制 CUDA 同步。原 `PilotPolicy` 已有部分精度处理，不能把父类问题误当成历史 BF16 分叉的已证实根因。|新 CPU 精度回归；真实 BF16/FP32 数值 gate|
|`src/rl/diffusion_trajectory.py`|原噪声可被广播，timestep/有限性检查不足；现在拒绝错误 shape/dtype/device、非有限 noise/epsilon/action 和不在 schedule 中的步。保留随机 DDIM 数学、action detach 和原 1e-6 回放阈值。|Diffusers 转移对照、Gaussian density、解析终局奖励梯度回归|
|`src/training/generator_adapter.py`|PEFT 加载只拒绝 unexpected keys，缺失 adapter 张量可能静默保留当前权重。现在先完整核对 adapter 键集、shape、dtype、有限性，再加载并逐张量验证精确值。|缺失/多余键、错误 dtype/shape、NaN 均在修改参数前拒绝|
|`src/mini_pilot/engine.py`|统一精度路径；拒绝空 batch 与非有限 replay error；按实际设备搬移轨迹；检查更新后 LoRA 有限性。checkpoint 增加 SHA-256、optimizer/clipping 合约及原子状态写入；支持 NumPy RNG 树比较；记录实际 optimizer 次数和 Adam dtype。|CPU 小 UNet 下一步恢复；真实预训练模型重复/续训 gate|
|`src/mini_pilot/benign_validation.py`|新增只含良性亮度、边缘结构代理的独立 reward；SSIM 与裁剪率仅作评估。新增固定/固定规则/UCB 搜索、源图配对核查。|奖励方向、UCB 冷启动、固定规则、seed 配对测试|
|`scripts/run_benign_ddpo_minimal.py`、`configs/benign_ddpo_minimal.json`|新增严格限时、固定协议、阶段 gate、完整拒绝记录的良性实验入口；不导入 victim 或安全指标优化器。|Slurm 90945 完成；运行源文件以完整 archive SHA 为准；原 CPU receipt 只覆盖部分测试，复核后已补全后续版本|
|`tests/test_benign_ddpo_audit.py`、`scripts/test_benign_ddpo_audit.py`|新增数值/恢复/奖励/搜索回归，与既有专项测试合并。|{cpu['tests_run']} 项通过，0 失败、0 error、0 skip|
|`scripts/benign_cluster_io.py`、`scripts/benign_remote_inventory.py`|只向独立 run 目录部署；保留源码快照、环境清单与实际命令；禁止同 run 二次 GPU 提交。|source_upload.json、submission.json、remote_inventory.json|
|`scripts/audit_benign_crop_posthoc.py`|新增运行后保存像素的 crop 复测，单独记录原始/修正奖励，禁止重写训练日志与 checkpoint。|126 个图像 CPU 复测；无新增模型或 optimizer 调用|
|`scripts/report_benign_ddpo_audit.py`、`scripts/render_benign_audit_diagram.py`|生成中文报告、CSV、文件哈希清单、原始图像配对预览与概念 PDF/SVG。|数字从 JSON/日志重算；图像与 PDF 已视觉检查|
|`ARCHITECTURE_DECISION.md`、`CODEX_START_HERE.md`、`README_FOR_CODEX_CN.md`、`docs/benign_starter_original/`|纳入入口资料，完整保留原附件；仅纠正当前架构文件的定义和证据状态。|原 ZIP SHA-256 和原文副本|
|`docs/benign_ddpo_method_outline.tex`|新增独立方法与 Algorithm 1；原稿未改写。|方法 outline，未声称整稿编译通过|

新 checkpoint schema 为 `benign_pilot_checkpoint_v2`。旧 schema 不具备本轮完整合约/sidecar，新的 `PilotTrainer.restore` 会显式拒绝；旧 checkpoint 文件未修改。仅验证项目自建、可信 checkpoint 的恢复，未实现任意不可信 pickle 的安全加载。

## 算法核查

现有路径不是需要重新发明的单标量优化器：随机去噪转移、终局奖励、逐转移 score-function 和重复更新已经存在。本轮核查其实际执行。`eta=1`，epsilon prediction，leading DDIM，`steps_offset=1`、`set_alpha_to_one=False`；末步方差仍为正。Frozen 和 Inner 使用相同 DDIM；原生 Euler 仅单独复现一次冻结推理，不混入对照。

log-prob 在坐标上取平均、loss 再在时间与 batch 上取平均；相对联合转移 log density 的求和只差固定正比例。没有 PPO ratio、当前 batch 标准化或旧轨迹训练复用；因此不能称 PPO/importance-sampling DDPO。终局奖励可用于所有时间步，不需要每步独立奖励。

本轮奖励为 `亮度目标误差减少 - 0.05*(1-边缘方向余弦)`，亮化目标为源图平均亮度 +0.05，暗化为 -0.05，截到 [0,1]。原运行意图排除白色 padding，但审阅发现训练源图 574996000 使用 floor 居中，与 Pillow 的 round 居中相差1像素，纳入一列 padding、遗漏一列内容；不能声称原训练严格排除了 padding。评估/迁移图的 crop 未受影响，修正后训练会产生什么结果未知。当前源码已改用 Pillow 一致的 round，并补奇数 padding 测试；这是运行后修复，未重跑 GPU。它是光度/结构代理，不是 edited-label ground truth，也不是语义身份认证；SSIM 未进入优化。没有 ASR、隐蔽性、victim 反馈、投毒比例或答案重定向信号。

## 历史材料的边界

`train_lighting_v3.py` 与 `src/mini_pilot/engine.py` 证明后续实现存在，不能继续沿用入口附件「现有代码只有一次更新」的概括。历史 `outputs/reward_debug_20261104T054605Z_s42/internal_pilot/numerical_witness/comparison.json` 记录 FP32 五次更新/恢复 PASS；其 `BF16_fix_claim=false`。旧报告标题或语句不替代当前源文件、日志与 checkpoint。本轮不重算或合并历史安全研究结果。

`E:/main9_revised.tex` 的良性 pilot 是另一配置（512px、rank8、两步、20/4/4），不能把本轮值替换进原表。原文保留 SHA-256，未改写。新增独立方法 outline、Algorithm 1 和概念图；没有把设想写成实验结果。

## 验证边界

BF16 与 FP32 都仅在本轮 256px/4步/rank2、batch2 配置通过。历史 BF16 不一致没有在旧配置上复现或修复认证。本机 Anaconda 在实际 import torch 时出现 OpenMP runtime 重复初始化退出，且缺少 Diffusers；未使用 `KMP_DUPLICATE_LIB_OK` 掩盖问题，实际测试在现有集群环境完成。

“FP32”表示 dtype 与计算路径；使用的源权重仍是固定 revision 的 `variant=fp16`，再上转为 FP32，并非另一套原生全精度权重。原运行没有验证 outer loop 的进程崩溃后恢复，也没有实际接受过正向候选；只验证了拒绝/恢复路径。

运行前 CPU receipt 未包含测试驱动自身和其余6个执行测试模块；完整 archive 包含全部文件，复核已逐字节对应。因此原 receipt 的覆盖范围需保留限定。当前测试驱动已将自身与全部执行模块纳入 SHA 清单。原始运行源码完整展开在 `experiments/benign_ddpo_audit_20261111_v1/executed_source/`，与复核后当前源码分开保存。

仓库开始时已有大量未提交修改；`git_status_before.txt` 与 commit 文件记录基线。未 reset、删除原始实验、修改集群 canonical 源码或干扰其他作业。当前新代码只在单独源快照中运行。
'''
    write('IMPLEMENTATION_AUDIT.md',audit_text)
    write('MINIMAL_EXPERIMENT_STATUS.md',f'''# 最小良性实验实际状态

**终态：{status['status']}；Slurm 90945 COMPLETED，运行中作业 0。**

|对象|实现|运行|验证结论|
|---|---|---|---|
|随机采样 / log-prob / FP32 LoRA-Adam / checkpoint|已实现|已运行|47 项 CPU 测试通过；本轮 BF16、FP32 精确重复/续训通过|
|Frozen vs Inner|已实现|已完成|4 次 Inner 更新；本轮代理效能无正向收益|
|fixed / rule / UCB Outer|已实现|已完成|9 个候选全拒绝；额外收益 0|
|可迁移 RSI improvement procedure|未建立|未运行独立 benchmark 验证|不能宣称 RSI；当前仅 HPO/recipe search|

## 实际命令、环境和时间

本地执行 `python scripts/benign_cluster_io.py stage`、`python scripts/benign_cluster_io.py cpu`、`python scripts/benign_cluster_io.py submit`；仅一次 GPU 提交。`status` 与 `fetch` 只读已有作业/结果。

实际 Slurm 内命令（外层 `timeout --signal=TERM 1080s`）：

```bash
{command}
```

- Slurm ID：90945；PID：{status['PID']}；GPU：{status['GPU']['name']}，1 块；seed：42。
- 开始（纽约）：{local_time(status['started_UTC'])}；结束：{local_time(status['finished_UTC'])}。
- 开始 UTC：{status['started_UTC']}；结束 UTC：{status['finished_UTC']}。
- Slurm 实际 reservation：304 GPU 秒 = {304/3600:.6f} GPUh；上限 1200 秒 = 0.333333 GPUh。包含模型加载、数值 gate、失败/拒绝候选和评估；没有追加 GPU 作业。
- 软件：现有 `.venv-editor`；Torch 2.6.0+cu124 / Diffusers 0.35.1 / PEFT 0.17.1 / Accelerate 1.11.0 / Transformers 4.57.1。详细环境见 `remote_inventory.json`。
- 模型：IP2Pix revision `31519b5cb02a7fd89b906d88731cd4d6a7bbf88d`、`variant=fp16`，复用缓存并上转为对应计算 dtype；256×256、4 个随机 DDIM 步、rank2/alpha4、FP32 对照、AdamW lr1e-5、batch2、clip0.5。
- 记录到的 peak GPU allocation：{status['peak_GPU_allocated_bytes']} bytes；FP32 路径 UNet predict 调用 {status['total_UNet_NFE_FP32']} 次。该计数不包含 activation-checkpoint 内部重算或 VAE/text encoder，因此不冒充全部 FLOPs。

## 数据划分

总计 14 个 source ID / content hash 全部互斥，来自历史可见 COCO/VQAv2 开发训练池。`test` 仅为本 run 内 source-held-out assessment，不是全项目封存的独立最终测试。历史 test 与 prospective reservation 均未打开。transfer 使用另 4 张源图与暗化指令；这是同一开发池的跨任务 smoke，不是独立 benchmark。

{split_table}

## 实测结果

{table}

Inner−Frozen 的亮化评估差为 {paired['test']['mean_reward_delta']:+.9f}，源图 bootstrap 95% 区间 {paired['test']['exploratory_source_bootstrap_95CI']}；暗化迁移差为 {paired['transfer']['mean_reward_delta']:+.9f}，区间 {paired['transfer']['exploratory_source_bootstrap_95CI']}。每个任务只有 4 张图、1 个训练 seed，区间只是探索性的源图重采样，不估计训练 seed 变异。不将负值包装成学习成功，也不据此宣称所有配置均无法学习。

Frozen 与 Inner 的评估 source、seed、sampler、精度和 query 数匹配。Frozen 没有训练成本；Inner 多出的训练成本单独记录，不声称两者总训练计算量相同。

## Outer 预算与拒绝记录

{outer_table}

每种方法从同一 Inner checkpoint 恢复；3 个候选 × 每候选 2 次更新 × batch2，另各 6 次 dev query、4 次 test 和4 次 transfer query。每候选 40 次 predict 调用，每种方法合计 152 次（含最终评估）。规则、UCB 与固定 recipe 均用同一个 dev 门槛。三种方法均接受 0/3，恢复到同一 Inner 状态，最终 test/transfer 图像与 Inner 相同；不能把零差解释为控制器收益。UCB 只有3轮，仍是小规模非上下文搜索。

本轮一共执行 {len(all_updates)} 次 optimizer 调用：BF16 数值诊断5 + FP32 数值诊断5 + Inner4 + Outer尝试18。被拒绝更新的成本保留，不把“尝试次数”混同为最终 checkpoint 的 retained step。

## 证据与恢复

- 原始事件：`remote/gpu/events.jsonl`；图像观测：`remote/gpu/fp32/observations.jsonl`；完整 stdout/stderr：`remote/slurm-90945.log`。
- 数值 gate：`remote/gpu/{{bf16,fp32}}/numerical_gate.json`；保留普通 A/B 更新及 resumed 下一步的张量 witness。
- Inner checkpoint：`remote/gpu/fp32/checkpoints/inner_final/training_state.pt`；SHA-256 `{status['inner']['checkpoint_SHA256']}`。
- 9 个 candidate 与3个 selected checkpoint 均保留；目录名、哈希、dev 变化和拒绝理由见 `outer_comparison.json` 与 events。
- 本地已重算 {len(observations)+len(bf16)} 个图像哈希和 {len(checks)} 个 checkpoint sidecar；完整文件 SHA 清单在 `LOCAL_EVIDENCE_VERIFICATION.json`。其中原 status.json 的 retained_bytes 在写入最终状态前计数，比实际 GPU 文件总量小85字节；以最终文件清单为准。
- Slurm 为 COMPLETED，无训练异常 trace；本地 OpenMP 启动失败与初次受限 DNS 检查属于环境审计问题，未启动本地 GPU 训练。独立实验效能、语义人工审核和跨 seed 验证仍未完成。
''')
    write('ARCHITECTURE_CORRECTIONS.md', '''# 架构与方法纠正

1. 原 ZIP 三份文档已纳入仓库，原文另存 `docs/benign_starter_original/`。`ARCHITECTURE_DECISION.md` 是待验证设计，不能当作结果。`CODEX_START_HERE.md` 与本次用户请求确定良性执行范围。
2. 纠正 §1.4–1.5：DDPO 可以只使用终局奖励；各随机去噪转移的 score-function 累加，不需要独立逐步奖励或 learned credit network。一轮 LoRA 更新也不能自动认证 DDPO。现有路径应精确称 on-policy score-function DDPO 类实现，无 PPO ratio。
3. 明确实际 sampler：原生 IP2Pix Euler 先单独复现；受控训练与 Frozen 对照改用同一随机 DDIM 合约。坐标平均 log density 是固定缩放，不能未经验证当成联合概率用于 importance ratio。
4. 纠正“DDPO/DRaFT 谁更省样本”的预设。DRaFT 是可微奖励梯度优化的不同对照；本轮未运行，不声称优劣。
5. 当前 Outer 为有限学习率臂的固定 recipe、固定规则和非上下文 UCB，再加 dev accept/reject。它是 HPO/recipe adaptation，不是已训练 Outer PPO，也不是已验证 RSI。暗化 smoke 未提供独立 benchmark 或改进程序迁移的证据。
6. BF16/FP32 短配置通过与历史 BF16 全面修复严格分开。技术通路可复现、真实参数有变化，不等价于任务效能变好。
7. 原 `E:/main9_revised.tex` 未覆盖；独立方法/Algorithm 1 在 `docs/benign_ddpo_method_outline.tex`（LaTeX outline，未进行整稿编译）。概念流程图在 `output/benign_ddpo_audit_20261111/`，提供 SVG 可编辑对象与 PDF。图是框架示意，不是经验效果图。

核对过的原始来源：

- [DDPO 原论文，§3/§4](https://proceedings.iclr.cc/paper_files/paper/2024/file/14f75513f0f1ca01de1e826b52e6b840-Paper-Conference.pdf)：终局 reward 与 diffusion MDP / policy gradient。
- [DDPO 官方 PyTorch 实现](https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py)：随机 DDIM 与 transition log-prob。
- [IP2Pix 官方仓库](https://github.com/timothybrooks/instruct-pix2pix)：图像条件编辑模型；作者主页本次抓取404，改核官方仓库。
- [DRaFT 原始发布](https://deepmind.google/research/publications/51081/)：通过采样链反传 reward gradient。
- [RRSI 官方仓库](https://github.com/google-research/rrsi)：将 evolve selection 与 held-out/OOD 评估区分。
- [AIDE² 原文](https://arxiv.org/abs/2609.26457)：作为研究背景，未在本仓库复现其算法或结果。

这些文献说明方法定义，不替代本仓库的实测证据。
''')
    write('NEXT_EXPERIMENT_DECISION.md', '''# 下一实验决策

**当前决定：停止扩大 GPU 实验。唯一的短作业已结束，没有排队或自动续跑的后继。**

已经通过：47项 CPU 测试；真实 IP2Pix BF16/FP32 短配置的采样/回放、有限 FP32 梯度与 Adam、原始梯度重复、完整 checkpoint 和下一步续训检查；source ID/content 划分互斥；冻结 base 未变。

未通过或尚缺：Inner−Frozen 在本轮亮化与暗化代理均为负；9个 Outer 候选全拒绝；没有有益编辑的人类/真实标签认证，没有训练 seed 变异估计，没有独立 benchmark 的 improvement-procedure 迁移，也没有旧 512px/rank8 BF16 配置的完整修复认证。4步输出质量可能严重偏离正常编辑，应先检查已保存的配对图，不能以提高 reward 数字代替质量。

继续的具体前置工作：在现有输出上进行人工审阅，明确亮度目标与结构代理是否符合编辑需求；给出小型、独立标注的普通编辑评估集和未用于调参的任务协议。不得在当前4张 assessment 图上试 seed/LR/奖励权重以制造正结果。

如以后开展新的短 pilot，应先固定独立 train/dev/test、合理推理步数、预期最小任务收益、内容保留/质量条件、源图与训练 seed 不确定性、共享初始化与完整计算账本。数值 gate 必须按新尺寸/步数/rank重做；任一 finite/replay/checkpoint gate 失败即停止。增加 GPU 时长或大规模扫描不在本轮授权范围内。

Outer 只有在 Inner 能产生可解释、可复现的正向编辑学习后，才值得研究更广的 recipe。要称 RSI，必须让改进程序本身跨独立 benign benchmark 保留收益，并与固定机制、普通重拟合/HPO/规则搜索在同预算下比较；目前保持“未验证”。

DRaFT 对照、本地 Anaconda 环境修复、历史 BF16 原配置复核均为 NOT RUN；不报告未经运行的效能值。所有新训练目标仍限普通图像编辑，不引入 ASR、隐蔽性或 victim 攻击反馈。
''')
    write('BENIGN_VALIDATION_REPORT_CN.md',f'''# IP2Pix / LoRA 良性验证中文报告

实际修复与短实验已完成。47项测试通过；BF16/FP32 在本轮配置下的重复与续训一致。**Inner 没有优于 Frozen，Outer 也没有额外价值证据。**

|事项|状态|实际结果|
|---|---|---|
|代码修复与 CPU 回归|已实现、已运行、已验证|47/47 通过|
|BF16/FP32 数值可靠性|本配置已验证|普通两步重复及恢复后的下一步均精确一致；不等于历史 BF16 全面修复|
|Frozen vs Inner|已运行|1 seed，train4/dev2/test4/transfer4；4次 Inner 更新；test Δ={paired['test']['mean_reward_delta']:+.9f}，transfer Δ={paired['transfer']['mean_reward_delta']:+.9f}|
|fixed / rule / UCB Outer|已运行|每种3候选、6次尝试更新、152次 predict；9候选全拒绝；最终相对 Inner Δ=0|
|RSI、跨 seed 和独立 benchmark|未验证|当前仅有小规模 recipe HPO；无效能或 RSI 主张|

一次 A100 作业90945：纽约时间 {local_time(status['started_UTC'])} 至 {local_time(status['finished_UTC'])}；Slurm实际304 GPU秒（0.084444 GPUh），上限1200秒。未追加长任务。所有原始日志、图像、trajectory、数值 witness 和 checkpoint 已回读到H盘，并生成 SHA 清单。

实际命令与逐图ID：[完整实验状态](MINIMAL_EXPERIMENT_STATUS.md)。

修改的核心文件为 `src/rl/diffusion_policy.py`、`src/rl/diffusion_trajectory.py`、`src/training/generator_adapter.py`、`src/mini_pilot/engine.py`；新增独立良性指标/控制器、配置、运行入口和测试。逐文件问题、修复及验证见 [实现审计](IMPLEMENTATION_AUDIT.md)。没有改动原稿 `E:/main9_revised.tex`，没有覆盖其他实验或启动任何安全指标优化回路。

{table}

奖励和SSIM均为图像代理；没有编辑 ground truth 或独立人工语义验证。原始图像配对预览见 `output/benign_ddpo_audit_20261111/paired_editing_examples.png`。低步数/小样本输出不支持实际编辑效能结论。当前暂停扩大训练，具体问题与下一门槛见 [下一实验决策](NEXT_EXPERIMENT_DECISION.md)。

[架构纠正与原始来源](ARCHITECTURE_CORRECTIONS.md)；[概念图 SVG](../output/benign_ddpo_audit_20261111/benign_validation_framework.svg)；[概念图 PDF](../output/benign_ddpo_audit_20261111/benign_validation_framework.pdf)；方法与 Algorithm 1 源码为 `docs/benign_ddpo_method_outline.tex`。

**运行后发现并修复的限制：** 1张训练图存在1像素 padding 裁剪偏差；当前源码已修正并新增回归，原 GPU 日志与 checkpoint 原样保留。评估与迁移的裁剪及上述数字不受该边界错误影响，但修复后的训练结果尚未运行，不能将旧 checkpoint 称作修复后产物。

独立只读完整性复核另见 `EXPERIMENT_AUDIT.md`；该复核为 same-family/provisional，不是跨家族或人工认证。数据和模型未复制到C盘；本轮临时目录在核验后清理，checkpoint和实验源码作为复现证据保留。
''')
    post_dir=RUN/'postrun_fix'
    if (post_dir/'cpu/cpu_tests.json').exists():
        post=read(post_dir/'cpu/cpu_tests.json'); crop=read(post_dir/'crop_audit.json')
        assert post['passed']
        for name,value in post['source_sha256'].items(): assert digest(ROOT/name)==value,name
        note=f"\n\n## 运行后修复验证（CPU only）\n\n新增 padding 取整与测试驱动 SHA 覆盖回归后，{post['tests_run']} 项 CPU 测试通过，零失败/跳过。已用修复后的度量重新检查 {crop['observations_checked']} 个保存图像；涉及源图 {crop['affected_source_IDs']}，各 split 最大 reward 绝对变化为 `{crop['max_abs_reward_delta_by_split']}`。这只是已保存像素的 CPU 复测，没有新的模型调用、optimizer 更新或 GPU 作业。训练原始奖励/轨迹/checkpoint 不改写；修复后模型学习效能仍未运行。详细证据在 `experiments/benign_ddpo_audit_20261111_v1/postrun_fix/`。\n"
        for name in ('IMPLEMENTATION_AUDIT.md','MINIMAL_EXPERIMENT_STATUS.md','BENIGN_VALIDATION_REPORT_CN.md','NEXT_EXPERIMENT_DECISION.md'):
            with (REPORTS/name).open('a',encoding='utf-8') as stream: stream.write(note)
    make_examples(inner)
    visual_note='\n\n## 保存图像的视觉检查\n\n配对预览中，电脑图与领带图出现明显纹理化和结构破坏，Frozen 与 Inner 都存在该问题。因此4步采样没有建立可用的普通编辑基线；本轮 P3 只能验证等预算提案/拒绝流程，不能作为控制器方法价值的有效效能实验。该观察来自保存图的技术审阅，不冒充独立人工质量评测。下一步应先在独立、固定的少量普通编辑样本上校准合理推理步数与视觉质量，再考虑新的学习实验；本轮不追加 GPU。\n'
    for name in ('IMPLEMENTATION_AUDIT.md','MINIMAL_EXPERIMENT_STATUS.md','BENIGN_VALIDATION_REPORT_CN.md','NEXT_EXPERIMENT_DECISION.md'):
        with (REPORTS/name).open('a',encoding='utf-8') as stream:stream.write(visual_note)
    if (REPORTS/'EXPERIMENT_AUDIT.md').exists():
        reviewed=[ROOT/'src/mini_pilot/benign_validation.py',ROOT/'src/mini_pilot/engine.py',
                  ROOT/'src/rl/diffusion_policy.py',ROOT/'src/rl/diffusion_trajectory.py',
                  ROOT/'src/training/generator_adapter.py',ROOT/'scripts/run_benign_ddpo_minimal.py',
                  ROOT/'scripts/test_benign_ddpo_audit.py',ROOT/'tests/test_benign_ddpo_audit.py',
                  ROOT/'configs/benign_ddpo_minimal.json',GPU/'status.json',GPU/'source_manifest.json',
                  GPU/'fp32/inner_comparison.json',GPU/'fp32/outer_comparison.json',
                  RUN/'postrun_fix/cpu/cpu_tests.json',RUN/'postrun_fix/crop_audit.json']
        audit_record={'audit_skill':'experiment-audit','verdict':'WARN',
            'reason_code':'single_seed_proxy_scope_and_corrected_training_not_run',
            'summary':'Artifact-level engineering results reproduced; editing efficacy and RSI unverified.',
            'reviewer_model':'gpt-6-astra','reviewer_reasoning':'ultra','reviewer_family':'openai',
            'executor_model':'codex','executor_family':'openai','agent_id':'/root/benign_integrity_review',
            'review_independence':'same-family','acceptance_status':'provisional',
            'trace_path':str((RUN/'REVIEW_TRACE.md').relative_to(ROOT)),
            'generated_at':datetime.now(timezone.utc).isoformat(),'date':'2026-11-11',
            'audited_input_hashes':{p.relative_to(ROOT).as_posix():'sha256:'+digest(p) for p in reviewed},
            'overall_verdict':'warn','integrity_status':'warn',
            'checks':{'gt_provenance':{'status':'pass'},'score_normalization':{'status':'pass'},
                      'result_existence':{'status':'pass'},'dead_code':{'status':'pass','limit':'positive promotion untested'},
                      'scope':{'status':'warn'},'eval_type':'self_supervised_proxy',
                      'synthetic_CPU_tests_eval_type':'simulation_only'},
            'final_review_residual_code_issues_resolved':True,
            'claims':[{'id':'bounded_engineering_execution','impact':'supported'},
                      {'id':'short_shape_precision_restore','impact':'qualified'},
                      {'id':'original_negative_paired_measurements','impact':'supported'},
                      {'id':'useful_editing_or_RSI','impact':'unsupported'},
                      {'id':'corrected_training_effectiveness','impact':'unmeasured'}]}
        (REPORTS/'EXPERIMENT_AUDIT.json').write_text(json.dumps(audit_record,indent=2),encoding='utf-8')
    if (RUN/'disk_after.json').exists() and (RUN/'final_resource_receipt.json').exists():
        before={r['Name']:r for r in read(RUN/'disk_before.json')}
        after={r['Name']:r for r in read(RUN/'disk_after.json')}
        own_output=ROOT/'output/benign_ddpo_audit_20261111'
        retained=sum(p.stat().st_size for base in (RUN,own_output) for p in base.rglob('*') if p.is_file())
        resource_note=f"\n\n## 磁盘与清理\n\n本轮证据和最终图件在H盘保留约 {retained/1024**2:.2f} MiB（含源码、checkpoint、失败/拒绝候选、数值witness与图像）。无模型/数据下载或训练checkpoint写入C盘。C盘可用空间审计点为 {before['C']['Free']/1024**3:.2f} → {after['C']['Free']/1024**3:.2f} GiB，差额未作归因；未删除Codex会话历史、模型或共享运行时。H盘可用 {after['H']['Free']/1024**3:.2f} GiB。图件中间渲染、本轮临时目录和新建bytecode缓存已清理；集群scratch同样不存在，canonical源码SHA与开始时相同。清理及账本证据见 `LOCAL_CLEANUP.json`、`disk_after.json`、`final_resource_receipt.json`。\n"
        for name in ('MINIMAL_EXPERIMENT_STATUS.md','BENIGN_VALIDATION_REPORT_CN.md'):
            with (REPORTS/name).open('a',encoding='utf-8') as stream:stream.write(resource_note)
    print(json.dumps({'verified_images':len(observations)+len(bf16),'verified_checkpoints':len(checks),
                      'optimizer_calls':len(all_updates),'candidates':len(all_candidate_entries),
                      'accepted_candidates':sum(row['accepted'] for row in all_candidate_entries),
                      'report_files':6,'retained_bytes':sum(row['bytes'] for row in inventory)},indent=2))


def make_examples(inner):
    from PIL import Image,ImageDraw,ImageFont
    destination=ROOT/'output/benign_ddpo_audit_20261111'
    destination.mkdir(parents=True,exist_ok=True)
    size=256; margin=18; row_height=292
    sheet=Image.new('RGB',(3*size+4*margin,70+4*row_height),'white')
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((margin,10),'Actual saved outputs - 4-step development smoke, not evidence of useful editing',fill='#24354b',font=font)
    for column,title in enumerate(['Source (padded)','Frozen DDIM','Inner: 4 updates']):
        draw.text((margin+column*(size+margin),40),title,fill='#24354b',font=font)
    for index,(baseline,candidate) in enumerate(zip(inner['frozen']['test'],inner['inner']['test'])):
        paths=[GPU/'fp32/sources'/f"{baseline['sample_id']}.png",GPU/baseline['image'],GPU/candidate['image']]
        for column,path in enumerate(paths):
            with Image.open(path) as image: sheet.paste(image.convert('RGB'),(margin+column*(size+margin),70+index*row_height))
        draw.text((margin,70+index*row_height+size+4),baseline['sample_id'],fill='#24354b',font=font)
    sheet.save(destination/'paired_editing_examples.png')


if __name__=='__main__': main()
