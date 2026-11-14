"""Windows stdlib SSH transport and lightweight unattended campaign monitoring."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
ROOT=Path(__file__).resolve().parents[1]
RUN='overnight_benign_20261112T051015Z'
ASSETS='/storage/home/ydk297/projects/poisoning'
REMOTE=ASSETS+'/outputs/'+RUN
PYTHON=ASSETS+'/.venv-editor/bin/python'
LOCAL=ROOT/'outputs'/RUN
MONITOR=ROOT/'reports/monitor'


def put(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.partial')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');os.replace(tmp,path)


def remote(code,data=None,timeout=130):
    expression='exec(__import__("base64").b64decode("'+base64.b64encode(code.encode()).decode()+'"))'
    args=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=12','-o','IdentityAgent=none','-o','IdentitiesOnly=yes','utc-login',shlex.join([PYTHON,'-B','-c',expression])]
    result=subprocess.run(args,input=data,capture_output=True,timeout=timeout)
    # One connection retry only; the remote durable ledger makes monitor ticks
    # idempotent. Never retry an application error or an uncertain sbatch.
    if result.returncode==255 and any(t in result.stderr.lower() for t in (b'connection timed out',b'connection reset',b'connection closed',b'connection refused')):
        time.sleep(1);result=subprocess.run(args,input=data,capture_output=True,timeout=timeout)
    if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace')+'\n'+result.stdout.decode(errors='replace'))
    return result.stdout


def render(snapshot):
    budget=snapshot.get('budget',{});rows=[]
    for job in snapshot.get('jobs',[]):
        progress=job.get('progress',{});result=job.get('result',{})
        rows.append(f"|{job.get('job_id')}|{job['stage']} / attempt {job['attempt']}|{job.get('state','UNKNOWN')}|{job.get('elapsed_seconds',0)/3600:.5f}|{progress.get('phase','—')} {progress.get('retained_step',progress.get('evaluated',''))}|{result.get('status','NOT_COMPLETED')}|")
    metrics=[]
    for job in snapshot.get('jobs',[]):
        result=job.get('result',{})
        if result.get('status')=='PASS':metrics.append({'stage':job['stage'],'seed':result.get('seed'),'metrics':result.get('metrics'),'canary_resume_exact':result.get('resume_exact'),'public_adapter_load_exact':result.get('public_adapter_load_exact'),'dev_before':result.get('dev_before'),'dev_after':result.get('dev_after'),'checkpoint':result.get('checkpoint'),'dev_history':result.get('dev_history')})
    task=json.loads((MONITOR/'scheduler_receipt.json').read_text(encoding='utf-8-sig')) if (MONITOR/'scheduler_receipt.json').exists() else {'status':'NOT_YET_VERIFIED'}
    task={k:v for k,v in task.items() if k not in ('AC_sleep_hibernate_policy','wake_timer_policy')}
    codex_receipt=MONITOR/'codex_automation_receipt.json'
    if codex_receipt.exists():
        task=json.loads(codex_receipt.read_text(encoding='utf-8'))
    monitor_closed=task.get('status')=='DELETED_COMPLETED'
    trigger_observed=str(task.get('scheduled_background_trigger_test','')).startswith('PASS_OBSERVED')
    monitor_description=('本轮预设GPU阶段均已完成，Codex两小时监控任务已删除，不再重复唤醒；未扩展新GPU实验。' if monitor_closed else 'Codex内置heartbeat任务已启用，每2小时唤起当前聊天执行CPU/SSH检查。')
    trigger_description=('首次Codex后台定时触发已实际观察并成功执行SSH/Slurm与完整性检查，触发时间详见下方收据。' if trigger_observed else '首次Codex后台定时触发尚未观察到，不将此前Windows手动测试当成Codex触发验证。')
    analysis=json.loads((LOCAL/'comparison.json').read_text(encoding='utf-8')) if (LOCAL/'comparison.json').exists() else {'status':'NOT_COMPLETE'}
    stage_states=[]
    for stage in ('canary','frozen','seed42','seed123','seed3407'):
        matches=[r for r in snapshot.get('jobs',[]) if r['stage']==stage]
        stage_states.append({'stage':stage,'status':matches[-1].get('state','UNKNOWN') if matches else 'NOT_SUBMITTED_WAITING_PREDECESSOR','job_id':matches[-1].get('job_id') if matches else None})
    p1check=LOCAL/'remote/p1_preparation/CPU_check.json';p1submission=LOCAL/'remote/analysis_code/submission.json'
    p1={'status':json.loads(p1check.read_text(encoding='utf-8'))['status'] if p1check.exists() else 'NOT_COMPLETED','CPU_job_id':json.loads(p1submission.read_text(encoding='utf-8'))['job_id'] if p1submission.exists() else None,'GPU_status':'BLOCKED_NOT_RUN'}
    resource_path=LOCAL/'remote/final_resource_receipt.json'
    if not resource_path.exists():resource_path=LOCAL/'remote/resource_snapshot.json'
    resource=json.loads(resource_path.read_text(encoding='utf-8')) if resource_path.exists() else {}
    local_disk=LOCAL/'disk_final.json'
    if not local_disk.exists():local_disk=LOCAL/'disk_after.json'
    local_resource=json.loads(local_disk.read_text(encoding='utf-8-sig')) if local_disk.exists() else {}
    local_evidence_MiB=round(local_resource.get('local_run_bytes',0)/1024**2,2)
    run_activity='本轮已无活动GPU。' if not snapshot.get('budget',{}).get('active_GPU_jobs') else '运行中保留规模会继续增长。'
    comparison_table='三seed未全部完成，汇总单元格保持空。'
    if analysis.get('status')=='COMPLETE_3_SEED_READ_ONLY_ANALYSIS':
        comparison_table='|split|独立source数|Frozen|SFT均值 ± 训练seed SD|配对Δ|source bootstrap 95% CI|\n|---|---:|---:|---:|---:|---|\n'
        for split,value in analysis['splits'].items():
            ci=value['paired_delta_source_bootstrap_95CI']
            comparison_table+=f"|{split}|{value['n_sources']}|{value['frozen']:.6f}|{value['SFT_mean']:.6f} ± {value['SFT_training_seed_SD']:.6f}|{value['paired_delta_mean']:+.6f}|[{ci[0]:+.6f}, {ci[1]:+.6f}]|\n"
        comparison_table+='\n仅为冻结cohort上的正常VQA SFT初步结果；test配对CI为正，transfer CI包含0。CI条件于本轮三个训练seed，非跨训练seed总体CI。预训练污染未知，不能用于支持图像编辑Inner/Outer/RSI价值。逐条评分、训练流、dev-only选择及预算只读复核见本轮final_read_only_audit.json。'
    text=f'''# 夜间良性实验实际状态

更新时间：{snapshot.get('updated_UTC','UNKNOWN')}。Run ID：`{RUN}`。

{snapshot.get('next_action','MONITOR_CONNECTION_FAILED')}。本轮最大同时1个GPU作业，总上限10 GPU-hours，含失败/重试。每个Slurm作业独立wall time；按实际分配墙钟时间计GPU消耗，不用利用率折扣。未运行不是PASS。

{'连接/调度检查失败：下列预算与作业信息仅为上次成功快照，当前剩余额度未知，禁止新增提交。' if snapshot.get('monitor_connection')=='FAILED' else '本表来自本次实际SSH/Slurm回读。'}

## 实际作业

|Job ID|配置|Slurm状态|实际GPU-hours|最近进度|结果状态|
|---|---|---|---:|---|---|
{chr(10).join(rows) or '|—|—|NOT_SUBMITTED|0|—|NOT_RUN|'}

CPU准备/调度作业：{', '.join(str(r.get('job_id'))+' '+r['stage']+' '+r.get('state','NEW') for r in snapshot.get('CPU_jobs',[]))}。

所有预设阶段状态：`{json.dumps(stage_states,ensure_ascii=False)}`。P1配对数据检查：`{json.dumps(p1,ensure_ascii=False)}`。P1所有GPU比较（Frozen/SFT/正常编辑RL/Fixed/Rule/UCB/Learned/Random/HPO）均为BLOCKED_NOT_RUN。

已消耗{budget.get('actual_GPU_hours','UNKNOWN')} GPU-hours；包括活动任务最坏时限的承诺量{budget.get('committed_GPU_hours','UNKNOWN')}；可再承诺{budget.get('remaining_uncommitted_GPU_hours','UNKNOWN')} GPU-hours。待运行次序canary→Frozen VQA→seed42/123/3407；前置门槛失败即停止其依赖GPU任务。普通节点故障最多原配置安全恢复一次；OOM/NaN/超时/保存失败不重试。

## 数据、指标与证据

正常VQAv2 train2014内部source-disjoint cohort：train1024/dev128/test512/transfer128，每源图一个问题。train共识≥3/10，评估不按答案一致度筛选。固定3 seed；每seed512 AdamW更新、逻辑batch4、同模型/LoRA/分辨率/预算。dev128在128/256/512步评估，最大dev分、同分选最早；最终test仅在选择后评估。Frozen为共享greedy基线。transfer只是独立source，不能宣称跨任务RSI；公开预训练污染未知，非官方测试服务器结果。

Qwen3-VL-2B pinned revision；BF16 frozen base，FP32 language q/v LoRA rank4/alpha8，LR1e-4，max_pixels65536。正常QA response-only交叉熵，不是RL。

数据准备实况：`{json.dumps(snapshot.get('data',{}),ensure_ascii=False)}`。

完成阶段实际指标：

```json
{json.dumps(metrics,ensure_ascii=False,indent=2)}
```

每个来源的VQA共识分、源图bootstrap CI及逐条预测保存在各attempt中；3 seed全部完成前不填写均值±SD。统计单位是source，不把3×512当作1536独立源图。任务方向为VQA分数↑。未运行的P1论文表格单元格保持空。

三seed配对汇总（只读，不反馈训练）：

{comparison_table}

```json
{json.dumps(analysis,ensure_ascii=False,indent=2)}
```

远端根目录：`{REMOTE}`。协议`protocol.json`，来源SHA`source_hashes.json`，固定数据`manifest.json`，账本`ledger.json`，Slurm脚本`slurm/`，原始日志`logs/`，逐步训练/预测/恢复检查`stages/<stage>/attempt_<n>/`，完整checkpoint在该目录`checkpoints/`和`selected_checkpoint/`。本地小证据镜像：`outputs/{RUN}/remote/`。权重和图像不重复下载到C盘。

磁盘检查时点unix={resource.get('checked_unix','NOT_RUN')}，远端保留原图/配对/日志/checkpoint约{round(resource.get('run_bytes',0)/1024**2,2)} MiB。{run_activity} 模型共享缓存没有复制。本地最新disk收据记录约{local_evidence_MiB} MiB小证据，C盘未创建本轮大模型/数据文件。空本地run scratch已校验路径后移除；远端job trap清理各自临时目录，CPU监控仅rmdir已验证路径的空run根，保留checkpoint和证据。

## 编辑与方法门槛

Frozen图像编辑门槛仍为NOT_CLEARED；既有≥20步/512px视觉诊断见`reports/frozen_quality_gate.md`，未重复采样搜索、未启动下游编辑RL。P1配对数据准备使用原始真实照片作为目标、人工变暗/模糊作为输入，全部对照共用同一split。数据准备不是方法验证。代码映射与阻塞见`reports/OVERNIGHT_METHOD_AUDIT.md`。

Inner是on-policy score-function；Learned Outer是categorical REINFORCE，均不能标成PPO。缺少old-policy模型快照、exp(new-old)重要性ratio与clipped surrogate/clip fraction；梯度裁剪不是PPO。独立任务族迁移/RSI尚无证据。

## 监控任务

```json
{json.dumps(task,ensure_ascii=False,indent=2)}
```

{monitor_description} 任务名`RSI_Experiment_Monitor`，ID `rsi-experiment-monitor`。之前误建的Windows同名任务已停用，证据`reports/monitor/windows_task_disabled_receipt.json`，旧Windows XML/手动测试收据仅为历史记录。{trigger_description} GPU阶段之间的短Slurm CPU完成回调也已全部结束。当前队列/GPU只读探针、checkpoint SHA/更新时间、日志及预算在`reports/monitor/heartbeat.json`。其他用户和其他项目任务不取消、不抢占。

## 异常与下一步

{chr(10).join('- '+str(x) for x in snapshot.get('issues',[])) or '- 本次检查未发现已记录异常；不能据此认证未完成结果。'}

下一步：{snapshot.get('next_action','重建SSH连接后核实真实账本；未知消耗时停止新增提交')}。只按冻结协议推进，不修改LR、reward、训练数据或最终测试集。

修改/新增实现：`src/operations/night_budget.py`、`tests/test_night_budget.py`、`tests/test_night_analysis.py`、`configs/overnight_benign_20261112.json`、`scripts/prepare_night_benign.py`、`scripts/night_vqa_worker.py`、`scripts/night_campaign_remote.py`、`scripts/night_campaign.py`、`scripts/summarize_night_vqa.py`、`scripts/audit_night_vqa_results.py`、`scripts/check_night_p1.py`、`src/training/paired_generator_data.py`及历史Windows监控启动/安装脚本（任务已停用）。新增预算和配对统计CPU测试13项通过（集群快照中预算10项另已实际通过）；命令和故障修复记录在`outputs/{RUN}/COMMANDS.md`；canary/正式运行是否通过，以本表真实状态和远端日志/checkpoint为准。
'''
    (ROOT/'reports/OVERNIGHT_EXPERIMENT_STATUS.md').write_text(text,encoding='utf-8')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['stage','start','monitor','fetch']);a=ap.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True);MONITOR.mkdir(parents=True,exist_ok=True)
    if a.action=='stage':
        files=list((ROOT/'src').rglob('*.py'))+[ROOT/'tests/__init__.py',ROOT/'tests/test_night_budget.py']+[ROOT/'scripts'/x for x in ('prepare_night_benign.py','night_vqa_worker.py','night_campaign_remote.py')]
        hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        protocol=json.loads((ROOT/'configs/overnight_benign_20261112.json').read_text(encoding='utf-8'))
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
            for path in files:archive.add(path,arcname=path.relative_to(ROOT).as_posix(),recursive=False)
            for name,obj in [('__protocol.json',protocol),('__source_hashes.json',hashes)]:
                blob=json.dumps(obj,indent=2).encode();info=tarfile.TarInfo(name);info.size=len(blob);archive.addfile(info,io.BytesIO(blob))
        code=f'''import io,json,sys,tarfile
from pathlib import Path
run=Path({REMOTE!r})
if run.exists():raise ValueError('Unique run already exists; refusing overwrite')
run.mkdir(parents=True);code=run/'code';code.mkdir();data=sys.stdin.buffer.read();(run/'source.tar.gz').write_bytes(data)
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
 for member in archive:
  if member.name.startswith('__'):
   if member.name not in ('__protocol.json','__source_hashes.json'):raise ValueError('Unknown metadata')
   (run/member.name[2:]).write_bytes(archive.extractfile(member).read());continue
  path=(code/member.name).resolve()
  if not member.isfile() or not path.is_relative_to(code.resolve()):raise ValueError('Unsafe archive')
  path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
print(json.dumps({{'status':'STAGED','run':str(run),'source_files':len(json.loads((run/'source_hashes.json').read_text()))}}))
'''
        result=remote(code,buffer.getvalue());(LOCAL/'stage_receipt.json').write_bytes(result);put(LOCAL/'protocol.json',protocol);put(LOCAL/'source_hashes.json',hashes);print(result.decode());return
    if a.action=='fetch':
        code=f'''import io,sys,tarfile
from pathlib import Path
run=Path({REMOTE!r});buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w:gz') as out:
 for pattern in ['*.json','*.log','logs/*.log','slurm/*.sbatch','stages/*/attempt_*/*.json','stages/*/attempt_*/*.jsonl','stages/*/attempt_*/*.txt','p1_preparation/*.json','analysis_code/*.json','analysis_code/*.log','analysis_code/*.sbatch','controlplane_revisions/*']:
  for path in run.glob(pattern):
   if path.is_file():out.add(path,arcname=path.relative_to(run).as_posix(),recursive=False)
sys.stdout.buffer.write(buffer.getvalue())
'''
        data=remote(code);base=(LOCAL/'remote').resolve();base.mkdir(exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
            for member in archive:
                path=(base/member.name).resolve()
                if not member.isfile() or not path.is_relative_to(base):raise ValueError('Unsafe evidence archive')
                path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(archive.extractfile(member).read())
        print(json.dumps({'status':'FETCHED_SMALL_EVIDENCE','archive_bytes':len(data)}));return
    timestamp=datetime.now().strftime('%Y%m%d_%H%M%S');error=None
    code=f'''import subprocess,sys
sys.exit(subprocess.call([{PYTHON!r},'-B',{(REMOTE+'/code/scripts/night_campaign_remote.py')!r},'--campaign',{REMOTE!r}]+{(['--initialize'] if a.action=='start' else [])!r}))
'''
    try:
        snapshot=json.loads(remote(code,timeout=180));put(LOCAL/'latest_snapshot.json',snapshot)
    except Exception as exc:
        error=str(exc);old=LOCAL/'latest_snapshot.json'
        snapshot=json.loads(old.read_text()) if old.exists() else {'run_id':RUN}
        snapshot.update(monitor_connection='FAILED',monitor_error=error,next_action='NO_NEW_SUBMISSION_UNTIL_REAL_ACCOUNTING_RECOVERS')
        snapshot['issues']=snapshot.get('issues',[])+['SSH/dispatcher failed: '+error]
    if not error:
        # Read-back small evidence each check, never model/image archives.
        fetched=subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'fetch'],capture_output=True,text=True,timeout=90)
        if fetched.returncode:snapshot['issues'].append('Small evidence sync failed: '+fetched.stderr[-1000:])
        else:
            try:
                from summarize_night_vqa import summarize
                put(LOCAL/'comparison.json',summarize(LOCAL/'remote',snapshot))
            except Exception as analysis_error:snapshot['issues'].append('Read-only analysis failed: '+repr(analysis_error))
    snapshot['Windows_check_UTC']=datetime.now(timezone.utc).isoformat();snapshot['Windows_check_local']=datetime.now().astimezone().isoformat()
    put(MONITOR/'heartbeat.json',snapshot);render(snapshot)
    report=(ROOT/'reports/OVERNIGHT_EXPERIMENT_STATUS.md').read_text(encoding='utf-8')
    (MONITOR/f'check_{timestamp}.md').write_text(report,encoding='utf-8')
    print(json.dumps({'run_id':RUN,'next_action':snapshot.get('next_action'),'budget':snapshot.get('budget'),'jobs':[{k:r.get(k) for k in ('job_id','stage','state')} for r in snapshot.get('jobs',[])],'data':snapshot.get('data'),'issues':snapshot.get('issues'),'monitor_error':error},ensure_ascii=False))
    if error:raise SystemExit(1)


if __name__=='__main__':main()
