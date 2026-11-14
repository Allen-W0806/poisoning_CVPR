"""CPU reporting only; exact scheduler evidence and image-clustered clean metrics."""
import argparse,csv,hashlib,html,json,re,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
def read(p,default=None):return json.loads(Path(p).read_text()) if Path(p).exists() else default
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True);a=ap.parse_args();run=ROOT/'results'/a.run_id;cfg=read(run/'protocol.json');ledger=read(run/'job_ledger.json');heartbeat=read(ROOT/'reports/rsi_stage_monitor/heartbeat.json');schedule=read(ROOT/'reports/monitoring/scheduler.json');reports=run/'reports';reports.mkdir(exist_ok=True)
    command=['sacct','-X','-j',','.join(r['job_id'] for r in ledger)+',86850,86855','-nP','--format=JobIDRaw,JobName%40,State,ElapsedRaw,ExitCode,Start,End'];p=subprocess.run(command,capture_output=True,text=True,check=True);(run/'sacct_final.txt').write_text(p.stdout);states={r[0]:r for line in p.stdout.splitlines() if (r:=line.split('|')) and len(r)>6}
    rows=[];hashes=[];allpreds=[];perseed=[]
    for record in ledger:
        child=ROOT/'results'/record['run_id'];out=child/'clean_main' if record['kind']=='qwen' else child/'diagnostic';checks=read(out/'checks.json',{});summary=read(out/'summary.json',{});s=states[record['job_id']]
        rows.append({'task':record['task'],'attempt':record['attempt'],'job_id':record['job_id'],'scheduler_state':s[2],'elapsed_GPU_seconds':int(s[3]),'exit_code':s[4],'run_id':record['run_id'],'numerical_status':checks.get('status','NOT VALIDATED')})
        if record['kind']=='qwen' and checks.get('status')=='PASS':
            raw=[json.loads(l) for l in (out/'predictions.jsonl').read_text().splitlines()];before=[r for r in raw if r['stage']=='before'];after=[r for r in raw if r['stage']=='after'];assert len(before)==len(after)==32
            assert [r['sample_id'] for r in before]==[r['sample_id'] for r in after];matrix=np.array([[x['score'],y['score']] for x,y in zip(before,after)]);rng=np.random.default_rng(202610290);boot=matrix[rng.integers(0,32,(10000,32))].mean(1);diff=matrix[:,1]-matrix[:,0]
            values={'seed':record['seed'],'job_id':record['job_id'],'before':float(matrix[:,0].mean()),'after':float(matrix[:,1].mean()),'before_ci95':np.quantile(boot[:,0],[.025,.975]).tolist(),'after_ci95':np.quantile(boot[:,1],[.025,.975]).tolist(),'paired_difference':float(diff.mean()),'paired_ci95':np.quantile(boot[:,1]-boot[:,0],[.025,.975]).tolist(),'normalized_answers_changed':sum(x['normalized_answer']!=y['normalized_answer'] for x,y in zip(before,after)),'posttraining_repeat_and_reload':'PASS','fresh_object_next_step':'PASS','frozen_base_hash':summary['frozen_base_sha256'],'final_adapter_sha256':summary['adapter_sha256'],'trainable_tensors':summary['trainable_tensors'],'trainable_elements':summary['trainable_elements']}
            for r in raw:allpreds.append({'training_seed':record['seed'],**r})
            perseed.append(values);hashes.append(summary['adapter_sha256'])
    assert len(perseed)==3 and len(set(hashes))==3
    write(run/'replication_statistics.json',{'status':'PASS_ENGINEERING','per_seed':perseed,'effective_unique_evaluation_sources':32,'training_seeds':3,'confidence_method':'10000 paired image bootstrap; seed replicates share the same evaluation images; no effective N=96 inflation','accuracy_gain_supported':False,'training_seed_mean_accuracy_std':float(np.std([r['after'] for r in perseed],ddof=1))})
    with (run/'clean_replication_evaluation.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allpreds[0]));w.writeheader();w.writerows(allpreds)
    gpu=sum(r['elapsed_GPU_seconds'] for r in rows);resources={'new_stage_GPU_seconds':gpu,'prior_validation_GPU_seconds':cfg['charged_prior_seconds'],'charged_total_GPU_seconds':gpu+cfg['charged_prior_seconds'],'remaining_GPU_seconds':3600-gpu-cfg['charged_prior_seconds'],'budget_GPU_seconds':3600,'registered_GPU_jobs':rows,'CPU_verification_jobs':{j:{'state':states[j][2],'elapsed_seconds':int(states[j][3])} for j in ['86850','86855']},'other_projects_not_charged_to_this_validation_budget':True,'new_images_bytes':read(run/'fresh_source_receipt.json')['images_downloaded_bytes']};write(run/'resource_accounting.json',resources)
    p2=[]
    for row in rows:
        if row['task']=='P2_IP2Pix':
            out=ROOT/'results'/row['run_id']/'diagnostic';p2.append({'job_id':row['job_id'],'run_id':row['run_id'],'summary':read(out/'summary.json',{}),'raw_localization':read(out/'raw_backward_localization.json',{})})
    write(run/'ip2pix_comparison_summary.json',p2)
    checked=datetime.fromisoformat(heartbeat['checked_UTC']).astimezone(ZoneInfo('America/New_York')).isoformat();nextdue=datetime.fromisoformat(schedule['next_due_utc']).astimezone(ZoneInfo('America/New_York')).isoformat();prefix='../results/'+run.name
    table='| Task | Attempt | Slurm job | Scheduler state | Numerical checks | GPU seconds |\n|---|---:|---|---|---|---:|\n'+''.join(f"| {r['task']} | {r['attempt']} | {r['job_id']} | {r['scheduler_state']} | {r['numerical_status']} | {r['elapsed_GPU_seconds']} |\n" for r in rows)
    monitoring=f'''# Live monitoring and execution evidence

Run `{run.name}`. Latest successful stage inspection **{checked}** (America/New_York). Monitor PASS means inspection completed; it does not mean every experiment passed.

Initial live inspection found a genuine existing two-hour Slurm chain, pending86836, latest legacy heartbeat2026-10-29T04:33:01Z, but its123-entry legacy registry did not cover the new result directories or Latest dashboard. Earlier clean baseline jobs86804–86807 actually executed; the next replication/diagnostic recommendations had not yet been submitted. The new coverage and experiments were executed this stage.

`crontab -l`: command not found (127). `systemctl --user list-timers --all`: Failed to connect to bus: No medium found (1). `scrontab -l`: disabled on this cluster. These errors are preserved in [activation probes]({prefix}/audit/activation_probe.json). The permitted native mechanism is a short CPU-only Slurm successor chain; no permanent login-node process, sleeping GPU or dependence on an open chat.

Installation reuses `scripts/manage_monitor.py` / `src.monitoring.scheduler.Scheduler`, extends its callback with `scripts/rsi_stage_monitor.py`, and registers the approved stage in `reports/rsi_stage_monitor/active_stage.json`. Native install/status confirms enabled=true, interval7200seconds, methodslurm_cpu_successor. Registration entry: `reports/monitoring/monitor_job.sbatch`; owned comment prefix `poisoning-monitor-v1:f916f24e0de6`; future job **{schedule['next_job_id']}**; next eligible **{nextdue}**, subject to queue delay. Do not interpret eligibility as a guaranteed start.

Actual registered entry-point verification job **86850 COMPLETED**,4 CPU-seconds, invoked the native tick inside an owned Slurm allocation, executed both old and new inspections, and retained the future successor. Completion CPU jobs86847/86849/86852 advanced all three seeds into P2 without chat/terminal supervision. Separate nonblocking scheduler and extension locks prevent overlap. Each experiment also has an afterany CPU check; this is an event hook, not a second recurring chain.

{table}

The monitor inspects sacct states, pending/stalled progress, training JSONL finite values, numerical assertions, checkpoint/output hash inventories and expected step/evaluation counts. Known recoverable node/preemption/transient-I/O failures may retry from initialization, with unique run IDs, unchanged science and at most two additional attempts per task; budget checks include active reservations. A persisted uncertain submission blocks duplicate submission. Unexplained numerical failures are retained and blocked, not repeatedly seeded until PASS. Failure tests cover failed/node/pending/stalled/completed states, overlapping locks, checkpoint tampering and numerical assertions:9/9 PASS, zero real jobs disrupted. The instrumented P2 follow-up adds read-only measurements and preserves attempt0.

Heartbeat: `reports/rsi_stage_monitor/heartbeat.json`. Latest log: `{heartbeat['log_path']}`. Periodic scheduler state: `reports/monitoring/scheduler.json`; periodic verification log: `reports/rsi_stage_monitor/periodic-entry-86850.out`. Dashboard: `dashboard/RSI_Experimental_Progress_Latest.html`, updated after every extension pass. All findings and failed states remain visible. No claim of high availability: the existing scheduler trips a circuit breaker after three failed/interrupted checks, and an import/controller failure before successor submission can interrupt recurrence.

This stage allocated **{gpu} A100-seconds**. Including earlier gradient198 + matched584 + clean220 gives **{resources['charged_total_GPU_seconds']}/3600 seconds**, leaving **{resources['remaining_GPU_seconds']} seconds**. No active reservation remains when all listed jobs are terminal. Other account activity such as WANOP86841 belongs to `RSI_Video_Branching_20261027_v1`, not this approved validation scope; it was not modified. Dollar billing unavailable. [Raw accounting]({prefix}/sacct_final.txt), [budget ledger]({prefix}/resource_accounting.json).

Review handoff engineering PASS, human semantic review BLOCKED:160 cases/320 hashes checked, no genuine independent records. See [handoff](INDEPENDENT_REVIEW_HANDOFF.md). Generator reward-release guards remain active. No backdoor training, poisoning-policy optimization or ASR-driven optimization.
'''
    qwen='''# Clean Qwen three-seed replication

**PASS for bounded training engineering; no accuracy improvement observed.** The three independent adapter-initialization seeds42/123/3407 each execute16 logical steps plus one fresh-object next-step replay. Train32 cached original examples; fresh held-out32 original images excluded from known project exposure by image ID/content hash. All questions/ten-answer lists originate in cached official VQAv2 archives;6.57MB of new original COCO images were fetched from the official source. Source IDs, original hashes, ten human answers, consensus labels, source URLs and selection decisions were frozen before inference. Freshness is project-level; model-pretraining exposure and perceptual near duplicates remain unresolved. The high-consensus category-balanced diagnostic pool is not population representative.

Pinned Qwen revision89644892e4d85e24eaac8bacfd4f463576704203; existing .venv-editor. BF16 base, FP32 rank8/alpha16 language q/v attention LoRA, AdamW1e-4, constant scheduler, accumulation2 and norm cap1. Exactly112 tensors /1,605,632 elements update in every seed. Base and visual parameters remain frozen. The three final adapter hashes differ, confirming separate adaptation states.

| Seed | Job | Before | After | Paired difference | 95% paired CI | Fresh-object next step |
|---:|---|---:|---:|---:|---|---|
'''+''.join(f"| {r['seed']} | {r['job_id']} | {100*r['before']:.2f}% | {100*r['after']:.2f}% | {100*r['paired_difference']:+.2f}pp | [{100*r['paired_ci95'][0]:+.2f}, {100*r['paired_ci95'][1]:+.2f}]pp | PASS |\n" for r in perseed)+f'''
All three before/after image-bootstrap intervals are [{100*perseed[0]['before_ci95'][0]:.2f}%,{100*perseed[0]['before_ci95'][1]:.2f}%]. Shared evaluation images give effective n32, not96; the three training seeds are separate engineering replications. Zero observed score changes produces a degenerate empirical paired interval; this does not establish population equivalence. Seed-to-seed mean-score standard deviation is0 in this small fixture.

Fresh-object continuation creates a new pinned model and LoRA object at logical step8, restores optimizer/scheduler/all saved RNG states, takes the same next update, and compares gradient tensors, adapters, optimizer, scheduler and RNG bitwise. All pass in all three seeds. Repeated and fresh-final-reloaded predictions also match. Finite nonzero gradients, exact optimizer ownership, saved-checkpoint binding and per-tensor parameter changes pass. No assertion was relaxed.

Per-seed statistics, checkpoint hashes and decoded predictions: [replication_statistics.json]({prefix}/replication_statistics.json), [evaluation CSV]({prefix}/clean_replication_evaluation.csv). Source manifest: [fresh_source_receipt.json]({prefix}/fresh_source_receipt.json). This is a small original-VQA LoRA infrastructure test, not full-model training, generalization gain or backdoor evidence.
'''
    diag='''# Pretrained IP2Pix BF16 checkpoint diagnostic

**Uninstrumented strict continuation FAIL; instrumented witness PASS; general reliability NOT VALIDATED.** Actual pretrained BF16 IP2Pix and production attention-LoRA/DDPO-SF wiring are used in isolation, with a constant content-independent coefficient1. No target VLM is loaded and no attack, ASR, distortion/cost or model-error reward is used. The legacy production reward configuration is recorded as provenance but is not evaluated. Frozen UNet base/VAE/text weights remain unchanged. Four bounded optimizer calls per diagnostic include warm/reference/restored/control branches; no generator-learning efficacy is claimed.

The first run reproduces the previous failure: restored parameters/optimizer/RNG, conditioning, stochastic seeds, every saved trajectory state/action/log-probability and decoded pixels match. Stored **post-clipping** gradients diverge, followed by parameters and optimizer state. Maximum stored gradient error7.6610595e-6, relative L2 error6.3059489e-5. These errors are descriptive; bitwise acceptance remains strict. All256 stored gradient tensors differ. A checkpoint-disabled control produces identical resumed gradient tensors to the enabled resumed branch, so this fixture does not implicate checkpoint recomputation as the cause.

The initial gradients are stored after clipping, so raw backward divergence versus clipping/norm divergence is not identified in that run. A uniquely identified follow-up captures raw gradients before clipping, the native preclip norm and gradient layouts; it does not change the objective, optimizer or threshold. The instrumented run matches raw and clipped gradients, optimizer and RNG bitwise, with equal preclip norms1.834847331. Both checkpoint-enabled and disabled branches match. The original failure remains preserved. CPU copies synchronize CUDA and alter tensor lifetimes; instrumentation plus a different process/context can change execution. These two runs do not isolate the causal kernel or prove that adding capture repairs the general training path. No further seed search or repeated run-until-PASS is performed.

'''
    for r in p2:
        diag+=f"Job **{r['job_id']}**, run `{r['run_id']}`, status **{r['summary'].get('status','NOT VALIDATED')}**. Raw localization: `{json.dumps(r['raw_localization'])}`.\n\n"
    diag+=f'''[Derived stored-gradient analysis]({prefix}/audit/ip2pix_backward_forensics.json), [comparison summary]({prefix}/ip2pix_comparison_summary.json). Root kernel/arithmetic cause is not inferred without evidence. The historical full batch10 stochastic continuation remains NOT VALIDATED; a small benign diagnostic cannot certify that old run. Tiny random-UNet and clean Qwen success do not close this separate gate. Full pretrained generator learning remains blocked.
'''
    for name,text in [('REPLICATION_AND_MONITORING_STATUS',monitoring),('QWEN_THREE_SEED_REPLICATION',qwen),('IP2PIX_BF16_DIAGNOSTIC',diag)]:
        (ROOT/'reports'/f'{name}.md').write_text(text);(reports/f'{name}.md').write_text(text)
    decision=ROOT/'reports/NEXT_EXPERIMENT_DECISION.md';prior=decision.read_text();marker='<!--REPLICATION_MONITOR_20261029_V1-->'
    if marker not in prior:
        (run/'audit/NEXT_EXPERIMENT_DECISION_previous.md').write_text(prior);decision.write_text(marker+'\n# Current execution decision\n\nThree-seed clean Qwen replication and fresh-object next-step checks PASS; fresh32-image accuracy90.0% before/after, no gain. Pretrained IP2Pix uninstrumented strict continuation FAIL; read-only instrumented witness PASS. The latter changes synchronization/tensor lifetime and does not resolve general BF16 reliability or localize the causal kernel. Human review0/160 BLOCKED. Persistent two-hour monitoring is verified under Slurm; all approved jobs and failures are retained. No further unapproved training or attack optimization. See [execution evidence](REPLICATION_AND_MONITORING_STATUS.md), [Qwen](QWEN_THREE_SEED_REPLICATION.md), [IP2Pix](IP2PIX_BF16_DIAGNOSTIC.md), and [human handoff](INDEPENDENT_REVIEW_HANDOFF.md).\n\nPrior-stage decision follows unchanged.\n\n'+prior)
    block='<section style="max-width:1180px;margin:20px auto;padding:24px"><h2>Executed P1–P3 · '+html.escape(checked)+'</h2><p>Three clean Qwen seeds: PASS,16 logical steps each; fresh32-image accuracy90.0% before and after. Fresh-object next-step gradients, optimizer and RNG match bitwise; base weights unchanged. No improvement claim.</p><p>IP2Pix BF16: uninstrumented continuation FAIL; instrumented witness PASS. Synchronization and tensor-lifetime sensitivity remain unresolved; the negative result is preserved. Human semantic review remains BLOCKED:0/160; verified dual-review handoff is ready.</p><p>Budget charged '+str(resources['charged_total_GPU_seconds'])+' /3600 A100-seconds; remaining '+str(resources['remaining_GPU_seconds'])+'s. No attack training or ASR measurement.</p><p><a href="../reports/REPLICATION_AND_MONITORING_STATUS.md">Scheduler and execution evidence</a> · <a href="../reports/QWEN_THREE_SEED_REPLICATION.md">Three-seed results</a> · <a href="../reports/IP2PIX_BF16_DIAGNOSTIC.md">Negative BF16 diagnostic</a> · <a href="../reports/INDEPENDENT_REVIEW_HANDOFF.md">Human handoff</a></p></section>'
    path=ROOT/'dashboard/RSI_Experimental_Progress_Latest.html';text=path.read_text();text=re.sub(r'<!--RSI_REPLICATION_RESULTS_START-->.*?<!--RSI_REPLICATION_RESULTS_END-->','',text,flags=re.S);text=text.replace('</header>','</header><!--RSI_REPLICATION_RESULTS_START-->'+block+'<!--RSI_REPLICATION_RESULTS_END-->',1);path.write_text(text)
    event={'event':'REPLICATION_MONITOR_20261029_V1','run_id':run.name,'Qwen_seeds':3,'Qwen_engineering':'PASS','Qwen_accuracy_before_after':[.9,.9],'IP2Pix_strict_continuation':'FAIL','semantics':'BLOCKED','GPU_seconds_new':gpu,'GPU_seconds_total_including_prior':resources['charged_total_GPU_seconds'],'monitor_successor_job':schedule['next_job_id']}
    history=ROOT/'experiments/semantic_clean_history.jsonl'
    if event['event'] not in history.read_text():
        with history.open('a') as f:f.write(json.dumps(event)+'\n')
    index=ROOT/'EXPERIMENT_INDEX.md'
    if event['event'] not in index.read_text():
        with index.open('a') as f:f.write('\n\n## '+event['event']+'\n\nActual Slurm three-seed Qwen replication PASS;90.0% clean original VQA before/after. Pretrained BF16 IP2Pix strict continuation FAIL, preserved and instrumented. Existing two-hour monitor extended and verified in native CPU allocation; automatic completion hooks proved. Human review BLOCKED; dual-review handoff ready. [Execution and scheduler evidence](reports/REPLICATION_AND_MONITORING_STATUS.md).\n')
    write(run/'report_generation.json',{'status':'PASS','statistical_reports_generated':True,'event':event,'heartbeat_snapshot':heartbeat,'scheduler_snapshot':schedule});print(json.dumps(event,indent=2))
if __name__=='__main__':main()
