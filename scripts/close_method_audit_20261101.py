"""Close the evidence packet and publish its scoped status in existing views.

No training, submissions, reward release, acceptance changes, or new timer.
"""
import hashlib, html, json, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'method_code_audit_20261101T0510Z_v1'

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, value):
    p = Path(p)
    temp = p.with_suffix(p.suffix + '.partial')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(p)

def dashboard(root):
    """Passive renderer called by the existing monitor's dashboard refresh."""
    root = Path(root)
    status = root / 'reports/METHOD_AUDIT_CLOSURE_20261101.json'
    if not status.exists():
        return
    s = read(status)
    data = read(root / 'dashboard/dashboard_data.json')
    data['method_implementation_audit_20261101'] = s
    encoded = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')
    for name in ['rsi_poisoning_zh.html', 'rsi_poisoning_en.html', 'index_zh.html', 'index_en.html', 'RSI_Experimental_Progress_Latest.html']:
        p = root / 'dashboard' / name
        if not p.exists():
            continue
        zh = '_zh' in name or 'Latest' in name
        title = '方法—执行代码核查：科学门槛 NO-GO' if zh else 'Method-to-code audit: scientific gate NO-GO'
        description = ('IP2Pix：1,594,368 个 LoRA 参数；实际算法 DDPO-SF，非 GRPO。72 次局部 VJP 位级一致，仅为局部工程 PASS；整图 BF16 重启仍未验证，独立人工标签为 0。' if zh else 'IP2Pix: 1,594,368 LoRA parameters; executed algorithm DDPO-SF. All 72 local VJP replays were bitwise equal: local engineering PASS only. Full-graph BF16 restart and independent edited labels remain unvalidated.')
        b = s['budget']
        block = '<aside id="method-audit-current" style="padding:20px;background:#fff3cd;color:#312300;overflow-wrap:anywhere"><h2>' + title + '</h2><p>' + html.escape(s['updated_UTC']) + '</p><p>' + description + '</p><p>Validation GPU ' + str(b['consumed']) + '/3600s; remaining ' + str(b['remaining']) + 's; reserved 0s. '
        block += ('较早预算卡片为各阶段历史快照。' if zh else 'Earlier budget cards are historical stage snapshots.') + '</p><p>'
        block += '; '.join(html.escape(x['job_id'] + ': ' + x['state'] + ' (' + x['exit_code'] + ')') for x in s['jobs'])
        block += '</p><p>' + ('保留唯一两小时 Slurm 监控链；此处为被动状态展示，不创建定时器或新 GPU 作业。' if zh else 'The existing single two-hour Slurm chain is retained. This passive renderer creates no timer or GPU job.') + '</p><p>'
        for label in ['METHOD_IMPLEMENTATION_AUDIT', 'METRIC_AND_NORMALIZATION_AUDIT', 'RL_RSI_CONTROLLED_VALIDATION', 'NEXT_EXPERIMENT_DECISION']:
            block += '<a href="../reports/' + label + '.md">' + label + '</a> · '
        block += '</p></aside>'
        text = p.read_text(encoding='utf-8')
        # The compatibility page uses valid HTML with an implicit body.
        # Make that element explicit before adding the same visible/data blocks.
        if not re.search(r'<body(?:\s|>)', text):
            assert '<header' in text and '</html>' in text
            text = text.replace('<header', '<body><header', 1)
            text = text.replace('</html>', '</body></html>', 1)
        text = re.sub(r'<!--METHOD_AUDIT_CURRENT_START-->.*?<!--METHOD_AUDIT_CURRENT_END-->', '', text, flags=re.S)
        text = re.sub(r'(<body[^>]*>)', lambda m: m[1] + '<!--METHOD_AUDIT_CURRENT_START-->' + block + '<!--METHOD_AUDIT_CURRENT_END-->', text, count=1)
        text = re.sub(r'(<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>).*?(</script>)', lambda m: m[1] + encoded + m[2], text, flags=re.S)
        if not re.search(r'<script[^>]*id="(?:boot-data|dashboard-state)"', text):
            text = text.replace('</body>', '<script type="application/json" id="dashboard-state">' + encoded + '</script></body>')
        p.write_text(text, encoding='utf-8')
    write(root / 'dashboard/state.json', data)
    write(root / 'dashboard/dashboard_data.json', data)

def main():
    import torch
    sys.path.insert(0, str(ROOT))
    run = ROOT / 'results' / RUN
    assert not (run / 'closure.json').exists(), 'Preserve completed closure; do not silently overwrite'
    ledger = read(run / 'job_ledger.json')
    ids = [ledger[k] for k in ['cpu_job', 'gpu_job', 'report_job', 'prefix_cpu_job']]
    raw = subprocess.check_output(['sacct', '-X', '-nP', '-j', ','.join(ids), '--format=JobID,State,Start,End,ElapsedRaw,AllocTRES,ExitCode'], text=True)
    jobs = []
    for line in raw.splitlines():
        v = line.split('|')
        if len(v) >= 7:
            jobs.append(dict(job_id=v[0], state=v[1], start_EDT=v[2], end_EDT=v[3], elapsed_seconds=int(v[4]), gpu_seconds=int(v[4]) if 'gres/gpu=1' in v[5] else 0, exit_code=v[6]))
    assert len(jobs) == 4 and all(x['state'] == 'COMPLETED' and x['exit_code'] == '0:0' for x in jobs)
    before = read(run / 'final_status.json')
    assert sum(x['gpu_seconds'] for x in jobs) == 35
    assert before['budget']['consumed'] == 2281
    # Explain contract mismatch without inferring it from a hash alone.
    contracts = []
    for row in read(run / 'historical_prefix_state_comparison.json')['rows']:
        if row['step'] != 0:
            continue
        a = torch.load(ROOT / 'experiments' / row['baseline'] / 'checkpoints/step_0000/training_state.pt', map_location='cpu', weights_only=False)['contract']
        b = torch.load(ROOT / 'experiments' / row['rsi'] / 'checkpoints/step_0000/training_state.pt', map_location='cpu', weights_only=False)['contract']
        contracts.append(dict(seed=row['seed'], baseline_contract=a, rsi_contract=b))
    write(run / 'historical_initial_contracts.json', contracts)
    original_index = read(run / 'complete_source_index.json')
    assert all(sha(ROOT / n) == v['sha256'] for n, v in original_index.items())
    assert sha(ROOT / 'src/rl/phase5_backend.py') == before['guard_sha256']
    # Freeze all new evidence, including checkpoint tensors and local VJP inputs.
    files = [dict(path=p.relative_to(run).as_posix(), bytes=p.stat().st_size, sha256=sha(p)) for p in sorted(run.rglob('*')) if p.is_file() and p.suffix == '.pt']
    assert files and any('after_diagnostic_update/training_state.pt' in p['path'] for p in files)
    write(run / 'new_tensor_integrity.json', dict(status='PASS_INTEGRITY_ONLY', files=files, retained_bytes=sum(p['bytes'] for p in files)))
    for name in ['scripts/close_method_audit_20261101.py', 'scripts/check_method_prefix_20261101.py', 'scripts/report_method_audit_20261101.py', 'scripts/manage_method_audit_20261101.py']:
        d = run / 'audit/executed_source' / name
        d.parent.mkdir(parents=True, exist_ok=True)
        if not d.exists():
            shutil.copyfile(ROOT / name, d)
    schedule = read(ROOT / 'reports/monitoring/scheduler.json')
    queue = subprocess.check_output(['squeue', '-h', '-j', str(schedule['next_job_id']), '-o', '%i|%T|%S|%k'], text=True).strip()
    assert schedule['enabled'] and schedule['interval_hours'] == 2 and queue.startswith(str(schedule['next_job_id']) + '|PENDING|')
    s = dict(run_id=RUN, updated_UTC=datetime.now(timezone.utc).isoformat(), scientific_gate='NO-GO', jobs=jobs, budget=before['budget'], local_VJP='PASS_LOCAL_VJP_ONLY', full_graph_BF16_restart='NOT VALIDATED', independent_edited_labels=0, actual_task_learning='NOT VALIDATED', scientific_improvement='NOT VALIDATED', four_arm_image_training='BLOCKED_NOT_LAUNCHED', monitor=dict(method=schedule['method'], last_success_UTC=schedule['last_success_utc'], next_job_id=schedule['next_job_id'], next_due_UTC=schedule['next_due_utc'], scheduler_evidence=queue, duplicate_created=False), tensor_integrity_files=len(files))
    write(run / 'closure.json', s)
    write(ROOT / 'reports/METHOD_AUDIT_CLOSURE_20261101.json', s)
    evidence = '\n## Final execution and claim boundary\n\n' + json.dumps(s, indent=2) + '\n\nDirect initial checkpoint contract values are retained in results/' + RUN + '/historical_initial_contracts.json. Both seeds initially share parameters, optimizer, scheduler and RNG. Before LR intervention at step4, all256 exp_avg and256 exp_avg_sq tensors differ, although saved RNG, data order, rollout seeds and LR match. This is a confounded historical comparison, not evidence of RSI benefit.\n'
    for name in ['METHOD_IMPLEMENTATION_AUDIT.md', 'METRIC_AND_NORMALIZATION_AUDIT.md', 'RL_RSI_CONTROLLED_VALIDATION.md', 'NEXT_EXPERIMENT_DECISION.md']:
        p = ROOT / 'reports' / name
        dest = run / 'audit/preclosure_reports' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        text = p.read_text(encoding='utf-8').replace('`' + RUN + '/actual_source_hash_recheck.json`', '`results/' + RUN + '/actual_source_hash_recheck.json`')
        p.write_text(text + evidence, encoding='utf-8')
    with (ROOT / 'reports/METHOD_IMPLEMENTATION_AUDIT.md').open('a', encoding='utf-8') as f:
        f.write('\n## Figure1 / Algorithm1 execution coverage\n\n| Stage | Execution evidence | Remaining boundary |\n|---|---|---|\n| Shared generator/policy | Actual pretrained IP2Pix LoRA and DDPO-SF | Algorithm1 separate phi/psi is not reconciled |\n| Original q/y retained | Official cached join verified1040 records | Edited factual labels require independent review |\n| External victim training | Clean Qwen LoRA engineering replication | Matched modified/clean victim lineage not established |\n| Delayed development feedback | Metric interface and conceptual diagram | Post-training credit to generator not implemented/released |\n| RSI revision | Historical LR cooldown; separate workflow-check harness | No demonstrated natural-image recursive scientific gain |\n| Untouched final test | Diagram requirement | Current real-image data exposed; no independent final claim |\n')
    backup = run / 'audit/preclosure_dashboard'
    backup.mkdir()
    for name in ['rsi_poisoning_zh.html', 'rsi_poisoning_en.html', 'index_zh.html', 'index_en.html', 'RSI_Experimental_Progress_Latest.html', 'state.json', 'dashboard_data.json']:
        p = ROOT / 'dashboard' / name
        if p.exists():
            shutil.copyfile(p, backup / name)
    dashboard(ROOT)
    data = read(ROOT / 'dashboard/state.json')
    assert (ROOT / 'dashboard/state.json').read_bytes() == (ROOT / 'dashboard/dashboard_data.json').read_bytes()
    views = []
    for name in ['rsi_poisoning_zh.html', 'rsi_poisoning_en.html', 'RSI_Experimental_Progress_Latest.html']:
        text = (ROOT / 'dashboard' / name).read_text(encoding='utf-8')
        payload = re.search(r'<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>(.*?)</script>', text, re.S)
        assert payload and json.loads(payload[1]) == data
        assert text.count('id="method-audit-current"') == 1
        views.append(name)
    write(run / 'dashboard_data_QA.json', dict(status='PASS_DATA_AND_HTML_CONTRACTS_ONLY', views=views, common_payload=True, browser_visual_QA='NOT PERFORMED', no_new_monitor=True))
    with (ROOT / 'EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as f:
        f.write('\n\nMethod audit closure: CPU88083/88085/88090 and GPU88084 COMPLETED0:0;35 new A100-seconds, cumulative2281/3600. Local VJP PASS only. Scientific four-arm image-edit training BLOCKED;0 real edited-label reviews; historical preintervention adapter and optimizer mismatch preserved.\n')
    print(json.dumps(s, ensure_ascii=False))

def repair_dashboard_only():
    run = ROOT / 'results' / RUN
    s = read(run / 'closure.json')
    assert s['budget']['consumed'] == 2281 and s['scientific_gate'] == 'NO-GO'
    failure = dict(job_ids=['88093','88094'], status='FAILED_PRESERVED', exit_codes=['1:0','1:0'], root_cause='Latest compatibility HTML uses an implicit body and has no embedded JSON. First repair added a JSON insertion rule but missed the absent explicit body tags. Original bilingual equality already passed.', fix='Make the existing implicit body explicit, then add the same authoritative payload/banner; retain the original equality assertion', numerical_tests_rerun=False, thresholds_changed=False)
    write(run / 'dashboard_closure_failure.json', failure)
    dashboard(ROOT)
    data = read(ROOT / 'dashboard/state.json')
    assert (ROOT / 'dashboard/state.json').read_bytes() == (ROOT / 'dashboard/dashboard_data.json').read_bytes()
    views = []
    for name in ['rsi_poisoning_zh.html', 'rsi_poisoning_en.html', 'RSI_Experimental_Progress_Latest.html']:
        text = (ROOT / 'dashboard' / name).read_text(encoding='utf-8')
        payload = re.search(r'<script[^>]*id="(?:boot-data|dashboard-state)"[^>]*>(.*?)</script>', text, re.S)
        assert payload and json.loads(payload[1]) == data
        assert text.count('id="method-audit-current"') == 1
        views.append(name)
    write(run / 'dashboard_data_QA.json', dict(status='PASS_DATA_AND_HTML_CONTRACTS_ONLY', views=views, common_payload=True, browser_visual_QA='NOT PERFORMED', no_new_monitor=True, original_failure=failure))
    for name in ['METHOD_IMPLEMENTATION_AUDIT.md', 'METRIC_AND_NORMALIZATION_AUDIT.md', 'RL_RSI_CONTROLLED_VALIDATION.md', 'NEXT_EXPERIMENT_DECISION.md']:
        p = ROOT / 'reports' / name
        with p.open('a', encoding='utf-8') as f:
            f.write('\n## Retained reporting failures and bounded repair\n\nCPU88093 and88094 failed the final dashboard-contract assertion: legacy Latest HTML used an implicit body and had no embedded JSON. Chinese/English data already matched. The passive renderer now makes that existing body explicit and adds the authoritative JSON/banner; the original payload-equality/unique-banner assertions pass. Evidence: results/' + RUN + '/dashboard_data_QA.json. This is an interface repair; no numerical test, reward or threshold was changed.\n')
    with (ROOT / 'EXPERIMENT_INDEX.md').open('a', encoding='utf-8') as f:
        f.write('\n\nMethod audit closure: CPU88083/88085/88090 and GPU88084 COMPLETED0:0;35 new A100-seconds, cumulative2281/3600. CPU88093 reporting-interface failure preserved; bounded CPU repair checks authoritative Chinese/English/Latest payload equality. Local VJP PASS only; real image-edit learning and scientific improvement NOT VALIDATED.\n')
    shutil.copyfile(Path(__file__), run / 'audit/executed_source/scripts/close_method_audit_repaired_v1.py')
    print(json.dumps(dict(status='PASS_DASHBOARD_DATA_REPAIR_ONLY', retained_failure=failure)))

if __name__ == '__main__':
    if '--repair-dashboard-only' in sys.argv:
        repair_dashboard_only()
    else:
        main()
