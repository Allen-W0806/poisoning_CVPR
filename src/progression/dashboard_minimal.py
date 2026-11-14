"""Both language views derive from one atomic JSON snapshot."""
import html
import json
import os
import uuid
from pathlib import Path
from ..monitoring.common import read_json,now
from ..utils.logging import write_json,git_commit

def atomic_text(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    tmp.write_text(text,encoding='utf8');os.replace(tmp,path)

def refresh(root,live=None):
    root=Path(root);base=root/'dashboard';base.mkdir(exist_ok=True)
    state=read_json(root/'reports/PHASE6_STAGE_STATUS.json',{})
    receipts={p.stem:read_json(p) for p in (root/'reports/phase6').glob('S*.json')}
    source={'schema_version':'phase6_dashboard_v1','updated_utc':now(),'source_commit':git_commit(root),
      'stage_status':state,'stage_receipts':receipts,'live':live or read_json(root/'reports/phase6/live.json',{}),
      'models':{'VLM':'Qwen/Qwen3-VL-2B-Instruct','generator':'timbrooks/instruct-pix2pix','trainable':'UNet attention LoRA rank8 alpha16'},
      'actual':{'new_gpu_hours':0,'new_generator_updates':0,'human_verified_labels':0},
      'historical':{'phase4_updates':32,'phase4_gpu_hours':13122/3600,'scientific_RSI_advantage':'not established','strict_continuation':'unresolved'},
      'synthetic':{'phase5_integrity_accuracy_before':.5,'phase5_integrity_accuracy_after':1.,'sources':32},
      'prepared':{'arms':['M0','M1','M2','M3','M4','M5'],'status':'PREPARED_NOT_RUN','historical_planning_gpu_hours':42.19540658619441},
      'authorized_budget':{'scientific_gpu_seconds':0},
      'unverified_claims':['natural-image semantic improvement','real post-training security effect','RL/RSI superiority'],
      'failure_history':read_json(root/'reports/phase6/failure_summary.json',{}),
      'sync_notice':'Local files update only after scripts/sync_phase6.ps1 and browser reload. This copy is a timestamped snapshot.'}
    encoded=json.dumps(source,ensure_ascii=False,allow_nan=False,indent=2)
    safe_encoded=encoded.replace('<','\\u003c')
    write_json(base/'state.json',source)
    # Immutable revision snapshot enables audit if two atomic page replacements straddle a sync.
    for lang,name in [('zh','index_zh.html'),('en','index_en.html')]:
        zh=lang=='zh';title='RSI 图像编辑实验进度' if zh else 'RSI image-editing research progress'
        stage_rows=''.join('<tr><td>'+html.escape(s)+'</td><td>'+html.escape(v['state'])+'</td><td>'+html.escape(v.get('reason',v.get('scope',''))+' '+v.get('completion_evidence_state',''))+'</td></tr>' for s,v in state.get('stages',{}).items())
        reward=receipts.get('S1_REWARD_AUDIT',{}).get('results',{})
        harness=receipts.get('S2_REAL_HARNESS',{}).get('results',{})
        selected=harness.get('selected_result',{})
        review=receipts.get('S1_REVIEW_PACKAGE',{})
        reviewlink='../experiments/'+review.get('experiment_id','')+'/index.html'
        bars=''.join(f'<div class="barrow"><span>{html.escape(k)}</span><div class="bar" style="width:{min(100,100*v.get("mean",0))}%"></div><span>{v.get("mean",0):.3f}</span></div>' for k,v in reward.get('component_distributions',{}).items() if k in ['semantic','answer_consistency','distortion','cost','invalid_penalty'])
        labels=['当前阶段与作业','模型与完整方法','Reward 审计','RSI 实测','历史结果与限制','下一步'] if zh else ['Experimental progress','Models and methodology','Reward audit','Verified workflow result','Historical results and limitations','Next milestones']
        badge='Scientific NO-GO / 科学训练未放行' if zh else 'Scientific NO-GO'
        detail=''
        if zh:
            detail='<section><h2>运行、排队、失败与修复</h2><pre>'+html.escape(json.dumps(source['live'],ensure_ascii=False,indent=2))+'</pre><p>已修复工程项：阶段门控、完成验证、双语同步、旧GPU恢复路径的零预算限制。科学问题不自动重试。</p><pre>'+html.escape(json.dumps(source['failure_history'],ensure_ascii=False,indent=2))+'</pre></section>'
        document=f'''<!doctype html><html lang="{lang}"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>{title}</title><style>*{{box-sizing:border-box}}body{{margin:0;background:#edf3f7;color:#152a3b;font:17px system-ui}}main{{max-width:1180px;margin:auto;padding:32px 22px}}header{{padding:24px 0}}h1{{font-size:38px;line-height:1.2}}h2{{font-size:25px}}section{{min-width:0;overflow-wrap:anywhere;background:white;padding:24px;margin:20px 0;border:1px solid #d4e0e8;border-radius:16px}}.badge{{display:inline-block;background:#7e2b1e;color:white;padding:8px 14px;border-radius:20px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.metric{{font-size:31px;font-weight:700}}table{{table-layout:fixed;width:100%;border-collapse:collapse}}td,th{{border-bottom:1px solid #d8e2e9;padding:12px;text-align:left;overflow-wrap:anywhere}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:14px}}a{{color:#145a88}}.barrow{{display:grid;grid-template-columns:170px 1fr 70px;gap:12px;align-items:center;margin:10px 0}}.bar{{height:18px;background:#237b9d;border-radius:4px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}h1{{font-size:30px}}table{{font-size:13px}}.barrow{{grid-template-columns:120px 1fr 60px}}}}</style><main><header><p>RSI • DIFFUSION POLICY • SEMANTIC VALIDATION</p><h1>{title}</h1><span class="badge">{badge}</span><p><a href="{'index_en.html' if zh else 'index_zh.html'}">{'English summary' if zh else '中文完整进度'}</a> · <a href="state.json">JSON evidence</a> · <a href="{reviewlink}">{'独立图像审核' if zh else 'Independent image review'}</a></p><p>Snapshot UTC: {source['updated_utc']}</p></header><div class="grid"><section><div class="metric">0 GPU h</div><p>New Phase6 scientific training</p></section><section><div class="metric">0 labels</div><p>Independent edited-image truth</p></section><section><div class="metric">42.20 h</div><p>Historical forecast; authorized budget0</p></section></div><section><h2>{labels[0]}</h2><table><thead><tr><th>Stage</th><th>Status</th><th>Evidence boundary</th></tr></thead><tbody>{stage_rows}</tbody></table></section><section><h2>{labels[1]}</h2><p>Qwen/Qwen3-VL-2B-Instruct (frozen) · timbrooks/instruct-pix2pix · UNet attention LoRA rank8/alpha16.</p><p>Source image + fixed benign instruction → stochastic diffusion trajectories → independently validated benign reward (currently BLOCKED) → on-policy DDPO-SF LoRA updates. Logical batch10 =10 source images,10 trajectories,100 transitions,10 accumulated microbatches,1 optimizer update.</p><p>Train trace → competing explanations → new diagnostic observations → restricted reusable workflow revision → fixed paired cost-aware Gate → accept/reject → immutable history and rollback.</p><p>L = −meanᵢ[Rᵢ meanₜ log pθ(zₜ₋₁ | zₜ, image, instruction)]. Historical diagnostic R = S +0.5A −0.2D −0.1C −U. Security ASR is excluded from optimization.</p></section><section><h2>{labels[2]}</h2><p>Saved observations: {reward.get('observations','pending')}; lower clipping: {reward.get('clip_lower_count','pending')}; independent-label agreement: unavailable.</p>{bars}<p>Component means are diagnostic proxies, not independently verified editing performance. Instruction-success measurement is missing; reward release remains blocked.</p></section><section><h2>{labels[3]}</h2><p>Recorded train investigations: {harness.get('investigated_real_train_episodes','pending')}; paired source groups: {harness.get('gate_unique_train_sources','pending')}; injected metadata controls: {harness.get('injected_metadata_controls','pending')}.</p><p>Selected integrity accuracy {selected.get('parent_accuracy','pending')} → {selected.get('candidate_accuracy','pending')}; diagnostic-operation saving {selected.get('diagnostic_operation_saving_fraction','pending')}. No natural-image semantic gain is established.</p></section>{detail}<section><h2>{labels[4]}</h2><p>Phase4:32 actual LoRA updates,3.645 A100 GPUh; no reliable RL/RSI advantage. Strict stochastic continuation remains unresolved. Phase5 synthetic integrity fixture:50%→100% on32 constructed sources; this is an engineering test.</p><p>v2: train/dev/Gate/test999/104/96/301. Dev/Gate historically exposed; no fresh confirmatory evidence. M0–M5 are prepared only. Real post-training security effect: unverified.</p></section><section><h2>{labels[5]}</h2><p>Independent seven-dimension image review and edited factual answers → reward construct validation → fresh private evaluator and matched initialization → separately authorized small pilot. All unresolved scientific gates remain enforced.</p><p>{source['sync_notice']}</p><p>Source commit: {html.escape(source['source_commit'])}</p></section><script type="application/json" id="dashboard-state">{safe_encoded}</script></main></html>'''
        if zh:
            translations={
              'New Phase6 scientific training':'Phase6 新增科学训练用量',
              'Independent edited-image truth':'独立确认的编辑后事实标签',
              'Historical forecast; authorized budget0':'历史方案估计；当前获批预算为0',
              'Source image + fixed benign instruction → stochastic diffusion trajectories → independently validated benign reward (currently BLOCKED) → on-policy DDPO-SF LoRA updates. Logical batch10 =10 source images,10 trajectories,100 transitions,10 accumulated microbatches,1 optimizer update.':'原图＋固定无害编辑指令 → 随机扩散轨迹 → 独立验证的编辑奖励（当前未放行）→ DDPO-SF 更新真实 LoRA。逻辑批次10＝10张源图、10条轨迹、100个去噪转移、10次微批累积、1次优化器更新。',
              'Train trace → competing explanations → new diagnostic observations → restricted reusable workflow revision → fixed paired cost-aware Gate → accept/reject → immutable history and rollback.':'训练轨迹 → 竞争原因假设 → 追加诊断观察 → 受限的可复用流程修订 → 固定配对成本 Gate → 接受或拒绝 → 不可变版本与回滚。',
              'Component means are diagnostic proxies, not independently verified editing performance. Instruction-success measurement is missing; reward release remains blocked.':'上图为诊断分量均值，不是独立确认的编辑性能。显式指令完成测量仍缺失，训练奖励仍未发布。',
              'No natural-image semantic gain is established.':'尚未证实自然图像语义效果提升。',
              'Independent seven-dimension image review and edited factual answers → reward construct validation → fresh private evaluator and matched initialization → separately authorized small pilot. All unresolved scientific gates remain enforced.':'先完成七维独立审核和编辑后事实答案，再验证奖励构念、建立新私有评价和匹配初始化；获得单独预算后才能执行小 pilot。未满足的科学条件持续阻断训练。',
              'Local files update only after scripts/sync_phase6.ps1 and browser reload. This copy is a timestamped snapshot.':'本地文件需运行 scripts/sync_phase6.ps1 并刷新浏览器后更新；当前页面只是带时间戳的快照。',
              'Phase4:32 actual LoRA updates,3.645 A100 GPUh; no reliable RL/RSI advantage. Strict stochastic continuation remains unresolved. Phase5 synthetic integrity fixture:50%→100% on32 constructed sources; this is an engineering test.':'Phase4 实际完成32次 LoRA 更新、耗用3.645 A100 GPU小时；未确认可靠的 RL/RSI 优势。严格随机续训可复现性仍未解决。Phase5 的50%→100%来自32个构造完整性样本，仅为工程测试。',
              'v2: train/dev/Gate/test999/104/96/301. Dev/Gate historically exposed; no fresh confirmatory evidence. M0–M5 are prepared only. Real post-training security effect: unverified.':'v2 train/dev/Gate/test＝999/104/96/301。dev/Gate曾暴露，不能作为新确认性证据。M0–M5仅完成准备。真实训练后安全效果尚未验证。'}
            marker='<script type="application/json"';visible,tail=document.split(marker,1)
            for a,b in translations.items():visible=visible.replace(a,b)
            document=visible+marker+tail
        atomic_text(base/name,document)
    return {'json':str(base/'state.json'),'zh':str(base/'index_zh.html'),'en':str(base/'index_en.html'),'snapshot_utc':source['updated_utc']}
