// The supplied complete UI remains the main renderer. This layer binds current evidence.
const suppliedRenderPage=renderPage;
function renderCurrentEvidence(){
 const en=document.documentElement.lang==='en',d=DATA,f=d.followup||{},r=f.results||{},a=r.reward_separation||{},h=r.new_harness_candidates||{},v=d.review_analysis||{},m=d.current.monitor||{};
 const e=escapeHTML;
 const content=(zh,english)=>en?english:zh;
 let panel=document.getElementById('finalized-evidence');
 if(!panel){panel=document.createElement('section');panel.id='finalized-evidence';panel.className='en-section';const anchor=document.querySelector('#phase6-gates')||document.querySelector('#setup');anchor.parentNode.insertBefore(panel,anchor);}
 const clipped=a.n?100*a.lower_clipped/a.n:0;
 panel.dataset.snapshot=d.updated_utc;
 panel.innerHTML=`<div class="section-head"><div><h2>${content('Phase 6 后续 · 已完成实验与当前决策','Phase 6 follow-up · finalized evidence and current decision')}</h2><p>${e(d.current.phase)} · ${e(d.updated_utc)} · Engineering: ${e(d.current.engineering_gate)}</p></div><span class="tag bad">${e(d.current.science_gate)}</span></div>
 <div class="grid cols-2-even"><article class="card"><h3>${content('本次新增的结果','New finalized results')}</h3><p class="mono" id="followup-experiment-id">${e(f.experiment_id||'No finalized follow-up')}</p>
 <div class="stat-row"><span>${content('核对图片对 / 图片文件','Verified pairs / image files')}</span><strong>${r.packet_audit?.pairs??'UNKNOWN'} / ${r.packet_audit?.image_files_verified??'UNKNOWN'}</strong></div>
 <div class="stat-row"><span>${content('独立来源核实后的共识图片对','Independently reviewed consensus pairs')}</span><strong id="current-independent-count">${v.independently_reviewed_consensus_pairs||0}</strong></div>
 <div class="stat-row"><span>${content('新修订：接受 / 拒绝','New revisions: accepted / rejected')}</span><strong>${h.newly_accepted_revisions??'UNKNOWN'} / ${h.decisions?.filter(x=>!x.accepted).length??'UNKNOWN'}</strong></div>
 <p>${content('仅为历史图片的结构完整性任务；未证明图像编辑语义改善。','Historical image-file integrity task only; semantic editing improvement is not established.')}</p>
 <div class="stat-row"><span>${content('新 GPU / 模型调用','New GPU / model calls')}</span><strong>${r.resources?.gpu_seconds??0}s / ${r.resources?.model_calls??0}</strong></div>
 <a class="btn" href="../reviews/workflow/index.html">${content('打开 24 对独立审查工作台','Open independent 24-pair review workflow')} ↗</a></article>
 <article class="card"><h3>${content('历史奖励审计 · 全部 500 条','Historical reward audit · all 500 observations')}</h3>
 <div class="stat-row"><span>${content('触及下限','Lower-clipped')}</span><strong>${a.lower_clipped??'UNKNOWN'} / ${a.n??'UNKNOWN'} (${clipped.toFixed(1)}%)</strong></div>
 <div class="stat-track"><div class="stat-fill" style="width:${Math.max(0,Math.min(100,clipped))}%;background:#c14a53"></div></div>
 <p>${content('原始测量与旧标量已分离。C、U 为常数；S=0.5A。指令完成度与校准不确定性缺失。','Raw measurements separated from legacy scalar. C and U constant; S=0.5A. Instruction success and calibrated uncertainty absent.')}</p>
 <p>${content('未更改系数，未发布新训练奖励。安全指标与良性编辑反馈保持分开。','Coefficients unchanged; no new training reward released. Security metrics remain separate from benign editing feedback.')}</p>
 <p class="mono">CPU ${r.resources?.cpu_seconds?.toFixed(3)??'UNKNOWN'}s · wall ${r.resources?.wall_seconds?.toFixed(3)??'UNKNOWN'}s</p></article></div>
 <article class="card" style="margin-top:14px"><h3>${content('监控与阻塞条件','Monitoring and blockers')}</h3><p id="current-monitor">${content('上次成功','Last success')}: ${e(m.last_success_utc||'UNKNOWN')} · ${content('下次作业','Next job')}: ${e(m.next_job_id||'UNKNOWN')} · ${e(m.reported_job_state)} · ${e(m.next_due_utc||'UNKNOWN')}</p><p>${content('调度器实际观测','Actual scheduler observation')}: ${e(m.scheduler_observed_at||'UNKNOWN')} · ${content('科学 GPU 任务','Scientific GPU jobs')}: ${d.current.active_scientific_jobs?.length??0}</p><p>${e(d.blockers.join(' · '))}</p><p>${content('下一项：独立人工审查与来源核实；无需 GPU。奖励、私有 Gate、干预前可比性和显式 GPU 预算仍阻塞试验。','Next: independent human review and provenance verification; no GPU required. Reward, private Gate, pre-intervention comparability and explicit GPU budget still block the pilot.')}</p></article>`;
 const now=document.getElementById('now-list');if(now)now.innerHTML=[['当前阶段',d.current.phase],['本次新增',d.current.activity],['科学决策',d.current.science_gate],['下一任务',d.current.next_gate]].map(([k,x],i)=>`<div class="overview-row"><div class="num">${i+1}</div><div><strong>${e(k)}</strong><p>${e(x)}</p></div></div>`).join('');
 const jobs=document.getElementById('jobs-current');if(jobs)jobs.innerHTML=`<p>${e(m.scheduler_observed_at||'UNKNOWN')}</p>`+(d.jobs||[]).filter(j=>j.scope==='this_project').map(j=>`<div class="stat-row"><span>${e(j.name)}</span><strong>${e(j.job_id)} · ${e(j.status)}</strong></div>`).join('')+`<p>${e(m.last_success_utc)} → ${e(m.next_due_utc)}</p>`;
 let latest=document.getElementById('finalized-metrics');if(!latest){latest=document.createElement('article');latest.id='finalized-metrics';latest.className='card';latest.style.marginTop='14px';panel.append(latest);}latest.innerHTML=`<h3>${content('最新已完成实验 · 原始指标摘要','Latest finalized experiments · recorded metric summary')}</h3>`+(d.live.recent_experiments||[]).filter(x=>!['RUNNING','CREATED','SUBMITTED'].includes(x.status)).slice(0,4).map(x=>`<details class="issue"><summary><strong>${e(x.id)}</strong> ${e(x.status)} · ${e(x.ended_at)}</summary><pre style="white-space:pre-wrap">${e(JSON.stringify(x.metric_summary,null,2))}</pre><a href="${e(x.metrics_path||'#')}">${content('完整原始指标','Complete recorded metrics')} ↗</a></details>`).join('');
 // The English page also exposes the synchronized issue/fix record.
 if(en){let issues=document.getElementById('current-issues');if(!issues){issues=document.createElement('section');issues.id='current-issues';issues.className='en-section';panel.after(issues);}issues.innerHTML='<h2>Current issues and fixes</h2><div class="issue-list">'+d.issues.filter(i=>i.id.startsWith('FOLLOWUP')).map(i=>`<article class="card"><strong>${e(i.title.split(' / ').pop())}</strong><p>${e(i.status)}</p><p>${e(i.action)}</p></article>`).join('')+'</div>';}
 const timestamp=document.getElementById('as-of');if(timestamp)timestamp.textContent=d.updated_utc;
 const refresh=document.getElementById('sync-status');if(refresh)refresh.textContent=location.protocol==='file:'?content('本地副本 · 每 25 秒重载；集群变化需同步','Local copy · reload every 25s; cluster changes require sync'):content('每 25 秒读取统一 JSON','Unified JSON polled every 25s');
}
renderPage=function(){suppliedRenderPage();renderCurrentEvidence();};
if(location.protocol==='file:'){
 // Local-file fetch is restricted by browsers. Reread the generated HTML snapshot.
 setInterval(()=>location.reload(),25000);
 const oldInit=initUI;initUI=function(){oldInit();const b=document.getElementById('refresh-btn');if(b)b.onclick=()=>location.reload();};
}
