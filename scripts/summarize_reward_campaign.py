"""Create evidence-bound reports/tables; missing victim metrics stay NOT TESTED."""
import argparse
import csv
import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

METHODS = {"frozen":"Frozen", "fixed":"Fixed (equivalent)", "ddpo_only":"DDPO-only",
           "rule_rsi":"Rule RSI", "rl_rsi":"RL-RSI"}
NT = "NOT TESTED"


def read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.is_file() else []


def mean(rows, key):
    return sum(r[key] for r in rows) / len(rows) if rows else None


def csv_file(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k:NT if v is None else v for k,v in row.items()} for row in rows)


def markdown_table(rows, columns):
    def display(value):
        return NT if value is None else f"{value:.6f}" if isinstance(value,float) else str(value)
    return ("| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n" +
            "\n".join("| " + " | ".join(display(r.get(k)) for k in columns) + " |" for r in rows)+"\n")


def paired_source_ci(first, second, key="common_reward"):
    import numpy as np
    if [r["sample_id"] for r in first] != [r["sample_id"] for r in second]:
        raise ValueError("Source pairing changed")
    differences = np.asarray([a[key]-b[key] for a,b in zip(first,second)])
    indices = np.random.default_rng(42).integers(0,len(differences),(10000,len(differences)))
    low,high = np.quantile(differences[indices].mean(1),[.025,.975])
    return {"difference":float(differences.mean()), "CI95_low":float(low), "CI95_high":float(high),
            "resampling_unit":"paired source image, one trained seed; NOT seed uncertainty"}


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument("--campaign",required=True,type=Path)
    p.add_argument("--publish-root",action="store_true")
    p.add_argument("--root-guard",type=Path,help="Optional previous root-report SHA256 receipt; refuse independent edits")
    a=p.parse_args()
    c=a.campaign.resolve(); root=Path(__file__).resolve().parents[1]
    out=c/"reports";out.mkdir(exist_ok=True)
    debug=read(c/"diagnostics/debug_results.json",{})
    internal=read(c/"internal_pilot/run_state.json",{})
    v3_calibration=read(c/"lighting_v3_calibration.json",{})
    internal_protocol=read(c/"internal_pilot/protocol.json",{})
    accounting=read(c/"GPU_ACCOUNTING.json",{})
    padding=read(c/"padding_audit.json",{})
    frozen_audit=read(c/"lighting_vqa_audit/summary.json",{})
    raw_audit=read(c/"internal_pilot_raw_audit.json",{})
    token_audit=read(c/"shadowcast/poison_tokenization_audit.json",{})
    external_stats=read(c/"shadowcast/verified_statistics.json",{})
    viz_stats=read(c/"vizwiz/verified_statistics.json",{})
    split_overlap=read(c/"shadowcast/exact_split_overlap_audit.json",{})
    stats={}
    rows=[]
    for method,label in METHODS.items():
        observed=internal.get("assessment",{}).get(method,[])
        arm=internal.get("arms",{}).get(method,{})
        frozen_group=frozen_audit.get("groups",{}).get(method,{})
        fractions=raw_audit.get("editing_fractions",{}).get(method,{})
        row={"Method":label,"scope":"benign lighting-v3 development","seed":42,
             "status":arm.get("status",internal.get("status","NOT STARTED")),
             "optimizer_steps":arm.get("optimizer_steps",0),"optimizer_target":"IP2Pix generator LoRA","ASR":NT,"ASR_unit":"fraction","ASR_numerator":NT,"ASR_denominator":NT,
             "clean_accuracy":NT,"Delta_Clean":NT,"clean_accuracy_scope":NT,
             "VQAv2_development_accuracy":NT,"VQAv2_development_Delta_Clean":NT,
             "frozen_VQA_quality_accuracy":frozen_group.get("source_answer_accuracy"),
             "frozen_source_prediction_agreement":frozen_group.get("source_prediction_agreement"),
             "poison_sample_fraction":NT,
             "edited_assessment_fraction":fractions.get("edited_assessment_fraction"),
             "canvas_area_fraction_tau0":fractions.get("canvas_area_fraction_tau0"),
             "content_area_fraction_tau0":fractions.get("content_area_fraction_tau0"),
             "LPIPS":mean(observed,"LPIPS"),"SSIM":mean(observed,"SSIM"),
             "lighting_reward":mean(observed,"common_reward"),"reward_version":internal_protocol.get("reward_version","lighting_v3_mean_progress_20261104"),
             "content_encoded_luma_delta":mean(observed,"content_encoded_luma_delta"),
             "content_linear_luma_delta":mean(observed,"content_linear_luma_delta"),
             "content_chromaticity_MAE":mean(observed,"content_chromaticity_MAE"),
             "content_brightness_progress":mean(observed,"brightness_progress"),
             "mean_target_linear_luma_error":mean(observed,"mean_brightness_error"),
             "GPU_h":NT,"GPU_h_scope":"shared internal allocation; see GPU_ACCOUNTING, not attributed per arm",
             "training_seed_CI":"NOT ESTIMABLE (n=1)"}
        rows.append(row)
    for count in [0,50]:
        ev=read(c/f"shadowcast/eval_{count}_summary.json",{})
        baseline=read(c/"shadowcast/eval_0_summary.json",{})
        costs=[r for r in accounting.get("records",[]) if r["stage"].startswith(("shadowcast","vizwiz"))]
        gpu_h=sum(r["elapsed_seconds"] for r in costs)/3600 if costs else None
        viz=read(c/f"vizwiz/summary_{count}.json",{})
        viz_baseline=read(c/"vizwiz/summary_0.json",{})
        clean=viz.get("clean_VizWiz_full_test_accuracy")
        delta=clean-viz_baseline["clean_VizWiz_full_test_accuracy"] if clean is not None and viz_baseline else None
        clean_dev=ev.get("clean_VQAv2_development_accuracy")
        delta_dev=clean_dev-baseline["clean_VQAv2_development_accuracy"] if clean_dev is not None and baseline else None
        trained=read(c/f"shadowcast/poison_{count}_seed42_lora/trainer_state.json",{})
        row={"Method":f"Shadowcast {'clean0 control' if count==0 else 'poison50'}",
             "scope":"external original Trump-to-Biden task; full VizWiz test utility","seed":42,
             "status":"MEASURED" if ev and viz else "ASR MEASURED; UTILITY PENDING" if ev else "IN PROGRESS","optimizer_steps":trained.get("global_step",NT),
             "optimizer_target":"LLaVA victim LoRA/projector","ASR":ev.get("ASR"),"ASR_unit":"fraction","ASR_numerator":ev.get("ASR_numerator"),"ASR_denominator":ev.get("ASR_denominator"),
             "clean_accuracy":clean,"Delta_Clean":delta,
             "clean_accuracy_scope":"VizWiz full8000 public test; original LLaVA prompts; fraction",
             "VQAv2_development_accuracy":clean_dev,"VQAv2_development_Delta_Clean":delta_dev,
             "frozen_VQA_quality_accuracy":NT,"frozen_source_prediction_agreement":NT,
             "poison_sample_fraction":count/(3439+count),
             "edited_assessment_fraction":NT,"canvas_area_fraction_tau0":None,"content_area_fraction_tau0":None,
             "LPIPS":None,"SSIM":None,"lighting_reward":NT,"reward_version":"external original-task ASR","content_encoded_luma_delta":NT,
             "content_linear_luma_delta":NT,"content_chromaticity_MAE":NT,
             "content_brightness_progress":NT,"mean_target_linear_luma_error":NT,
             "GPU_h":gpu_h if count==50 else NT,
             "GPU_h_scope":"all external stages including clean control, failed allocations, poison and eval; counted once",
             "training_seed_CI":"NOT ESTIMABLE (n=1)"}
        quality=read(c/"shadowcast/poison_quality.json",{})
        if count==50:row.update(LPIPS=quality.get("LPIPS"),SSIM=quality.get("SSIM"),
                               canvas_area_fraction_tau0=quality.get("pixel_area_fraction_tau0"),
                               content_area_fraction_tau0=quality.get("pixel_area_fraction_tau0"))
        rows.append(row)
    csv_file(out/"BASELINE_COMPARISON.csv",rows)
    assessment=internal.get("assessment",{})
    if assessment.get("rl_rsi") and assessment.get("ddpo_only"):
        stats["RL_RSI_minus_DDPO_lighting_v3"] = paired_source_ci(assessment["rl_rsi"],assessment["ddpo_only"])
        stats["RL_RSI_minus_DDPO_content_luma"] = paired_source_ci(assessment["rl_rsi"],assessment["ddpo_only"],"content_encoded_luma_delta")
        for method in ["ddpo_only","rule_rsi","rl_rsi"]:
            stats[f"{method}_minus_Frozen_lighting_v3"] = paired_source_ci(assessment[method],assessment["frozen"])
        stats["comparison_scope"]="RL-minus-DDPO is the requested comparison; additional frozen comparisons are descriptive, one fixed seed, no multiplicity-adjusted efficacy claim"
    (out/"SOURCE_PAIRED_STATISTICS.json").write_text(json.dumps(stats,indent=2)+"\n")
    control=[]
    for variant in ["identity","exposure_plus3pct","exposure_minus3pct","channel_swap","unrelated_image"]:
        group=[r for r in debug.get("calibration",[]) if r["variant"]==variant]
        revised=[r for r in v3_calibration.get("controls",[]) if r["variant"]==variant]
        control.append({"Control":variant,"n":len(group),"legacy_reward":mean(group,"legacy_common_reward"),
                        "rejected_v2_reward":mean(group,"common_reward"),"lighting_v3_reward":mean(revised,"common_reward"),"LPIPS":mean(group,"LPIPS")})
    csv_file(out/"REWARD_CONTROLS.csv",control)
    quality=[]
    for setting in ["legacy10_g75_eta1","steps50_g75_eta1","steps50_g5_eta1","steps50_g5_eta0"]:
        group=[r for r in debug.get("quality",[]) if r["setting"]==setting]
        quality.append({"setting":setting,"n":len(group), **{k:mean(group,k) for k in
            ["LPIPS","SSIM","content_encoded_luma_delta","content_chromaticity_MAE"]}})
    csv_file(out/"EDITING_PARAMETER_AUDIT.csv",quality)
    stamp=datetime.now(timezone.utc).isoformat()
    bf=debug.get("BF16_ordinary_repeat",{})
    frozen_rows=[{"Group":group,**values} for group,values in frozen_audit.get("groups",{}).items()]
    frozen_text=(markdown_table(frozen_rows,["Group","n","source_answer_accuracy","source_prediction_agreement"])
                 if frozen_rows else "Frozen Qwen质量测量尚未完成。\n")
    stats_text=(json.dumps(stats,indent=2,ensure_ascii=False) if stats else "完成全部固定checkpoint测量后计算；当前不发布训练中差值。")
    conclusion="最终开发评估尚未完成；不提前判定方法优劣。\n\n"
    comparison=stats.get("RL_RSI_minus_DDPO_lighting_v3")
    if comparison:
        rl=next(row for row in rows if row["Method"]=="RL-RSI")
        originals=assessment["frozen"]
        identity_error=sum(abs(r["target_mean_linear_luma"]-r["source_mean_linear_luma"]) for r in originals)/len(originals)
        improved=sum(r["brightness_progress"]>0 for r in assessment["rl_rsi"])
        worsened=sum(r["brightness_progress"]<0 for r in assessment["rl_rsi"])
        below_identity=sum(r["common_reward"]<0 for r in assessment["rl_rsi"])
        interval_contains_zero=comparison["CI95_low"]<=0<=comparison["CI95_high"]
        inference=("本轮未证实RL-RSI优于DDPO" if interval_contains_zero else
                   "该固定模型的开发源图比较偏向RL-RSI；不能外推为ASR或跨种子优势" if comparison["difference"]>0 else
                   "该固定模型的开发源图比较偏向DDPO；不能外推为跨种子结论")
        conclusion=(f"内部最终16图：RL-RSI−DDPO共同奖励差{comparison['difference']:+.6f}，配对源图95%区间"
                    f"[{comparison['CI95_low']:+.6f}, {comparison['CI95_high']:+.6f}]；{inference}。\n\n"
                    f"RL-RSI平均亮度目标进展{rl['content_brightness_progress']:.6f}，色度MAE{rl['content_chromaticity_MAE']:.6f}，"
                    f"平均目标线性亮度误差{rl['mean_target_linear_luma_error']:.6f}；像素不编辑参考误差{identity_error:.6f}，其v3 Reward为0。"
                    f"相对各自原图，目标误差改善{improved}/16张、恶化{worsened}/16张，{below_identity}/16张共同Reward低于像素不编辑参考；这些是逐图描述计数，不是后门ASR。"
                    "当前不支持只改目标属性、其他语义完整保持的主张。BF16仍未修复，采用通过实际数值门的FP32替代路径。\n\n")
    report=(f"# Reward / Editing / Numerical Debug Report\n\n更新时间 UTC：{stamp}\n\n"
        f"本轮只跑 seed=42，累计授权20GPU小时，历史关闭额度没有重开。内部状态：{internal.get('status','IN PROGRESS')}。\n\n"
        "## 本轮实测结论\n\n"+conclusion+
        "## Reward 的实际问题\n\n"
        "旧验证集原图CLIP均值0.136749707628，LPIPS自比较0，因此2×CLIP−LPIPS=0.273499415256，算术正确。"
        "用户列出的−0.014295来自另一组development assessment，而非这16张validation；"
        "相同validation上的旧RL为−0.0154535，DDPO为−0.0271254，Frozen为−0.0959000。"
        "训练提高了旧代理Reward，却仍使内容亮度下降，显示代理目标和任务方向不一致。"
        "CLIP cosine不是成功概率或指令完成率；原图不要求得分为零。源图已低于提问中的0.2阈值。"
        "真实CLIP输出embedding已经归一化；本轮四张训练图上cosine与归一化点积最大差为0。"
        "LPIPS使用校准AlexNet和[-1,1]输入，CLIP使用RGB PIL与固定processor，未发现这些实现错误。\n\n"
        "实际问题是目标错配：旧奖励在校准控制上偏好不编辑，轻微提亮反而降低得分。"
        "加大LPIPS惩罚会更偏好不编辑。仅减去源图CLIP能让原图得分归零，"
        "但不会改变同一源图的候选排序，也不能单凭归零宣称学习方向修好。\n\n"
        "对实际PilotTrainer更新代码的静态复查未发现策略梯度符号反转："
        "loss=−R×mean(logp)，使用原始终局奖励和零基线，未用会在B=1时消除梯度的批内归一化。"
        "这说明损失符号与最大化期望奖励一致，不等于有限样本Adam更新必然提高任务质量；"
        "实际效果仍以固定checkpoint的开发评估为准。\n\n"
        "初版lighting_v2通过了解析控制，但在44个实际开发训练输出中43个亮度项硬截断为−1，"
        "标准差仅1.192e-5。该开发实验被拒绝并取消，全部日志、图片和checkpoint保留在"
        "`internal_pilot_v2_saturation_probe`；它不是完成的对比实验。2840秒取消作业照常计费。\n\n"
        "初始停止判断时为42/43张饱和，取消生效前又保存了一张，最终为43/44；"
        "冻结源码和原始reward receipt中的42/43保留为当时的记录，最终统计使用全部44张，不覆盖历史证据。\n\n"
        "采用显式新版本lighting_v3_mean_progress_20261104：\n\n"
        "`R = w_sem*(CLIP_edit-CLIP_source + mean_brightness_progress) - color_cost - w_lpips*LPIPS`。\n\n"
        "目标是本次良性任务的+3%线性光曝光对应的内容区域平均亮度。"
        "令x=(原图到目标的平均亮度误差−编辑图到目标的平均亮度误差)/max(原图误差,0.001)，"
        "progress=x/(1+abs(x))，连续保留方向和偏差排序；色度误差/0.02仍截断至5。"
        "空间保持仍由独立LPIPS和色度项约束；平均属性本身不能证明对象完整保持。"
        "亮度/色度只看原图内容区域；CLIP/LPIPS/SSIM仍覆盖完整画布。"
        "这增加了明确的任务项，属于科学目标修订，不能冒充通用后门奖励修复；不能跨版本比较原始Reward大小。"
        f"同44个已保存开发输出的v3重评分亮度项标准差为{v3_calibration.get('v3_progress_SD',NT)}，"
        "不再硬饱和。该重评分未调用新模型，也未冒充v3训练结果。新pilot从相同初始状态重新训练。\n\n"+
        markdown_table(control,list(control[0]))+
        "\n校准图为预先选定的前4张训练图，解析曝光仅是正/负控制，未替代InstructPix2Pix生成器。"
        "校准通过不代表生成器已经学会提亮、保留全部语义或提高ASR。\n\n"
        "## 编辑质量\n\n"
        "旧pilot实际是DDIM10步，而非2步。原图被白色填充到512方形。"
        "旧RL验证输出全画布编码亮度差−0.1093863，内容区域−0.0406355；"
        "白边放大负变化，内容仍变暗。不能把边框变化当作目标属性变化。\n\n"+
        markdown_table(quality,list(quality[0]))+
        "\n这四张验证图由固定顺序选定。50步/guidance5/eta1改善LPIPS和色度，仍未实现平均提亮。"
        "eta0明显改变部分场景；它只用于推理诊断，不能直接用于要求正Gaussian方差的现有DDPO训练。"
        "新pilot采用50步、guidance5、imageguidance1.5、eta1、rank16/alpha32、LR5e-6、clip0.5。"
        "这些参数共同改变，不能据此单独归因rank或学习率。Frozen已出现色偏，故rank不是其原因。\n\n"
        "## BF16 与 FP32\n\n"
        "在本轮v2目标诊断下，FP32 LoRA/linear计算/Adam已实际应用，且梯度和Adam状态有限；但BF16普通同一步重复仍失败："
        f"{bf.get('different_elements',NT)}个元素不同，最大绝对差{bf.get('max_absolute_error',NT)}，"
        f"exact={bf.get('exact',NT)}，tolerance={bf.get('tolerance',NT)}。Torch/CUDA RNG均一致。"
        "差异已出现在raw gradients，不能归因于随机种子或简单地用梯度裁剪解释。\n\n"
        "进一步源码复查发现，现有DiffusionPolicy构造器已经关闭BF16低精度归约；"
        "进程在构造前的默认True不代表训练时实际设置。因此这项候选在提交前撤销，新增GPU消耗0，不能当作新修复。\n\n"
        f"内部新pilot全FP32数值门：{internal.get('numerical',{}).get('status',NT)}。"
        "它上采样现有FP16-variant权重到FP32，不恢复权重已经丢失的精度。"
        "普通第一/第二更新和磁盘重载后的下一更新分别比较1538个张量；"
        "通过这一局部门不等于BF16路径修复或普遍数值认证。\n\n"
        "## 内部pilot与外部复现\n\n"+
        markdown_table(rows,["Method","status","optimizer_steps","lighting_reward","LPIPS","SSIM","ASR","clean_accuracy"])+
        "\n内部为32/16/16良性开发实验，使用v3共同评价器；最终16图来自历史暴露的训练池。"
        "Frozen=Fixed为用户指定的冻结生成器、相同prompt等价性对照，仍执行图像编辑采样；像素不编辑对照另列，v3 Reward为0。"
        "内部没有训练poisoned victim，ASR/ΔClean不填代理值。三次controller更新不是独立种子，也不是正式accepted revision。\n\n"
        "最终内容区域属性如下：亮度差为编码域平均变化，目标误差为线性光亮度的绝对误差。"
        "平均亮度上升本身不能证明每张图接近1.03倍曝光目标，也不能证明颜色和对象保持。\n\n"+
        markdown_table(rows[:5],["Method","content_encoded_luma_delta","content_chromaticity_MAE",
                                "content_brightness_progress","mean_target_linear_luma_error"])+"\n"
        "额外冻结Qwen3-VL-2B检查使用全部16张development assessment的原问题、原10人答案和greedy回答；"
        "原图及四个物理编辑条件共80次查询，Fixed共享Frozen观测。这是预先登记的事后质量审查，未用于controller或模型选择。"
        "单个原问题不能覆盖整图所有颜色、对象、数量或证明完整语义保持；冻结模型也可能答错。\n\n"+
        frozen_text+"\n"
        "Shadowcast采用官方代码固定提交b13f877a，完整Trump-to-Biden标签任务，3439干净+50投毒。"
        "PGD采用已发布代码的4000步，与论文Sec4.1所述2000步不同；该差异不为追求结果而调整。"
        "模型LLaVA1.5-7B、CLIP-L/14@336均固定revision。ASR使用全部200官方测试图，"
        "成功要求整词Biden存在且Trump不存在；错误不得静默跳过。干净对照也评估相同200图，记录自然目标响应率。"
        f"277张源图及277张guidance构造图与200张ASR图的文件/解码RGB精确重复审计：overlap_found={split_overlap.get('exact_overlap_found',NT)}；"
        "这不覆盖近重复或预训练污染。"
        "主要清洁效用为VizWiz公开完整8000题，使用原始LLaVA问题和prompt、2026年4月官方公开的10人参考答案；"
        "这是完整test自评分；原论文Table3未明确EvalAI分区，是否相同分区尚未核实。参考答案不进入模型，输出全部锁定后才评分。"
        "补充效用为104条已暴露VQAv2 development，两个数据集分别报告，GQA未测。"
        f"两模型均配置one epoch，实际optimizer更新数为clean0={external_stats.get('actual_training',{}).get('0',{}).get('global_step',NT)}、"
        f"poison50={external_stats.get('actual_training',{}).get('50',{}).get('global_step',NT)}；描述性差值不是等更新数的独立因果估计。"
        f"对选定投毒文本的官方预处理静态重放：{token_audit.get('nonzero_supervision_count',NT)}/50有非零监督token，"
        f"整条忽略数{token_audit.get('zero_supervision_count',NT)}。这仅验证标签资格，未证明采样器实际消费了每个样本。"
        "软件Torch2.1.2、单卡eager和4×32微批差异均披露，不宣称逐位复现或与内部良性实验匹配。\n\n"
        "BadVLM按用户最新决定跳过。提供的BadVLA仓库是机器人项目。BadSem/CBV仅列已核实文献边界，"
        "未用自行编写的重着色函数冒充论文复现。\n\n"
        "## 统计与资源\n\n"
        f"实际/含在途预留GPU小时：{accounting.get('actual_GPU_seconds',0)/3600:.4f} / "
        f"{accounting.get('committed_GPU_seconds',0)/3600:.4f}，上限20。"
        "失败、取消、烟测及干净模型均计费；详见GPU_ACCOUNTING.json。单训练种子不能估计seed均值±std或seed级95%CI。"
        "配对源图bootstrap区间只描述这个固定训练种子的源图不确定性，不能证明方法泛化显著性。\n\n"
        "ASR的经验源图bootstrap在全成功/全失败时可能退化；这种退化不表示总体错误率不确定性为零。\n\n"
        "```json\n"+stats_text+"\n```\n\n"
        "外部原任务ASR与清洁效用的固定模型源图区间分别保存在"
        "`shadowcast/verified_statistics.json`和`vizwiz/verified_statistics.json`。"
        f"Shadowcast poison50 ASR：{external_stats.get('poison_50',{}).get('ASR',{})}。\n\n"
        f"VizWiz poison50−clean0 配对差值：{viz_stats.get('Delta_Clean_poison50_minus_clean0','完整评估尚未完成')}。\n\n"
        "所有模型与大文件均在集群，H盘仅镜像精选结果；C盘无本轮模型下载。"
        "保留源码、参数/优化器见证、checkpoint、逐图结果及失败记录。运行结束清理专属tmp目录。\n")
    (out/"DEBUG_REPORT.md").write_text(report,encoding="utf-8")
    completed=internal.get("status")=="COMPLETED"
    ext_done=all((c/f"shadowcast/eval_{count}_summary.json").is_file() for count in [0,50])
    viz_done=all((c/f"vizwiz/summary_{count}.json").is_file() for count in [0,50])
    next_steps=("# Next Steps / Actual Status\n\n"
        "- Reward算术、归一化、输入范围：审计完成；旧公式未发现所述cosine计算Bug。\n"
        "- 任务目标修订：v2解析控制通过但真实输出饱和，已拒绝/保留；v3连续平均属性项通过控制和真实输出区分检查，identity=0；不能泛化为ASR修复。\n"
        f"- Seed42新pilot：{'完成' if completed else internal.get('status','待运行')}。\n"
        "- BF16：未修复；全FP32路径通过实际重复/恢复局部门，采用此路径完成pilot。\n"
        f"- Shadowcast干净/投毒模型及真实ASR：{'完成' if ext_done else '执行中；只报告已经完成的测量'}。\n"
        f"- Shadowcast完整VizWiz8000题clean0/poison50效用：{'完成' if viz_done else '固定协议已注册、数据就绪，等待固定模型评估'}。GQA未测。\n"
        "- BadVLM：按用户决定跳过；无正确作者代码/全文时不把BadVLA或猜测实现当作BadVLM。\n"
        "- 多seed：本轮只seed42；不生成不存在的跨seed方差或显著性。\n"
        "- Tables5/6/7：提供真实测量的插入稿，未测ASR/调度器/PBT/匹配外部比较保留NOT TESTED；不为80%覆盖率造数。\n"
        "- 后续科学工作：先验证目标属性和独立语义保持，再预注册内部victim训练、触发定义与ASR分母；"
        "新增独立训练seed及真正matched baseline之后才评估方法优势。\n")
    (out/"NEXT_STEPS.md").write_text(next_steps,encoding="utf-8")
    # Paper insertion rows separate real benign metrics from unmeasured attack endpoints.
    table5=[{"Method":row["Method"],"ASR":row["ASR"],"Delta_Clean":row["Delta_Clean"],
             "Sample_frac":row["poison_sample_fraction"],"Area_frac_tau0_canvas":row["canvas_area_fraction_tau0"],"LPIPS":row["LPIPS"],
             "SSIM":row["SSIM"],"GPU_h":row["GPU_h"]} for row in rows[:5]]
    csv_file(out/"TABLE5_MAIN_INSERTION.csv",table5)
    ex=read(c/"shadowcast/eval_50_summary.json",{})
    ext_viz=read(c/"vizwiz/summary_50.json",{})
    ext_viz0=read(c/"vizwiz/summary_0.json",{})
    table6=[{"Method":"Shadowcast","Literature_setting":"LLaVA1.5 / CC-SBU; multiple counts/tasks",
             "Literature_ASR":">95% at50 poisons, Trump-to-Biden (Sec4.2)", "Our_reproduction_ASR":f"{100*ex['ASR']:.2f}%" if ex else None,
             "Our_VizWiz_full8000_clean_accuracy":ext_viz.get("clean_VizWiz_full_test_accuracy"),
             "Our_VizWiz_Delta_Clean":ext_viz["clean_VizWiz_full_test_accuracy"]-ext_viz0["clean_VizWiz_full_test_accuracy"] if ext_viz and ext_viz0 else None,
             "Ours_method_matched":NT,"Status":"original-task measured" if ex else "IN PROGRESS",
             "Source":"https://arxiv.org/html/2402.06659"},
            {"Method":"BadVLM","Literature_setting":"captioning/VQA,0.1% poison; institutional abstract",
             "Literature_ASR":">99% (abstract claim)","Our_reproduction_ASR":NT,
             "Our_VizWiz_full8000_clean_accuracy":NT,"Our_VizWiz_Delta_Clean":NT,"Ours_method_matched":NT,
             "Status":"SKIPPED by user","Source":"https://espace.inrs.ca/id/eprint/17209/"},
            {"Method":"BadSem","Literature_setting":"SIMBad,4VLMs; aggregate abstract",
             "Literature_ASR":">98% average (abstract claim)","Our_reproduction_ASR":NT,
             "Our_VizWiz_full8000_clean_accuracy":NT,"Our_VizWiz_Delta_Clean":NT,"Ours_method_matched":NT,
             "Status":"No reproduction","Source":"https://arxiv.org/abs/2506.07214"},
            {"Method":"CBV","Literature_setting":"COCO/VQAv2,4VLMs; aggregate abstract",
             "Literature_ASR":">80% (abstract claim)","Our_reproduction_ASR":NT,
             "Our_VizWiz_full8000_clean_accuracy":NT,"Our_VizWiz_Delta_Clean":NT,"Ours_method_matched":NT,
             "Status":"No reproduction","Source":"https://proceedings.mlr.press/v306/guo26q.html"}]
    csv_file(out/"TABLE6_EXTERNAL_INSERTION.csv",table6)
    table7=[]
    recorded_actions=json_lines(c/"internal_pilot/controller_actions.jsonl")
    recorded_rules=json_lines(c/"internal_pilot/rule_actions.jsonl")
    for label,arm in [("Fixed_config (DDPO)","ddpo_only"),("Ordinary_scheduler",None),("Rule_RSI","rule_rsi"),
                      ("PBT_config_search",None),("RL_controller","rl_rsi")]:
        info=internal.get("arms",{}).get(arm,{}) if arm else {}
        observed=assessment.get(arm,[]) if arm else []
        table7.append({"Configuration":label,"ASR":NT,"Accepted_revisions":"N/A: no formal acceptance gate" if arm else NT,
                       "Recorded_config_actions":len(recorded_actions) if arm=="rl_rsi" else len(recorded_rules) if arm=="rule_rsi" else 0 if arm else NT,
                       "Controller_updates":info.get("controller_updates",0) if arm else NT,
                       "Generator_updates":info.get("optimizer_steps",0) if arm else NT,
                       "Lighting_reward_v3":mean(observed,"common_reward"),"Status":info.get("status",NT)})
    csv_file(out/"TABLE7_ABLATION_INSERTION.csv",table7)
    (out/"TABLES5_6_7_README.md").write_text("# Tables5/6/7 evidence boundary\n\n"+
        "Table5内部LPIPS/SSIM来自良性开发图像，ASR/poison fraction未测。Table6区分文献、外部原任务复现和Ours匹配结果；"
        "三者不能混写。Table7 Fixed_config对应配置固定的DDPO，不是Frozen编辑器；未运行scheduler/PBT不填0。"
        "Area_frac按整个512画布任意非零整数RGB变化计像素；BASELINE_COMPARISON另列排除白边的content版本。"
        "edited_assessment_fraction只描述16张开发评估图中被改变的比例，不等于victim训练poison_sample_fraction。"
        "正式accepted revisions机制不存在，不能拿controller action数代替。报告不追求用NOT TESTED伪装80%实测覆盖率。\n\n"
        "内部每行n=16历史暴露开发源图、seed=42，每个训练条件100次更新；LPIPS/SSIM/面积均为无量纲。"
        "Table6自复现ASR以百分数列出，分母200；完整VizWiz为8000题、7928个图像hash，准确率及DeltaClean以fraction列出，"
        "−0.0034375等于−0.34375个百分点。clean0/poison50实际26/27次更新，PGD代码4000步与论文2000步不同，"
        "历史EvalAI分区是否匹配尚未核实。区间是固定模型源图区间，不是训练种子区间。\n")
    table_notes={
        5:"Internal n=16 exposed development sources, seed 42; 100 updates per trained arm. Frozen/Fixed share one control. "
          "LPIPS, SSIM and area are unitless. Area uses the full 512x512 canvas, any nonzero integer RGB change (tau=0). "
          "Sample fraction means victim poison fraction, not the edited-assessment fraction. Internal GPU time is shared, not attributed per arm. ",
        6:"Own ASR is a percentage over 200 original-task images; accuracy and DeltaClean are fractions on all 8000 VizWiz questions "
          "(7928 source-image hashes). The measured DeltaClean -0.0034375 equals -0.34375 percentage points. "
          "Clean0/poison50 completed 26/27 victim updates, so the lift is descriptive, not equal-update causal evidence. "
          "Released-code PGD 4000 differs from paper 2000; single-A100 eager 4x32 packing differs from the released training setup. "
          "Exact historical EvalAI partition match is unverified. Literature values, this original-task reference, and matched Ours are separate. ",
        7:"Internal n=16 exposed development sources, seed 42. The common lighting-v3 reward has an objective-specific scale. "
          "Fixed-config means DDPO with fixed configuration, 100 generator updates; it is distinct from the frozen-editor Fixed equivalence row. "
          "Actions and controller updates are observed mechanism counts. N/A means no formal acceptance gate; scheduler/PBT and all internal ASRs remain NT. "}
    def tex(value):
        if value is None or value==NT:return "NT"
        if isinstance(value,float):return f"{value:.4f}"
        return str(value).replace("\\","\\textbackslash{}").replace("_",r"\_").replace("%",r"\%").replace("&",r"\&").replace(">","$>$").replace("<","$<$")
    fragments=["% Evidence-bound insertion drafts. Requires booktabs and graphicx.\n"
               "% Do not force table counters; integrate at the intended Table5/6/7 positions.\n"]
    for number,table,columns,caption,label in [
        (5,table5,list(table5[0]),"Available benign development quality measurements (seed42). Post-training attack endpoints remain unmeasured; these are not main backdoor results.","campaign_quality"),
        (6,table6,["Method","Literature_ASR","Our_reproduction_ASR","Our_VizWiz_full8000_clean_accuracy","Our_VizWiz_Delta_Clean","Ours_method_matched","Status"],
         "Literature claims, our external original-task reproduction, and matched Ours results are separate. Shadowcast's threshold is from Section4.2; other percentages are abstract-level claims. Settings differ; this is not a leaderboard.","campaign_external"),
        (7,table7,list(table7[0]),"Controller mechanism records for the benign pilot. Actions or controller updates are not formally accepted revisions. Scheduler and PBT were not run.","campaign_ablation")]:
        fragments.append(f"% Intended Table{number}\n\\begin{{table*}}[t]\n\\centering\n\\scriptsize\n"
            f"\\caption{{{caption}}}\n\\label{{tab:{label}}}\n\\resizebox{{\\textwidth}}{{!}}{{%\n"
            "\\begin{tabular}{"+"l"*len(columns)+"}\n\\toprule\n"+
            " & ".join(tex(k) for k in columns)+" \\\\\n\\midrule\n"+
            "\n".join(" & ".join(tex(row.get(k)) for k in columns)+" \\\\" for row in table)+
            "\n\\bottomrule\n\\end{tabular}}\n\\par\\smallskip\\raggedright\\scriptsize "
            "NT: not tested. One seed cannot estimate between-seed uncertainty. "
            +table_notes[number]+
            "No internal ASR, matched external superiority, or full semantic-preservation claim is made.\n\\end{table*}\n")
    (out/"TABLES5_6_7_INSERTION.tex").write_text("\n".join(fragments),encoding="utf-8")
    if a.publish_root:
        if a.root_guard:
            guard=json.loads(a.root_guard.read_text(encoding="utf-8-sig"))
            expected={r["name"]:r["SHA256"].lower() for r in guard["reports"]}
            if set(expected)!={"DEBUG_REPORT.md","BASELINE_COMPARISON.csv","NEXT_STEPS.md"}:
                raise ValueError("Incomplete root publication guard")
            for name,digest in expected.items():
                target=root/"reports"/name
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                    raise RuntimeError("Independent root report edit; no publication: "+name)
        for name in ["DEBUG_REPORT.md","BASELINE_COMPARISON.csv","NEXT_STEPS.md"]:
            target=root/"reports"/name
            if target.exists():
                backup=c/"prior_root_reports"/name;backup.parent.mkdir(exist_ok=True)
                if not backup.exists():shutil.copy2(target,backup)
            shutil.copy2(out/name,target)
    print(json.dumps({"output":str(out),"internal_complete":completed,"external_ASR_complete":ext_done,
                      "external_VizWiz_complete":viz_done,"source_statistics":stats},indent=2))


if __name__=="__main__":main()
