"""Answer the ten requested questions from saved evidence, without model execution."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.mini_pilot.data import write_json
from src.mini_pilot.reporting import METHODS, generate_reports, paired_bootstrap


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output = (ROOT / args.output_dir).resolve()
    if not output.is_relative_to(ROOT / "outputs"):
        raise ValueError("Expected a mini-pilot output under this project")
    protocol = json.loads((output / "protocol.json").read_text())
    state = json.loads((output / "run_state.json").read_text())
    review_path = output / "qualitative_review.json"
    if review_path.exists():
        state["qualitative_review"] = json.loads(review_path.read_text())
    elif (output / "initial_visual_review.json").exists():
        state["qualitative_review"] = json.loads((output / "initial_visual_review.json").read_text())
    rows = generate_reports(output, protocol, state)
    history = state.get("failure_history", [])
    if history:
        fast = output / "reports/FAST_VALIDATION_STATUS.md"
        fast.write_text(fast.read_text(encoding="utf-8") +
                        "\n## Preserved execution history\n\n" +
                        "\n".join(f"- {item}" for item in history) +
                        "\n\nThe time-cap continuation and CPU-only plotting repair are recorded separately. "
                        "Completed measurements were preserved; see reporting_repair/receipt.json.\n", encoding="utf-8")
    log = output / "training.jsonl"
    updates = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    actual = {method: sum(row.get("scope") == "arm_training" and row.get("arm") == method for row in updates)
              for method in protocol["methods"]}
    controller_path = output / "controller_updates.jsonl"
    controller = [json.loads(line) for line in controller_path.read_text().splitlines()] if controller_path.exists() else []
    actions_path = output / "controller_actions.jsonl"
    actions = [json.loads(line) for line in actions_path.read_text().splitlines()] if actions_path.exists() else []
    complete = state["status"] == "COMPLETED"
    measured = [row for row in rows if row["common_reward"] is not None]
    best = max(measured, key=lambda row: row["common_reward"]) if measured else None
    best_names = ", ".join(row["Method"] for row in measured if abs(row["common_reward"] - best["common_reward"]) <= 1e-12) if best else "NOT TESTED"
    numerical = state.get("numerical", {})
    lines = ["# 小数据无害编辑 Pilot：中文进度与结果", "",
             f"运行 ID：`{output.name}`。实际状态：**{state['status']}**。", "",
             "本次使用 VQAv2 历史开发数据池中的 64 张图构建 32/16/16 划分。"
             "16 张 mini-test 仅是本次开发评估，不是未接触的最终测试集。历史封存测试池保持未打开。", "",
             "1. **真正完成的实验。** " +
             ("三组真实生成器 LoRA 训练均完成 100 次更新、3 轮验证；冻结编辑器完成基线评估。"
              if complete else "以下只报告实际日志记录的工作；尚未完成的比较不能称为完整实验。") +
             f"普通更新/恢复数值检查完成 {numerical.get('actual_optimizer_updates', 0)} 次实际更新。"
             f"各组实际训练更新：{json.dumps(actual, ensure_ascii=False)}。"
             "Frozen 与 Fixed 按用户确认共享同一 Prompt、权重、采样和观测，是等价性对照。", "",
             "2. **最新 Baseline 核心数据。** 共同指标为 `2 × CLIP cosine − LPIPS`；所有组按相同固定权重评估。", "",
             "| 方法 | 实际训练更新 | 开发评估图数 | 共同 Reward ↑ | CLIP cosine ↑ | LPIPS ↓ | SSIM ↑ | ASR |",
             "|---|---:|---:|---:|---:|---:|---:|---|"]
    def number(value):
        return "NOT TESTED" if value is None else f"{value:.6f}"
    for row in rows:
        lines.append(f"| {row['Method']} | {actual[row['arm']]} | {row['measured_assessment_images']} | " +
                     " | ".join(number(row[key]) for key in ["common_reward", "CLIP_cosine", "LPIPS", "SSIM"]) +
                     " | NOT TESTED |")
    lines += ["", "3. **哪个方法最好。** " +
              (f"按已测共同 Reward，数值最高的是 {best_names}（{best['common_reward']:.6f}）。"
               "这只是单训练种子的描述性排序，不是普遍优越性结论。" if best else "尚无完成的锁定 checkpoint 开发评估，不能排序。") +
              (" 当前比较未完成，不能当作最终排名。" if not complete else ""), "",
              "4. **BF16 与可信度。** BF16 原路径的问题尚未宣告修复。本次采用现有 FP16 缓存权重上转 FP32 的计算路径，"
              "LoRA 存储/线性计算和 Adam moments 均为 FP32；这不恢复缓存中原本不存在的权重精度。"
              f"本次数值 gate 为 {numerical.get('status', 'NOT TESTED')}。"
              "精确一致与数值容差分别检查普通第一步、第二步，以及磁盘重载后真实完成的下一步。"
              "通过范围限于记录的节点、模型、数据和短更新序列。训练日志的有限梯度并不等于语义正确、ASR 有效或跨硬件复现。", "",
              "5. **实际视觉编辑。** " +
              json.dumps(state.get("qualitative_review", {"status": "NOT PERFORMED"}), ensure_ascii=False) +
              " 该检查不是独立人工标签，不能换算成语义保持准确率。", "",
              "6. **DDPO Reward 是否上升。** 以下为固定 16 张验证图、固定噪声种子下的共同 Reward，避免把训练样本变化或动态权重当作改进："]
    for method in ["ddpo_only", "rule_rsi", "rl_rsi"]:
        points = state.get("validation", {}).get(method, [])
        if points:
            values = [row["common_reward"] for row in points]
            lines.append(f"   - {METHODS[method]}：" + " → ".join(f"{value:.6f}" for value in values) +
                         f"；首末变化 {values[-1]-values[0]:+.6f}；完成 {len(points)-1} 轮验证。")
    audit_path = output / "final_evidence_audit/results.json"
    if audit_path.exists():
        final_audit = json.loads(audit_path.read_text())
        identity = final_audit["identity_mean"]["common_reward"]
        best_edit = max(value["common_reward"] for value in final_audit["edited_means"].values())
        lines += ["", f"   同一 CPU 评估路径下，不编辑原图的共同 Reward 为 {identity:.6f}，"
                  f"最终编辑方法的最高验证均值为 {best_edit:.6f}。"
                  + ("这提示该 Proxy 偏好保留原图；Reward 上升不能单独作为完成提亮指令的证据。" if identity > best_edit else
                     "该诊断应与图像质量和实际指令完成情况一起解释。") +
                  f" CPU/GPU 指标最大绝对差：{final_audit['CPU_vs_original_GPU_max_absolute_difference']}。"]
        luma = final_audit["mean_RGB_luma_change"]
        luma_conclusion = ("各组均值均下降，当前结果不能支持已完成“轻微提亮”的结论。"
                           if all(value < 0 for value in luma.values()) else
                           "亮度方向应结合逐图结果与独立指令完成度评价解释。")
        lines += ["", "   只读亮度方向审计：平均编码 RGB 亮度代理变化为 " +
                  "、".join(f"{method} {value:+.6f}" for method, value in luma.items()) +
                  "。单位为归一化 RGB 加权值，未作 gamma 线性化；这是事后诊断，不是指令成功率。"
                  + luma_conclusion]
        index = next(i for i, line in enumerate(lines) if line.startswith("4. **BF16"))
        repeat = final_audit["state_comparisons"].get("untriggered_rule_vs_DDPO_100_update_state_comparison")
        if repeat:
            lines[index] += (f" 未触发规则的 DDPO/Rule 两次 100 步运行，其 adapter、Adam、Torch/CUDA RNG、"
                             f"step/version/contract 共 {repeat['tensor_count']} 个比较张量："
                             f"精确一致={repeat['exact']}、容差通过={repeat['tolerance']}。"
                             "未保存所有 100 步的完整原始梯度；这不等于 BF16 修复或独立种子复现。")
        for filename in ["IMAGE_EDIT_QUALITY_AUDIT.md", "NEXT_EXPERIMENT_DECISION.md"]:
            path = output / "reports" / filename
            path.write_text(path.read_text(encoding="utf-8") +
                            f"\n## Read-only validation audit\n\nSame CPU evaluator: identity common reward {identity:.6f}; "
                            f"best edited common reward {best_edit:.6f}. "
                            f"Mean signed encoded-RGB luma changes: {json.dumps(luma)}. " +
                            ("The unchanged source scores higher. " if identity > best_edit else "The unchanged-source comparison is retained. ") +
                            ("All mean luma changes are negative. " if all(value < 0 for value in luma.values()) else "Luma directions are retained without a success label. ") +
                            "Validate instruction completion "
                            "and reward sensitivity before scaling. This is a post-hoc diagnostic, not measured ASR or brightness accuracy.\n",
                            encoding="utf-8")
    lines += ["", "7. **学习型 Controller 是否实际工作。** "
              f"记录了 {len(actions)} 次动作、{len(controller)} 次真实策略优化；其中 "
              f"{sum(row['controller_max_parameter_change'] > 0 for row in controller)} 次观测到策略参数实际改变。"
              f"动作序列：{[row['action'] for row in actions]}（0 Keep，1 增 LR，2 减 LR，3 增语义权重）。"
              "经验记忆来自每轮真实验证共同 Reward 的变化；没有 Victim 微调或 ASR 反馈。"
              "三轮最多三个策略更新，只能验证学习代码被执行，不能说明学会了稳定的优化策略。", "",
              "8. **RSI 独立贡献。** "]
    assessment = state.get("assessment", {})
    if assessment.get("rl_rsi") and assessment.get("ddpo_only"):
        delta, low, high = paired_bootstrap(assessment["rl_rsi"], assessment["ddpo_only"], protocol["seed"])
        lines[-1] += (f"RL-RSI − DDPO-only 的共同 Reward 差为 {delta:+.6f}，配对图像 bootstrap 95% CI "
                      f"[{low:+.6f}, {high:+.6f}]。" +
                      ("区间包含 0，目前没有明确的正向证据。" if low <= 0 <= high else
                       "区间在 0 以上，可记录为单种子、开发图像上的初步迹象。" if low > 0 else
                       "区间在 0 以下，当前观察方向对 RSI 不利。") +
                      "该区间仅重采样 16 张源图，不包含训练随机种子的不确定性，也未证明真实 ASR 或泛化收益。")
        lines[-1] += " 尚未比较随机 Controller 或 LR/语义权重匹配的固定配置，不能将该差值单独归因于学习型策略。"
    else:
        lines[-1] += "NOT TESTED：尚无两组完整锁定 checkpoint 的配对开发评估。"
    lines += ["", "9. **可以写入论文的内容。** 可以在明确注明“无害编辑、单种子、开发评估”的 Pilot 表中使用实际测得的 "
              "CLIP cosine、LPIPS、SSIM、共同 Reward、真实更新次数及其数值验证范围。"
              "Table 8 中的真实 ASR、ΔClean、Victim 微调有效性以及跨种子显著性仍为 NOT TESTED。"
              "不得将 Frozen/Fixed 重复行当作独立证据，也不得把本次 FP32 通过写成 BF16 根因修复。", "",
              "10. **下一步最值得投入什么。** " +
              ("先核对源图/编辑图中的颜色、物体、文字与问题答案是否被保持，并检查共同 Reward 的分项及负结果。"
               "只有质量与比较证据支持后，再提出相同预算的多独立训练种子复现；不自动扩展到主实验或封存测试集。"
               if complete else "先处理当前未完成或失败条件，保留现有日志与 checkpoint；不以扩大数据、追加步数或更改阈值追求正结果。"), "",
              "实验代码和重现方法见项目 `docs/MINI_PILOT.md`。CSV 的配对 95% 区间只描述开发图像抽样。"
              "七个要求的报告、原始观测、训练/Controller 日志、checkpoint、数值 witness 和源代码快照保留在本次独立输出目录。", ""]
    (output / "reports/PROGRESS_REPORT_CN.md").write_text("\n".join(lines), encoding="utf-8")
    print(output / "reports/PROGRESS_REPORT_CN.md")


if __name__ == "__main__":
    main()
