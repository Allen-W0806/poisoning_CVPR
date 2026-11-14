# Image-edit quality audit

All numbers below come from saved observations. This is a benign lighting-editing pilot. CLIP cosine is semantic text alignment, not ASR, instruction correctness, or VQA accuracy. The 16 mini-test images come from a historically exposed training pool and are development assessment. Frozen edits images with unchanged pretrained weights; it is not a pixel identity baseline. Fixed shares the identical prompt, seed, weights, sampler and observations by explicit user choice; it is an equivalence control, not independent evidence.

| Method | status | optimizer_steps | common_reward | CLIP_cosine | LPIPS | SSIM | delta_common_vs_Frozen | ASR |
|---|---|---|---|---|---|---|---|---|
| Frozen | MEASURED | 0 | -0.051463 | 0.152862 | 0.357187 | 0.595992 | 0.000000 | NOT TESTED |
| Fixed Editing (equivalent control) | MEASURED_EQUIVALENT_CONTROL | 0 | -0.051463 | 0.152862 | 0.357187 | 0.595992 | 0.000000 | NOT TESTED |
| DDPO-only | MEASURED | 100 | -0.020859 | 0.153833 | 0.328526 | 0.619803 | 0.030603 | NOT TESTED |
| Rule-based RSI | MEASURED | 100 | -0.020859 | 0.153833 | 0.328526 | 0.619803 | 0.030603 | NOT TESTED |
| RL-RSI | MEASURED | 100 | -0.014295 | 0.153746 | 0.321787 | 0.625512 | 0.037168 | NOT TESTED |

SSIM and LPIPS use the same aspect-preserving, padded 512 × 512 source as generation. Calibrated LPIPS receives RGB [-1,1]; CLIP receives RGB PIL via its processor. Semantic preservation is UNREVIEWED until actual source/edit pairs are inspected. No visual superiority sentence is generated from an arm name.

Qualitative inspection record: {
  "status": "AGENT_VISUAL_INSPECTION_NOT_INDEPENDENT_GROUND_TRUTH",
  "selection": "开发评估集中预先固定的前四张源图；未挑选最好看的样例",
  "sample_ids": [
    "340345009",
    "155177008",
    "245528000",
    "94416001"
  ],
  "evidence": "figures/source_edit_contact_sheet.png",
  "observations": [
    "长颈鹿场景在各编辑方法中均出现明显黄绿偏色，动物及草地颜色偏离原图；大体布局仍在。",
    "电脑桌场景中原本较暗的电脑外壳被改成青绿色、黄色，桌面和背景色调也变化，不符合仅轻微提亮且保持颜色的描述。",
    "喷水人物与自行车场景的大体构图保持，背景色调和白色填充边缘发生变化；不能由这四图推断所有问题答案保持。",
    "摩托车场景保留主体位置，车身、地面和边缘填充区域的色彩/对比度发生变化。",
    "DDPO-only 与 Rule-based RSI 在这些图中看起来相同；策略运行未触发规则减半 LR。RL-RSI 仍可见偏色，图上细微差别不足以证明语义保持或方法普遍优越。"
  ],
  "limits": "四张源图的自动代理目视检查；不是盲审、人工独立标签或语义正确率。ASR、Delta Clean 与 Victim 微调仍未测试。"
}

## Read-only validation audit

Same CPU evaluator: identity common reward 0.273499; best edited common reward -0.015453. Mean signed encoded-RGB luma changes: {"frozen": -0.11243483232629072, "ddpo_only": -0.10622581312759255, "rule_rsi": -0.10622581312759255, "rl_rsi": -0.10938630244030674}. The unchanged source scores higher. All mean luma changes are negative. Validate instruction completion and reward sensitivity before scaling. This is a post-hoc diagnostic, not measured ASR or brightness accuracy.
