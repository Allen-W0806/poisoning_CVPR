# P0.3 干净 VLM 技术闭环

**Can run clean VLM end-to-end? YES（有界LoRA工程验证）。** 本轮重新运行，不仅引用旧报告。

- 模型：Qwen/Qwen3-VL-2B-Instruct，revision `89644892e4d85e24eaac8bacfd4f463576704203`；BF16冻结base，FP32语言attention q/v LoRA rank4/alpha8，视觉编码器冻结。
- 数据：4张正常train QA，dev/test/transfer各4张独立source；原始公开COCO图与VQAv2十个人类答案，>=6/10规范化共识作为训练回答。无编辑/后门/投毒数据参与SFT。只监督assistant回答，prompt/image token为-100，前缀必须精确一致。
- AdamW lr1e-4，accum2，clip norm1，constant scheduler；保留2步，实际调用3次（包含一次fresh-object replay）；112个adapter张量在第二步有变化。
- checkpoint后重新加载固定base、公开adapter、新optimizer/scheduler，执行相同下一步。比较562个张量，差异元素0，最大绝对误差0.0。原始梯度、参数、Adam、scheduler及全部保存RNG精确一致。
- 最终adapter/optimizer再次由新对象经完整training_state恢复，dev预测逐条相同。该路径会覆盖public adapter张量，因此不单独认证仅adapter文件的推理加载；其文件内容与state一致性另作核验。base聚合SHA `390847f15e9f70f12042da573246290cdef48b6150bbfce52b279d1a47ff0458` 前后一致，最终checkpoint SHA `ace072cfbbb3150b334fd252deedd3198fa737281514d197c368390d00a3ed97`。

|划分|n源图|VQA consensus score|源图SD|探索性源图bootstrap95%CI|
|---|---:|---:|---:|---|
|dev_before|4|0.9750|0.0500|[0.9249999999999999, 1.0]|
|dev|4|0.9750|0.0500|[0.9249999999999999, 1.0]|
|test|4|0.7250|0.4856|[0.25, 1.0]|
|transfer_clean_QA|4|1.0000|0.0000|[1.0, 1.0]|

每划分只有4张图、1训练seed，不能作为完整模型方法验证或泛化改善。dev训练前后分数未变。transfer是另一组正常QA源图，不是独立编辑任务族。source-bootstrap不估计训练seed变异，n=4全对时退化区间不证明总体无不确定性。

预先固定native_30编辑的dev图也做了QA：原始参考答案一致性=0.975，仅为原标签proxy；编辑后的正确答案未独立重新标注，不称“编辑图真实accuracy”。

实际Slurm命令：

```bash
/storage/home/ydk297/projects/poisoning/.venv-editor/bin/python /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1/code/scripts/run_p0_foundations.py --assets-root /storage/home/ydk297/projects/poisoning --run /storage/home/ydk297/projects/poisoning/experiments/p0_foundations_20261111_v1
```

本地提交/查看命令为 `python scripts/p0_cluster_io.py submit` / `status` / `fetch`。单run禁止二次提交，重现实验须创建新的run标识；现有结果不覆盖。

训练日志：`experiments/p0_foundations_20261111_v1/remote/clean_vlm/training.jsonl`；预测：`predictions.jsonl`；resume二进制witness：`reference_next_update.pt`、`fresh_next_update.pt`；checkpoint：`checkpoints/step1`、`checkpoints/final`。规范输出为`outputs/clean_vlm_metrics.json`和`.csv`，原始metrics保留在run内。


## 原始标注复核

已从缓存的官方questions/annotations ZIP直接提取所选16个原始记录，核对全部问题、image/question IDs、160个人类答案及其顺序、共识数和训练回答。原始提取与archive SHA在`experiments/p0_foundations_20261111_v1/original_annotation_extract.json`；没有复制完整大archive或打开历史封存test。
