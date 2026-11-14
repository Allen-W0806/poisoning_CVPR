# P0.4 固定协议与预算

配置：`configs/p0_foundations_20261111.json`；manifest：`experiments/p0_foundations_20261111_v1/remote/manifest.json`。protocol SHA `d991e10ed9428a306f9cdf73af77dfe0514aa2128f5b1664e8e1c0bacfb2a0dd`；manifest SHA `fa3126f4618eaab5d0e42ce61a077e43c7e2e5644b48090423938ee0dc77d4a9`。全部在模型调用前锁定。

train/dev/test/transfer=4/4/4/4，16个源图ID、内容SHA、question ID全互斥。新原始图片约2.39MB，排除先前缓存COCO ID和已有exposure census；没有打开历史test或prospective annotation。新独立source并不保证模型pretraining未见过。数据在独立run目录，不修改原始官方archive。

指标：clean VQA使用官方兼容ten-answer consensus score（越高越好）；SSIM越高表示源图结构相似；LPIPS越低表示源图感知距离小；亮度变化只表示方向，不能替代指令成功。没有ASR/隐蔽性优化。源图bootstrap10000次、analysis_seed20261111，报告n、SD、区间；单seed不能给跨seedCI。

16源图的额外64bit dHash筛查见`split_perceptual_audit.json`，仅为启发式，不宣称排除所有感知近重复。

预算：主P0 job90965预约1200秒、实际170秒（应用166.804秒）；结束后预处理job90978预约300秒、实际57秒。第二次提交前按前一作业终态实际170+新预约300=470秒检查，仍小于1200秒累计承诺上限；单个申请分别1200/300，不能说只有一次GPU提交。合计227 GPU秒（0.063056 GPUh）。包含模型加载/哈希、VAE/Frozen、clean SFT、重复/恢复、评估和全部后续dev诊断。90965记录的峰值allocated GPU为8599158272 bytes；90978没有记录峰值，因此整个两作业最大显存未知。沒有P1学习任务。账本见`reports/reproduction_ledger.md`，不得挪用其它项目历史预算。

P1论文表的未运行单元格保持空白（`reports/P1_results_unrun.csv`），不填0或PASS。当前尚缺>=3seeds、普通编辑SFT/可微reward对照、random/HPO完整控制组和独立任务族经验迁移。


## 原始标注复核

已从缓存的官方questions/annotations ZIP直接提取所选16个原始记录，核对全部问题、image/question IDs、160个人类答案及其顺序、共识数和训练回答。原始提取与archive SHA在`experiments/p0_foundations_20261111_v1/original_annotation_extract.json`；没有复制完整大archive或打开历史封存test。
