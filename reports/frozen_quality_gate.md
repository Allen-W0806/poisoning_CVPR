# P0.2 Frozen 推理质量门槛

状态：**PARTIAL_NOT_CLEARED_FOR_P1**。真实运行，不是旧4步smoke的重命名。固定4张新dev源图，512px，原生Euler20/30/50步和随机DDIM30步，seed/source/指令固定；所有配置0个optimizer更新。原生和DDIM分别记录，不把两种sampler混为一个policy。test没有用于配置选择。

另运行有界预处理诊断（job90978，57 GPU秒）：同一4-dev、指令/seed、30步，去掉人工白色方形padding，按长边512/8倍数网格保留全景；再记录预先固定的text/image guidance5/3。后续dev诊断不更改原manifest或原对照，不调reward。新函数`src/editing/native_preprocessing.py:native_aspect_image`和3项CPU测试保留；旧square-pad默认及历史checkpoint合约未覆盖。不同输入网格改变噪声几何；guidance5/3同时改变两项设置，不能定位唯一原因。

|配置|n|SSIM ↑|LPIPS ↓|平均亮度变化|方向变亮的源图|
|---|---:|---:|---:|---:|---:|
|vae_roundtrip|4|0.652760|0.082231|-0.006544|0/4|
|native_20|4|0.552083|0.174997|-0.028025|3/4|
|native_30|4|0.565553|0.167602|-0.025375|2/4|
|native_50|4|0.570467|0.155478|-0.026174|2/4|
|ddim_30|4|0.552687|0.196627|-0.009602|2/4|
|native_aspect_30|4|0.564799|0.135883|-0.030945|2/4|
|native_aspect_preserve_30|4|0.506705|0.226840|+0.001357|3/4|


指令：`Slightly brighten the lighting. Keep all objects, their colors, counts and positions unchanged.`。LPIPS为校准AlexNet v0.1，输入[-1,1]，trunk/calibration SHA已检查。只在与Pillow一致的content box上评估，SSIM/LPIPS是源参考诊断，不是任务成功或语义认证。变亮比例只是全局亮度方向代理。

VAE encode(mode)→decode(mode)独立重建检查已运行。其均值SSIM=0.652760、LPIPS=0.082231，不能以重建不完美自动断言代码错误。原生官方pipeline已直接运行，模型、预处理、scheduler、解码路径及权重SHA均记录；没有调整RL reward。

技术视觉检查：512px/20–50步下主要场景可辨认，未见上一轮部分4步样本那样的全面纹理崩坏；但斑马图明显偏黄、棒球场饱和度改变，固定亮化指令未稳定满足。去除白色方形padding使部分感知距离下降，但不能单独消除偏色；guidance 5/3减轻偏色同时引入颗粒纹理。没有找到可同时认证指令符合与内容保留的配置，保留全部负结果。。检查者为Codex技术审阅，不冒充独立人工质量标签。图为`outputs/p0_foundations_20261111_v1/paired_examples.png`；源图、原尺寸生成PNG、编辑指令、seed、逐图指标在`experiments/p0_foundations_20261111_v1/remote/frozen/`。

预处理配对图为`outputs/p0_foundations_20261111_v1/paired_preprocessing.png`，原始PNG/新source grid在`remote/quality_followup/`。去padding的LPIPS均值0.135883低于方形30步0.167602，但亮度均值仍负；guidance5/3虽平均亮度略正0.001357，LPIPS变差且有颗粒纹理。没有挑一个单项正值作为整体PASS，未将不同网格的指标当作严格配对效能检验。

P1决定：**NOT_RUN：Frozen普通编辑任务质量/指令门槛未清，不能把P0技术闭环当成方法验证**。未将不同源图/分辨率的旧4步实验与本轮做成配对因果结论；没有用test重新挑选seed或奖励权重。
