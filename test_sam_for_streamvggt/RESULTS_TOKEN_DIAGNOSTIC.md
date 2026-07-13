# SAM3 / StreamVGGT Token 跨视角诊断结果

## 实验目的

判断 SAM3 与 StreamVGGT 的哪些中间特征包含稳定的跨视角实例对应信息，为后续融合层选择提供依据。

固定输入为 ScanNet++ 场景 `00a231a370`、实例 `37`（cabinet）和五帧序列 `[133, 162, 520, 566, 477]`。使用参考帧 GT instance mask 内的 token 均值作为 query prototype，在其余可见帧中计算余弦相似度与实例定位指标。

## 关键结果

| Feature | Shape | Top-1 命中率 | Top-5 命中率 | 正负区域 margin | 定位误差 |
|---|---:|---:|---:|---:|---:|
| SAM3 encoder FPN2 | `[5,256,72,72]` | 0.50 | 0.50 | 0.129 | 0.108 |
| SAM3 actual spatial memory | `[5,64,72,72]` | 0.50 | 0.50 | 0.077 | 0.287 |
| SAM3 oracle spatial memory | `[5,64,72,72]` | **1.00** | **0.90** | 0.237 | 0.051 |
| StreamVGGT layer 4 | `[5,2048,22,37]` | 0.50 | 0.50 | 0.086 | 0.154 |
| StreamVGGT layer 11 | `[5,2048,22,37]` | 0.50 | 0.40 | 0.088 | 0.140 |
| StreamVGGT layer 17 | `[5,2048,22,37]` | **1.00** | 0.60 | **0.295** | **0.032** |
| StreamVGGT layer 23 | `[5,2048,22,37]` | 0.50 | 0.30 | 0.112 | 0.064 |

Top-1/Top-5 和 margin 越高越好，定位误差越低越好。

## 已验证结论

1. **StreamVGGT layer 17 是当前样本中最强的跨视角实例定位特征。** 它在两个非参考可见帧中均命中目标实例，目标与背景的相似度 margin 为 `0.295`，定位误差仅为归一化图像对角线的 `3.2%`。
2. **SAM3 原始 FPN2 只有有限的跨视角可分性。** 其 Top-1 命中率为 `0.5`，说明仅依赖当前外观特征不能稳定处理该大视角序列。
3. **SAM3 memory 的结构能力存在，但实际 memory 已受到错误传播影响。** oracle memory 的 Top-1 为 `1.0`、定位误差为 `0.051`；actual memory 则退化到 Top-1 `0.5`、定位误差 `0.287`。因此问题主要出现在目标存在判断、预测 mask 和 memory 写入形成的传播链，而不是 memory 表示完全没有能力。
4. **StreamVGGT 最深层不是最佳融合层。** Layer 23 的目标余弦相似度很高，但背景相似度同样很高，导致 margin 和 Top-K 命中率明显低于 layer 17。此前单层融合默认取最后一层，会丢失 layer 17 的优势。
5. **早期 layer 4/11 更偏全局相似，实例定位不足。** 二者有效秩较高，但目标与背景 margin 较小，不能单独作为可靠的实例重定位信号。

## 尚未证明

本实验是冻结特征上的离线 prototype 检索，只使用一个场景、一个实例和两个非参考可见帧，因此尚不能证明：

- Layer 17 注入 SAM3 后一定能提高最终 mask IoU；
- 提升能够泛化到其他实例、场景和序列；
- StreamVGGT geometry 是因果因素，而不是特征容量或训练参数量带来的收益；
- oracle memory 可以在真实推理中获得，因为该分支使用了 GT visibility。

## 下一步验证

固定使用 StreamVGGT layer 17，并在相同网络结构、随机种子和训练配置下比较：

1. zero geometry；
2. aligned layer-17 geometry；
3. shuffled layer-17 geometry。

只有 aligned 同时优于 zero 与 shuffled，才能说明收益来自正确的跨视角几何对应。

## Layer-17 控制实验结果

固定使用 StreamVGGT layer 17，冻结 SAM3 tracker，仅训练同构 fusion adapter。三组实验使用相同的五帧序列、随机种子和训练步数。

| Experiment | Final cross-view IoU | Best cross-view IoU | Recall | Absent FP |
|---|---:|---:|---:|---:|
| Zero geometry | 0.8930 | 0.8951 | 1.0 | 0.0 |
| Aligned geometry | **0.9262** | **0.9273** | 1.0 | 0.0 |
| Shuffled geometry | 0.9238 | 0.9238 | 1.0 | 0.0 |

### 当前可以得出的结论

1. **非零 layer-17 特征对最终分割有帮助。** Aligned 相比 zero 的 final cross-view IoU 提高约 `0.0333`，best IoU 提高约 `0.0322`。这与前述 token 检索结果一致，说明 layer 17 不只是具有离线可分性，也能通过 fusion adapter 改善 mask 输出。
2. **三组均恢复了所有可见帧，且没有在目标消失帧产生误检。** 因此更强的 fusion residual 本身已经能帮助冻结的 SAM3 tracker 越过 object-presence gate；geometry 的主要收益体现在 mask 精度，而非本组实验中的 Tracking Recall。
3. **正确帧对齐的额外收益目前很弱。** Aligned 只比 shuffled 的 final IoU 高约 `0.0024`，best IoU 高约 `0.0035`。单个随机种子下，这个差距不足以证明正确的跨视角对应是性能提升的原因。

### Shuffled 对照的限制

当前 shuffled 模型在固定循环排列 `[4, 0, 1, 2, 3]` 上训练并评估 700 步。由于训练序列固定，模型可能记住错误排列或把 geometry 当作与帧无关的特征先验。因此“shuffled 接近 aligned”不等于空间对应无用，也可能是当前控制实验允许模型适应固定错位。

### 下一步严格验证

使用 aligned checkpoint，在不继续训练的情况下执行三种推理：

1. aligned geometry；
2. zero geometry；
3. shuffled geometry。

这种同一 checkpoint 的 inference-time perturbation 能排除不同模型各自过拟合的问题。若 aligned checkpoint 在推理时打乱或移除 geometry 后明显下降，才能证明模型实际依赖正确的 layer-17 对应。随后应增加多个随机种子和未参与训练的帧/实例验证稳定性。
