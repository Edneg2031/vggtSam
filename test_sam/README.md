# SAM3 + StreamVGGT 跨视角一致性消融

这个目录只验证一个问题：StreamVGGT 的隐式几何特征能否提高 SAM3 对同一实例的跨视角追踪一致性。

实验不构建 pointmap head，不读取 GT pointmap，也不使用 camera token。融合方法输出 SAM3 FPN residual，然后进入同一套 SAM3 原生 tracker：

```text
SAM3 tracker FPN2 [T, 256, 72, 72] ─┐
                                     ├─ Fusion ─ FPN2 residual
StreamVGGT aggregator feature(s) ────┘              │
                                                    ▼
                                SAM3 memory attention + mask decoder
                                                    │
                                                    ▼
                                           instance mask sequence
```

## 融合消融

| 方法 | 作用 |
|---|---|
| `sam_only` | 不加载 StreamVGGT；同容量 SAM residual adapter 基线 |
| `add` | 对齐通道与空间后逐元素相加 |
| `concat_conv` | 拼接 SAM/geometry feature，再用卷积细化 |
| `film` | 用全局 geometry feature 调制 SAM 通道；不保留空间对应 |
| `cross_attention` | SAM token 作 query，StreamVGGT token 作 key/value |
| `gated_cross_attention` | cross-attention 后增加逐 token gate |
| `multilevel_cross_attention` | 逐层融合 StreamVGGT 4/11/17/23 层，再与 SAM FPN2 卷积合并；最接近 3AM |

`--zero-geometry` 会保留同一个融合器和参数量，但把 StreamVGGT 特征清零，是判断提升是否真正来自 3D 信息的关键对照。

默认只向 FPN2 注入 residual：这是进入 SAM3 memory attention 的深层特征；FPN0/FPN1 保持原样，为 mask decoder 保留高分辨率边界信息，也与 3AM “不修改浅层 Hiera feature”的设计一致。`fusion.inject_levels` 可用于额外的注入层消融。

可视化中的 `Original SAM3` 使用原版 predictor API（reference GT box + 类别文本），用于直观参考；严格的同构对照是 `sam_only` 和 `cross_attention --zero-geometry`，因为它们与 3D 实验使用完全相同的 source-flow、mask prompt 和损失。

推荐按以下顺序跑，而不是一开始把所有方法铺开：

1. `sam_only`：确认 SAM3 source-flow 与数据监督本身能训练。
2. `cross_attention --zero-geometry`：控制融合器容量，但不提供 3D 信息。
3. `cross_attention`：只改变 geometry 输入，和第 2 组直接比较。
4. `multilevel_cross_attention`：检验 3AM 式多层几何融合是否优于单层。
5. `concat_conv`、`add`：判断复杂 attention 是否真的必要。
6. `gated_cross_attention`、`film`：分别检查选择性注入和无空间几何调制。

## 训练目标

- reference frame：选择目标实例面积最大的帧，只在该帧输入 GT mask prompt。
- 后续帧：完整经过 SAM3 tracker memory、object-presence gate 和 mask decoder。
- mask：focal loss + Dice loss。
- presence：逐帧监督目标是否可见，对应 3AM/SAM 的 occlusion/object-score 训练。
- 默认只训练 fusion adapter；`--train-tracker` 额外训练 SAM3 memory attention 和 mask decoder，接近 3AM 的训练范围。

## 输出

- `training_history.csv`：loss、正样本 IoU、Tracking Recall、消失帧误检率、residual RMS。
- `training_curves.png`：loss、跨视角指标和误检率曲线。
- `frame_metrics.csv`：每帧 fused/original SAM3 IoU 与 object score。
- `visualizations/`：RGB、GT、Original SAM3、Fused SAM3 四列对比。
- `checkpoints/`：融合器参数；启用 `--train-tracker` 时同时保存对应 SAM3 子模块。

论文依据：[3AM](https://arxiv.org/html/2601.08831) 使用多层 3D foundation-model 特征，经逐层 cross-attention 与卷积细化后并入 SAM2 特征，再完整经过 memory attention 和 mask decoder；[Multimodal SAM Adapter](https://arxiv.org/html/2509.10408) 还提供了 concat 与 cross-attention injector 这两类有价值的对照。当前 `multilevel_cross_attention` 保留 3AM 主线，但有意去掉 point/ray positional encoding 与 camera token，以隔离“隐式 3D feature fusion”本身。

`dataset.frame_indices` 的书写顺序就是送入 SAM3/StreamVGGT 的流式顺序，代码不会自动排序。做真实时序实验时应按采集顺序填写；故意构造跳视角序列时也要明确该顺序代表的传播过程。

## 一键运行

```bash
bash test_sam/run_all_ablations.sh
```

默认每组运行 700 step。快速检查全部链路可使用：

```bash
ITERATIONS=2 OUTPUT_ROOT=outputs/test_sam_ablation_smoke \
  bash test_sam/run_all_ablations.sh
```

所有实验顺序执行，日志保存在各实验目录的 `run.log`，最终汇总写入 `ablation_summary.csv`。
