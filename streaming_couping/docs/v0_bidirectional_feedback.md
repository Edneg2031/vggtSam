# V0 显式空间—时间反馈诊断

这个实验把 Gemini 提出的三个 primitive 退回到当前冻结的 V0 单场景：

```text
clip=00a231a370_90_525_step15_37_68_54
scene=00a231a370
frames=90,105,...,525
```

运行入口：

```bash
zsh streaming_couping/commands_run_v0_bidirectional_feedback.txt
```

输出默认写到：

```text
${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0_bidirectional_feedback/
```

## 实验边界

该命令复用已经完成的 V0 cache，不重新运行 StreamVGGT 或 SAM3.1，不修改
world pointmap，也不更新任何参数。V0 当前没有真实的逐像素
`vggt_tracking_loss_map`，因此 `StaticBackgroundPoseOptimizer` 只做合成梯度
smoke；它没有被冒充成真实 pose 训练结果。

候选生成阶段只读取冻结的 V0 几何、深度、mask、track 和 pose 字段。深度候选和
时序投影候选冻结后，才打开 GT 做评估。结果的 `decision` 固定为
`DIAGNOSTIC_ONLY`，不会自动把一个启发式提升写成新的 baseline。

## 三个模块

`streaming_couping/src/bidirectional_feedback.py` 提供：

1. `StaticBackgroundPoseOptimizer`：合并 SAM foreground mask，取 background
   complement，并对已有 loss map 做 masked mean/sum。
2. `DepthGuidedMaskRefiner`：对 mask 内正深度排序，按固定绝对/相对 gap 切成
   1-D cluster，保留最大 cluster。它是可审计 heuristic，不适用于直接证明
   mask ownership。
3. `TemporalPromptProjector`：将 global 3-D center 用
   `T_w2c = inverse(T_c2w)` 变换，再按 pinhole model 投影成 `(u,v)`；返回有效
   prompt 坐标、object ID、深度和 validity mask。

## V0 诊断内容

### Depth-guided mask

使用固定默认参数：

```text
absolute_gap=0.05
relative_gap=0.05
```

在 depth grid 上运行 refinement，再映射回 pointmap grid。GT 评估固定使用 raw
V0 track-to-instance assignment；GT mask 通过仓库已有的
`streamvggt_label_to_grid`（含 crop/pad）映射到对应的 stream/pointmap 网格，比较：

```text
raw_v0
depth_refined_v0
gt_mask_oracle   # 仅用于上界参考
```

`gt_mask_oracle` 不参与候选生成或阈值选择。

### Temporal prompt prior

每一帧只使用该帧之前最后一次有足够高置信点的对象中心；不会使用当前帧 GT 或
当前帧 GT mask 生成 center。分别用 V0 选定的 QK pose 和 raw StreamVGGT pose
做 projection control，统计：

- 是否在图像内；
- 是否落在当前 raw SAM mask；
- sealed evaluation 后是否落在对应 GT mask；
- history gap 和 center support 数量。

这些只是“如果把点喂给 SAM，空间先验有多可信”的指标。当前版本没有把点真正
传给 SAM，因此不能报告 temporal flicker 改善或 SAM re-segmentation 改善。

## 输出文件

```text
summary.json
copyable_result.txt
depth_refinement.csv
temporal_projection_causal.csv
temporal_projection.csv
temporal_projection_summary.csv
tracking_metrics.csv
tracking_object_metrics.csv
map_metrics.csv
map_object_metrics.csv
```

## 如何解释

重点看 `copyable_result.txt` 和 `summary.json`：

- `depth_refined_v0` 同时改善 tracking 和 map 指标，才说明该启发式值得进入
  下一轮 SAM 重分割实验；单个指标改善不足以证明有效。
- `gt_mask_oracle` 与 raw 的差距仍然很大时，主要瓶颈仍可能是 mask/identity，
  而不是 geometry feedback。
- temporal projection 的 `gt_mask_hit_rate` 低，说明直接把中心点作为正 prompt
  风险较高；高命中率也只说明 prompt prior 可行，不等于闭环已验证。
- 任何结论都必须结合单场景限制。该实验不提供跨场景泛化证据。
