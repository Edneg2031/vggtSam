# Explicit Dual-Framework Baseline

这个目录实现新方案的第一阶段可验证基线。它不再混合 SAM3 与
StreamVGGT 的隐空间 token，而是让两个冻结主干通过可解释的几何量交互。

## 当前数据流

```text
RGB 序列 + 首个可见帧的 GT instance prompt
        |                         |
        v                         v
Frozen SAM3 video tracker   Frozen StreamVGGT (causal KV cache)
mask / instance ID / score  world points / confidence / pose / intrinsics
        |                         |
        +------------+------------+
                     v
          ExplicitGeometryBridge
  reliable mask -> instance world-point map
  low SAM3 score -> project historical object points into current view
                     |
                     v
       SAM3 original / 3D prior / bridged mask
```

GT 只承担两件事：在参考帧生成给 SAM3 的 box prompt、所有帧的离线指标
计算。默认 `map_update_source: sam3`，包括参考帧在内的物体地图都从 SAM3
输出 mask 采样，不读取 GT mask。`oracle` 只保留作诊断上限，不能作为主结果。

## 门控

- `update_map`：SAM3 presence proxy、SAM3 mask 区域的几何置信度、持续观测次数以及
  当前 mask 与历史 3D 投影的一致性都可靠时更新地图。
- `use_fallback`：SAM3 分数低，且历史实例点与当前 pointmap 在相同像素处的
  深度一致支持率足够高时启用重投影。StreamVGGT confidence 只用于记录，不再
  被误当作实例对应置信度。

这两个 gate 必须分开。否则“跟踪分数高才允许 fallback”会使丢失恢复永远
无法触发。

当前 SAM3 predictor 的 `out_probs` 在传播阶段沿用初始检测分数，不是完整的
逐帧 matching score。因此代码将“该实例 ID 是否仍有输出”与 `out_probs`
组合成 presence proxy，并在 CSV 中保留该值。3D prior 会要求历史点深度与
当前 pointmap 深度双向接近；不再使用旧版只检查“历史点不在当前表面后方”的
单向条件，避免物体离开后把更远的背景误判为当前实例。

## 控制实验

一次运行会共享同一组冻结主干输出并比较：

- `zero`：禁用几何桥，等价于原始 SAM3 输出。
- `aligned`：使用时间对齐的 pointmap 和相机。
- `shuffled`：循环错位几何帧，检查收益是否依赖正确时空对应。

结果写入 `summary.csv`、各模式的 `frame_metrics.csv` 和
`tracking_report.png`。每个有效模式同时写出 `object_map.npz`，字段为
`points/confidence/instance_id/label/observations`。`static_centroid_diagnostics.csv`
是阶段 A 的只读诊断：
它衡量同一 GT 实例在 StreamVGGT 世界点中的重心漂移，但当前版本不会假装
StreamVGGT 支持在线 BA，也不会修改相机位姿。

## 运行

```bash
bash test_dframework/run_controls.sh
```

## 当前边界

这是 Stage B 的单实例 MVP，验证的是“显式 3D 历史能否在 SAM3 丢失时提供
正确候选区域”。它尚未把候选区域作为纠错 prompt 重新送入 SAM3 decoder，
也未实现 3D-aware memory positional encoding、滑窗位姿精修或最终多实例语义
点云地图。这些应在 aligned 明显优于 zero 和 shuffled 后再逐项加入。
