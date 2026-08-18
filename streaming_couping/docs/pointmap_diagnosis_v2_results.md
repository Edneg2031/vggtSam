# Pointmap Diagnosis V2

## 固定边界

- 正式 V0 不变：QK pose、full-history raw world pointmap、SAM persistent semantic lifting。
- V2 只读 V0 cache 和 QK pose artifact，不训练，不更新 pose，不更新 pointmap，也不写回 semantic map。
- 旧 V1 pointwise surfel/correspondence update 已删除，因为 correct-ID support 虽然更相干，但实际点更新使 correct-support 几何性能下降（RMSE 上升）。

## 实验顺序

1. **D0 Depth × Pose Oracle**：固定 processed calibration K 和同一份 V0 reference Sim(3)，比较 raw/GT depth × raw/QK/GT pose，并保留 direct raw pointmap。`GT depth + GT pose` 坐标闭环不通过时立即停止。
2. **D1 Raw Pointmap Error Decomposition**：分别统计 instance interior、inner boundary、outer boundary、unprompted complement；边界宽度固定测试 3 px 和 5 px；另做 per-instance、confidence quantile、risk-coverage 和 high-confidence-wrong。
3. **T0 Independent Triangulation**：运行 oracle 2D/real frozen RGB-patch correspondence × GT/QK pose。SAM 只限定同一 persistent slot 的搜索范围；correct-ID 与 shuffled-ID 使用完全相同的 current queries。`wardrobe` 仍保留在 V0 语义地图和 D1 分解中，但因区域过大不进入 T0 稀疏 anchor probe。
4. **TrackHead preflight**：按真实 API 输入 cached DPT token levels，visibility/confidence 使用 head 内部 sigmoid 后的固定 0.5 阈值。若仍无有效观测，记录 `TRACKHEAD_UNUSABLE`，不放宽阈值。

T0 只输出稀疏诊断 anchors，不写回 V0。只有 high-quality triangulation 优于同像素 raw point，并且 correct-ID 优于 shuffled-ID，才允许下一阶段讨论 T1。

D0 中 depth head 按其训练目标作为独立 scale/shift-invariant 输出处理：只在 reference frame 用 GT depth 拟合一次 robust affine，然后冻结到整个序列。直接套用 point-head Sim(3) scale 的旧假设仅作为 report-only 对照。T0 oracle 的两端坐标均由同一个 GT world point 投影，原始整数像素仅用于取得同位置 raw/GT 评分点。

## 运行

```bash
zsh streaming_couping/commands_v2_pointmap_diagnosis.txt
```

复制终端中的 `COPYABLE_V2_POINTMAP_DIAGNOSIS` 结果用于下一轮分析。正式结果同时写入：

```text
outputs/streaming_couping_v0/pointmap_diagnosis_v2/
```
