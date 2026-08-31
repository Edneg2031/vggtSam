# 多场景 causal affine / depth-head ray 诊断

入口：

```bash
zsh streaming_couping/commands_analyze_multiclip_affine_geometry.txt
```

该实验复用已经完成的四个 scene-disjoint V0 frozen cache，不重跑
StreamVGGT/SAM，不训练，也不修改正式 V0 pointmap。默认使用每个 clip 的前
70% 帧做 calibration、后 30% 帧做 holdout；四个 30-frame pilot clip 因此是
21/9 帧，合计 120 帧。每个 clip 独立拟合 affine，不能跨场景共享尺度参数。

## Causal pair

候选阶段只使用 raw SAM object mask、raw StreamVGGT world pointmap、depth
head、confidence、intrinsics 和 raw StreamVGGT pose。对当前帧的每个 slot，
只收集严格早于当前帧的同 slot 点云，将它投影到当前视角并与当前 mask 相交：

```text
depth_head_to_historical_pointmap:
    current depth-head Z  ->  earlier pointmap Z

pointmap_z_to_historical_pointmap:
    current raw pointmap Z -> earlier pointmap Z
```

两种 source 必须分别拟合；不能把 depth head 和 pointmap 的 self-consistency
结果混成一个证据。拟合形式为：

```text
historical pointmap Z = scale * current source Z + shift
```

使用 robust positive-slope affine fit，参数只由 calibration prefix 产生。对于
holdout，历史点可以包含之前已经到达的 holdout 帧，但 holdout 当前值不会参与
参数拟合。

`selected_v0` pose 只作为 self-consistency control。所有 geometry branch 都
固定使用 raw StreamVGGT pose，避免把 QK pose 改善与 depth 几何改善混在一起。

## Ray reconstruction branches

候选 geometry 会写入独立的 `candidate_geometry_<clip>.pt`，并且只替换 raw SAM
object union 内的点，背景保留原始 pointmap：

```text
raw_v0
pointmap_self_affine_holdout
depth_head_ray_identity_holdout
depth_head_ray_self_affine_holdout
depth_head_ray_self_affine_all
```

depth-head branch 先将 depth map 调整到 pointmap 网格，使用调整后的内参在同一
像素射线上做：

```text
Z' = scale * Z + shift
Xc' = normalize(K^-1 [u,v,1]) * Z'
Xw' = inverse(raw world_to_camera) Xc'
```

非法深度、负深度或无法重建的像素自动保留 raw point。`identity_holdout` 是
“depth head 与 pointmap 已同 gauge”的控制，不是部署结论。

## Evaluation-only 指标和 gate

所有 candidate 文件冻结后才打开 GT。输出同时报告 calibration、holdout、all
三个 frame scope，以及 `all_confident` 和 `raw_object_union` 两个 depth scope：

- depth RMSE、median、P90；
- object accuracy/completeness、paired RMSE；
- F-score@5cm、voxelIoU@5cm、ghost-point ratio；
- 每 clip 结果和 macro aggregate。

`depth_head_ray_self_affine_holdout` 是主 geometry branch。通过条件是：

- 至少 3/4 clip 的 depth-head self-affine holdout median 和 P90 同时下降，且
  scale 为正；
- 主 branch 的 holdout voxelIoU/F-score 总体不低于 raw；
- 至少 3/4 clip 不低于 raw；
- completeness error 增幅不超过 5%，ghost rate 不增加超过 5 个百分点。

脚本即使显示 `gate=GO`，也只表示诊断 gate 通过，`decision` 仍为
`DIAGNOSTIC_ONLY`，不会自动把 affine correction 写入正式 baseline。四个场景
属于已有 pilot/development audit，不能据此宣称新的独立 test 泛化。

主要输出：

```text
candidate_metadata.json
candidate_geometry_<clip>.pt
scale_shift_pairs_causal.csv
scale_shift_metrics.csv
depth_source_metrics.csv
map_metrics.csv
map_object_metrics.csv
tracking_metrics.csv
per_clip_metrics.csv
aggregate_metrics.csv
summary.json
copyable_result.txt
```
