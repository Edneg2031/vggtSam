# V0 联合几何反馈诊断

运行命令已归档；本文仅保留该历史诊断的设置、结果和结论。

该实验只读取已经完成的 V0 frozen cache，不重跑 StreamVGGT 或 SAM，不训练、不
修改 pose/pointmap，也不会把任何修正写回正式 baseline。候选生成结束后才打开 GT，
最终 `decision` 固定为 `DIAGNOSTIC_ONLY`。

## 历史点云深度 Veto

对每个当前 frame/slot，只收集严格早于当前 frame 的同一 slot 点云，并用当前冻结
pose 将这些 world points 变换到当前 camera 坐标。历史深度的中位数/MAD 区间和
q05/q95 区间分别作为候选 reference；当前深度只作为待检测值。

`current_depth_reference` 是显式循环论证 control：它用当前 mask 的当前 depth
构造 reference，不应被解释为部署方法。

由于 StreamVGGT 输出可能处于 arbitrary scene gauge，Veto 的固定 padding 虽然保留
了命令行历史名称 `absolute_padding_m`，实际单位是输入 depth 的 native units，
不是保证为 metric metres。候选阶段不能读取 GT Sim(3) scale 来换算单位。

## Scale-shift 诊断

无 GT 的 `self_consistency` 使用当前 depth 与历史点投影到当前 mask 后的深度配对，
并采用固定前 70%/后 30% 的时间切分拟合：

```text
projected_history_depth ~= scale * current_depth + shift
```

evaluation-only 的两条 GT 行则使用同一 pointmap pixel 的缓存 GT depth：

```text
GT depth ~= scale * predicted_pointmap_native_depth + shift
GT depth ~= scale * baseline_depth_head + shift
```

第二种对比避免了将“GT 点云再经过 predicted pose”的结果当成像素级 target，因而
不会把 pose 错误混入 depth target。GT 行只用于诊断是否存在可泛化的 affine depth
关系，不会自动校正 pointmap。

## 输出

主要文件：

```text
summary.json
copyable_result.txt
candidate_metadata.json
depth_veto_causal.csv
depth_veto_metrics.csv
scale_shift_pairs_causal.csv
scale_shift_metrics.csv
tracking_metrics.csv
tracking_frame_metrics.csv
map_metrics.csv
per_scene_metrics.csv
```

重点查看 Veto 的 `removed_ratio`、`removed_foreground_fraction`、tracking/map
delta 和 worsening ratios；scale-shift 则查看 calibration/holdout 的 RMSE、
median、P90 是否同时下降。单场景 positive diagnostic 只能决定是否值得在新的
scene-disjoint validation 上继续，不能直接证明正式反馈模块有效。
