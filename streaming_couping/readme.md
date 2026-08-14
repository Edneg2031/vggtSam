# streaming_couping

本目录只保留两个 V0 入口。

## 1. Baseline

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

Baseline 包含：

- SAM3.1 多 prompt、多实例、forward-only discovery/tracking；
- future late birth、永久 slot、persistent ID；
- StreamVGGT 几何辅助 SAM mask prompt/competition；
- 冻结 StreamVGGT first-frame anchor + native QK Top-4 history retrieval；
- 同一次QK replay运行camera/depth/point heads并保存joint geometry artifact；
- QK artifact 缺失时显式回退 raw StreamVGGT pose；
- 不缓存 SAM appearance token，不训练 pose model，不使用 pose loss。

30帧单序列结果（frame 90为gauge，其余29帧评分）：center error
`0.120983 → 0.107762`（改善`10.9278%`），rotation
`2.55064° → 2.39375°`（改善`6.1511%`）。该结论只适用于当前单序列；不声明
SAM identity改善pose，也不声明跨场景泛化。

## 2. Semantic map A/B

```bash
zsh streaming_couping/commands_v0_semantic_map_ab.txt
```

该命令使用同一SAM persistent-slot masks/IDs、RGB和锁定的raw-confidence像素支持，比较：

- `raw_depth_pose`：full-history raw depth/K/pose；
- `qk_pose_raw_depth`：只替换QK pose的已证伪负对照；
- `qk_joint_depth_pose`：同一次QK replay输出的depth/K/pose；
- `raw_pointmap`与`qk_joint_pointmap`：两个上下文各自的point head输出。

depth-backprojection与point-head分别从各自raw reference拟合一次固定Sim(3)，再固定用于同族所有分支，
不再混用两个head的尺度。GT只在全部native-coordinate maps生成后用于评分。

输出包括：

- `semantic_maps.pt`：点、RGB、semantic slot、confidence、frame和prompt/track metadata；
- 五个分支的binary PLY；
- 全场景、全部SAM区域聚合、逐persistent instance的paired RMSE与融合点云双向距离；
- `frame_metrics.csv`和`instance_metrics.csv`。

该路径同时验证QK joint depth/pointmap，但不把prompt标签当作GT语义分类评价。

r2最终结果：raw-depth reference Sim(3)拟合RMSE为`0.02370 m`，协议有效；把QK pose用于相同raw depth
后，paired RMSE退化`0.4588%`、融合点云双向距离退化`2.7279%`，map pass为0。该结论现在作为
`qk_pose_raw_depth`负对照保留，而不是用来判断joint replay。

- pose输出：`retrieve_qk`（当前单序列center/rotation均改善）；
- semantic map部署在r3结果出来前仍为`raw_semantic_map.ply`；
- 只有joint depth或joint pointmap同时通过全场景与SAM区域gate，才允许替换raw map。

这次r3不是恢复旧ICP或单独调外参，而是直接验证RetrieveVGGT式的上下文联合几何输出。

## Evidence archive

失败候选的代码、配置和命令已删除；V4–V9.8及后续验证/证伪结论保存在：

[实验正负证据归档](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)
