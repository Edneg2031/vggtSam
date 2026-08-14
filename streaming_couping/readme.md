# streaming_couping

本目录只保留两个 V0 入口。V0 已冻结，不再包含 SAM mask/identity 位姿优化实验。

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
后，paired RMSE退化`0.4588%`、融合点云双向距离退化`2.7279%`，map pass为0。该结论作为
`qk_pose_raw_depth`负对照保留。

r3联合回放也已完成：

- `qk_joint_depth_pose`的全场景paired/fused分别退化`2.8689%/5.0488%`；
- `qk_joint_pointmap`的全场景paired/fused分别退化`8.8613%/16.3464%`；
- joint depth在全部SAM区域聚合上改善`1.0502%/7.1129%`，但逐实例只有两个bed track改善，
  wardrobe和只可见2帧的chair退化；
- 两个joint分支的全场景+SAM区域联合gate都为0。

因此当前只保留`retrieve_qk` pose输出（单序列center/rotation均改善）；semantic map仍部署
`raw_semantic_map.ply`。r3证伪了“QK稀疏历史上下文联合重跑冻结geometry heads就能改善整体点云”，
但保留了逐SAM实例分析作为后续显式几何约束的依据。

## Evidence archive

失败候选的代码、配置和命令已删除。增量审计确认QK不是普通截断：它优于recent和3-seed random
pose对照，同时QK pose与raw pointmap的一致性指标改善。随后SAM persistent-ID probe未通过：
correct ID没有同时改善center/rotation，也没有优于shuffled-ID和shifted-mask controls。因此V0中
SAM固定为语义追踪/投影角色，不参与pose候选生成。V4–V9.8及其他验证/证伪结论保存在：

[实验正负证据归档](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)
