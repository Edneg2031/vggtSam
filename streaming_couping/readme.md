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
- 冻结 StreamVGGT first-frame anchor + native QK Top-4 pose retrieval；
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

该命令固定相同的raw StreamVGGT depth/K、SAM persistent-slot masks/IDs、RGB和
confidence support，只比较raw pose与selected QK pose。地图先在StreamVGGT native
reference坐标生成；GT和固定raw-reference Sim(3)只在两份地图完成后用于评分。

输出包括：

- `semantic_maps.pt`：点、RGB、semantic slot、confidence、frame和prompt/track metadata；
- `raw_semantic_map.ply` / `selected_semantic_map.ply`；
- paired RMSE、融合点云双向nearest-neighbor/F-score与逐帧CSV。

该路径不使用QK candidate pointmap，也不把prompt标签当作GT语义分类评价。

## Evidence archive

失败候选的代码、配置和命令已删除；V4–V9.8及后续验证/证伪结论保存在：

[实验正负证据归档](docs/weekly_experiment_summary_2026-08-03_to_2026-08-06.md)
