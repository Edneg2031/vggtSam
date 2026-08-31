# streaming_couping

本目录只保留 `V0` SAM3.1 × StreamVGGT 动态实例几何 baseline。当前做法、效果、实验结果和结论边界统一记录在：

[当前状态与实验结论](docs/current_status.md)

运行：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

该入口支持多 prompt、多实例、未来帧 birth、永久 slot、几何辅助 mask、静态实例几何 pose residual
和无有效实例时的 exact camera-baseline fallback。它不缓存 SAM appearance token，也不声称 SAM token
能改善 pose。旧版本 runner、配置、命令和专用测试已删除；模型、数据、`externals/`、`outputs/`
均不在本次清理范围。

## 语义地图 pipeline

新增的 `src/semantic_mapping/` 是独立于具体模型的语义实例地图层。它使用统一的
`GeometryFrame`、`ObjectObservation` 和 `SegmentationFrame` contract；当前的
StreamVGGT 与 SAM3.1 通过 adapter 接入，后续替换 HorizonStream 时只需新增几何 adapter，
不需要修改体素融合和导出逻辑。

从 RGB 和文本 prompt 运行当前模型：

```bash
PYTHONPATH=. python -m streaming_couping.scripts.run_semantic_map \
  --frames /path/to/rgb_frames \
  --prompts bed wardrobe \
  --output-dir outputs/semantic_map
```

如果已经有 V0 cache，可以不重新运行模型：

```bash
PYTHONPATH=. python -m streaming_couping.scripts.run_semantic_map \
  --cache outputs/streaming_couping_v0/cache/<clip>.pt \
  --output-dir outputs/semantic_map
```

输出包括 `semantic_map.pt`、按实例着色的 `semantic_map.ply`、RGB 版本的 `rgb_map.ply`、
`object_tracks.ply` 和 `map_summary.json`。静态目标写入世界坐标体素地图；动态目标保存在
独立 object track 中，避免污染静态地图。默认使用当前 V0 的 raw pointmap，不启用被否决的
temporal point prompt、历史深度 Veto 或 affine depth correction。
