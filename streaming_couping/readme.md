# streaming_couping V0

V0 是冻结、免训练的 StreamVGGT + SAM3.1 语义地图 pipeline：

- QK retrieval 只为 camera head 选择历史并输出相机轨迹；
- full-history raw StreamVGGT world pointmap 作为地图几何；
- SAM3.1 persistent mask、slot 和 track ID 作为点云语义；
- SAM 不参与 pose，点云不做后处理优化。

模型主运行入口：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

运行流程包括：缓存 StreamVGGT 几何和 SAM3.1 多 prompt 实例轨迹；使用 first-frame
anchor + native QK Top-4 运行 pose-only camera replay；固定 pose 完成 tracking/pose
审计；最后将 SAM persistent 实例标签投影到未修改的 raw world pointmap。

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

## 辅助入口

```bash
zsh streaming_couping/commands_plot_v0_poses.txt
zsh streaming_couping/commands_semantic_mapping.txt
zsh streaming_couping/commands_semantic_mapping_v1.txt
zsh streaming_couping/commands_semantic_mapping_v1_evaluation.txt
zsh streaming_couping/commands_check_scannetpp_data.txt
zsh streaming_couping/commands_audit_scannetpp_processed.txt
zsh streaming_couping/commands_run_v0_bidirectional_feedback.txt
zsh streaming_couping/commands_run_v0_temporal_prompt_matrix.txt
zsh streaming_couping/commands_analyze_multiclip_affine_geometry.txt
```

后续实验命令均为冻结 cache 上的诊断或离线评估，不会自动修改正式 V0。实验记录见：

- [当前状态与实验结论](docs/current_status.md)
- [系统 pipeline](docs/system_pipeline.md)
- [语义地图实验路线](docs/experiment_route.md)

当前单序列证据只支持 QK pose 改善；V0 不声明 SAM 改善 pose 或几何。SAM 的正式作用
是实例发现、跨帧 persistent identity 和语义投影。新方法必须在 scene-disjoint 数据上
与 raw V0 按同一协议比较，并报告 tracking、depth、map、恶化帧比例和 fallback/coverage。
