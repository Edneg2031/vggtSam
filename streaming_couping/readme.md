# streaming_couping V0

V0 是冻结、免训练的 StreamVGGT + SAM3.1 语义地图 pipeline：

- QK retrieval只为camera head选择历史，输出相机轨迹；
- full-history raw StreamVGGT world pointmap作为地图几何；
- SAM3.1 persistent mask/slot/track ID作为点云语义；
- SAM不参与pose，QK分支不运行depth/point heads，点云不做后处理优化。

模型主运行入口：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

命令包含四个阶段：

1. 缓存full-history StreamVGGT几何和SAM3.1多prompt实例轨迹；
2. 使用first-frame anchor + native QK Top-4运行pose-only camera replay；
3. 固定QK pose并完成tracking/pose审计；
4. 将SAM persistent实例标签投影到未修改的raw world pointmap并导出语义地图。

辅助入口：

```bash
zsh streaming_couping/commands_plot_v0_poses.txt
zsh streaming_couping/commands_semantic_mapping.txt
zsh streaming_couping/commands_semantic_mapping_v1.txt
zsh streaming_couping/commands_semantic_mapping_v1_evaluation.txt
zsh streaming_couping/commands_check_scannetpp_data.txt
zsh streaming_couping/commands_audit_scannetpp_processed.txt
```

其中 `commands_semantic_mapping.txt` 只读取已经完成的 V0/V6 cache，离线比较 raw SAM、V6、
GT-mask oracle 和 object-memory map-write gate，不重新运行模型。

`commands_semantic_mapping_v1.txt` 在同一 frozen cache 上导出 V1 persistent 3-D instance memory
语义地图；`commands_semantic_mapping_v1_evaluation.txt` 比较 V0 track identity 与 V1 object
identity，并输出 fragmentation、merge error、re-entry 和 object-level map 指标。两条 V1 命令都
不重新运行 StreamVGGT/SAM3。

主要输出：

```text
outputs/streaming_couping_v0/
├── baseline_summary.json
├── poses.pt
├── qk_pose_retrieval/
│   ├── qk_pose_output.pt
│   └── qk_pose_summary.json
└── semantic_map/
    ├── semantic_map.pt
    ├── semantic_map.ply  # 不同persistent instance使用固定不同颜色
    ├── rgb_map.ply       # 原始RGB外观
    ├── semantic_map_summary.json
    └── tracks.csv

V1 导出写入 `outputs/streaming_couping_v1/semantic_map/`，额外包含
`object_memory.json`、`object_memory.csv`、`association_events.csv`，并在
`semantic_map.pt` 中同时保存 `sam_track_ids` 和 `persistent_object_ids`。
```

绘制 V0 raw pose 与 QK pose 的相机轨迹、GT 对比和逐帧误差：

```bash
zsh streaming_couping/commands_plot_v0_poses.txt
```

结果写入 `outputs/streaming_couping_v0/pose_plots/`：
`pose_trajectory.png`、`pose_trajectory.svg`、`pose_metrics.csv` 和
`pose_plot_summary.json`。图中的 GT 会使用 V0 cache 的
`point_alignment_scale` 转换到 StreamVGGT native gauge；这只用于可视化，
不会修改 V0 的 pose artifact。

当前单序列证据只支持QK pose改善（center `10.93%`、rotation `6.15%`）。V0不声明SAM改善
pose或几何；SAM的正式作用是实例发现、跨帧persistent identity和语义投影。

当前系统与实验路线：

- [系统 pipeline](docs/system_pipeline.md)
- [语义地图实验路线](docs/experiment_route.md)

仓库只保留当前主线和必要的只读数据检查入口。已停止的 residual、Direct PointHead、R3、
temporal residual、Stage2 训练及其命令/配置已经移除，不应再从旧输出目录继续启动。
