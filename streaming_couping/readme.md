# streaming_couping V0

V0 是冻结、免训练的 StreamVGGT + SAM3.1 语义地图 pipeline：

- QK retrieval只为camera head选择历史，输出相机轨迹；
- full-history raw StreamVGGT world pointmap作为地图几何；
- SAM3.1 persistent mask/slot/track ID作为点云语义；
- SAM不参与pose，QK分支不运行depth/point heads，点云不做后处理优化。

唯一运行入口：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

冻结 V0 之后的只读 pointmap 诊断入口是：

```bash
zsh streaming_couping/commands_v2_pointmap_diagnosis.txt
```

它按 D0（depth × pose 坐标闭环）、D1（区域/置信度误差分解）、T0（独立稀疏三角化）顺序运行，
不会覆盖 V0 的 pose、pointmap 或 semantic map。

命令包含四个阶段：

1. 缓存full-history StreamVGGT几何和SAM3.1多prompt实例轨迹；
2. 使用first-frame anchor + native QK Top-4运行pose-only camera replay；
3. 固定QK pose并完成tracking/pose审计；
4. 将SAM persistent实例标签投影到未修改的raw world pointmap并导出语义地图。

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
```

当前单序列证据只支持QK pose改善（center `10.93%`、rotation `6.15%`）。V0不声明SAM改善
pose或几何；SAM的正式作用是实例发现、跨帧persistent identity和语义投影。完整方法见
[V0方法说明](docs/v0_method.md)。
