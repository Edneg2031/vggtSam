# V6：StreamVGGT 几何前置引导 SAM3.1 分割

V6 当前研究目标改为：先把 StreamVGGT 几何转换成空间提示，再让 SAM3.1 生成 mask。旧的
V6 camera-token overfit 代码仅作为历史机制消融保留，不再代表当前 V6 主线；V4/V5 不改。

## 1. 数据流

```text
参考帧 mask
    ↓ 在 StreamVGGT pointmap 中采样
reference-only object point map
    ↓ 使用当前 StreamVGGT pose + intrinsics 投影
projected points + current-pointmap support
    ↓ robust geometry box
SAM3.1(text + geometry box)
    ├── geometry-box mask
    └── 同一 session 内使用几何正/负点细化选中实例
    ↓ 双向二维覆盖 / 可选三维注册
接受候选，否则逐帧回退 raw SAM3.1 mask
```

几何 box 在 mask 生成前进入 `add_prompt`。SAM3.1 不允许 point 与 text/box 在同一次
调用中混合，因此 point 分支严格遵循官方交互协议：先用 `text + geometry box` 生成并
按几何选择实例 ID，再在同一 session 中对该 ID 添加 point prompt。

- 正点来自“参考物体投影且得到当前 StreamVGGT pointmap 支持”的像素；
- 负点来自“参考物体发生投影、但被当前 pointmap 否定”的像素，并排除正点邻域；
- 默认使用官方 `sam3.1_multiplex.pt`；`use_fa3=false` 以兼容共享的 24 GiB Ampere GPU。

## 2. 安全约束

- 后续帧 GT mask 和 GT pose 不参与 prompt、候选排序或回退；
- 后续预测 mask 暂不更新 object point map，避免错误身份污染几何记忆；
- 几何不可见、投影被拒绝、SAM3.1 无候选或候选不达阈值时，逐帧回退 raw SAM3.1；
- V6-3D 只有在 translation-only 3D registration 被接受时才替换 raw mask；
- 参考帧仍沿用现有实验协议中的 GT instance mask，尚不是完全自动初始化。
- 配置中某个实例若在参考帧不可见，会保留为空槽且不进入汇总指标；同一 clip 中其他
  参考帧可见实例仍正常运行。

## 3. 一次运行的消融

| variant | 含义 |
|---|---|
| `raw_sam31` | 原始参考 mask + SAM3.1 标准 tracking |
| `current_sam3_late_geometry` | 当前缓存中的 SAM3 后验几何恢复结果 |
| `v6_sam31_geometry_box` | StreamVGGT geometry box 前置提示 SAM3.1 |
| `v6_sam31_geometry_points` | box 定位后，用几何正/负点细化 |
| `v6_sam31_geometry_box_3d` | geometry-box mask 再通过三维注册确认 |
| `v6_sam31_geometry_points_3d` | point-refined mask 再通过三维注册确认 |

候选排序不使用 GT。二维版本综合：

```text
0.55 × projected-support recall
+ 0.30 × candidate-inside-geometry-box precision
+ 0.15 × SAM3 score
```

三维版本综合：

```text
0.35 × projected-support recall
+ 0.20 × box precision
+ 0.35 × registration score
+ 0.10 × SAM3 score
```

其中 `box precision` 会惩罚面积很大、只覆盖少量投影点的相似外观错误 mask；这正是旧的
单向 coverage 排序缺少的约束。

## 4. 运行与输出

```bash
zsh streaming_couping/commands_v6_geometry_segmentation.txt
```

不训练 camera 或 pointmap head。命令先校验并复用已有 StreamVGGT feature cache，然后
运行 raw SAM3.1、geometry-box SAM3.1 和 geometry-point SAM3.1。缓存中的
`current_sam3_late_geometry` 仍来自原 SAM3，是旧方法对照，不应与 raw SAM3.1 混称。

```text
outputs/streaming_couping_v6_geometry_segmentation/
├── v6_segmentation_summary.csv
├── v6_segmentation_frames.csv
├── v6_geometry_prompt_diagnostics.csv
├── masks/<clip>.pt
└── visualizations/<clip>/<frame>.png
```

短表主要看：

- `mean_iou_visible`、`mean_precision_visible`、`mean_recall_visible`；
- `mean_iou_delta_from_raw`；
- `improved_frames` 与 `worse_frames`；
- GT 不可见时的 `false_positive_absent_frames`。

可视化按列显示 RGB、GT 和上述六个 variant。

## 5. 第一轮判定

第一轮不调单帧阈值，先判断路径：

1. box 分支若提高 precision 和 IoU，说明“几何先提示、SAM3.1 再分割”有效；
2. point 分支进一步改善，说明 pointmap 的支持/否定关系可作为有效交互提示；
3. point 分支退化而 box 有效，优先检查负点是否落在真实物体或遮挡边界；
4. 2D 提高而 3D 版本大量回退，说明旧 3D registration 对局部可见仍过严；
5. 当前 clip 有效后，再研究安全的多帧 object-map 更新和 SAM3.1 memory writeback。
