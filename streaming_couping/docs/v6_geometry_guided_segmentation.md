# V6：轻量几何辅助 SAM3.1 分割

V6 当前只保留一个部署策略：
`v6_sam31_adaptive_positive_compete_010`。历史参数 sweep 已完成，代码中不再保留
safe、005、015、points-only、负点和旧 late-geometry 分支。V4/V5 继续保留原始 SAM3，
不与 V6 的 cache 或结果混用。

## 1. 数据流

```text
参考帧实例 mask
  → 从 StreamVGGT reference pointmap 取可信 3D 点
  → 用当前 pose + intrinsics 投影到目标帧
  → 当前 pointmap 距离一致性筛选
  → box_mask + positive_mask
  → SAM3.1 text/box 定位，再加入 positive points
  → compete_010 无 GT 门控
  → 最终 mask
```

几何模块只需输出两张图像空间布尔 mask：

- `box_mask`：目标的大致范围；
- `positive_mask`：确认属于目标的稀疏支持点。

因此后续替换 StreamVGGT 时，只需实现相同接口，不必修改 SAM3.1 或门控代码。当前流程
不使用 hidden feature、ICP、多帧 object-map 写回或负点。

## 2. 最终门控

候选分数为：

```text
0.55 × positive-support recall
+ 0.30 × candidate-inside-box precision
+ 0.15 × SAM3.1 score
```

raw mask 明显不可靠时，候选的 support 和总分都至少提高 `0.05`；raw mask 看起来可靠时，
两项都至少提高 `0.10`。候选相对 raw 的面积还必须位于 `[0.5, 4.0]`。任一条件不满足，
或几何提示不存在，就逐帧保留 raw SAM3.1。

该策略在已完成的选择实验中：

- train + validation 加权 IoU delta：`+0.040819769`；
- development：`0 worse_frames`；
- test report-only delta：`+0.020205715`；
- test：`0 worse_frames`。

## 3. 如何进入 V6 主链

最终 mask 同时用于：

- 实例几何与 identity 状态；
- depth、pointmap 和 ray support 采样；
- SAM3.1 `detector_fpn2` appearance pooling；
- 后续 instance token 与 camera/pointmap 实验。

cache 位于 `outputs/streaming_couping_v6_sam31/cache`，并记录 SAM 版本、checkpoint、
分割策略和 appearance source，避免复用旧 SAM3 cache。

## 4. 复现分割

```bash
zsh streaming_couping/commands_v6_geometry_segmentation.txt
```

输出只保留 raw 与最终策略：

```text
outputs/streaming_couping_v6_geometry_segmentation/
├── v6_segmentation_summary.csv
├── v6_segmentation_frames.csv
├── v6_geometry_prompt_diagnostics.csv
├── masks/<clip>.pt
└── visualizations/<clip>/<frame>.png
```

GT 仅用于离线 CSV 指标和可视化，不参与提示、候选排序或回退决策。
