# Streaming Coupling Paired Memory Test

该实验只验证一个变量：**共同恢复 mask 是否写入 SAM3 memory**。
SAM3、StreamVGGT 均冻结。恢复模块使用一次单帧 text-conditioned SAM3
重查询，但其临时 ID 会被丢弃，不替换原视频 tracker 的持久实例 ID。
两条分支使用独立但相同初始化的原视频 session，并在恢复帧前后采用相同的
分段传播时序；代码会检查恢复前输出一致，避免共享状态或执行顺序污染对照。

## 共同数据流

```text
RGB 序列 + reference GT instance mask
        |                         |
        v                         v
SAM3 原视频 session        StreamVGGT causal geometry
持久 obj_id                 reference instance world points
        |                         |
        |                  投影并与当前 pointmap 检查
        |                         v
        +-----------> geometry box + 3 个支持正点
                              |
                              v
             text-conditioned SAM3 单帧重查询
                              |
                              v
                    共同 dense recovery mask
                              |
               +--------------+--------------+
               |                             |
          no-memory                    existing obj_id memory
```

只有 reference GT mask 用于初始化。其他帧 GT 只计算指标，不参与候选生成、
恢复帧选择或 SAM3 修正。

## 唯一消融变量

恢复帧之前，两条分支完全相同；恢复帧使用逐像素相同的 dense mask。临时
重查询只负责生成共同观测，不把临时 SAM3 ID 带入任何对照分支。

- `no_memory`：恢复帧显示修正 mask，但不改变未来 tracker state；未来帧使用
  未写回修正的原 SAM3 轨迹。
- `memory`：将同一个 dense mask 通过 SAM3 原生 memory encoder 写入同一
  `obj_id`，并执行 SAM3 原生 existing-object refine 的激活 bookkeeping，
  未来帧从修正后的 memory 继续传播。该步骤不会创建新实例 ID。

代码会检查 recovery frame 的两份 mask 是否逐像素相同。判断 memory 效果只看：

```text
no_memory_post_recovery_iou
memory_post_recovery_iou
```

## 运行

完整服务器命令见 `streaming_couping/commands.txt`。入口为：

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_bridge \
  --config streaming_couping/configs/default.yaml
```

旧 manifest 的 RGB 若仍指向不可读 NAS，先运行 `commands.txt` 中的
`scripts/cache_scannetpp_rgb.py`。`--allow-summary-fallback` 仅适合调试，
不能用于最终定量结果。

## 输出

- `summary.csv`：no-memory、memory 的总体及恢复后指标。
- `frame_metrics.csv`：逐帧候选质量、分支 IoU、score 与恢复帧标记。
- `paired_memory_report.png`：同图逐帧比较 no-memory 与 memory。
- `resolved_config.json`：实际运行配置。

`summary.csv` 还会显式记录 `same_obj_id=1`、
`paired_branch_redetection_used=0`、`paired_causal_split=1`，便于检查对照条件。

## 当前边界

- 当前只测试一次 aligned geometry correction 对后续 memory 的影响。
- 持久实例身份始终由原视频 tracker 的同一个 `obj_id` 维持。
- 临时 text 重查询只生成恢复 mask，不承担持久身份。
- 不包含点云地图更新、相机优化或联合训练。

## GT Mask 几何可行性实验

`run_gt_mask_pose_refinement` 是独立的反向验证，不改变上述 memory 实验。
它冻结 StreamVGGT，只用 GT instance mask 从每帧预测 pointmap 中选出同一
静态物体的点，再用 trimmed point-to-point ICP 求当前帧的 `SE(3)` 修正。
Reference 默认取序列中最早可见帧，确保任意当前帧只使用历史信息；也可用
`--reference-sequence-index` 显式指定，但指定帧必须包含目标实例。

当前默认 `--pose-refinement-mode translation_only`：固定 StreamVGGT 的旋转，
只迭代估计实例点云支持的平移增量。`full_se3` 保留为上一版对照，不作为默认
camera correction。

为消除 StreamVGGT 的任意坐标系和尺度，raw/refined 两条路径共享一次仅由
reference frame 估计的 `Sim(3)`。GT pose 和 GT pointmap 在 ICP 中不使用，
只负责公共坐标对齐和最终评价。

同时保留一个不参与 ICP 的诊断分支：使用 StreamVGGT `depth_head` 的深度和
`camera_head` 的内外参反投影出 camera-consistent pointmap。由于它与
`point_head` 输出处在不同的原生 gauge，该分支也只在同一个 reference frame
独立估计一次 `Sim(3)`，并固定用于全部后续帧。这样比较的是两种几何输出的
跨帧一致性，不会用每帧 GT 做对齐。

主要输出：

- `frame_metrics.csv`：逐帧 ICP、pose、全场景/实例 pointmap 误差。
- `summary.csv`：非 reference 可见帧的 raw/refined 均值。
- `transforms.json`：共享 Sim(3) 与每帧 ICP 修正矩阵。
- `pointmaps/*_streamvggt_native.ply`：模型原生坐标系的 pointmap。
- `pointmaps/*_depth_camera_{native,aligned}.ply`：由 depth head 和 camera head
  反投影得到的 camera-consistent pointmap 诊断分支。
- `pointmaps/*_{raw,refined,gt}.ply`：逐帧和整段对齐后场景点云。
- `pointmaps/*_object.ply`：由 GT instance mask 选出的实例点云。

这里的 `raw` 指经过 point-head 分支公共 Sim(3) 坐标对齐、但没有 ICP 修正的
StreamVGGT pointmap。运行命令见 `streaming_couping/commands.txt`。

## Mask-gated Track BA 消融

`run_mask_track_ba` 继续验证反向的 SAM3 -> geometry 约束。它从 reference
实例 mask 内采样点，用冻结的 StreamVGGT `track_head` 做 reference-current
pairwise 跟踪，并比较三条路径：

- `raw`：冻结的 depth/camera 输出，不优化。
- `unmasked_ba`：所有可靠且在图像内的 track 参与 pose-only BA。
- `mask_gated_ba`：同一批 track 还必须落在当前实例 mask 内。

两条 BA 共享查询点、track、初始 camera、深度和超参数，唯一变量是当前实例
mask 门控。优化变量只是 StreamVGGT 初始位姿附近的小 `SE(3) delta`，损失由
robust 2D 重投影误差和原位姿先验组成；尺度固定，不参与 BA。GT pose 与
pointmap 仍只用于 reference-frame `Sim(3)` gauge 和最终指标。

当前 `track_head` 采用 reference-current pairwise 诊断，不读取未来帧，但它还
不是最终的流式滑窗 BA。只有 oracle GT-mask 消融有效后，才把门控替换为 SAM3
预测 mask 并接入在线窗口。输出包括 `summary.csv`、`frame_metrics.csv`、
`transforms.json`、`pointmaps/*.ply` 和 `track_visualizations/*.png`。

`--track-score-mode threshold` 使用 StreamVGGT visibility/confidence；`ignore`
只用于诊断坐标是否有效，并让两条 BA 同时忽略这些分数，不能作为最终方法。
