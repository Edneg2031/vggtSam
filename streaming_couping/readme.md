# Streaming Coupling Paired Memory Test

该实验只验证一个变量：**共同恢复 mask 是否写入 SAM3 memory**。
SAM3、StreamVGGT 均冻结。恢复模块使用一次单帧 text-conditioned SAM3
重查询，但其临时 ID 会被丢弃，不替换原视频 tracker 的持久实例 ID。
两条分支使用独立但相同初始化的原视频 session，并在恢复帧前后采用相同的
分段传播时序；代码会检查恢复前输出一致，避免共享状态或执行顺序污染对照。

## 研究定位与创新点

目标是构建**提示驱动的流式 3D 实例建图闭环**，而不是简单拼接两个 backbone：

- 当 SAM3 在大视角或重现帧失效时，用 StreamVGGT 历史实例几何重查询目标，
  并恢复原 persistent `obj_id`，不创建替代身份。
- 将同一恢复观测分别写入 SAM3 tracking memory 和 3D instance memory，并用
  mask、深度、几何与时序置信度控制每个实例的独立更新，抑制错误累积。
- 输出持续一致的 2D mask、实例身份与对象级 3D 地图，形成 geometry -> SAM3
  恢复和 SAM3 -> geometry 融合的双向在线协作。

普通特征融合和单实例 ICP 相机修正不作为创新点；后者仅保留为诊断基线。

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

同时运行独立的 camera-consistent 消融分支：使用 StreamVGGT `depth_head` 的
深度和 `camera_head` 的内外参反投影 pointmap，再用相同 GT instance mask 和
ICP 参数估计自己的相机增量。由于它与 `point_head` 输出处在不同的原生 gauge，
两条分支分别只在同一个 reference frame 估计一次 `Sim(3)`，并固定用于全部
后续帧；二者不共享 ICP delta，也不会用后续帧 GT 做坐标对齐。

主要输出：

- `frame_metrics.csv`：逐帧 ICP、pose、全场景/实例 pointmap 误差。
- `summary.csv`：非 reference 可见帧的 raw/refined 均值。
- `camera_trajectories.png`：point-head ICP 的 GT、raw、refined 轨迹。
- `camera_trajectories_depth_camera.png`：depth-camera ICP 的独立轨迹；绿色箭头
  均表示各分支自己的 raw -> refined 相机中心变化。
- `transforms.json`：共享 Sim(3) 与每帧 ICP 修正矩阵。
- `pointmaps/*_streamvggt_native.ply`：模型原生坐标系的 pointmap。
- `pointmaps/*_depth_camera_{native,aligned}.ply`：由 depth head 和 camera head
  反投影得到的 camera-consistent pointmap。
- `pointmaps/*_depth_camera_refined.ply`：由实例 ICP 相机增量一致修正后的
  depth-camera pointmap。
- `pointmaps/*_{raw,refined,gt}.ply`：逐帧和整段对齐后场景点云。
- `pointmaps/*_object.ply`：由 GT instance mask 选出的实例点云。

这里的 `raw` 指经过 point-head 分支公共 Sim(3) 坐标对齐、但没有 ICP 修正的
StreamVGGT pointmap。运行命令见 `streaming_couping/commands.txt`。

## SAM3 Mask 实例点云融合

`run_sam3_mask_object_fusion` 比较三种 mask：`GT oracle`、原始 SAM3、几何恢复
并写入 memory 后的 SAM3。三条分支共享 reference GT mask、StreamVGGT 几何、
reference-frame Sim(3) 和 ICP 参数。局部 ICP 只修正 mask 选中的实例点，不修改
相机和场景其他点，从而避免单物体约束破坏全局重建。

主要看 `summary.csv` 的 cross-view mask IoU、实例 Chamfer 及其改善量；
`pointmaps/object_*_refined.ply` 用于检查实例融合是否仍有重影。旧的
`run_gt_mask_pose_refinement` 保留为相机修正失败诊断和回退基线。
若序列没有通过门控的恢复帧，实验不会强制制造恢复；geometry-memory 分支与
original 分支保持相同，并在 `summary.csv` 记录 `recovery_triggered=0`。

## 3D-aware Memory Warping 独立消融

`run_memory_warp_ablation` 单独验证 `method.md` §3.1，不使用几何 fallback、
重检测、恢复 mask 写回或相机优化。首帧 GT mask 只初始化同一个 SAM3
`obj_id`，后续 GT 仅评价。

StreamVGGT 的 `depth_head + camera_head` 先生成 camera-consistent 3D 点；历史
SAM3 memory token 的位置经“历史像素 -> 3D -> 当前视角”重投影，再用投影位置
采样当前网格位置编码，替换原 `maskmem_pos_enc`。memory feature、object pointer、
memory attention 和 mask decoder 都保持 SAM3 原实现。

四条对照为：

- `original`：原始 SAM3。
- `identity`：安装同一个 hook，但不改变位置编码；必须与 original 完全一致。
- `aligned`：使用时序正确的 StreamVGGT 几何。
- `shuffled`：循环打乱几何帧对应关系，排除“任意扰动都有效”。

主要比较 `summary.csv` 的 `cross_view_iou`、`cross_view_recall` 和
`absent_fp_ratio`。只有 `identity == original`，且 aligned 稳定优于 original
和 shuffled，才能支持“几何对齐 memory 位置先验有效”。逐帧结果和 warp 有效
比例分别写入 `frame_metrics.csv`、`memory_warp_pairs.csv`，总览图为
`memory_warp_report.png`。运行命令见 `streaming_couping/commands.txt`。

当最终 mask 仍为空时，额外按以下顺序诊断：

1. `geometry_projection_metrics.csv` 和 `geometry_projection_report.png`：只把
   reference GT 物体 token 投到后续帧，GT 仅评价投影命中率、覆盖率和 IoU。
2. `frame_metrics.csv` 和 `soft_response_report.png`：读取 SAM3 硬 presence gate
   之前的 decoder mask、`object_score_logits`、GT/背景响应差和 soft IoU。

若 aligned 投影命中高但 `presence_logit <= 0` 或 soft margin 不提高，问题在
SAM3 对几何位置先验的消费方式；若 aligned 投影本身不优于 shuffled，则先修正
StreamVGGT 几何/坐标变换。soft 诊断要求 `soft_capture_count=1`，以保证它对应
当前单实例 tracker；identity 的 hard/soft 输出都必须与 original 相同。
