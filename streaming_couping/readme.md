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

普通特征拼接和单实例 ICP 相机修正不单独作为创新点；后者仅保留为诊断基线。
当前重点是“显式 memory 几何对齐 + 可控 residual adapter + 原生 gate”的组合。

## SAM 跟踪只保留两条主线

1. **Hard recovery**：现有 `run_bridge`。当 SAM3 丢失目标时，显式几何候选
   触发 mask 重查询，并可写回同一 `obj_id` 的 memory。该路线已经通过当前
   大视角压力片段，代码保持不变。
2. **Learned §3.1**：`train_geometry_memory_adapter`。StreamVGGT 第
   `4/11/17` 层先做多层 attention merger，再与 SAM3 tracker FPN2 卷积融合，
   生成置信度门控 residual；同时保留显式 memory position warping。融合结果
   完整经过 SAM3 原生 memory attention、mask decoder、object score gate 和
   memory encoder，不使用 hard fallback、重检测或恢复 mask 写回。

§3.1 只训练轻量 adapter，SAM3 与 StreamVGGT 冻结。训练时 GT instance mask
提供 focal/Dice，GT visibility 监督 `object_score_logits` 并用于 teacher forcing；
最终消融推理关闭 teacher forcing，仍使用原生 `object_score_logits > 0` gate。
最终五种模式共享同一个训练后 adapter checkpoint，属于推理时移除/打乱组件的
对照，不是五个分别训练的模型。

主要输出为 `training_history.csv`、`training_curves.png`、
`ablation_summary.csv`、`ablation_frame_metrics.csv` 和 `ablation_report.png`。
完整命令见 `commands.txt`；首次运行可先把 `--iterations 700` 改为 `2` 做冒烟。

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

## 四分支因果消融

`run_recovery_writeback_ablation` 在同一 RGB 序列上分别评估实例，不混合
不同实例的 SAM3 memory。当前主测试使用 cabinet(37) 和 wardrobe(68)，暂不
使用首帧局部可见且面积过大的 bed(54)。

四条分支为：

1. `original`
2. `geometry_recovery_no_memory`
3. `geometry_recovery_same_id_memory`
4. `shuffled_geometry_same_id_memory`

第 2、3 条分支共享逐像素完全相同的全图文本候选 mask，并在相同恢复帧做因果
切分；代码同时检查 no-memory 切分没有改变原 SAM3 预测、恢复前两分支完全一致、
写回后仍使用原 `obj_id`。shuffled 分支固定 reference 几何，只循环打乱后续
StreamVGGT 输出，RGB、SAM3 文本候选和阈值不变。

主要输出为 `summary.csv`、`frame_metrics.csv`、
`candidate_diagnostics.csv`，以及每个实例目录中的
`recovery_writeback_report.png`。候选 CSV 中的 GT IoU 只用于 oracle 诊断，
不参与候选选择。

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

## SAM3 Mask Point-head 点云增量

`run_sam3_mask_camera_refinement` 比较
`GT oracle / SAM3 original / SAM3 hard-memory` 三种实例 mask。当前正式配置使用
StreamVGGT `point_head` 直接预测的世界点云；`depth_camera` 仍可作为相机一致性
诊断。三条分支共享 reference 观测、Sim(3) 和 ICP 参数。几何置信度保留两个
独立入口：`alignment_confidence_threshold` 只固定 reference Sim(3)，
`icp_confidence_threshold` 只筛选 reference/current 实例 ICP 点。

每帧由 mask 选出的静态实例点与 reference 实例点估计 `Delta T`；通过 ICP
门控后，`Delta T` 作用于整帧 pointmap，而不是只移动实例点。当前 point-head
基线保持 `translation_only` 并应用完整 `Delta t_ICP`，目标是点云配准质量。
同步变换后的 camera pose 只作为 proxy 诊断，因为 point-head 与 camera-head
不存在严格解析约束；不能把 point-head ATE 直接解释为真实相机优化。

同一次运行比较两种实例几何目标：`reference_only` 始终对齐首帧实例点；
`causal` 在可靠 ICP 后将修正后的新实例表面体素合并到持久 object map，后续帧
对齐逐步扩展的历史地图。后者不读取后续 GT，且漏检或 ICP 被拒绝时不更新，
用于验证持续实例地图能否缓解首帧只观察到局部表面的问题。

`scene_consistency=guard` 进一步维护只含历史预测的因果场景地图。对实例 ICP
提出的平移依次检查 `0/0.25/0.5/0.75/1.0`，用当前帧非实例高置信点与历史
场景的 trimmed nearest-neighbor RMSE 和 fitness 选择不破坏全局重合度的最大
增量。该 guard 不读取 GT；当场景重叠不足时保留原实例 ICP，避免无证据拒绝。
当前压力片段中该 guard 对所有可见帧均选择完整增量，虽然内部 scene NN 指标
改善，却不能预测 GT 全场景误差，因此仅保留为负结果诊断，不作为正式配置。
正式下一步固定 causal map，扫描 translation `alpha={0,0.25,0.5,0.75,1}`，
检查实例与全场景误差是否存在共同最优点。
为保持单变量，causal object map 在各 alpha 分支中都使用完整实例 ICP 后的点
更新；alpha 只控制该增量应用到整帧 pointmap/pose proxy 的比例。

主要比较 `summary.csv` 中三条分支的 full/object point RMSE 与 Chamfer；ATE
使用固定 reference Sim(3) 后的全部相机中心计算，在 point-head 模式下仅作
辅助诊断，不对 refined 轨迹再次对齐。
`camera_trajectories_*.png` 显示 GT、raw、refined 轨迹，`pointmaps/` 保存各
mask source 和 alpha 的整场景 PLY。GT oracle 是可行性上限；hard-memory 的
后续 mask 不使用 GT。

当前单实例调参模式不再输出 `sam3_original` 分支或逐帧 PLY。GT oracle 只保留
CSV 指标；每个 hard-memory alpha 仅保存 `scene_*.ply` 与 `object_*.ply`，另有
共享的 `scene_raw.ply`、`scene_gt.ply`、`object_raw.ply` 和 `object_gt.ply`。
重跑时会清理该目录中的旧 PLY，避免历史调试文件混入结果。
`mask_sources.png` 仍保留 SAM3 original，按 RGB / GT / SAM3 original /
hard-memory 四列展示，作为恢复前后的直观对照。

## 多实例共享点云增量

`run_multi_instance_camera_refinement` 在同一 RGB 序列上只运行一次 StreamVGGT，
并加载一次 SAM3 模型。每个实例使用独立 tracker session、persistent memory 和
causal object map，避免实例 ID/外观记忆互相污染；几何优化阶段则是联合的：
每帧只在各实例自己的 object map 内建立对应，先做实例内置信度/ICP 门控，再对
各实例建议平移做一致性筛选和实例等权平均，最终只产生一个共享 `Delta T_t`。
默认至少两个实例通过门控才修正整帧 pointmap，不能把三个独立平移依次叠加。

主要输出为 `summary.csv`（共享轨迹/整场景指标）、`instance_summary.csv`
（各实例 mask 与点云指标）、`mask_sources.png`、共享修正后的场景/对象 PLY，
以及 `semantic_map/object_{id}.ply + object_registry.json`。

## Memory Warping 诊断消融

`run_memory_warp_ablation` 只诊断 §3.1 的位置编码分支，不使用几何 fallback、
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

`presence_threshold_sweep.csv` 在不改变 SAM3 memory 的前提下，对所有模式应用
完全相同的 presence 阈值和 `soft_mask_threshold=0.5`。它只用于判断 aligned
增量能否越过统一 gate，同时保持 absent FP；不能在当前序列选择最佳阈值后直接
作为最终结果。阈值 `0.5` 对应 SAM3 原始 `object_score_logits > 0` gate，`0.0`
则展示完全移除 presence gate 时的 soft mask 上限和误检代价。
