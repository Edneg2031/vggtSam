# 当前物体点云独立对齐 Pipeline、实验结果与结论

更新时间：2026-09-09  
当前实现提交：`75e3a5e`

本文总结当前实验主线：保持 HorizonStream 预估的相机位姿不变，只使用 SAM 实例 mask 选出的物体点云进行物体级 6DoF 修正。

## 1. 目标与基本假设

目标不是重新估计整段相机轨迹，而是验证：

> 如果同一个 SAM instance 在不同帧中对应的是同一个静态物体，那么可以用该物体的跨帧点云一致性估计一个小的刚体修正，并只把修正应用到这个物体的点云。

因此当前实验明确区分两种变换：

```text
真实相机位姿：
    p_world = T_raw_camera_to_world[f] * p_camera

实验性的物体点云修正：
    p_aligned_object = C[f, instance] * p_raw_world
```

`C[f, instance]` 是实验用的物体点云修正，不再被解释为真实相机位姿。当前 pipeline 中：

- HorizonStream 的 `raw_camera_to_world` 保持不变；
- 背景点云保持不变；
- 非 SAM 物体点保持不变；
- 同一帧内不同 instance 可以有不同的 6DoF；
- 只有通过 gate 的 instance 才会应用修正；
- 物体指标只统计物体级点云，不把背景点云混入物体级汇总。

## 2. 当前 Pipeline

```text
ScanNet++ RGB 帧
        │
        ├── HorizonStream
        │      ├── depth / pointmap
        │      ├── geometry confidence
        │      ├── intrinsics
        │      └── raw camera-to-world pose
        │
        └── SAM3.1
               ├── prompt mask
               ├── persistent instance ID
               ├── track score
               └── static score
                         │
                         ▼
             mask 内 camera-space 物体点云
                         │
                         ▼
          与该 instance 的历史观察建立最近点对应
                         │
                         ▼
              每个 instance 独立优化一个 6DoF
                         │
                         ▼
          只修正当前帧该 instance 的 object points
                         │
                         ▼
       raw_pose 分支 / object_pose_object_only_per_instance 分支
```

### 2.1 几何和 SAM 输入

当前 ScanNet++ 场景为 `00a231a370`，prompt 为：

```text
bed, wardrobe, chair, rug, dustbin
```

HorizonStream 生成并缓存：

- 每帧深度或点图；
- 几何置信度；
- 相机内参；
- 原始 `camera_to_world`；
- 处理后的 RGB 输入。

SAM3.1 提供 mask、持久 instance ID、track score 和 static score。物体修正不重新改变 SAM 的 ID，也不修改 HorizonStream 内部状态或 KV cache。

### 2.2 观测过滤

当前 100/200/all-frame 命令使用的主要过滤参数为：

| 条件 | 当前值 |
|---|---:|
| 最低 track score | `0.50` |
| static score threshold | `0.20` |
| 最低 geometry confidence | `0.30` |
| 最低 mask 像素数 | `32` |
| 最大 mask 面积比例 | `0.85` |
| 每次观测最多点数 | `256` |
| 每次观测最少有效几何点 | `24` |

默认 `require_static_score=False`。也就是说，缺失 static score 时不会自动过滤；有 score 时才根据 `0.20` 判断静态性。

### 2.3 历史参考点云

- 前 `5` 个输入帧只作为 anchor，不做修正；
- 每个 instance 最多保留 `3` 个 anchor observation；
- 另外使用最近 `2` 个 history observation；
- 当前帧只和同一个 `instance_id` 的历史点云匹配；
- 每个物体独立构造参考，不把同一帧其他物体合并进来。

这与之前的 shared 版本不同。shared 版本把一帧内多个物体的约束合到同一个 6DoF；当前 per-instance 版本为每个物体单独优化。

### 2.4 最近点匹配和损失

当前对应关系不是 GT 对应，而是由预测点云自己产生：

1. 将当前 camera-space 点云用 raw HorizonStream pose 变换到 world frame；
2. 对当前点和历史参考点计算 `torch.cdist`；
3. 采用 mutual nearest neighbor；
4. 只保留距离不超过 `0.25 m` 的对应；
5. 按距离排序后保留较近的 `70%`，至少满足每对 `8` 个匹配；
6. 所有参考对合计至少需要 `16` 个匹配。

优化变量是 6D 增量：3D rotation vector 加 3D translation。它左乘到 raw pose 上：

```text
T_candidate = Exp(delta_6d) * T_raw
```

实际损失为加权的 robust nearest-point loss：

```text
L_match = weighted_mean( Huber(||T_candidate p_current - p_reference||) )
L_total = L_match + 0.02 * L_pose_prior
```

当前参数：

```text
Huber delta                 = 0.05 m
Adam outer iterations       = 4
每次 outer 的 optimizer step = 30
learning rate               = 0.03
pose prior weight           = 0.02
最大 rotation correction   = 10 deg
最大 translation correction = 0.25 m
```

### 2.5 接受条件

候选修正需要满足：

- 最终总匹配数不少于 `16`；
- 保留下来的匹配 pair 数不少于初始 pair 数的 `50%`；
- object loss 相对下降至少 `2%`；
- 每个存活 pair 最少有 `8` 个匹配。

注意：这个 gate 只判断“预测历史点云之间的 loss 是否下降”，不判断 GT 指标是否变好。

### 2.6 点云写回范围

独立模式下，当前帧每个 instance 都有自己的修正矩阵：

```text
object_point_transforms[frame_id][instance_id] -> 4x4
```

映射阶段只对以下点应用矩阵：

```text
static SAM-selected object points of this instance
```

`refined_camera_to_world` 在 independent 模式下仍保存 raw pose，因此评估输出中只有 `raw_pose` 位姿分支。这是有意设计的：物体点云修正不是相机位姿修正。

## 3. 评估方式

物体级评估使用 GT instance mask 和 GT object pointmap，仅在离线评估阶段读取 GT，不参与候选生成和优化。

当前主要指标：

| 指标 | 方向 |
|---|---|
| `object_accuracy_m` | 越低越好 |
| `object_completeness_m` | 越低越好 |
| `fscore_5cm` | 越高越好 |
| `voxel_iou_5cm` | 越高越好 |
| `ghost_point_ratio` | 越低越好 |

日志中的多物体汇总是“匹配物体的 macro mean”，不是整场景点云指标；背景点云没有进入这个 object-only 汇总。`ATE/RPE` 只是额外确认 raw HorizonStream 相机位姿，没有被当前物体修正改变。

评估器对预测 object 和 GT object 使用类别兼容性加 voxel IoU 的一对一 Hungarian assignment。因而：

- 同一 GT 物体不能被多个预测 object 同时正式匹配；
- 未匹配但有点的预测 object 会进入 duplicate/unmatched 统计；
- `object=dustbin` 这一列来自 GT label，当前简洁打印脚本没有同时打印 `predicted_category`。

所以仅凭日志中出现多个 `dustbin`，还不能直接断言都是 SAM 误检；需要查看 `map_objects.csv` 中的 `gt_instance_id`、`predicted_instance_id`、`predicted_category` 和 assignment。若场景 GT 确实只有一个 dustbin，理论上只有一个 GT dustbin 行可以正式匹配，其余应表现为未匹配或 duplicate track。

## 4. 已完成实验结果

### 4.1 可公平比较的 shared object-only 100 帧实验

帧段：`90–189`，共 100 帧。相机位姿仍为 HorizonStream raw，但一帧内所有物体共享一个 6DoF 修正。

| 指标 | raw | shared object-only | 变化 |
|---|---:|---:|---:|
| object accuracy | `0.0735035` | `0.0765191` | 变差 |
| object completeness | `0.0861875` | `0.0829532` | 改善 |
| F5cm | `0.4742214` | `0.4879335` | 改善 |
| ghost ratio | `0.2463200` | `0.2612594` | 变差 |

bed 单独的结果当时比较明显：

| 指标 | raw | shared object-only | 变化 |
|---|---:|---:|---:|
| accuracy | `0.0709000` | `0.0587268` | 改善 |
| completeness | `0.0907613` | `0.0843056` | 改善 |
| F5cm | `0.3974930` | `0.5243894` | 明显改善 |
| ghost ratio | `0.2346191` | `0.1762695` | 改善 |

这个结果说明：只修正物体点云、保持相机位姿不变，确实可以让某些物体（尤其 bed）变好。但 shared 变换是多个物体的折中，不代表每个物体都能同时变好。

### 4.2 当前 per-instance 100 帧实验

帧段：`90–189`，共 100 帧。每个 instance 独立优化 6DoF。

运行摘要：

```text
independent_instance_poses=True
raw_matched_objects=6
accepted_frames=93
object_point_correction_frames=93
object_point_correction_instances=206
pose=raw_pose
ATE_RMSE=0.1192301 m
RPE_t_RMSE=0.0090871 m
RPE_r_RMSE=0.3176797 deg
```

这里的 `object_point_correction_instances=206` 是“帧-实例修正次数”，不是场景中有 206 个不同物体。

匹配物体的 macro mean：

| 指标 | raw | per-instance | 变化 | 结论 |
|---|---:|---:|---:|---|
| object accuracy | `0.0735035` | `0.0708708` | `-0.0026327` | 改善 |
| object completeness | `0.0861875` | `0.0881111` | `+0.0019236` | 变差 |
| F5cm | `0.4742214` | `0.4464289` | `-0.0277925` | 变差 |
| voxel IoU 5cm | `0.1484792` | `0.1376399` | `-0.0108393` | 变差 |
| ghost ratio | `0.2463200` | `0.2362977` | `-0.0100222` | 改善 |

逐物体结果：

| object / predicted ID | F5cm raw → aligned | voxel IoU raw → aligned | ghost raw → aligned | 结论 |
|---|---:|---:|---:|---|
| bed / 0 | `0.397493 → 0.402995` | `0.103926 → 0.087386` | `0.234619 → 0.189941` | 混合；F5cm、ghost 改善 |
| dustbin / 2 | `0.677055 → 0.763275` | `0.200820 → 0.227273` | `0.020384 → 0.020384` | 明显改善 |
| dustbin / 3 | `0.734813 → 0.534510` | `0.243523 → 0.207650` | `0.018066 → 0.025879` | 明显变差 |
| chair / 4 | `0.138428 → 0.136272` | `0.054054 → 0.057834` | `0.500341 → 0.484642` | 混合；整体接近 |
| wardrobe / 5 | `0.818529 → 0.762512` | `0.276576 → 0.233720` | `0.037842 → 0.030273` | 混合；F5cm、IoU 下降 |
| dustbin / 6 | `0.079010 → 0.079010` | `0.011976 → 0.011976` | `0.666667 → 0.666667` | 未改变 |

因此当前 per-instance 版本的准确结论是：

- bed 仍然有小幅提升，但不如 shared 版本明显；
- dustbin ID 2 有明显提升；
- dustbin ID 3 和 wardrobe 的 F5cm/voxel IoU 下降；
- chair 的不同指标方向不一致；
- 汇总后的 F5cm 和 voxel IoU 没有提升。

### 4.3 不可直接作为结论的 first-100 结果

此前 `frame_start=0` 的 first-100 实验只有 `2` 个匹配物体，raw ATE 约 `0.732 m`，与 `90–189` 实验的 `6` 个匹配物体、raw ATE `0.119 m` 不同。因此它不能用于判断方法好坏，也不能和 `90–189` 的结果直接比较。

### 4.4 全量帧实验状态

全量帧命令已经准备并推送：

```bash
zsh streaming_couping/commands_run_scannet_object_pose_loss_object_per_instance_allf.txt
```

配置为：

```text
frame_start=0
frame_stride=1
frame_count=0  # 由输入解析器定义为剩余全部帧
```

输出目录：

```text
/data184/open_source/vggtSam/outputs/semantic_map_allframes_horizonstream_object_pose_loss_object_per_instance_v1
```

截至本文撰写时，全量帧实验还没有新的终端结果。因此不能提前声称 200 帧或全量帧会提升；它只能检验更多时间参考是否改善匹配稳定性。

## 5. 当前结论

### 5.1 已经验证的部分

1. **相机位姿没有被物体修正改变。**  
   per-instance 输出中的 `pose=raw_pose` 与 raw branch 相同，修正保存在 `object_point_corrections.pt`，只作用于物体点云。

2. **每个物体独立 6DoF 的机制已经真正生效。**  
   `independent_instance_poses=True`，且 `object_point_correction_instances=206`；不是把一个 shared 变换错误地广播给所有物体。

3. **物体级修正确实能提升局部物体。**  
   bed 和 dustbin ID 2 的部分指标提升，说明“raw pose + object-only correction”这个方向不是完全无效。

4. **当前 loss 不能保证 GT 点云质量提升。**  
   per-instance 版本的内部历史匹配 loss 可以下降，但 macro F5cm 和 voxel IoU 反而下降。

### 5.2 当前不能得出的结论

不能根据当前一次 100 帧 per-instance 实验得出“独立 6DoF 思路错误”。更准确的说法是：

> 独立 6DoF 已正确实现，但当前的最近点对应、参考点云、接受 gate 和 SAM track 质量还不足以保证每个物体的真实 GT 几何指标提升。

同样，也不能因为日志里有三个 `dustbin` 就直接认定是 SAM 分错。必须查看 GT instance 数量、`predicted_category` 和 duplicate/unmatched 行。

## 6. 表现不稳定的主要原因

### 6.1 优化目标和评价目标不一致

优化使用的是预测历史点云之间的 self-consistency：

```text
当前预测点云 ↔ 历史预测点云
```

评价使用的是：

```text
当前累计点云 ↔ GT 点云
```

如果历史点云已经带有同方向的系统误差，或者多个帧都把背景写进了物体 mask，那么修正可以降低 self-consistency loss，同时离 GT 更远。

### 6.2 最近点对应不一定正确

当前 `max_match_distance_m=0.25` 相对宽。对于小偏移、平面、遮挡或稀疏深度，多个点可能具有相近的最近邻。mutual nearest 和 70% trimming 能减少一部分 outlier，但无法证明保留下来的点是物理上同一个表面点。

这正是当前假设最敏感的地方：

```text
最近点对应正确  →  6DoF 优化可能有效
最近点对应错误  →  loss 仍可能下降，但会把点云带偏
```

### 6.3 SAM mask/track 的错误会被独立 6DoF 放大

同一个 prompt 可能产生：

- 同一物体被拆成多个 track；
- 一个 track 只覆盖物体局部；
- mask 混入墙、地面或其他物体；
- 遮挡后重新出现时 ID 不稳定；
- 同类别多个候选之间发生身份混淆。

shared 6DoF 会把这些错误平均掉一部分；per-instance 6DoF 则给每个错误 track 更大的自由度，容易单独过拟合。

### 6.4 单个物体的几何约束可能退化

每次观测最多 256 点，但经过 mask、confidence、mutual match、距离和 trimming 后，真正参与优化的点可能不多。床、衣柜、墙边等点云经常接近平面，导致某些 rotation/translation 方向不可观测或不稳定。

`16` 个总匹配只说明数量达到下限，不代表三维空间中有足够的非共面约束。

### 6.5 当前接受 gate 偏向训练匹配 loss

当前主要接受标准是相对 loss 改善 `2%`。它没有：

- 独立 hold-out 参考帧验证；
- 最终绝对 RMSE 上限；
- 点云秩/非退化检查；
- 对应点的空间分布检查；
- 预测 mask 污染比例检查；
- 同类重复 track 抑制。

所以一个错误但容易拟合的物体修正也可能被接受。

### 6.6 参考点云可能发生误差传播

前 5 帧 anchor 是固定的，但后续通过 gate 的修正 observation 会进入 history。若某一帧错误修正被接受，后续帧可能继续对齐到这个错误 history，从而形成局部 drift。

### 6.7 F5cm、voxel IoU 与 accuracy 不完全等价

一个修正可能让平均最近距离下降，却让点云整体边界、占用体素或 recall 变差。因此不能只看 accuracy：当前 per-instance 结果正好表现为 accuracy、ghost 改善，但 F5cm、voxel IoU 下降。

## 7. 为什么之前 shared 版本的 bed 更好

shared 版本把一帧内多个物体的观测合并来估计一个 6DoF，具有两个效果：

1. 约束数量更多，优化 basin 更稳定；
2. 多个物体对同一个变换构成隐式正则，减少单个物体退化或错误最近点的影响。

如果 bed 的误差方向与 shared 估计方向一致，它会得到明显改善；但同一个 shared 变换不可能同时适合所有物体，所以其他物体可能下降。

per-instance 版本去掉了这个隐式正则，理论自由度更合理，但对 mask、对应点和 gate 的要求更高。这解释了为什么 bed 仍有提升，而 dustbin ID 3、wardrobe 等物体反而下降。

合理的后续设计不是把 shared 6DoF 作为最终结果广播给所有物体，而是可以把 shared 结果仅用作：

- per-instance ICP/优化的初始化；
- 小范围搜索的中心；
- per-instance correction 的先验或上限；
- 检测同一帧内明显冲突的异常物体。

最终写回仍保持每个物体自己的 correction。

## 8. 当前实验命令和输出

| 用途 | 命令/路径 |
|---|---|
| shared object-only 100 帧 | `streaming_couping/commands_run_scannet_object_pose_loss_object_only_100f.txt` |
| per-instance 100 帧，90–189 | `streaming_couping/commands_run_scannet_object_pose_loss_object_per_instance_100f.txt` |
| per-instance 200 帧，90–289 | `streaming_couping/commands_run_scannet_object_pose_loss_object_per_instance_200f.txt` |
| per-instance 全部帧 | `streaming_couping/commands_run_scannet_object_pose_loss_object_per_instance_allf.txt` |
| 当前 per-instance 评估摘要 | `gt_evaluation/object_pose_loss_object_per_instance_metrics.txt` |
| 逐物体 CSV | `gt_evaluation/object_pose_loss_object_per_instance_metrics.csv` |
| refinement summary | `object_pose_refinement/pose_refinement_summary.json` |
| 逐次优化 trace | `object_pose_refinement/optimization_trace.json` |
| accepted/rejected edges | `object_pose_refinement/accepted_edges.json`、`rejected_edges.json` |
| per-instance correction artifact | `object_pose_refinement/object_point_corrections.pt` |

实际输出根目录取决于实验帧段。例如已完成的 90–189 实验为：

```text
/data184/open_source/vggtSam/outputs/semantic_map_100frames_horizonstream_object_pose_loss_object_per_instance_90_189_v1
```

## 9. 建议的下一步判断顺序

1. 先运行全量帧实验，保持当前参数不变，观察是否仍然只有少数物体受益。
2. 读取 `map_objects.csv`，确认三个 `dustbin` 是三个 GT instance、三个预测 track，还是 GT/预测类别显示混淆。
3. 按 instance 汇总 accepted/rejected correction、最终 RMSE、匹配数和 correction 大小。
4. 对每个 instance 增加 hold-out 历史帧验证：训练匹配 loss 下降但 hold-out 变差时拒绝修正。
5. 收紧最近点 gate，例如先试 `0.05–0.08 m`，并增加绝对 RMSE 和三维非退化检查。
6. 对明显重复的同类 SAM track 做 duplicate suppression，再比较 per-instance 点云指标。

在这些检查完成前，最稳妥的结论是：

> HorizonStream raw pose 应继续作为相机位姿；物体级独立 6DoF 可以作为点云修正实验保留，但当前 loss 和匹配 gate 还不能保证物体点云整体提升。
