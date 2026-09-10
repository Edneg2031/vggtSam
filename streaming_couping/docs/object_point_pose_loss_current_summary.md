# HorizonStream + SAM3.1 物体点云对齐实验总结

更新时间：2026-09-10
代码分支：`main`  参考提交：`f3509e7`

## 摘要

本阶段验证的问题是：

> 在保持 HorizonStream 原始相机位姿不变的情况下，能否利用 SAM3.1 的物体 mask 和跨帧物体匹配，把同一物体的点云对齐得更紧，从而提升物体级点云质量？

当前得到的最准确结论是：

> 物体级 6DoF 点云修正已经真正生效，并且可以改善部分物体；但是在当前 ScanNet++ 场景和当前 SAM 跟踪、最近点匹配及接受规则下，整体物体点云指标没有稳定提升。100 帧片段中曾出现局部增益，300 帧连续实验反而进一步暴露了长序列中的跟踪、误差累积和参考污染问题。因此目前只能说方法“局部有效”，还不能宣称 SAM 能够可靠地提升所有物体的点云质量。

本实验不是闭环位姿估计实验，也不训练模型。当前最终目标是“只修正物体点云”，而不是修改 HorizonStream 的相机轨迹。

## 1. 实验问题与假设

### 1.1 目标

对同一个被 SAM3.1 跟踪的静态物体，将不同帧中由 HorizonStream 生成的物体点云变换到统一世界坐标系，再用跨帧点云一致性估计一个小刚体修正。修正只作用于该物体的点云。

### 1.2 最近点假设

当前方法假设：当相邻帧之间的运动足够小、SAM mask 确实对应同一个物体时，预测点云之间的最近点大概率对应同一物理表面。于是可以使用最近点残差优化小幅 6DoF 修正。

这个假设只有在以下条件同时成立时才可靠：

```text
同一物体身份正确
        + mask 没有明显混入背景
        + 相邻帧位移较小
        + 点云具有足够的三维约束
        + 最近点对应落在同一表面
```

如果最近点对应错误，优化 loss 仍然可能下降，但修正后的点云可能离 GT 更远。

## 2. 当前 Pipeline

```text
ScanNet++ RGB 帧
        │
        ├── HorizonStream
        │      ├── depth / pointmap
        │      ├── geometry confidence
        │      ├── camera intrinsics
        │      └── raw camera pose
        │
        └── SAM3.1
               ├── text-prompted object masks
               ├── persistent instance IDs
               ├── track score
               └── static score
                         │
                         ▼
             mask 内的 camera-space 物体点云
                         │
                         ▼
          与同一 instance 的历史观察建立最近点对应
                         │
                         ▼
        shared 或 per-instance 的 6DoF loss optimization
                         │
                         ▼
          只变换通过 gate 的物体点云
                         │
                         ▼
          raw object map vs aligned object map
```

### 2.1 几何输入

HorizonStream 先独立生成并缓存每帧几何，物体对齐阶段复用该缓存。相机位姿和背景几何不因 SAM 修正而改变。

当前主要场景是 ScanNet++：

```text
scene = 00a231a370
```

### 2.2 SAM3.1 的 prompt 方式

当前 object-pose-loss 主命令使用文本 prompt：

```text
bed, wardrobe, chair, rug, dustbin
```

因此当前主实验只会请求这些类别，不是完全无 prompt 的开放世界自动检测。代码中另有基于规则网格正点的 class-agnostic visual-point proposal 分支，但它是独立的 SAM 自动提议诊断，没有接入本总结中的主要 object-pose-loss 结果。

普通的 `SAM31SegmentationAdapter` 在 RGB 模式下要求至少提供一个文本 prompt。SAM3.1 负责生成 mask 和跟踪 ID；它本身并不负责通过当前 loss 自动纠正 HorizonStream 的相机位姿。

### 2.3 物体点云修正的坐标含义

HorizonStream 模型输出 convention 是 `world_to_camera`；几何接口内部使用 `camera_to_world` 将 camera-space 点变换到世界坐标：

```text
p_world = T_raw_camera_to_world[f] · p_camera
```

独立物体模式中，每个帧-实例有自己的点云修正：

```text
p_aligned_object[f, i] = C[f, i] · p_raw_world[f, i]
```

其中 `C[f, i]` 是实验性的物体点云变换，不应解释为真实相机位姿。当前 independent 分支中：

- HorizonStream 的 raw camera pose 保持不变；
- `pose=raw_pose` 仍是唯一的相机位姿分支；
- 背景点云不变；
- 非目标物体点不变；
- 同一帧内不同 instance 可以使用不同的 6DoF；
- 只有通过接受条件的 object points 才应用修正。

### 2.4 历史参考与优化

当前主要设置如下：

| 参数 | 值 |
|---|---:|
| anchor 帧数 | `5` |
| 每个 instance 最多 anchor observation | `3` |
| 每个 instance 最多 recent history observation | `2` |
| 每次观测最多点数 | `256` |
| 每次观测最少有效点数 | `24` |
| 最低 track score | `0.50` |
| 最低 geometry confidence | `0.30` |
| 最低 mask 像素数 | `32` |
| 最大 mask 面积比例 | `0.85` |
| 最大匹配距离 | `0.25 m` |
| 最近点保留比例 | `70%` |
| 每个 pair 最少匹配数 | `8` |
| 总匹配数下限 | `16` |
| 最大 rotation correction | `10°` |
| 最大 translation correction | `0.25 m` |

优化变量是 rotation vector 加 translation 的 6D 增量。候选变换左乘 raw pose：

```text
T_candidate = Exp(delta_6d) · T_raw
```

损失由 robust nearest-point loss 和 raw pose prior 组成：

```text
L = weighted_Huber(nearest_point_residual) + 0.02 · pose_prior
```

需要强调：这个 loss 比较的是预测点云之间的 self-consistency，不是直接比较 GT 点云。

### 2.5 shared 与 per-instance 的区别

| 模式 | 6DoF 数量 | 修正范围 | 主要问题 |
|---|---|---|---|
| shared object-only | 每帧一个 | 当前帧所有通过筛选的物体点 | 多个物体互相折中 |
| per-instance object-only | 每帧每个 instance 一个 | 当前 instance 的物体点 | 自由度高，容易被错误 track 或错误对应带偏 |

per-instance 模式不是为所有物体估计真实相机 pose，而是分别修正每个物体的点云。

## 3. 评价方式

物体级评价只在离线阶段读取 GT instance mask 和 GT object point cloud；GT 不参与 SAM proposal、track、最近点匹配或优化。

主要指标：

| 指标 | 趋势 | 含义 |
|---|---|---|
| `object_accuracy_m` | 越低越好 | 预测点到 GT 的平均距离 |
| `object_completeness_m` | 越低越好 | GT 到预测点的平均距离 |
| `fscore_5cm` | 越高越好 | 5 cm 阈值下的 precision/recall 综合指标 |
| `voxel_iou_5cm` | 越高越好 | 5 cm 体素占用重合度 |
| `ghost_point_ratio` | 越低越好 | 物体点中落在错误区域的比例 |

日志中的多物体汇总通常是 matched objects 的 macro mean，不是全场景点云指标。评价时 raw 和 aligned 使用同一评价范围，不能只看一个指标判断改进。

`predicted_instance_id` 来自 SAM tracking。出现多个 `dustbin` 标签不等于场景中有多个真实 dustbin；需要结合 `map_objects.csv` 中的 `gt_instance_id`、`predicted_instance_id`、`predicted_category` 和 assignment 判断是否是重复 track、类别显示或匹配错误。

## 4. 已完成实验结果

### 4.1 V1 object point alignment：没有形成有效修正

设置：100 帧，`frame_start=90`，raw pose 用于两个分支。

V1 输出显示：

```text
mean_correction_rotation_deg = 0.0
mean_correction_translation_m ≈ 5.8e-17
max_correction_translation_m  ≈ 3.2e-16
```

raw 和 V1 aligned 的所有点云指标完全相同：

| 指标 | raw | V1 aligned |
|---|---:|---:|
| accuracy | `0.0882103` | `0.0882103` |
| completeness | `0.0718032` | `0.0718032` |
| F5cm | `0.3480705` | `0.3480705` |
| voxel IoU | `0.1063202` | `0.1063202` |
| ghost ratio | `0.1846276` | `0.1846276` |

因此这次 V1 run 实际没有应用有效点云变换，不能作为“V1 点云提升”的证据。更可能是 V1 artifact 与当前 raw-pose/object-only 接口之间没有产生可用 correction。

### 4.2 shared object-only loss：bed 有明显局部提升

设置：100 帧，`frame_start=90`，`frame_stride=1`。保持 raw 相机位姿，只对目标物体点应用一帧共享的 6DoF。

整体 matched-object macro mean：

| 指标 | raw | shared aligned | aligned - raw | 趋势 |
|---|---:|---:|---:|---|
| accuracy | `0.0735035` | `0.0765191` | `+0.0030156` | 变差 |
| completeness | `0.0861875` | `0.0829532` | `-0.0032343` | 改善 |
| F5cm | `0.4742214` | `0.4879335` | `+0.0137121` | 改善 |
| ghost ratio | `0.2463200` | `0.2612594` | `+0.0149395` | 变差 |

bed 的结果：

| 指标 | raw | shared aligned | 趋势 |
|---|---:|---:|---|
| accuracy | `0.0709000` | `0.0587268` | 改善 |
| completeness | `0.0907613` | `0.0843056` | 改善 |
| F5cm | `0.3974930` | `0.5243894` | 明显改善 |
| ghost ratio | `0.2346191` | `0.1762695` | 改善 |

这说明“保持 raw pose、只改物体点云”确实能让部分物体变好。但 shared correction 是多个物体的折中，不能保证所有物体同时改善。

### 4.3 V3 instance point alignment：内部 loss 下降，但 GT 指标几乎不改善

设置：100 帧，`frame_start=90`，raw HorizonStream pose 用于两个分支；背景和全场景几何保持不变，仅修改 object points。

运行审计：

```text
camera_pose_modified=False
full_scene_geometry_modified=False
object_points_modified=True
decision_count=294
bootstrap_count=9
accepted_count=78
rejected_count=188
mean_initial_rmse_m=0.0369341
mean_final_rmse_m=0.0108241
mean_relative_improvement=0.5855030
```

整体 object-map 结果：

| 指标 | raw | aligned | aligned - raw | 趋势 |
|---|---:|---:|---:|---|
| accuracy | `0.0882103` | `0.0886103` | `+0.0004000` | 变差 |
| completeness | `0.0718032` | `0.0719695` | `+0.0001662` | 变差 |
| F5cm | `0.3480705` | `0.3484249` | `+0.0003546` | 极小改善 |
| voxel IoU | `0.1063202` | `0.1050106` | `-0.0013097` | 变差 |
| ghost ratio | `0.1846276` | `0.1860131` | `+0.0013856` | 变差 |

选定的 3 个 aligned objects 的 macro mean：

| 指标 | raw | aligned | aligned - raw |
|---|---:|---:|---:|
| accuracy | `0.0917350` | `0.0924016` | `+0.0006666` |
| completeness | `0.0533775` | `0.0536547` | `+0.0002772` |
| F5cm | `0.5477839` | `0.5486108` | `+0.0008269` |
| voxel IoU | `0.1676975` | `0.1646416` | `-0.0030559` |
| ghost ratio | `0.2465198` | `0.2497528` | `+0.0032330` |

结论是：算法内部的对齐 RMSE 明显下降，但 GT 几何质量没有同步改善。这是“self-consistency loss 下降不等于真实点云质量提升”的直接例子。

### 4.4 per-instance object-only loss：部分物体提升，整体 F5cm 下降

设置：100 帧，`frame_start=90`，`frame_stride=1`，每个 persistent SAM instance 独立优化一个 6DoF；相机位姿仍为 raw。

运行摘要：

```text
independent_instance_poses=True
raw_matched_objects=6
common_matched_objects=6
accepted_frames=93
object_point_correction_frames=93
object_point_correction_instances=206
pose=raw_pose
```

`object_point_correction_instances=206` 表示帧-实例修正次数，不是场景中存在 206 个物体。

整体 matched-object macro mean：

| 指标 | raw | per-instance aligned | aligned - raw | 趋势 |
|---|---:|---:|---:|---|
| accuracy | `0.0735035` | `0.0708708` | `-0.0026327` | 改善 |
| completeness | `0.0861875` | `0.0881111` | `+0.0019236` | 变差 |
| F5cm | `0.4742214` | `0.4464289` | `-0.0277925` | 变差 |
| voxel IoU | `0.1484792` | `0.1376399` | `-0.0108393` | 变差 |
| ghost ratio | `0.2463200` | `0.2362977` | `-0.0100222` | 改善 |

逐物体结果：

| 预测 track / 显示类别 | F5cm raw → aligned | voxel IoU raw → aligned | ghost raw → aligned | 简要判断 |
|---|---:|---:|---:|---|
| `0 / bed` | `0.397493 → 0.402995` | `0.103926 → 0.087386` | `0.234619 → 0.189941` | F5cm、ghost 改善，IoU 下降 |
| `2 / dustbin` | `0.677055 → 0.763275` | `0.200820 → 0.227273` | `0.020384 → 0.020384` | 明显改善 |
| `3 / dustbin` | `0.734813 → 0.534510` | `0.243523 → 0.207650` | `0.018066 → 0.025879` | 明显变差 |
| `4 / chair` | `0.138428 → 0.136272` | `0.054054 → 0.057834` | `0.500341 → 0.484642` | 指标方向混合 |
| `5 / wardrobe` | `0.818529 → 0.762512` | `0.276576 → 0.233720` | `0.037842 → 0.030273` | F5cm、IoU 下降 |
| `6 / dustbin` | `0.079010 → 0.079010` | `0.011976 → 0.011976` | `0.666667 → 0.666667` | 没有变化 |

这个结果是目前最接近目标的实验。它证明：

- 独立 per-instance 6DoF 确实被应用到物体点云；
- bed 和 track 2 的部分指标有增益；
- 但是 track 3、wardrobe 等物体明显下降；
- 六个匹配物体的总体 F5cm 和 voxel IoU 下降。

### 4.5 per-instance object-only loss：300 帧连续实验

设置：前 300 帧，`frame_start=0`，`frame_stride=1`，每个 persistent SAM instance 独立优化一个 6DoF；相机位姿仍为 raw。该实验已经成功完成，包括长序列 GT evaluation。

运行摘要：

```text
independent_instance_poses=True
raw_matched_objects=2
common_matched_objects=2
accepted_frames=266
object_point_correction_frames=266
object_point_correction_instances=687
pose=raw_pose
```

整体 matched-object macro mean：

| 指标 | raw | per-instance aligned | aligned - raw | 趋势 |
|---|---:|---:|---:|---|
| accuracy | `0.2079359` | `0.2097914` | `+0.0018556` | 变差 |
| completeness | `0.2766066` | `0.2946774` | `+0.0180708` | 变差 |
| F5cm | `0.0410128` | `0.0235497` | `-0.0174631` | 明显变差 |
| voxel IoU | `0.0079089` | `0.0086149` | `+0.0007060` | 极小改善 |
| ghost ratio | `0.8160400` | `0.8259277` | `+0.0098877` | 变差 |

逐物体结果：

| 预测 track / 显示类别 | F5cm raw → aligned | voxel IoU raw → aligned | ghost raw → aligned | 简要判断 |
|---|---:|---:|---:|---|
| `0 / bed` | `0.022955 → 0.020802` | `0.004389 → 0.005639` | `0.829346 → 0.832275` | F5cm、ghost 变差，IoU 极小上升 |
| `7 / chair` | `0.059071 → 0.026297` | `0.011429 → 0.011591` | `0.802734 → 0.819580` | F5cm、ghost 明显变差 |

300 帧结果说明：

- 300 帧中有 `266` 个帧级修正被接受，但最终只有 `2` 个 matched objects；高 accepted count 不能说明 SAM instance 身份一直正确。
- aligned 分支只有 voxel IoU 出现极小的数值上升，accuracy、completeness、F5cm 和 ghost ratio 全部变差；从整体几何质量看，这是失败结果。
- bed 的 F5cm 从 `0.022955` 降到 `0.020802`，chair 的 F5cm 从 `0.059071` 降到 `0.026297`，说明更长历史没有自动解决错误匹配问题。
- 与 `90–189` 的 100 帧结果相比，300 帧实验还改变了时间段和 matched-object 数量，因此不能把两者的绝对指标直接当作严格的帧数 ablation。

这次实验没有支持“连续帧越多，最近点匹配就越可靠”。更可能的现象是：在当前实现中，长序列增加了可用观测，但也增加了 track identity switch、错误 history 写入、点云污染和累积误差的机会。

### 4.6 first-100 与 90–189 不是同一个实验条件

曾经运行过 `frame_start=0` 的 100 帧实验，只有 2 个 matched objects，raw ATE 约为 `0.732 m`。而 `frame_start=90` 的 100 帧实验有 6 个 matched objects，raw ATE 约为 `0.119 m`。

两者的场景时间段、跟踪稳定性和几何质量不同，不能直接比较，也不能把 first-100 的结果当作方法整体性能。

## 5. 相关但不属于主目标的位姿实验

### 5.1 在线 object pose loop：loss 降低但相机位姿和地图变差

这是一个较早的共享位姿在线修正实验，不是当前的 object-only 方案。100 帧、`frame_start=90` 的结果：

| 指标 | raw pose | object-pose refined |
|---|---:|---:|
| ATE RMSE | `0.1192303 m` | `0.1243705 m` |
| RPE translation | `0.0090871 m` | `0.0095768 m` |
| RPE rotation | `0.317706°` | `0.3773099°` |
| map F5cm | `0.3480705` | `0.3378542` |
| map voxel IoU | `0.1063202` | `0.1061097` |

虽然 object loss 从 `0.00015093` 降到 `0.00007921`，但 GT pose 和地图指标变差。这进一步说明，不能用优化 loss 单独代表真实质量。

### 5.2 GT feedback POC：验证的是外部位姿累积器，不是 SAM

另一个 50 帧、修正帧 `t=15` 的 GT correction POC 不使用 SAM。它用于验证：如果已知 GT pose，外部 pose accumulator 是否能把 correction 传递到后续绝对位姿。

| 分支 | ATE RMSE | RPE translation | RPE rotation |
|---|---:|---:|---:|
| Raw | `0.057001 m` | `0.005613 m` | `0.125996°` |
| Post-hoc | `0.056748 m` | `0.009018 m` | `0.169455°` |
| Feedback | `0.055676 m` | `0.008833 m` | `0.168344°` |

在 `t+1` 到 `t+10`：

```text
translation improved = 10/10
rotation improved    = 6/10
posthoc future unchanged = true
```

这个 POC 说明外部位姿累积器可以接受一个已知正确的 pose correction；但它不证明 SAM 能估计出正确 correction，也不证明 HorizonStream 的 KV/GLA 状态会读取 pose correction。它与当前“只压物体点云”的主目标是分开的。

## 6. 当前结论

### 6.1 已经验证

1. **物体点云独立修正机制已经真正生效。**
   per-instance 实验中存在有效帧-实例 correction，且 raw pose 分支没有被修改。

2. **SAM 引导的物体级对齐对部分物体有效。**
   bed 和至少一个 dustbin track 的 F5cm、voxel IoU 或 ghost ratio 有改善。

3. **对齐 loss 下降不保证 GT 点云质量提升。**
   V3 和 per-instance 实验都表现出 loss/局部指标下降，但 aggregate F5cm 或 voxel IoU 下降。

4. **shared correction 曾经让 bed 明显改善。**
   这说明多个物体共同约束能够提供额外正则，但 shared 变换不一定适合每一个物体。

5. **raw HorizonStream pose 可以保持不变。**
   当前 object-only pipeline 不需要闭环，也不需要修改 HorizonStream KV、GLA cache 或在线相机位姿。

### 6.2 目前不能声称

目前不能写成：

> SAM3.1 已经能够稳定优化 HorizonStream 位姿或稳定提升场景中所有物体的点云质量。

更准确的表述是：

> SAM3.1 提供的 persistent object mask 可以作为物体级点云对齐的约束；在当前实验中，该约束对部分物体产生了增益，但由于实例身份、mask 质量和最近点对应仍不稳定，整体物体点云质量尚未稳定提升。

如果只讨论当前独立物体点云实验，可以写成：

> The per-instance 6DoF correction is effective at the implementation level and improves selected objects, but the current self-consistency objective is not yet a reliable proxy for ground-truth object reconstruction quality.

## 7. 结果不稳定的原因分析

### 7.1 SAM track 可能不是稳定的物体身份

同一个文本 prompt 可能导致：

- 一个物体被拆成多个 track；
- track 只覆盖物体的一部分；
- mask 混入墙、地面或相邻物体；
- 遮挡后重新出现时身份变化；
- 同一类别的不同物体发生身份混淆。

独立 6DoF 会给每一个错误 track 更高的自由度，因此容易把错误拟合得更好。

### 7.2 最近点对应可能错误

当前最大匹配距离是 `0.25 m`。对于平面、遮挡、稀疏点云和边界噪声，这个范围仍可能把点匹配到错误表面。mutual nearest 和 70% trimming 只能降低 outlier 数量，不能保证物理对应正确。

### 7.3 优化目标和 GT 评价目标不同

优化的是：

```text
当前预测 object cloud ↔ 历史预测 object cloud
```

评价的是：

```text
累计预测 object cloud ↔ GT object cloud
```

如果历史点云已经带有共同的系统误差，或者 mask 一直带入相同背景区域，self-consistency 会变好而 GT 距离不会变好。

### 7.4 单物体几何可能退化

单个物体可见区域经常接近平面，导致部分旋转或平移方向不可观测。`16` 个匹配点只是数量下限，不代表这些点在三维空间中具有足够的非共面约束。

### 7.5 reference history 可能传播错误

如果错误修正通过 gate，后续帧会把这次修正后的 observation 放入 history，之后可能持续对齐到错误参考。当前前 5 帧 anchor 固定，但 recent history 仍可能发生误差传播。

### 7.6 指标之间并不等价

accuracy 下降不代表 F5cm、voxel IoU 一定上升；ghost ratio 下降也不保证完整性变好。当前 per-instance 结果正是 accuracy 和 ghost ratio 改善，但 F5cm 和 voxel IoU 下降。

## 8. 300 帧实验状态与采样说明

300 帧命令已经准备好，当前设置为：

```text
frame_start=0
frame_stride=1
frame_count=300
```

之前 300 帧运行是在 GT evaluation 的 `torch.quantile()` 大输入处失败。该评估内存问题已经在 `main` 修复；300 帧脚本的默认输出目录也已切换为新的 `..._v2`，避免复用旧的失败半成品。修复后的 300 帧实验已经成功完成，但结果整体变差，详见第 4.5 节。

需要注意：如果数据帧率和 `frame_stride` 不变，改成 300 帧并不会改变相邻两帧之间的实际运动距离；它增加的是连续观测数量和可用历史参考。只有减小 `frame_stride` 或使用更高帧率，才会真正减小相邻输入之间的位移。

运行命令：

```zsh
zsh streaming_couping/commands_run_scannet_object_pose_loss_object_per_instance_300f.txt
```

对应输出目录：

```text
/data184/open_source/vggtSam/outputs/semantic_map_300frames_horizonstream_object_pose_loss_object_per_instance_v2
```

## 9. 建议的下一步

300 帧结果已经表明，单纯增加连续观测数量不能保证最近点匹配更可靠。下一步应优先定位失败来源，而不是继续盲目增加帧数。

之后按以下顺序分析：

1. 查看 `map_objects.csv`，确认重复 `dustbin` 行对应的是预测 track、GT instance 还是 assignment 展示问题。
2. 对每个 instance 同时记录 accepted correction 数、匹配点数、初始/最终 loss、correction 大小和 GT 指标变化。
3. 将 reference observation 分成优化用和 hold-out 验证用，防止只对训练匹配 loss 过拟合。
4. 对候选 correction 增加三维非退化检查和空间分布检查。
5. 以同一时间段做帧数 ablation，例如固定 `frame_start=90`，比较 `100/200` 帧；不要把起始帧和帧数同时改变。
6. 在固定时间段基线之后，再单独测试更严格的匹配距离，例如 `0.05–0.08 m`，不要和帧数变化同时修改。
7. 对 duplicate / identity-switch 的 SAM track 做过滤后，再比较 per-instance 结果。

## 10. 可引用的最终阶段性结论

> 我们实现了一个保持 HorizonStream 原始相机位姿不变的 SAM3.1-guided object-level point-cloud alignment pipeline。该方法从 persistent SAM instance mask 中提取物体点云，并为每个物体估计独立的小幅 6DoF correction，只将修正写回对应物体点云。100 帧实验中，部分物体获得了局部增益；300 帧连续实验中，内部修正被大量接受，但整体 F5cm、accuracy、completeness 和 ghost ratio 变差。结果表明，SAM 物体匹配提供了有用但不稳定的局部几何约束，当前 self-consistency loss 还不能可靠地转化为 GT reconstruction gain。
