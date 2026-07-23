# 实例引导的点云与相机位姿联合优化：当前最终方法

## 先用直白的话说明方法

### 系统到底输出什么

对一组按指定顺序输入的 RGB 图像，系统输出三类结果：

1. 每帧、每个 persistent instance 的 SAM3 分割/追踪 mask；
2. 原始 StreamVGGT 点云和修正后的点云；
3. 原始 StreamVGGT 相机位姿和修正后的相机位姿。

最终推理可以概括成：

```text
RGB 序列
  ├─ SAM3：从参考帧 mask 出发，追踪同一个物体
  └─ StreamVGGT：预测初始点云、相机旋转和平移
             ↓
固定的 learned adapter
  ├─ 根据跨帧实例信息，小幅修正 pointmap token → 新点云
  └─ 根据跨帧实例信息，小幅修正 camera token → 新旋转
             ↓
无需训练的几何求解器
  └─ 用“mask 内世界点必须落在对应相机射线上”重新求相机平移
             ↓
mask + refined pointmap + refined camera pose
```

### 为什么方法里需要训练

“训练”和“每个新序列上的优化”是两件不同的事。

需要训练的是一个很小的残差 adapter。输入实例外观、mask、跨帧 ID 和三维统计量后，究竟
应该怎样改 StreamVGGT 的 patch/camera token，没有一个能够直接写出的闭式公式。因此在
原训练帧上利用 GT pointmap/pose loss，学习这部分映射：

```text
实例证据 → pointmap token 应改多少
实例证据 → camera token 应改多少
```

SAM3 和 StreamVGGT 主干始终冻结，并不是重新训练两个大模型。只训练新加入的 instance
encoder、cross-attention、gate 和 zero projection。zero projection 保证开始训练前输出与
原始 StreamVGGT 完全相同，adapter 只能学习残差。

不需要训练的是最后的相机平移求解。给定 pointmap、rotation、内参和实例 mask 后，相机
中心只有三个未知数，可以通过 point-to-ray 几何约束直接求解。

### 训练一次还是每个序列重新训练

当前协议是训练一次、测试时固定权重：

| 模块 | 原序列训练阶段 | 新序列测试阶段 |
|---|---|---|
| SAM3 / StreamVGGT 主干 | 冻结 | 冻结 |
| learned pointmap/rotation adapter | 训练 | 固定 checkpoint，不训练 |
| SAM3 tracking/cache | 为训练序列计算 | 为新序列重新计算 |
| ray translation solver | 按帧求解 | 按帧重新求解，无梯度 |
| GT | 训练 loss 和评估 | 只做最终评估 |

因此新序列上的位姿提升不是“拿新序列 GT 再训练”得到的。它使用原先固定 checkpoint，
只针对当前图像重新计算实例追踪和几何平移。

### 当前已知效果边界

原序列 held-out 帧上，learned pointmap 分支和最终位姿都明显提升。新的
`105 109 113 122 254` 序列使用固定权重后：

- ATE RMSE 从 `0.1414 m` 降到 `0.0771 m`；
- translation RPE 从 `0.1220 m` 降到 `0.0576 m`；
- 5/5 ray fits accepted；
- 全场景点云 mean 只从 `0.0957 m` 降到 `0.0936 m`；
- bed 点云改善，但 cabinet 和 wardrobe 退化。

所以当前可以得出的结论是：位姿方法已经表现出跨序列价值；learned pointmap 分支只在原
训练视角上稳定，新视角下需要加入几何可靠性门控或无 GT 自适应，不能声称已经稳定泛化。

## 1. 研究目标

本项目的目标不是只改善 SAM3 跟踪，也不是只修复 StreamVGGT 位姿，而是同时得到：

1. 更准确、跨帧更一致的全场景和实例点云；
2. 更准确的相机旋转与平移；
3. 点云和相机位姿位于同一个内部坐标系，可以作为一套完整重建结果使用。

当前系统采用分阶段的因果混合优化，而不是把 SAM3 token 直接写入 StreamVGGT
所有 token，也不是端到端迭代 bundle adjustment：

```text
原始 StreamVGGT 几何
    -> 几何筛选和恢复 SAM3 跟踪
    -> 稳定 persistent instance IDs / masks

persistent instances
    -> V2 patch geometry branch
    -> refined world pointmap

persistent instance appearance
    -> V2 camera branch
    -> refined camera rotation

refined pointmap + refined rotation + reference predicted K
    -> V3 angular-Huber ray-center solve
    -> refined camera translation
```

最终可部署结果是：

```text
V2 refined world pointmap + V3 refined camera pose
```

二者共同使用 StreamVGGT native gauge。无 GT 时，这就是完整的方法输出。

## 2. Persistent instance 表示

SAM3 和 StreamVGGT 主干保持冻结。每个跨帧实例构建一个因果 persistent token，输入包括：

- SAM3 mask 内冻结外观特征的均值和方差；
- persistent instance ID 和当前 tracker score；
- 当前三维中心与历史中心；
- 点云协方差特征值和三维 extent；
- 实例 ICP 位移建议、fitness 和 RMSE ratio；
- point confidence、mask 面积和点数；
- geometry confidence 和 static score；
- 当前观测、EMA 历史记忆及二者差值；
- 记忆年龄和帧间隔。

tokenizer 只因果读取当前帧及历史信息，不读取未来帧。一个实例必须有可信历史记忆，
当前 token 才会参与 cross-attention。动态、跟踪不可靠或无有效观测的实例不会产生更新。

## 3. 点云优化：V2 geometry branch

### 3.1 最终结构

最终几何分支沿用消融阶段选出的 tracker-gated patch 更新：

```text
patch_sam_geometry_tracker_gate
```

persistent instance token 作为 key/value，StreamVGGT 多层 DPT patch tokens 作为 query：

```text
patch_token = patch_token
            + sigmoid(gate) * zero_proj(cross_attention(patch_token, instance_tokens))
```

只更新 `patch_start_idx` 之后的 patch tokens。camera/register 前缀完全保持不变，避免实例
语义污染相机头。更新后的多层 tokens 送入冻结的 StreamVGGT point/depth heads，得到新的
world pointmap、point confidence 和 depth。

几何分支拥有独立的 instance tokenizer、cross-attention、gate 和 projection，不与位姿分支
共享学习模块，防止 pointmap loss 与 pose loss 通过同一个瓶颈互相冲突。

### 3.2 门控设计

早期严格 geometry gate 会错误拒绝已经正确跟踪的 bed 等实例。因此最终模式采用：

- tracker confidence 负责硬门控；
- geometry confidence 和 static score 仍作为 token 特征；
- 网络学习如何使用几何质量，而不是在进入网络前将实例全部丢弃。

### 3.3 几何损失

训练使用：

- fixed-reference Sim(3) aligned pointmap loss；
- scale-invariant depth loss；
- fixed-reference depth loss；
- 跨帧静态实例 trimmed Chamfer/刚体一致性；
- 实例质心一致性；
- 小权重 residual regularization。

### 3.4 为什么点云能够提升

原始 StreamVGGT 主要依靠整帧视觉上下文。persistent instances 增加了跨帧稳定的对象级锚点：

1. persistent ID 提供跨帧对应关系；
2. 外观特征区分不同实例，减少错误关联；
3. 三维中心、协方差和 extent 提供形状与位置先验；
4. static/geometry confidence 降低遮挡和错误点图的影响；
5. 只更新 patch token，使几何监督直接作用于 point/depth head；
6. 独立分支避免相机与点图优化争夺同一表示。

### 3.5 当前点云结果

训练帧为 `90 105 119 130 140`，held-out 帧为 `210 240`：

| 方法 | held-out 全场景 pointmap mean error |
|---|---:|
| 原始 StreamVGGT | 0.15258 m |
| 单独最佳 geometry branch | **0.06053 m** |
| 最终 decoupled dual branch | **0.06358 m** |

最终双分支方法相对原始 StreamVGGT 降低约 `58.3%`。单独 geometry branch 略好，但不包含
完整位姿分支；三方最终比较中的 `ours.ply` 使用同时优化点云和位姿的
`decoupled_dual_branch`。

## 4. 位姿优化第一阶段：V2 learned rotation

### 4.1 Camera-token fusion

最终位姿学习分支沿用消融阶段选出的 SAM-appearance camera 更新：

```text
camera_sam_only
```

每帧 camera token 查询 persistent SAM3 appearance tokens，再通过 zero-initialized residual
写回 camera token。该分支只使用外观、persistent memory 和 tracker confidence，不通过几何
特征侧信道直接拉动相机。

zero projection 保证初始化及 module-off 时严格恢复原始 StreamVGGT 输出。SAM3 和
StreamVGGT 主干均冻结，只训练 instance encoder、cross-attention、gate 和 projection。

### 4.2 位姿学习损失

- camera encoding loss；
- all-pair relative rotation loss；
- translation-direction loss；
- 静态实例刚体一致性；
- 实例质心一致性。

### 4.3 V2 得到的结论

V2 显著改善旋转和相对平移，但自由回归的绝对相机平移失败：

| 指标 | 原始 StreamVGGT | learned V2 |
|---|---:|---:|
| ATE RMSE | 0.36879 m | 0.37535 m |
| rotation RPE | 3.518° | **1.171°** |
| translation RPE RMSE | 0.23981 m | **0.17627 m** |
| pair translation direction | 44.22° | **34.68°** |

因此 V3 保留 V2 的 refined rotation，替换 learned absolute translation。

## 5. 位姿优化第二阶段：V3 analytic translation

### 5.1 Point-to-ray 约束

对像素 `u`，refined pointmap 给出世界点 `X_u`。使用 refined rotation 和参考帧预测内参
构造世界射线：

```text
r_u = normalize(R_c2w K^-1 [u_x, u_y, 1]^T)
```

相机中心 `C`、像素射线和对应世界点应共线，其垂直残差为：

```text
e_line(C) = (I - r_u r_u^T)(X_u - C)
```

普通加权最小二乘只有三个未知量，闭式正规方程为：

```text
A = sum_u w_u (I - r_u r_u^T)
b = sum_u w_u (I - r_u r_u^T) X_u
C = A^-1 b
```

最终方法使用 angular-Huber IRLS：

```text
e_angle(C) = ||e_line(C)|| / ||X_u - C||
L(C) = sum_u w_u Huber(e_angle(C))
```

### 5.2 最终选定配置

- pointmap：V2 refined world pointmap；
- rotation：V2 refined rotation；
- intrinsics：原始 StreamVGGT 参考帧预测 K，在整段序列中固定；
- spatial scope：三个 persistent instance masks 的并集；
- solver：angular-Huber IRLS；
- minimum points：1024；
- condition-number gate：`1e8`；
- point-to-ray RMSE gate：`0.20` native units；
- center-shift gate：`0.75` native units；
- 失败时精确回退到 V2/baseline center。

### 5.3 为什么平移能够提升

1. 输入 pointmap 已比原始结果准确约 58%；
2. 输入 rotation 已显著改善，像素世界射线方向更可靠；
3. 相机平移由三个未知量和大量像素约束确定，不再自由回归；
4. persistent instance 区域比背景 pointmap 更稳定；
5. 多个空间分散实例提供非平行射线，避免单实例质心/ICP 退化；
6. angular residual 降低深度尺度误差的影响；
7. Huber IRLS 抑制错误 mask、遮挡和点图离群值；
8. reference K 抑制逐帧焦距漂移；
9. 所有失败条件可在无 GT 情况下检测并回退。

空间消融支持这一判断：

| 求解范围 | held-out ATE RMSE |
|---|---:|
| 全图 angular-Huber | 0.18165 m |
| background only | 0.19273 m |
| persistent instances only | **0.14387 m** |
| 全图 GT-K oracle | 0.14615 m |

实例区域使用预测 reference K 仍略优于全图 GT-K oracle，说明主要收益来自更干净的实例
几何约束，而不是读取 GT 内参。

## 6. 最终位姿结果

held-out 帧 `210 240`：

| 指标 | 原始 StreamVGGT | learned V2 | 最终 V3 |
|---|---:|---:|---:|
| ATE RMSE | 0.36879 m | 0.37535 m | **0.14387 m** |
| translation error mean | 0.35304 m | 0.37342 m | **0.14371 m** |
| translation RPE RMSE | 0.23981 m | 0.17627 m | **0.08986 m** |
| rotation RPE | 3.518° | **1.171°** | **1.171°** |
| pair translation-direction error | 44.22° | **34.68°** | 40.94° |

最终 V3：

- ATE 相对原始 StreamVGGT 降低约 `61.0%`；
- translation RPE 相对原始降低约 `62.5%`；
- translation RPE 相对 learned V2 降低约 `49.0%`；
- 保留 learned V2 的旋转提升；
- 两个 held-out fit 均 accepted；
- mean point-to-ray RMSE 从 `0.14058` 降至 `0.02678` native units；
- maximum condition number 为 `6.69`，远低于退化阈值。

限制是 pair translation-direction error 虽优于 raw，但不如 learned V2。当前不能声称所有位姿
指标均改善。

## 7. 最终输出与公平三方比较

### 7.1 无 GT 可部署结果

```text
final_instance_ray_pose_v3/<clip>/deployable_native/
  full_scene.ply
  instance_37.ply
  instance_68.ply
  instance_54.ply
  camera_poses.csv
  camera_poses.npz
```

这里没有 GT。V2 refined pointmap 和 V3 pose 使用同一个 StreamVGGT native gauge。

分割/追踪 mask 单独导出到：

```text
final_instance_ray_pose_v3/<clip>/segmentation_masks/
  sequence_overview.png
  overlays/
    seq_000_frame_<frame>.png
    ...
  binary/
    instance_37/
    instance_68/
    instance_54/
    union/
  mask_summary.csv
  legend.csv
```

`overlays/` 是 RGB 与彩色实例 mask 的叠加图；`binary/instance_<id>/` 是每实例 0/255
二值 PNG；`binary/union/` 是实例几何求解使用的 mask 并集。参考帧是初始化 prompt mask，
后续帧是方法实际消费的 SAM3 persistent tracking/recovery mask。

### 7.2 GT / raw StreamVGGT / ours 公平比较

```text
final_instance_ray_pose_v3/<clip>/comparison_gt_world/
  full_scene/
    ground_truth.ply
    streamvggt_raw.ply
    ours.ply
    overlay.ply
  instance_37/
  instance_68/
  instance_54/
  pointcloud_metrics.csv
  camera_poses.csv
  camera_pose_metrics.csv
  camera_poses.npz
  pose_comparison.png
  pose_comparison.pdf
  pose_comparison.svg  # matplotlib 不可用时的无依赖 fallback
```

比较协议：

- 三者使用相同帧；
- 三者只使用 GT/raw/ours 公共有限像素；
- 三者使用相同确定性采样；
- 只从 raw StreamVGGT 参考帧 pointmap 拟合一次 Sim(3)；
- 同一个 Sim(3) 原样应用于 raw 和 ours，禁止分别重新拟合；
- 所有比较 PLY 均位于 ScanNet++ GT-world；
- overlay 颜色为 GT=绿色、raw StreamVGGT=红色、ours=蓝色。

`pose_comparison.png/.pdf` 包含四个子图：

- GT/raw/ours 三维相机轨迹及相机前向方向；
- 自动选择变化最大的两个世界坐标轴生成轨迹投影；
- 每帧相机中心误差；
- 每帧旋转误差。

帧 `210、240` 使用黑色外圈/竖线标记为 held-out。三条轨迹使用与点云比较完全相同的
shared GT-world Sim(3)，不会分别拟合对齐。服务器存在 matplotlib 时输出 PNG 和 PDF；
否则自动输出不依赖 matplotlib 的 SVG，导出命令不会因此失败。

GT PLY 是选定相机视角下由 ScanNet++ mesh rasterize 得到的可见 GT pointmaps，不是完整扫描
mesh。这保证了逐像素点图比较公平。

运行当前冻结结果：

```bash
zsh streaming_couping/commands_final_joint_pointcloud_pose.txt
```

## 8. 当前研究结论和边界

当前方法已经实现：

```text
tracking -> 改善 pointmap 和 rotation
refined pointmap -> 解析改善 translation
```

它是一个学习前端与解析几何后端结合的因果系统，不是完整的迭代联合 BA。尚未实现：

```text
refined pose -> 再更新 pointmap -> 再优化 pose
```

此外：

- 当前只在 ScanNet++ 场景 `00a231a370` 上验证；
- 训练只使用帧 `90 105 119 130 140`；
- 主要 held-out 指标来自帧 `210 240`；
- 最终 spatial scope 是查看该场景消融后选定的；
- 因此这是单场景 proof-of-concept，不是无偏跨场景泛化结论；
- 不主张当前 depth 指标已经同步达到相同幅度的改善。

当前最可靠的结论是：persistent instance guidance 明显改善 pointmap 和相机旋转；在此基础上，
使用静态实例区域的 robust point-to-ray 几何后端可以显著修复 StreamVGGT 的绝对相机平移。

## 9. 已冻结的消融结论

旧实现代码已删除，以下结果保留作为最终选择依据：

| 被淘汰方向 | 观察结果 | 最终决定 |
|---|---|---|
| fused token 写入 SAM3 | 没有形成可靠的同实例恢复，且会耦合两个冻结主干 | SAM3 只提供 mask、ID、score 和 pooled appearance |
| camera geometry-only / combined fusion | 不如 SAM-appearance camera branch 稳定 | pose branch 只使用 appearance + tracker gate |
| all-token fusion | pointmap 与 pose 梯度互相干扰 | camera 与 patch 使用独立 tokenizer/attention |
| geometry strict gate | 会拒绝 bed 等已正确跟踪但几何分数暂低的实例 | geometry/static 作为特征，只有 tracker 做硬门控 |
| learned absolute translation | held-out ATE `0.37535 m`，没有优于 raw `0.36879 m` | 学习分支只保留 rotation，translation 交给解析 solver |
| 全图/background ray solve | held-out ATE 分别为 `0.18165/0.19273 m` | 只使用 persistent-instance mask 并集 |
| 逐帧 K、trimmed LS、GT-K oracle | 不如 reference predicted K + angular-Huber instance solve | 固定最终无 GT solver |

清理后的代码只允许：

```text
decoupled_dual_branch + aligned/module_off
ray_refined_pointmap_refined_rotation_reference_k_instances
```

`module_off` 仅用于严格确认关闭实例模块时逐元素恢复原始 StreamVGGT，不是另一种候选方法。
