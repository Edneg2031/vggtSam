# SAM3.1 视频跟踪与几何约束阶段性报告

更新时间：2026-09-04

项目主线：`SAM3.1 + HorizonStream` 三维语义实例地图

## 1. 项目目标

本阶段的目标是摸底 SAM3.1 的视频实例跟踪能力，并验证 SAM 提供的跨帧物体身份信息
能否进一步约束几何，从而实现多视角物体匹配，改善三维语义地图的精度和跨场景泛化性。

核心假设是：

```text
RGB
 ├─ HorizonStream → depth / confidence / intrinsics / online pose
 └─ SAM3.1       → mask / track score / persistent instance ID
                         ↓
              object identity and mask evidence
                         ↓
             multi-view geometric verification
                         ↓
                    semantic map
```

需要区分两个问题：

1. SAM3.1 能否在视频中稳定保持同一物体的 mask 和 persistent ID；
2. persistent ID 是否足以帮助得到可靠的多视角几何 correspondence。

当前实验表明，第一个问题存在可利用的信号，但第二个问题不能通过简单的同 ID 匹配或
appearance residual 直接解决。

## 2. 当前 baseline

当前正式 baseline 是一个免训练、冻结模型的两阶段 pipeline：

```text
selected RGB frames + text prompts
          │
          ├─ HorizonStream
          │    └─ metric depth + confidence + intrinsics + online pose
          │                 ↓
          │       horizonstream_geometry.pt
          │
          └─ SAM3.1
               └─ text masks + persistent instance IDs + track scores
                                ↓
                    depth backprojection to world coordinates
                                ↓
                       5 cm voxel-based fusion
                                ↓
             full scene map + semantic map + object-level PLYs
```

baseline 的约束如下：

- HorizonStream、SAM3.1 权重均冻结；没有训练、adapter 或 token fusion；
- SAM3.1 只负责类别、mask、实例身份和 track score，不修改 HorizonStream pose/depth；
- mapping 层使用 depth、intrinsics 和 camera pose 反投影，再进行世界坐标体素融合；
- 不启用 BA、ICP、affine correction、historical-depth veto 或语义反馈几何；
- GT 只用于独立 evaluation，不参与候选生成、tracking 或 map write 决策；
- HorizonStream 和 SAM 按两个阶段运行，geometry cache 作为两阶段之间的数据边界。

当前 HorizonStream 几何设置为：

```text
window=10
sliding=1
image_size=518
patch_size=14
crop=true
precision=float16
pose=online motion averaging
```

`sliding=1` 表示首个 10 帧窗口之后每次新增 1 帧，并不表示模型只使用 1 帧上下文。
当前主要用于降低显存峰值并保持连续位姿。它会增加长序列的运行时间，因此实验报告中
需要显式记录该设置。

## 3. 实验协议

### 3.1 旧 V0/StreamVGGT 诊断数据

历史 geometry 诊断使用四个 scene-disjoint clip，每个 clip 30 帧，共 120 帧：

```text
train:
  00a231a370_uniform30_r1
  0a184cf634_uniform30_r1
validation:
  1a8e0d78c0_uniform30_r1
test:
  1eacc65607_uniform30_r1
```

多场景 affine 实验对每个 clip 使用前 21 帧 calibration、后 9 帧 holdout。所有 affine
参数只由 calibration prefix 拟合。

### 3.2 HorizonStream baseline

此前已在 `00a231a370` 上完成 50 帧 smoke/runtime 验证。当前新增命令会对以下两个
外部 RGB 数据集进行全量帧 baseline 测试：

```text
/home/bod/182Nas/share/BaZhuaYu/DemoData/xiaoyi/wuhan_A3/images
/home/bod/182Nas/share/BaZhuaYu/DemoData/xiaoyi/blind_data/meeting_room_a02/frames
```

命令文件为：

```zsh
zsh streaming_couping/commands_run_two_dataset_baseline.txt
```

该测试使用 `fusion-policy=raw`，输出位于：

```text
/data184/open_source/vggtSam/outputs/wuhan_A3_horizonstream_sam3_baseline_allf
/data184/open_source/vggtSam/outputs/meeting_room_a02_horizonstream_sam3_baseline_allf
```

截至本报告生成时，这两个全量数据集的正式精度结果尚未回填，因此不能用它们宣称
HorizonStream 已经改善地图。

## 4. 实验结果

### 4.1 单场景 V0：SAM3.1 跟踪与语义地图 baseline

在一个封存场景的 30 帧 raw V0 结果中，得到：

| 指标 | raw V0 |
|---|---:|
| mean frame IoU | 0.41345 |
| frame IDF1 | 0.50510 |
| pixel IDF1 | 0.78665 |
| voxelIoU@5cm | 0.09964 |
| F-score@5cm | 0.34401 |
| ghost point ratio | 0.25789 |

GT-mask oracle 仅作为 evaluation-only 上界：

| 指标 | GT-mask oracle |
|---|---:|
| mean frame IoU | 0.81773 |
| frame IDF1 | 0.89973 |
| voxelIoU@5cm | 0.18513 |
| F-score@5cm | 0.50325 |

解释：预测 mask/实例 ownership 与 GT 之间仍有明显差距。oracle 与 raw 的差距说明，
最终地图误差不能简单归因于 geometry；mask 质量、实例关联和几何一致性都可能是瓶颈。
oracle 只用于判断上限，不是可部署方法。

### 4.2 Causal affine / depth-head ray 诊断

该实验复用冻结的 V0 cache，不重跑模型、不训练，也不修改正式 baseline。候选 geometry
冻结后才打开 GT 做 evaluation。

候选规模为：

```text
clips=4
total_frames=120
pair_rows=7680
pair_samples=454608
calibration/holdout=21/9 per clip
```

主分支尝试在同一像素 ray 上对 depth head 做 per-clip positive-slope affine 修正，
然后重新构建 object 点云和地图。

gate 结果：

```text
affine evidence gate: 1/4 clips passed
geometry gate:        2/4 clips passed
aggregate decision:   NO_GO
```

holdout aggregate 的代表性结果：

| 分支 | voxelIoU@5cm | F-score@5cm |
|---|---:|---:|
| raw_v0 | 0.00554 | 0.02582 |
| depth-head self-affine | 0.00428 | 0.01763 |

历史深度 veto、temporal point prompt 等因果 feedback 分支也出现明显恶化，没有进入
正式系统。

结论：单场景或少量 clip 上的深度关系不能直接推广成稳定的跨场景 affine correction。
当前 StreamVGGT geometry 不足以支持继续做简单局部 affine 修补。

### 4.3 DINOv3 persistent object geometry

该实验使用 SAM persistent track ID 构造 object feature，并比较：

```text
raw
geometry_only
single_view_dino
persistent_dino
shuffled_persistent_dino
```

StreamVGGT、SAM3.1 和 DINOv3 均冻结，只训练小型 residual geometry head，输出为：

```text
X_new = X_raw + ΔXYZ
```

修正只写入 SAM object mask 内，背景使用 raw fallback。

object-level 结果：

| split / branch | object RMSE | object F-score@5cm |
|---|---:|---:|
| validation / raw | 0.20261 | 0.44985 |
| validation / persistent_dino | 0.20676 | 0.36965 |
| validation / shuffled | 0.20559 | 0.40016 |
| test / raw | 0.47011 | 0.11466 |
| test / persistent_dino | 0.41798 | 0.13442 |
| test / shuffled | 0.44202 | 0.11459 |

test 上 persistent DINO 有一定正向信号，但 validation 上反而破坏结果，说明不能据此
证明泛化。

semantic-map readout：

| split / branch | map voxelIoU | map F-score@5cm | ghost |
|---|---:|---:|---:|
| validation / raw | 0.08063 | 0.28214 | 0.11424 |
| validation / persistent_dino | 0.03346 | 0.10802 | 0.22449 |
| test / raw | 0.01906 | 0.05944 | 0.26118 |
| test / persistent_dino | 0.02023 | 0.08686 | 0.20811 |

最终 decision 为 `NO_GO`。

这不表示 DINOv3 没有 appearance/identity 信息，而是说明把 appearance feature 直接
用于预测 metric `ΔXYZ` 职责不匹配。DINO 更适合做 re-identification 或 correspondence
候选排序，不适合单独决定世界坐标修正。

### 4.4 DINO patch retrieval / projection memory

在 patch retrieval 中，正确 persistent track 比 shuffled/unrestricted 更可靠，但没有
超过同一物体内的随机历史 patch：

| split | correct-track DINO RMSE | random same-object RMSE |
|---|---:|---:|
| validation | 0.42286 | 0.39022 |
| test | 0.54247 | 0.53040 |

因此可以认为：

- object identity 对限制搜索空间有帮助；
- 目前没有证据证明 DINO 已经找到了稳定的同一局部表面 correspondence；
- 历史同物体几何可能有帮助，但必须经过 world-space consistency 验证。

projection-memory 实验的四 clip aggregate 也为 `NO_GO`。held-out clip 中：

```text
persistent DINO fused F5cm = 0.14577
geometry-only fused F5cm  = 0.15979
shuffled control F5cm     = 0.16071
```

geometry-only 的 fused ghost rate 约为 `0.52`，DINO 分支约为 `0.55`。这说明未经严格
确认的历史 object point fusion 会污染地图，不能直接作为部署策略。

### 4.5 SAM instance-guided HorizonStream pose refinement B0

该实验是 post-geometry、training-free 的可关闭 ablation：

```text
HorizonStream depth/pose/intrinsics
        + SAM persistent ID/mask
        → mask-constrained feature matching
        → local-camera 3D correspondence
        → RANSAC / Procrustes
        → pose graph refinement
```

在 `00a231a370` 的 50 帧实验中：

```text
tracked_instances=7
candidate_pairs=40
```

使用无额外模型的 RGB patch backend：

```text
accepted_edges=0
rejected_edges=40
candidate_feature_matches_mean=0.075
candidate_valid_3d_mean=0.025
reject_reason=too_few_valid_3d_correspondences
```

切换 DINOv3 backend 后：

```text
accepted_edges=0
rejected_edges=40
candidate_feature_matches_mean=1.3
candidate_valid_3d_mean=0.1
optimizer_attempted=false
```

两次实验的 raw/refined 地图完全相同：

```text
scene_voxels=84126
semantic_voxels=9909
objects=7
pose correction=0
```

结论：当前失败点不是 pose graph，而是 mask 内有效 feature match 和 3D correspondence
数量不足。persistent ID 可以告诉系统“可能是同一个物体”，但不能直接提供精确像素对应。
在没有足够 correspondence 时，保守拒绝 edge 是正确行为。

## 5. 综合结论

### 5.1 关于 SAM3.1

SAM3.1 的 persistent instance ID 是有价值的长期语义信息，可以用于：

- 记录同一物体的跨帧观察；
- 限制历史候选搜索范围；
- 做 object-level re-entry 和 identity consistency；
- 为最终语义地图生成独立的实例点云。

但是：

```text
same persistent ID ≠ precise pixel correspondence
same object        ≠ same visible surface point
```

因此不能使用 object centroid matching，也不能因为两个观察拥有相同 ID 就无条件加入
pose constraint 或融合历史点云。

### 5.2 关于几何约束

已有实验共同说明：

- 简单 affine correction 不具备稳定跨场景泛化；
- 历史深度 veto 和 temporal point prompt 可能删除有效点或引入错误点；
- DINO appearance 不应直接预测 metric world-space correction；
- 直接历史 object point fusion 会产生 ghost 和 double surface；
- object pose refinement 的首要瓶颈是可靠的 mask-constrained 2D/3D correspondence。

### 5.3 当前项目定位

当前最稳妥的项目结论是：

> SAM3.1 负责语义 mask 和 persistent identity，HorizonStream 负责冻结的 metric geometry，
> mapping 层负责世界坐标反投影与体素融合。persistent identity 具备帮助多视角匹配的
> 潜力，但必须结合 appearance、深度、几何一致性和可拒绝的 robust registration；目前
> 尚未验证任何会自动修改 HorizonStream pose/depth 的方法。

## 6. 下一步建议

### 第一优先级：完成新 geometry baseline 摸底

先完成两个外部数据集的 raw baseline，并记录：

- 实际帧数、有效深度比例、运行时间和 peak VRAM；
- SAM track 数、每个类别/实例的出现帧数、mask 面积稳定性；
- scene RGB map、scene semantic map、object tracks 和每个实例 PLY；
- 是否存在明显的帧层错位、double surface、ghost 和实例错配。

没有 GT 时先做定性可视化和 tracking statistics，不把 voxel 指标写成精度结论。

### 第二优先级：分离验证 SAM 与 geometry 的贡献

建议固定相同 RGB 和 prompt，分别报告：

```text
SAM mask/track quality
HorizonStream depth/pose quality
raw semantic-map fusion quality
```

有 GT 时增加 frame IoU、IDF1、ID switches、re-entry、object completeness、voxelIoU、
F-score 和 ghost rate；没有 GT 时至少保留 mask coverage、track duration、实例点数和
可视化结果。

### 第三优先级：重新设计 object correspondence

如果继续探索几何约束，建议采用以下职责分工：

```text
SAM persistent ID       → 限制候选物体范围
DINO/局部 feature       → 产生 appearance correspondence candidates
HorizonStream geometry  → 提供 depth / ray / pose
3D consistency          → RANSAC、depth residual、visibility 检验
robust optimizer        → 只接收高置信 object edge
```

先实现 `world-memory-only` 和严格的 correspondence diagnostic，再考虑引入 DINO。所有
object edge 必须保留 provenance，并允许因 mask、匹配、深度或 pose disagreement 被拒绝。

## 7. 可复现实验入口

当前主要命令：

```zsh
# 当前 HorizonStream + SAM3.1 semantic-map baseline
zsh streaming_couping/commands_run_semantic_map.txt

# 两个外部数据集的全量帧 baseline
zsh streaming_couping/commands_run_two_dataset_baseline.txt

# SAM instance-guided HorizonStream pose refinement ablation
zsh streaming_couping/commands_run_object_pose_refinement.txt
```

历史实验的详细设置和完整 artifact 路径见：

- [`current_status.md`](current_status.md)
- [`multiclip_affine_geometry.md`](multiclip_affine_geometry.md)
- [`dinov3_object_geometry_current_status.md`](dinov3_object_geometry_current_status.md)
- [`object_pose_refinement.md`](object_pose_refinement.md)
- [`current_pipeline.md`](current_pipeline.md)

当前双数据集 baseline 命令已提交到 `main`，commit 为 `6cf9bf1`。本报告只记录实验事实
和阶段性判断，不自动把任何诊断分支升级为正式 baseline。
