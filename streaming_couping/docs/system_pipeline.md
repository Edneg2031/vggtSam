# StreamVGGT + SAM3.1 语义地图系统

更新时间：2026-08-24

## 1. 系统目标

当前系统的主要目标是构建稳定的 streaming semantic/object map：

```text
RGB sequence
    ├── StreamVGGT → camera pose + dense pointmap + confidence
    └── SAM3.1    → object mask + persistent track ID + visibility
                                      ↓
                         object-level 3D semantic map
```

系统中的两个模型承担不同职责：

```text
StreamVGGT：where
    相机位姿、深度、世界坐标点云、多视图几何

SAM3.1：which
    实例 mask、语义类别、persistent ID、遮挡和 re-entry
```

当前主线是：

```text
explicit geometry → SAM measurement/ownership → object map fusion
```

当前不把 SAM hidden token 直接接到 StreamVGGT 的 SE(3)、depth 或 XYZ head。任何未来的语义到
几何反馈，都必须通过显式的 static/dynamic、correspondence 或 confidence factor，并同时评价
pose、pointmap 和最终语义地图。

## 2. 已验证的 V0 pipeline

### 2.1 RGB 和数据入口

系统从 ScanNet++ processed manifest 读取：

- RGB image path；
- scene ID 和 frame index；
- instance/semantic annotation；
- mesh-rasterized GT pointmap（只用于 evaluation 或训练监督）。

原始 RGB 采用只读引用，不在实验中复制图片。

当前可用的完整 scene 为：

```text
00a231a370
0a184cf634
1a8e0d78c0
1eacc65607
```

`1be2c31cac` 由于 pinhole、annotation 和 mesh 资产不完整，不参与当前协议。

### 2.2 StreamVGGT geometry branch

V0 使用冻结的 StreamVGGT full-history inference：

```text
RGB frames
    ↓
StreamVGGT aggregator/backbone
    ↓
CameraHead → world_to_camera + intrinsics
PointHead  → dense world pointmap + confidence
```

点云保留 StreamVGGT 的第一帧参考坐标系。当前正式 semantic-map baseline 不修改原始 XYZ，
不使用 ICP、BA、三角化或点云后处理来制造几何提升。

V0 的 pose 输出可以使用：

```text
first-frame anchor + native QK Top-4 history → frozen CameraHead
```

但 QK pose branch 只改变相机轨迹输出，不自动改变 full-history PointHead 的点云。因此：

```text
pose improvement ≠ pointmap improvement
```

当前 map 的几何主体仍是 raw full-history StreamVGGT pointmap。

### 2.3 SAM3.1 tracking branch

SAM3.1 通过概念 prompt 发现对象，并使用 forward-only causal tracking：

```text
concept prompt
    ↓
SAM3.1 online discovery
    ↓
first visible frame = birth
    ↓
permanent slot / persistent track ID
    ↓
forward masks + track scores
```

对象 birth 只根据当前及过去信息确定；slot 不复用。当前 prompt 是 annotation-only 的固定词表，
用于模拟人工给定的对象概念，不作为自动开放词汇识别能力的证据。

### 2.4 V6 geometry-guided SAM

V6 是当前最强的跨模型正结果。它不修改 StreamVGGT 点云，而是把 frozen geometry 转换成当前
图像上的显式 prompt：

```text
reference object mask + trusted StreamVGGT points
    ↓
project into current frame
    ↓
coarse box + positive geometry support
    ↓
SAM3.1 candidate masks
    ↓
geometry support / box precision / SAM score
    ↓
adaptive candidate competition
```

V6 的 selection gate 只使用显式几何支持和 SAM candidate score。GT 不参与候选生成；GT 只在
候选冻结后用于评价。

已观察结果：

```text
train + validation weighted IoU delta = +0.040819769
test report-only IoU delta            = +0.020205715
worse-frame count                      = 0
```

这个结果证明的是逐帧 mask measurement 改善。它还没有证明 IDF1、ID switch、re-entry 或 3D
semantic map 一定改善。

### 2.5 Semantic map lifting

当前 V0 map 的写入方式是把 SAM mask lift 到 StreamVGGT world pointmap：

```text
StreamVGGT world pointmap [frame, H, W, 3]
SAM stream-grid mask       [frame, object, H, W]
confidence + track score
    ↓
persistent slot ownership
    ↓
object-colored world point cloud
```

同一 persistent slot 在不同 frame 中保持同一颜色和 ID。点的位置来自 raw StreamVGGT，SAM 只
负责 ownership/semantic label。

当前系统可以输出：

- RGB point cloud；
- semantic-colored point cloud；
- persistent track metadata；
- camera trajectory；
- dense 和 retained map points。

### 2.6 GT 边界

候选生成阶段的原则是：

```text
RGB + frozen StreamVGGT + SAM → candidate masks/tracks
```

GT instance mask、GT pointmap 和 Sim(3) 只在候选生成完成之后打开，用于：

- reference-frame Sim(3) 对齐；
- pose/pointmap evaluation；
- tracking evaluation；
- semantic-map evaluation；
- 训练监督。

## 3. 当前已验证结论

### 已成立

1. StreamVGGT full-history 可以提供可用于 semantic lifting 的 dense world pointmap。
2. SAM3.1 可以提供 persistent instance slot 和 causal mask tracking。
3. V0 可以生成可视化的 persistent semantic point map。
4. first-frame anchor + native QK Top-4 在当前单序列上改善了 pose：center error 约 `10.93%`，
   rotation error 约 `6.15%`。
5. 显式 StreamVGGT geometry → SAM3.1 candidate selection 在已有协议上改善了逐帧 mask IoU。

### 尚未成立

1. SAM 改善 StreamVGGT pose 或 pointmap。
2. SAM hidden token 能稳定改善跨场景几何。
3. V6 的 mask IoU 改善会自动转化成 IDF1 或 map quality 改善。
4. QK pose improvement 会自动改善 semantic map。
5. 当前系统已经具备可靠的 occlusion recovery 或 re-entry。

## 4. 当前代码入口

已验证或已有的主要入口：

```bash
zsh streaming_couping/commands_v0_baseline.txt
zsh streaming_couping/commands_plot_v0_poses.txt
zsh streaming_couping/commands_semantic_mapping.txt
zsh streaming_couping/commands_semantic_mapping_v1.txt
zsh streaming_couping/commands_semantic_mapping_v1_evaluation.txt
zsh streaming_couping/commands_check_scannetpp_data.txt
zsh streaming_couping/commands_audit_scannetpp_processed.txt
```

前五个命令分别用于运行 V0、可视化 pose、离线比较 Stage 0 分支、导出 V1 persistent object
memory 地图、以及比较 V0/V1 identity；后两个命令只读检查服务器上的 ScanNet++ 源数据和已处理
manifest。V1 只在 SAM observation 到 object memory 的身份层工作，不修改 StreamVGGT、SAM3 或
SAM hidden memory。

## 5. 系统定位

当前最合理的系统表述是：

> Frozen StreamVGGT provides metric geometry, while SAM3.1 provides object ownership and persistent
> identity. Explicit geometry-guided segmentation and object-level memory are evaluated by their effect
> on the final 3D semantic map.

当前优先级是把这个系统的 tracking、identity 和 map-level evidence 补齐，而不是继续把更多 token
直接拼到 pose head。
