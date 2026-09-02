# 当前语义地图实现

更新时间：2026-09-02

本文描述当前 `main` 分支实际使用的语义地图 pipeline。当前主线不是旧的
StreamVGGT/V0 实验，也不是 3D Gaussian Splatting 训练流程，而是一个免训练的
`HorizonStream + SAM3.1` 三维语义实例地图系统。

## 1. 系统目标

输入一段 RGB 图像和若干文本 prompt，输出：

- 完整场景 RGB 点云；
- 带实例颜色的完整场景语义点云；
- 仅包含 prompt 物体的语义地图；
- 每个物体独立的实例点云，例如 `bed_0.ply`、`rug_1.ply`；
- 跨帧保持一致的 persistent instance ID。

整体数据流如下：

```text
RGB 图像 + text prompts
        │
        ├─ HorizonStream
        │    └─ metric depth / confidence / intrinsics / online pose
        │                 ↓
        │       horizonstream_geometry.pt
        │
        └─ SAM3.1
             └─ text mask / track score / instance ID
                              ↓
                 depth backprojection to world coordinates
                              ↓
                    5 cm voxel-based fusion
                              ↓
          full scene map + semantic map + object-level PLYs
```

## 2. 当前 baseline 配置

运行入口是：

```bash
zsh streaming_couping/commands_run_semantic_map.txt
```

当前命令文件顶部的默认配置为：

```text
dataset              processed ScanNet++
manifest             /data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d/manifest.json
scene                00a231a370
frame selection      start=90, stride=1, count=50
selected positions   90, 91, ..., 139
prompts              bed wardrobe chair rug dustbin
voxel size           0.05 m
geometry confidence  0.30
SAM track score      0.50
HorizonStream device cuda:0
SAM device           cuda:2
Python env           /home/huawei/miniconda3/envs/horizonstream/bin/python (默认两阶段共用)
output               /data184/open_source/vggtSam/outputs/semantic_map_50frames_horizonstream_consecutive
```

`FRAME_START`、`FRAME_STRIDE` 和 `FRAME_COUNT` 是整个 manifest 展开后的序列位置，
不是图像文件名。当前场景从 795 帧中取连续 50 帧。没有写入 prompt 的类别不会被
赋予语义实例标签；当前 prompt 使用的是 `rug` 和 `dustbin`，不是 `mat`。

## 3. 两阶段执行方式

两个模型按顺序运行。当前命令文件默认让两个阶段共用
`/home/huawei/miniconda3/envs/horizonstream/bin/python`；如果服务器上的 SAM 依赖仍需
旧环境，可以通过 `STREAMING_COUPING_PYTHON` 覆盖第二阶段解释器。

### 阶段 A：HorizonStream 几何

第一阶段使用 `horizonstream` 环境运行
`generate_horizonstream_geometry_cache.py`：

```text
输入 RGB 清单
    → HorizonStream
    → depth、confidence、pose、intrinsics、processed RGB
    → horizonstream_geometry.pt
```

当前参数为：

```text
window_size  = 10
sliding_size = 1
image_size   = 518
patch_size   = 14
crop         = center crop
precision    = float16
pose         = online motion averaging
offline pose = disabled
```

HorizonStream 的当前输出是每帧的深度图、置信度和相机位姿，不是已经融合好的单个
场景点云。所有选中的 50 帧会先由 dataloader 读到 CPU；随后按 chunk 送入 GPU：

```text
第一个 chunk：10 帧
后续 chunk：每次新增 1 帧
```

`window_size` 表示模型使用的时序上下文，`sliding_size` 在当前封装中表示首个
窗口之后每个 chunk 新增多少帧。`sliding_size=1` 不是把窗口改成 1，而是以一帧
一帧的方式推进，并通过 HorizonStream 的 KV cache 保留历史上下文。

当前使用 `sliding_size=1` 主要是因为较大的增量 chunk 会显著增加显存峰值，且此前
`sliding_size=10/21` 的运行出现过显存不足或更明显的位姿偏移。

HorizonStream 生成的 cache 会保存：

- 选中的绝对 RGB 路径和原始 manifest 位置；
- `depth[S,H,W]`；
- 归一化到 `[0,1]` 的 confidence；
- `world_to_camera[S,3,4]`；
- `intrinsics[S,3,3]`；
- center-cropped 后的 `processed_rgb[S,H,W,3]`；
- checkpoint、源码 revision、预处理和推理参数。

如果图片顺序、图片文件状态、checkpoint、源码 revision 或 HorizonStream 参数发生
变化，cache 会被判定为无效并重新生成。

### 阶段 B：SAM3.1 语义追踪与建图

HorizonStream 进程结束后，才启动 `run_semantic_map.py`。第二阶段
加载几何 cache，不会加载或运行 StreamVGGT：

```text
horizonstream_geometry.pt
    + processed RGB
    → SAM3.1 text tracking
    → SemanticMapBuilder
    → PLY / PT / JSON
```

SAM 使用 cache 中保存的 processed RGB。代码会把它们写入临时目录，再交给 SAM3.1，
使 mask 与 HorizonStream depth 使用完全一致的裁剪后像素坐标。

## 4. SAM3.1 实例追踪

每一个 prompt 使用一个独立的 SAM3.1 forward tracking session：

```text
prompt
  → 从第 0 帧开始检测
  → 目标首次出现时创建 track
  → 只向后续帧传播
  → 转换为 pipeline 内的 persistent instance ID
```

当前追踪策略包括：

- forward-only，不使用未来帧回溯修正过去帧；
- hot-start delay 设置为 0；
- 连续检测确认阈值设置为 1；
- grounding batch size 为 1；
- 视频帧默认 offload 到 CPU，以降低显存峰值；
- 每个 prompt 最多保留 16 个目标；
- 所有 prompt 合计最多保留 16 个实例；
- birth mask 少于 128 像素或覆盖超过 90% 图像时丢弃；
- 不同 prompt 的轨迹在共同可见帧上 IoU 达到 0.80 时去重；
- 实例 ID 按首次出现帧、prompt 顺序和 SAM 原始 ID 确定。

SAM 的原始 track ID 只作为 provenance 保存。最终导出的 `instance_id` 是 pipeline
内部的连续 ID，不等于 SAM 模型内部的 object ID。

## 5. 从深度到世界坐标

`HorizonStreamGeometryCacheAdapter` 将 cache 中的 `world_to_camera` 求逆得到
`camera_to_world`。对像素 `(u,v)` 和 camera-z depth `z`，先计算相机坐标：

```text
X_camera = ((u-cx)/fx * z,
            (v-cy)/fy * z,
            z)
```

再变换到统一世界坐标：

```text
X_world = R_camera_to_world * X_camera + t_camera_to_world
```

每一帧因此得到一张 `[H,W,3]` 的世界坐标点图。有效点必须满足：

```text
depth 有限且大于 0
geometry confidence >= 0.30
```

## 6. 世界坐标体素融合

`SemanticMapBuilder` 按帧 ID 递增处理 geometry frame 和 segmentation frame。当前
体素大小为 5 cm：

```text
voxel = floor(X_world / 0.05)
```

### 完整场景地图

所有有效深度点都会进入完整场景 voxel map，不要求像素属于某个 prompt。每个体素
保存：

- confidence 加权后的点位置；
- confidence 加权后的 RGB；
- 观测次数；
- 如果有语义观测，则保存 dominant category 和 instance ID。

### 语义实例地图

对每个 SAM observation：

1. 将 mask 对齐到几何图像尺寸；
2. 取 `valid_geometry AND mask` 的像素；
3. 使用 `geometry_confidence * SAM_track_score` 作为点权重；
4. 每个 observation 最多保留 8,000 个最高权重点；
5. 写入对应实例的语义 voxel map。

当前 live SAM observation 没有额外的 `static_score`，而配置
`require_static_score=false`，所以通过 score 和几何过滤的 live observation 会被
当作静态实例写入语义地图。单个实例的 track 原始点最多保留 250,000 个。

需要注意：当前 `SemanticMapPipeline.run()` 会先让两个 provider 完成整段序列的
推理，然后按照递增帧 ID 调用 mapper。因果性主要由 HorizonStream 的缓存推理和
SAM 的 forward tracking 保证；mapper 本身按帧更新、不读取未来观测，但当前命令
不是严格意义上的一帧输入、一帧立即输出的低延迟版本。

## 7. SAM instance-guided pose refinement（默认关闭）

设置命令文件中的 `OBJECT_POSE_REFINEMENT=1` 可运行独立的 training-free pose-graph
ablation。它使用 SAM persistent instance ID 选择大时间间隔的 frame pair，在两个 mask 内
做 RGB patch matching，把各自 depth/intrinsics 反投影成 local-camera 3D correspondence，
经 RANSAC/weighted SVD 和 conservative pose-consistency gate 后，加入 HorizonStream
相邻帧 relative pose 组成 robust pose graph。只有 camera pose 被替换，原始 depth、
intrinsics、SAM mask/ID 和 HorizonStream cache 都不变；随后复用原 mapper 生成 refined
map。

该开关默认关闭，因此普通 baseline 命令的行为不变。开启后输出 `raw_pose/` 和
`object_pose_refined/` 两份地图，以及 `object_pose_refinement/` 下的候选、接受/拒绝
edge、轨迹和统计。完整参数与 A0/B0 运行方式见
[`object_pose_refinement.md`](object_pose_refinement.md)。

## 8. 输出文件

输出目录由命令文件中的 `OUTPUT_DIR` 指定，当前为：

```text
/data184/open_source/vggtSam/outputs/semantic_map_50frames_horizonstream_consecutive
```

| 文件 | 内容 |
|---|---|
| `scene_rgb_map.ply` | 所有帧的有效世界点，经过 5 cm 体素融合，使用 RGB 着色 |
| `scene_semantic_map.ply` | 与完整场景相同；有 prompt 标签的实例使用实例颜色，未标注区域使用变暗 RGB |
| `semantic_map.ply` | 仅静态 prompt 实例的体素融合地图，按实例着色 |
| `rgb_map.ply` | 与 `semantic_map.ply` 使用相同的物体体素，但按 RGB 着色 |
| `objects/<category>_<id>.ply` | 单个实例的体素融合点云，例如 `bed_0.ply` |
| `object_tracks.ply` | 每帧物体观测点直接按 track 拼接，未做体素融合，并保留 `frame_id` |
| `object_tracks.json` | 每个实例的类别、帧范围、点数、包围盒和融合 PLY 路径 |
| `semantic_map.pt` | 场景体素、语义体素、实例 track、权重和运行元数据 |
| `map_summary.json` | 输出文件索引、类别列表、实例统计和后端信息 |
| `horizonstream_geometry.pt` | 两个 conda 环境之间传递的深度、pose、内参和 processed RGB cache |

其中：

- `scene_*` 和 `objects/*` 是世界坐标下的体素融合结果；
- `object_tracks.ply` 是未体素融合的逐帧观测集合，因此同一物体可能出现重复点或
  多层点；
- PLY 中保存的是 `category_id`、`instance_id`、`evidence_weight`、`observations`
  和 `frame_id` 等数值字段，类别字符串和文件命名信息保存在 JSON 中。

## 8. 代码结构与可扩展性

核心实现位于 [`streaming_couping/src/semantic_mapping/`](../src/semantic_mapping/)：

- `contracts.py`：定义统一的 `GeometryFrame`、`SegmentationFrame` 和 provider 接口；
- `adapters.py`：适配 HorizonStream cache、StreamVGGT 和 SAM3.1；
- `geometry.py`：深度反投影、RGB 和 mask 尺寸处理；
- `mapping.py`：世界坐标点云、语义实例和体素证据融合；
- `pipeline.py`：串联 geometry provider、segmentation provider 和 mapper；
- `export.py`：导出 PLY、PT、JSON。

因此以后替换为 HorizonStream 之外的几何后端时，主要新增一个符合
`GeometryProvider` 的 adapter，mapping、SAM 和导出层不需要重写。接口既支持直接
提供 world pointmap，也支持提供 `depth + intrinsics + camera_to_world`，后者由通用
mapper 反投影。

## 9. 当前明确未启用的内容

当前 baseline 没有：

- 训练或微调；
- StreamVGGT 几何推理；
- ICP、BA 或 loop closure；
- affine correction；
- historical-depth veto 或 quantile veto；
- 使用 SAM 结果修改 HorizonStream pose/depth；
- 语义反馈几何；
- GT 参与候选生成或正式建图。

旧的 StreamVGGT/V0、历史深度 veto 和多场景 affine 实验仍属于诊断或兼容代码，不能
当作当前 HorizonStream baseline 的性能结论。当前 HorizonStream 的真实性能应以服务器
实际运行后的几何轨迹、完整场景点云、实例点云和评测结果为准。

## 10. 已实现但默认关闭的下一步实验层

为了让后续优化可以复用同一套 pipeline，当前代码还提供两个显式 opt-in 的模块，默认
不会改变 baseline：

- `--geometry-guidance`：使用当前实例在更早帧中的世界点，投影成当前图像的 box 和
  positive points，向 SAM3.1 发起第二个候选，并在候选通过面积、历史支撑和几何分数
  门控后才替换 raw mask。提示只读历史帧，不修改 HorizonStream 的 depth、pose 或
  SAM 的 persistent track ID；候选失败自动回退 raw mask。
- `--map-write-gate`：为每个实例维护短期 observation memory 和长期高质量 anchor，
  对首次出现、重入、面积突变、三维中心不一致或低置信度观测延迟/拒绝语义体素写入。
  物体仍保留在 `object_tracks` 中，因此该门控只影响 semantic map，不会删除追踪结果。

命令文件顶部的 `GEOMETRY_GUIDANCE=0` 和 `MAP_WRITE_GATE=0` 保证默认输出仍是 raw
SAM + HorizonStream。启用任一模块时应使用新的 `OUTPUT_DIR`，并比较 `map_summary.json`
中的 `segmentation_guidance_summary`、`object_memory`、`map_write_gate` 与 raw baseline。
这两个模块目前是可审计的实验组件，不应在新的 scene-disjoint 验证通过前宣称为正式改进。
