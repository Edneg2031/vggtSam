# HorizonStream + SAM3.1 语义地图系统

更新时间：2026-08-31

## 1. 系统目标

输入一段 RGB 和若干文本 prompt，输出带完整场景上下文、persistent instance ID 和每个物体
独立点云的语义地图：

```text
RGB + prompts
  ├─ geometry provider     → metric depth, confidence, K, camera_to_world
  └─ segmentation provider → category, mask, persistent instance ID, score
                                  ↓ frame-aligned contract
                        causal world-coordinate fusion
                                  ↓
                   scene map + semantic map + object PLYs
```

mapping 层不依赖具体几何网络。HorizonStream 是当前默认 provider；StreamVGGT 只是兼容
adapter，未来替换其他 streaming geometry backend 时无需修改融合与导出逻辑。

## 2. 两进程执行

HorizonStream 要求独立的 PyTorch 2.8 环境，SAM3.1 当前运行在 `3am`。唯一 zsh 入口按顺序
执行：

```text
process A: horizonstream env
  selected image list → HorizonStream → horizonstream_geometry.pt
  exit and release GPU/model memory

process B: 3am env
  geometry cache + aligned RGB → SAM3.1 → SemanticMapBuilder → exports
```

两个模型不会同时驻留 GPU，也不把 HorizonStream 安装进 `3am`。选帧由共享的
`rgb_inputs.py` 完成，两阶段再次核对完整绝对路径顺序，避免 geometry 第 n 帧与 SAM 第 n 帧
对应不同图片。

## 3. HorizonStream 几何

第一阶段复用上游 `horizonstream_infer.yaml` 的模型结构和 release checkpoint：

```text
window_size              10
sliding_size             21
input long edge          518
patch size               14
center crop              enabled
metric readout token     enabled
pose                     online motion averaging
offline/loop closure     disabled
output offload           CPU after every chunk
```

每个 chunk 调用 `HorizonStreamModel.forward_chunk`。上游 `depth` 是 OpenCV camera-z metric
depth；`pose_encoding_to_extri_intri` 返回 `[R|t]` 形式的 `world_to_camera` 和像素内参。cache
adapter 将 pose 求逆成 `camera_to_world`，通用 backprojection 使用：

```text
X_camera = [(u-cx)/fx * z, (v-cy)/fy * z, z]
X_world  = R_camera_to_world * X_camera + t_camera_to_world
```

上游 `depth_conf` 使用 `1 + exp(x)`，不是 `[0,1]` 概率。cache 生成器与旧 baseline 一致，
按每帧 q05/q95 做 robust normalization，再应用 mapping 的 confidence threshold。

## 4. 几何 cache contract

`horizonstream_geometry.pt` 是两个环境之间唯一的模型数据边界：

```text
schema/schema_version
backend = horizonstream
image_paths
frame_ids/source_positions
depth                 [S,H,W]
confidence            [S,H,W], normalized to [0,1]
world_to_camera       [S,3,4]
intrinsics            [S,3,3]
processed_rgb         [S,H,W,3], uint8
processed_size/source_sizes
scale_type = metric
window_size/sliding_size/checkpoint
request fingerprint
```

保存 `processed_rgb` 是必要的像素对齐措施。HorizonStream 的 center crop 不等价于把原图简单
resize 到 `[H,W]`；第二阶段将这些精确像素暂存为 JPEG 交给 SAM，并让 mask 直接输出到同一
`processed_size`。否则边缘区域的 mask 会投影到错误的深度像素。

cache reuse 同时比较图片顺序、checkpoint 路径和文件状态、上游 YAML hash、子模块 revision、
window/sliding/crop/precision 等设置。任一项变化就重新推理。

## 5. SAM 与实例 ID

每个 prompt 使用独立的 forward-only SAM3.1 session：

```text
prompt → online discovery → birth frame → permanent track ID → forward masks
```

不同 prompt 的 track 在 birth 时做保守 overlap 去重，再分配 pipeline 内永久 instance ID。
SAM 只负责 category、ownership 和 identity，不读取未来帧来修改过去结果，也不写回
HorizonStream pose/depth。

## 6. 地图融合

对每一帧按顺序处理：

1. depth、K 和 camera pose 反投影为 world points；
2. confidence gate 过滤无效几何，并写入完整 scene voxel map；
3. SAM mask 选择属于各实例的 world points；
4. track score/static gate 决定 observation 是否接受以及是否写入静态语义地图；
5. 同一 world voxel 累积 confidence-weighted position/RGB/category/instance evidence；
6. 动态或不确定目标保留在 object track，但不污染静态 map。

场景点云仍然来自多帧融合，而不是网络直接输出一个无重复 mesh。HorizonStream 为每帧提供
depth 和共享坐标系位姿；系统必须把各帧反投影后融合，才能得到可导出的全场景和实例点云。

## 7. 输出语义

- `scene_rgb_map.ply`: 所有帧的有效世界点经过体素融合，RGB 着色；
- `scene_semantic_map.ply`: 同一完整场景，静态 prompt 实例覆盖 instance 颜色；
- `semantic_map.ply`: 仅静态 prompt 实例的融合体素；
- `rgb_map.ply`: 同一 object-only map 的 RGB 版本；
- `objects/<category>_<id>.ply`: 每个静态实例的独立融合点云；
- `object_tracks.ply`: 未体素融合的逐帧实例观测点，保留 `frame_id`；
- `semantic_map.pt`: 上述 tensor、track 和 provenance；
- `map_summary.json`: 统计与文件索引。

## 8. 当前边界

- 当前没有训练、BA、ICP、loop closure 或语义反馈几何；
- HorizonStream metric scale 与在线 pose 的真实跨场景质量尚需服务器实验验证；
- prompt 不是 class-agnostic 全图识别，未写入 prompt 的类别不会获得语义标签；
- 当前静态判断主要沿用 track/static gate，不等价于完整动态场景建模；
- 历史 affine、depth veto 和 temporal point prompt 均为 NO-GO，不自动启用。

主运行入口是 `streaming_couping/commands_run_semantic_map.txt`。默认场景、帧数、权重、两个
Python 路径、GPU 与 cache 都集中写在该文件顶部。
