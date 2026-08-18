# StreamVGGT + SAM3.1 V0 语义地图方法

## 1. 任务

输入流式 RGB 序列和一组用户 prompt，输出：

- 相机轨迹；
- 共享世界坐标系中的彩色点云；
- 每个点的 SAM persistent slot、track ID 和 prompt 语义；
- 每个实例的出生帧、可见帧数和点数。

场景按静态世界处理。StreamVGGT 和 SAM3.1 均冻结，V0 不训练任何融合或位姿网络。

## 2. 固定方法

```text
RGB ──→ full-history StreamVGGT point head ──→ raw world pointmap ──┐
 │                                                                 │
 ├──→ first-frame anchor + native QK Top-4 ──→ camera head ──→ pose│
 │                                                                 ├──→ semantic_map.pt / .ply
 └──→ SAM3.1 prompt tracking ──→ persistent mask/slot/track ID ────┘
```

### 2.1 位姿

对当前帧只使用 RGB 产生的 StreamVGGT native Q–K similarity 检索历史帧。历史预算固定为5帧：
第一帧 anchor 加 QK Top-4。检索后只运行冻结的 camera head；不运行候选 depth/point head，
也不读取 SAM mask、ID、hidden feature 或 GT。

若 QK artifact 缺失，代码具备 raw pose fallback；正式 V0 验收要求 QK artifact 有效且没有触发 fallback。

### 2.2 点云

点云固定使用 full-history StreamVGGT point head 的原始 `world_pointmap`。该 pointmap 已位于
StreamVGGT 第一帧参考的共享世界坐标系，不再用 QK pose 对它进行二次变换，也不使用 QK 上下文
重新生成 pointmap。

### 2.3 SAM 实例语义

SAM3.1 根据配置中的 prompt 在线发现并追踪实例。每个实例第一次出现时占用一个永久 slot，slot
不复用，因此支持 late birth。每个像素的语义归属规则是：

当前语义地图 prompt 为 `wardrobe / chair / rug / desk / cabinet / nightstand / dustbin /
box / guitar case`。它们来自独立的 annotation-only 场景盘点；盘点结果只用于模拟人工查看 RGB 后
选择 prompt，不进入模型候选生成。`wardrobe` 作为重要场景语义保留在 V0 地图中；只读几何诊断
可单独排除或分组统计大区域，但不会改变这组 V0 prompts。其余 8 类是边界明确、重复可见的局部物体。
提示词不限制同类实例数量，永久 registry 总容量仍为 16。

1. 过滤低于 `track_score_threshold` 的 mask；
2. mask 重叠时选择 track score 更高的 slot；
3. 将 slot、SAM track ID 和 prompt 投影到同位置的 raw world pointmap；
4. 没有被任何 mask 覆盖的点保留在地图中，语义 slot/track ID 为 `-1`。

点坐标只经过置信度过滤和固定数量采样，不进行 ICP、BA、object fusion 或几何修正。
每个永久slot对应一个固定的高区分度颜色，因此同一实例跨帧始终同色，不同实例使用不同颜色；
未标注点显示为灰色。原始RGB另外保存，不被语义着色覆盖。

## 3. 坐标和模块边界

- `selected_world_to_camera`：QK retrieval camera head 输出；
- `world_points`：full-history raw StreamVGGT world pointmap；
- `semantic_slots/semantic_track_ids`：SAM mask 与 persistent registry；
- pose 与 pointmap 共享第一帧 reference，但来自不同 head/context；
- SAM 只改变点的语义属性，不改变相机位姿和三维坐标。

## 4. 输出

一次命令生成：

- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/baseline_summary.json`：tracking 与 pose 结果；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/poses.pt`：raw/QK 相机轨迹；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/semantic_map.pt`：完整结构化语义地图；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/semantic_map.ply`：按persistent slot着色的语义点云；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/rgb_map.ply`：保留原始图像颜色的点云；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/tracks.csv`：实例元数据；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/prompt_discovery.csv`：逐提示词的原始发现、出生门控、最终保留和 mask 支持统计；
- `${VGGT_SAM_STORAGE_ROOT}/outputs/streaming_couping_v0/semantic_map/semantic_map_summary.json`：pipeline 审计。

`semantic_map.pt`保存 world point、原始RGB、语义RGB、固定slot调色板、confidence、frame index、
semantic slot、SAM track ID、track metadata，以及raw/QK两条相机轨迹。

## 5. 运行

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

如果需要先人工选择更合适的 SAM prompt，可以单独运行：

```bash
zsh streaming_couping/commands_scene_object_inventory.txt
```

该命令只读取数据集实例标注与 mask，输出当前 30 帧可见的类别、实例数、可见帧数和面积，且明确
标记为 `manual_prompt_planning_only`。其结果不会被正式 V0 候选生成读取，因此不能作为模型自动识别
场景物体的实验结论；它只相当于人工查看场景后整理提示词。

该命令依次完成缓存、QK pose、tracking/pose审计和语义地图导出。通常复用已有缓存；显式重建方式：

```bash
BASELINE_REBUILD_CACHE=1 BASELINE_REBUILD_QK=1 \
zsh streaming_couping/commands_v0_baseline.txt
```

## 6. 已验证结论与限制

当前30帧单序列上，QK pose相对full-history raw pose：center error改善`10.93%`，rotation error
改善`6.15%`。正式点云保持raw world pointmap不变，因此V0不声明点云几何得到优化；SAM提供的是
实例语义与跨帧persistent identity。

生成 QK candidate 和 semantic map 时不读取GT。当前研究命令仍使用数据集GT在候选冻结后评价pose，
所以它是完整的离线实验pipeline；要支持任意无GT RGB目录，还需要把数据加载/缓存阶段的评测字段
拆成可选模块。

V0允许的结论是：训练自由的QK检索在当前序列改善pose，并使用未修改的StreamVGGT world pointmap
与SAM persistent tracking生成语义地图。V0不声明SAM改善pose、SAM改善几何或跨场景泛化。
