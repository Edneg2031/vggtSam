# streaming_couping V0

V0 是冻结、免训练的 StreamVGGT + SAM3.1 语义地图 pipeline：

- QK retrieval 只为 camera head 选择历史并输出相机轨迹；
- full-history raw StreamVGGT world pointmap 作为地图几何；
- SAM3.1 persistent mask、slot 和 track ID 作为点云语义；
- SAM 不参与 pose，点云不做后处理优化。

当前唯一的主运行入口是下面的语义地图命令；不需要先运行 V0 baseline：

```bash
zsh streaming_couping/commands_run_semantic_map.txt
```

## 语义地图 pipeline

新增的 `src/semantic_mapping/` 是独立于具体模型的语义实例地图层。它使用统一的
`GeometryFrame`、`ObjectObservation` 和 `SegmentationFrame` contract；当前的
StreamVGGT 与 SAM3.1 通过 adapter 接入，后续替换 HorizonStream 时只需新增几何 adapter，
不需要修改体素融合和导出逻辑。

从 RGB 和文本 prompt 运行当前模型：

```bash
zsh streaming_couping/commands_run_semantic_map.txt \
  --frames /path/to/rgb_frames \
  --prompts bed wardrobe \
  --output-dir outputs/semantic_map
```

也可以直接修改 `commands_run_semantic_map.txt` 顶部的 `FRAME_PATHS`、`PROMPTS` 和
`OUTPUT_DIR`，然后不带参数执行；命令行参数仍然可以临时覆盖这些默认值。帧选择使用
展开后排序的序列位置：`FRAME_START` 从 0 开始，`FRAME_STRIDE` 是步长，
`FRAME_COUNT=0` 表示取完剩余帧。例如 `FRAME_COUNT=30` 表示取前 30 帧。

默认 `MANIFEST_PATH` 是之前实验一直使用的 processed ScanNet++ 数据：
`/data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d/manifest.json`，
默认场景为 `00a231a370`，当前 795 帧 manifest 上默认采样序列为
`90,104,...,776`，共 50 帧。因此直接修改 prompt
和帧选择参数即可；不需要另填 RGB 目录。只有在把 `MANIFEST_PATH` 设为空时，才使用
`FRAME_PATHS` 中的显式图片文件或目录。当前 50 帧默认输出目录是
`/data184/open_source/vggtSam/outputs/semantic_map_50frames`，不会覆盖之前的 30 帧结果。

50 帧 SAM3.1 传播默认使用 `SAM_GROUNDING_BATCH_SIZE=4`，并启用
`SAM_OFFLOAD_VIDEO_TO_CPU=1`。前者只缩小高分辨率 grounding 的帧批次，后者把解码后的
视频帧保存在 CPU 按需送入 GPU；两者都不拆分 video session，也不重置 persistent track ID。
若服务器显存仍不足，可继续把 batch size 从 4 降为 2 或 1，代价是运行时间增加。

命令文件中 `INPUT_MODE="rgb"` 时读取 `FRAME_PATHS`；切换为
`INPUT_MODE="cache"` 时读取 `CACHE_PATH`。当前默认 cache 路径是：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_v0/cache/00a231a370_90_525_step15_37_68_54.pt
```

它只在 cache 模式使用；RGB 模式不会自动读取旧 cache。`OUTPUT_DIR` 默认是
`/data184/open_source/vggtSam/outputs/semantic_map`。

如果已经有 V0 cache，可以不重新运行模型：

```bash
zsh streaming_couping/commands_run_semantic_map.txt \
  --cache outputs/streaming_couping_v0/cache/<clip>.pt \
  --output-dir outputs/semantic_map
```

输出会同时保留完整场景、静态语义地图和逐帧 object track：

| 文件 | 包含内容 | 跨帧处理 |
|---|---|---|
| `scene_rgb_map.ply` | 所有通过置信度门限的场景点，使用 RGB 着色 | 所有选中帧按世界坐标叠加后做体素融合 |
| `scene_semantic_map.ply` | 同一完整场景；静态 prompt 物体按实例着色，其他场景区域显示为较暗 RGB | 所有选中帧按世界坐标叠加后做体素融合 |
| `semantic_map.ply` | 只包含静态 prompt 物体，按 persistent instance ID 着色 | 多帧体素融合 |
| `rgb_map.ply` | 与 `semantic_map.ply` 完全相同的物体体素，改用观测 RGB 着色 | 多帧体素融合 |
| `objects/<category>_<instance_id>.ply` | 每个静态实例单独保存，例如 `bed_0.ply`、`rug_3.ply` | 与语义地图相同的多帧体素融合，使用观测 RGB 着色 |
| `object_tracks.ply` | 每个 persistent track 的逐帧观测点，包含静态和动态目标 | 多帧点直接汇总，不做体素融合；点中保留 `frame_id` |

`semantic_map.pt` 保存上述完整场景体素、语义体素、轨迹和元数据，`object_tracks.json`
保存每个实例的类别、帧范围和包围盒，`map_summary.json` 保存输出索引和统计。
`object_tracks.ply` 不是由折线构成的运动轨迹图，而是各帧 mask 对应 3D 点的集合；因此同一
物体在多帧中的点可能重叠。静态目标写入语义体素地图，动态目标只保存在独立 track 中，避免
污染静态地图。默认使用当前 V0 的 raw pointmap，不启用被否决的 temporal point prompt、
历史深度 Veto 或 affine depth correction。

当前场景此前冻结实验使用的类别词是 `rug`（地毯）和 `dustbin`（垃圾桶），不是
`mat`。默认命令使用 `bed wardrobe chair rug dustbin`；完整的历史固定词表还包括
`desk cabinet nightstand box "guitar case"`。

## 历史实验

旧 baseline、诊断和优化实验的 `commands_*.txt` 已从当前运行目录移除；实验结果和结论
仍保留在文档中，仅作为历史记录，不属于当前 pipeline 的运行步骤。当前文档见：

- [当前状态与实验结论](docs/current_status.md)
- [系统 pipeline](docs/system_pipeline.md)
- [语义地图实验路线](docs/experiment_route.md)

当前单序列证据只支持 QK pose 改善；V0 不声明 SAM 改善 pose 或几何。SAM 的正式作用
是实例发现、跨帧 persistent identity 和语义投影。新方法必须在 scene-disjoint 数据上
与 raw V0 按同一协议比较，并报告 tracking、depth、map、恶化帧比例和 fallback/coverage。
