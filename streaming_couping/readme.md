# HorizonStream + SAM3.1 语义地图

当前主线是免训练的 RGB + 文本 prompt 语义实例地图：

- HorizonStream 输出 metric depth、相机内参、在线因果位姿和置信度；
- SAM3.1 输出 prompt 对应的 mask 与 persistent instance ID；
- 通用 mapping 层把每帧 mask 内的深度反投影到世界坐标并做多帧体素融合；
- 每个静态实例单独输出 `<category>_<instance_id>.ply`。

StreamVGGT 仍可作为兼容后端，但不再是默认几何来源。历史优化实验没有通过多场景 gate，
当前 pipeline 不启用 temporal point prompt、historical-depth veto 或 affine correction。

## 直接运行

服务器先同步代码和子模块：

```bash
git checkout main
git pull --ff-only origin main
git submodule update --init externals/horizonstream externals/sam3
zsh streaming_couping/commands_run_semantic_map.txt
```

唯一需要日常修改的文件是 `commands_run_semantic_map.txt` 顶部配置。默认值为：

```text
scene                 00a231a370
manifest positions    90,104,...,776
frame count           50
prompts               bed wardrobe chair rug dustbin
HorizonStream weight  /home/bod/86Nas/95_data_bak/FoundationModels/HorizonStream.pt
HorizonStream env     /home/huawei/miniconda3/envs/horizonstream/bin/python
SAM env               /home/huawei/miniconda3/envs/3am/bin/python
output                 /data184/open_source/vggtSam/outputs/semantic_map_50frames_horizonstream
```

不需要先运行 V0 baseline，也没有训练步骤。命令串行启动两个独立进程：

1. `horizonstream` 环境生成 `horizonstream_geometry.pt`，随后进程退出并释放模型和显存；
2. `3am` 环境加载几何 cache，运行 SAM3.1 并构建语义地图。

这样不会把 HorizonStream 的 PyTorch 2.8 依赖安装进 `3am`，也不会让两个大模型同时驻留
GPU。cache 会校验原图顺序、权重、源码 revision 和推理设置；完全匹配时自动复用。修改
帧选择或 HorizonStream 参数后会自动重建。若想强制重建，设置
`HORIZON_REUSE_CACHE=0`。

显存不足时，先把 `HORIZON_SLIDING_SIZE` 从 `21` 调成 `1`；SAM 默认使用稳妥的
`SAM_GROUNDING_BATCH_SIZE=1`，确认显存充足后可调成 `2` 或 `4`。两阶段使用不同进程，因此也可以分别
设置 `HORIZON_CUDA_VISIBLE_DEVICES` 和 `SAM_CUDA_VISIBLE_DEVICES`。

## 输入与帧数

`FRAME_START`、`FRAME_STRIDE`、`FRAME_COUNT` 都是展开后的序列位置，`FRAME_COUNT=0`
表示取完剩余帧。默认 manifest 是此前一直使用的 processed ScanNet++ 数据：

```text
/data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d/manifest.json
```

设置 `MANIFEST_PATH=""` 后可通过 `FRAME_PATHS` 指定图片或目录。HorizonStream 和 SAM 使用
同一个冻结图片清单；HorizonStream 实际看到的 center-cropped RGB 也保存在 cache 中并交给
SAM，避免 depth 与 mask 像素错位。

`INPUT_MODE="cache"` 是旧 V0 cache 的离线重放模式。`GEOMETRY_BACKEND="streamvggt"`
可临时回退到旧后端。

## 输出

| 文件 | 内容 |
|---|---|
| `scene_rgb_map.ply` | 所有有效深度按世界坐标融合后的完整 RGB 场景 |
| `scene_semantic_map.ply` | 完整场景，prompt 实例使用 instance 颜色 |
| `semantic_map.ply` | 仅静态 prompt 实例，多帧体素融合 |
| `rgb_map.ply` | 与 `semantic_map.ply` 相同的体素，使用 RGB 着色 |
| `objects/<category>_<id>.ply` | 单个静态实例的融合点云，例如 `bed_0.ply` |
| `object_tracks.ply` | 各帧 object mask 对应 3D 点的直接汇总，保留 `frame_id`，不做体素融合 |
| `semantic_map.pt` | 完整场景、语义体素、轨迹和运行元数据 |
| `map_summary.json` | 输出索引、实例统计和后端信息 |
| `horizonstream_geometry.pt` | 两个 conda 环境之间传递的 depth/pose/intrinsics/RGB cache |

这些 PLY 都来自多帧观测。`scene_*` 和 `objects/*` 是世界坐标下的体素融合结果；
`object_tracks.ply` 才是未融合的逐帧观测集合。当前场景使用 `rug` 和 `dustbin`，不是
`mat`。

## 代码边界

`src/semantic_mapping/` 只依赖统一 contract：

- `GeometryFrame`: `depth + intrinsics + camera_to_world` 或直接 world pointmap；
- `SegmentationFrame`: 每帧 prompt mask、分数和 persistent instance ID；
- `SemanticMapBuilder`: 后端无关的反投影、track 保存和体素融合；
- adapters: HorizonStream cache、StreamVGGT、SAM3.1 和旧 V0 cache。

HorizonStream 源码固定为子模块 `externals/horizonstream`，当前 revision 为 `9602f53`；权重
保留在 NAS，不提交到 Git。详细设计见 [系统 pipeline](docs/system_pipeline.md)，已有实验证据
见 [当前状态](docs/current_status.md)。
