# NAS 存储与 ScanNet++ 重新处理

大体积数据统一放在挂载目录：

```text
/home/bod/184Nas/process_data/vggtSam/
├── source/
│   ├── scannetpp_pinhole/
│   │   ├── data/<scene_id>/
│   │   │   ├── images/
│   │   │   ├── colmap/
│   │   │   └── mesh_aligned_0.05.ply
│   │   ├── metadata/semantic_classes.txt
│   │   └── scannetpp_part_valid_scenes.txt
│   └── scannetpp/data/<scene_id>/scans/
│       ├── mesh_aligned_0.05_semantic.ply
│       ├── segments.json
│       └── segments_anno.json
├── data/processed/scannetpp_pinhole_2d/
│   ├── manifest.json
│   └── <scene_id>/
│       ├── semantic_masks/
│       ├── instance_masks/
│       └── pointmaps/
└── outputs/
    ├── streaming_couping_v0/
    ├── streaming_couping_t1_feature_tta/
    └── processed_dataset_audit/
```

代码仍保留在 `/home/wlh50060092/vggtSam`，StreamVGGT 与 SAM3.1 checkpoint
仍使用原来的 `/home/bod/86Nas/...`，不复制到新挂载目录。

## 统一路径变量

服务器命令默认使用：

```bash
export VGGT_SAM_STORAGE_ROOT=/home/bod/184Nas/process_data/vggtSam
```

如果以后挂载点变化，只需要覆盖这个变量。V0 配置、恢复配置、T1 配置和
ScanNet++ 预处理配置都会展开该变量。

## 推荐处理顺序

先把原始场景文件放进上面的 `source/` 结构。然后依次执行：

如果服务器环境缺少预处理依赖，先运行一次：

```bash
/home/huawei/miniconda3/envs/3am/bin/python -m pip install -e '.[preprocess]'
```

然后依次执行：

```bash
# 1. 只检查输入文件，不生成 mask/pointmap，也不覆盖正式 manifest
SCANNETPP_PREPROCESS_MODE=preflight \
SCANNETPP_SCENES="00a231a370 1d34f56de8" \
zsh streaming_couping/commands_prepare_scannetpp_nas.txt

# 2. 先生成一个场景的 5 帧与可视化，检查投影/坐标是否正确
SCANNETPP_PREPROCESS_MODE=debug \
SCANNETPP_SCENES="00a231a370" \
zsh streaming_couping/commands_prepare_scannetpp_nas.txt

# 3. 检查无误后，增量生成完整场景；已有完整帧会跳过
SCANNETPP_PREPROCESS_MODE=full \
SCANNETPP_SCENES="00a231a370 1d34f56de8" \
zsh streaming_couping/commands_prepare_scannetpp_nas.txt
```

也可以不设置 `SCANNETPP_SCENES`，此时优先读取
`scannetpp_part_valid_scenes.txt`，不存在时扫描 `data/` 下的场景。

每次非预检运行后会进行只读资产审计，检查 RGB 和 pointmap 是否齐全，
并把结果写到：

```text
${VGGT_SAM_STORAGE_ROOT}/outputs/processed_dataset_audit/summary.json
```

该审计不读取 GT pointmap 数值，只看 manifest、文件存在性和文件大小。
默认不重复复制 RGB，也不保存后续实验不需要的 z-buffer/raster 中间文件，
以减少 NAS 占用。

## 重新生成 V0

处理后的 manifest 固定为：

```text
${VGGT_SAM_STORAGE_ROOT}/data/processed/scannetpp_pinhole_2d/manifest.json
```

随后直接运行：

```bash
BASELINE_REBUILD_CACHE=1 BASELINE_REBUILD_QK=1 \
zsh streaming_couping/commands_v0_baseline.txt
```

新的 V0 cache、QK pose 和语义地图都会写入 NAS 的 `outputs/`。

## 旧文件处理原则

这些命令不会自动移动、覆盖或删除仓库本地的旧 `data/`、`outputs/`。确认
NAS 新结果完整后，再由人工决定是否删除旧副本。不要在正式重处理前把旧
manifest 与新 manifest 混合。
