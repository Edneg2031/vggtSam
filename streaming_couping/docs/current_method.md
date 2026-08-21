# 当前方法说明

## 1. 研究目标

在冻结 StreamVGGT 几何 backbone 的基础上，利用多视角 RGB 序列和 object-level structure，改善 dense world-coordinate pointmap 的质量。

核心问题是：

```text
StreamVGGT 提供 metric geometry：where
SAM 提供 object partition：which surface/object
```

SAM 不负责预测 depth 或 XYZ，只提供物体分组、边界和跨帧实例对应关系。

## 2. 当前 baseline

```text
RGB sequence
    ↓
冻结的 StreamVGGT full-history inference
    ↓
geometry features / raw pointmap / camera pose
    ↓
pointmap evaluation
```

Residual baseline 使用：

```text
X_enhanced = X_raw + ΔX
```

其中 StreamVGGT 和原始 PointHead 冻结，只训练 residual decoder。候选点云生成阶段不使用 GT、SAM 或训练数据。

Pose 和 pointmap 分开评价。Pose 提升不等于 world-coordinate pointmap 一定提升。

## 3. 计划中的主方法

```text
StreamVGGT geometry feature F_geo
            +
SAM instance masks / persistent IDs / boundary
            ↓
Object-aware geometry decoder
            ↓
ΔXYZ
            ↓
X_enhanced = X_raw + ΔXYZ
```

第一版只使用 SAM 的 object structure，不使用 SAM hidden token：

```text
F_obj(i) = Pool(F_geo inside instance i)

F_decoder = local F_geo + corresponding F_obj + boundary feature
```

StreamVGGT 和 SAM backbone 都冻结，只训练 geometry decoder，并使用 GT pointmap 监督 XYZ 修正。

## 4. 数据与验证协议

当前有 4 个完整可用场景：

- `00a231a370`：从 `/home/bod/184Nas/open_source/scannet_pp_pinhole` 读取 pinhole 数据。
- 其他三个场景：从 `/data184/open_source/vggtSam/source/scanet_pp_pinhole` 读取。
- semantic/instance annotations：从 `/home/bod/184Nas/open_source/scannet_pp/data` 读取。
- 生成结果：写入 `/data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d`。

`1be2c31cac` 的 pinhole/annotation 资产不完整，已排除，不参与当前实验。

原始 RGB 只读引用，不复制图片。

当前 4 场景 pilot 使用 scene-disjoint protocol：

```text
Train: 2 scenes
Val:   1 unseen scene
Test:  1 completely unseen scene
```

当前固定 split 为：`Train={00a231a370, 0a184cf634}`、
`Val={1a8e0d78c0}`、`Test={1eacc65607}`。模型选择和 checkpoint 选择只能使用
validation scene。最终 test scene 只评价一次。条件允许时，进行 4-fold scene rotation；
要恢复 `3/1/1`，还需要新增一个完整场景。

## 5. 实验顺序

### Stage 0：数据完整性

确认每个场景的 RGB、COLMAP、semantic mask、instance mask 和 GT pointmap 完整。

运行文件：

```bash
zsh streaming_couping/commands_generate_scannetpp_data.txt
```

### Stage 1：geometry-only baseline

比较：

1. 原始 StreamVGGT full-history PointHead。
2. 冻结 StreamVGGT + trainable residual decoder。
3. 冻结 StreamVGGT backbone + trainable geometry PointHead/decoder。

当前入口：

```bash
zsh streaming_couping/commands_prepare_multiscene_geometry_cache.txt
zsh streaming_couping/commands_train_multiscene_geometry.txt
zsh streaming_couping/commands_train_multiscene_geometry_direct.txt
```

Stage 1 的 C 分支是预先登记的 Direct Geometry Decoder：冻结
StreamVGGT aggregator/backbone 和 CameraHead，只训练 released original
PointHead。它使用同一个 multi-scene cache、validation-only checkpoint selection
和一次性 test scoring；不能因为 residual 的 test 结果而修改 residual 或 Direct
分支的固定训练协议。当前 pilot 是 `2 train / 1 validation / 1 test`，不是最终的
`3 / 1 / 1` 五场景 protocol。

### Stage 2：object-aware geometry

在 Stage 1 最稳定的 geometry decoder 上加入：

1. 正确的 SAM instance grouping。
2. shuffled instance ID control。
3. shifted mask control。

只有正确的 SAM structure 在 held-out scenes 上稳定优于 geometry-only 和 controls，才能声称 object structure 对 metric geometry 有帮助。

## 6. 已知结果与限制

- QK retrieval 对 pose 有过提升，但小预算 history retrieval 没有稳定改善 pointmap。
- SAM-guided history retrieval、SAM memory 和多种 token fusion 已有实验，但尚未得到稳定的跨场景 causal improvement。
- 单场景 residual 在 `0a184cf634` 上提升，在 `1a8e0d78c0` 上恶化，因此目前不能声称 residual 已经跨场景泛化。
- RetrieveVGGT-style medium-budget segment sampling 尚未作为独立 baseline 完成验证。

因此当前重点是：先完成多场景 geometry-only baseline，再判断 SAM object-aware decoder 是否真正带来额外收益。
