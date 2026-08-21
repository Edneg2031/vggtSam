# StreamVGGT + SAM3.1 最近实验总结

更新时间：2026-08-21

本文记录从 V0 到当前多场景 geometry-only pilot 的方法、数据、有效结果和无效结果。
结论只在对应实验协议和场景范围内成立，不把单场景结果写成跨场景泛化结论。

## 1. 项目目标和正式系统

当前项目优先目标是交付一个稳定的 streaming semantic mapping system：

```text
RGB sequence
    ├── StreamVGGT → camera pose + raw world pointmap
    └── SAM3.1    → instance mask + persistent instance ID
                                      ↓
                         3D semantic / instance map
```

正式 V0 不训练模型，也不修改原始点云：

```text
Pose     = first-frame anchor + native QK Top-4 → frozen CameraHead
Pointmap = full-history StreamVGGT → original PointHead
Semantic = SAM3.1 persistent mask/slot/track ID lifted onto raw pointmap
```

因此 pose、pointmap 和 semantic label 是三个分开的输出。QK pose 变好，不代表 pointmap 变好；
SAM 产生语义属性，不改变 XYZ。

## 2. V0：冻结的 semantic mapping baseline

### 方法

1. StreamVGGT full-history PointHead 生成原始 world-coordinate pointmap。
2. 第一帧 anchor 加 native QK Top-4，只用于 CameraHead pose replay。
3. SAM3.1 根据 prompt 做实例发现和跨帧 persistent tracking。
4. 每个实例第一次出现时分配永久 slot，不复用 slot。
5. 将 mask、persistent slot 和 track ID lift 到 raw world pointmap。
6. 输出彩色点云、语义/实例点云、相机轨迹和实例统计。

SAM 不参与 QK retrieval，也不参与 pointmap 生成；不使用 ICP、BA、三角化或点云后处理。

### V0 使用的 prompt

```text
wardrobe, chair, rug, desk, cabinet, nightstand,
dustbin, box, guitar case
```

这些 prompt 来自 annotation-only 的人工场景盘点，只用于模拟人工选择 prompt，不进入候选生成，
也不能作为自动语义识别能力的证据。

### 有效结果

在原始 30 帧单序列上，QK pose 相对 raw pose：

```text
camera center error 改善：10.93%
rotation error 改善：6.15%
```

V0 已验证的结论是：训练自由的 QK history retrieval 可以改善当前序列的 pose；SAM 可以提供
persistent instance identity 并把语义投影到 raw world pointmap。

### V0 没有证明的结论

- 没有证明 SAM 改善 pose。
- 没有证明 SAM 改善 pointmap 几何。
- 没有证明 QK retrieval 改善 pointmap。
- 没有证明跨场景泛化。

正式 V0 运行入口：

```bash
zsh streaming_couping/commands_v0_baseline.txt
```

## 3. History retrieval 和 pointmap 试验

### 已测试

- full-history pointmap。
- QK Top-K history retrieval。
- recent-frame retrieval。
- geometry-diverse retrieval。
- SAM-guided retrieval、SAM memory、mask 内 QK。
- SAM local token、cross-attention、geometry Value 和 shuffled/wrong ID controls。

### 有效部分

QK retrieval 对 pose 有实际帮助，尤其是当前 V0 的 first-frame anchor + Top-4 camera replay。

### 无效或未稳定复现的部分

- 小预算 QK Top-4 对 pointmap 没有稳定提升。
- recent Top-K 对 geometry 的表现较差。
- geometry-diverse Top-K 没有超过 full-history。
- SAM 用来挑 history 没有形成稳定的 geometry causal gain。
- shuffled ID 有时不比正确 ID 差，不能证明 persistent ID 本身改善了 geometry。
- RetrieveVGGT 风格的完整 Segment Sampling + medium-budget `K=16/24/32` 尚未正式完成，
  因此不能据此否定 RetrieveVGGT-style retrieval。

早期 pointmap 对比中曾观察到类似趋势：full-history 优于小预算 retrieval；该结果只作为工程诊断，
不作为当前正式跨场景结论。

## 4. Phase 1：单场景 residual development

### 原始 protocol

旧实验只使用一个场景、30 帧 temporal split：

```text
train = 18 frames
val   = 6 frames
test  = 6 frames
```

模型读取冻结 StreamVGGT cache 中的 DPT point-head features，训练外接 residual head：

```text
X_new_native = X_raw_native + gate * ΔX
```

StreamVGGT backbone、原始 PointHead、CameraHead 和 SAM 全部冻结。GT 只用于训练监督、validation
model selection 和 test scoring，不参与候选生成。

### r2 的改进

r1 直接使用 final epoch，出现 train loss 继续下降而 validation 早期变差的问题。r2 改为：

```text
best checkpoint = minimum validation RMSE
tie-break       = minimum validation P90
```

test 只在 checkpoint 冻结后评价一次。

### 跨场景 R3 结果

使用 r2 `residual_no_gate` checkpoint，在两个 held-out scene 上做冻结 inference：

| Scene | Raw RMSE | Residual RMSE | Raw P90 | Residual P90 | Improved frames | Decision |
|---|---:|---:|---:|---:|---:|---|
| `0a184cf634` | 1.234221 | 1.220961 | 2.165761 | 2.137511 | 23/30 = 0.7667 | GO |
| `1a8e0d78c0` | 0.159830 | 0.172990 | 0.225987 | 0.250606 | 5/30 = 0.1667 | NO_GO |

### 结论

单场景 residual 有局部 development signal，但跨场景泛化没有建立。一个场景提升、另一个场景恶化，
说明当前不能继续把问题解释成“只需要调 gate 或 loss”。更合理的怀疑是训练场景、尺度、误差幅度和
几何分布不足以代表 held-out scene。

旧的单场景 residual 和 R3 command 已归档，不作为当前新实验入口。

## 5. 当前数据状态

### 可用场景

processed audit 已通过，当前有 4 个完整场景，共 1950 帧：

| Scene | Frames | Pinhole source | Annotation source |
|---|---:|---|---|
| `00a231a370` | 795 | `/home/bod/184Nas/open_source/scannet_pp_pinhole` | `/home/bod/184Nas/open_source/scannet_pp/data` |
| `0a184cf634` | 332 | `/data184/open_source/vggtSam/source/scanet_pp_pinhole` | `/home/bod/184Nas/open_source/scannet_pp/data` |
| `1a8e0d78c0` | 338 | `/data184/open_source/vggtSam/source/scanet_pp_pinhole` | `/home/bod/184Nas/open_source/scannet_pp/data` |
| `1eacc65607` | 485 | `/data184/open_source/vggtSam/source/scanet_pp_pinhole` | `/home/bod/184Nas/open_source/scannet_pp/data` |

每个可用场景都已确认：

```text
RGB image       complete
GT pointmap     complete
semantic mask   complete
instance mask   complete
```

生成结果统一写入：

```text
/data184/open_source/vggtSam/data/processed/scannetpp_pinhole_2d
```

原始 RGB 使用 source reference，不复制图片。

### 排除场景

`1be2c31cac` 的 pinhole/annotation 资产不完整，缺少 images、COLMAP、pinhole mesh、semantic mesh、
`segments.json` 和 `segments_anno.json`，已从当前实验排除，不伪造 label。

## 6. 当前 multi-scene geometry-only pilot

由于当前只有 4 个完整场景，暂时使用：

```text
Train: 00a231a370, 0a184cf634
Val:   1a8e0d78c0
Test:  1eacc65607
```

这是 pilot protocol，不足以支撑很强的泛化声明。后续可做 4-fold scene rotation；要恢复
`3 train / 1 val / 1 test`，需要再补一个完整场景。

### 已写入的实验代码

第一阶段先不加 SAM，只缓存冻结 StreamVGGT 的 geometry features 和 raw pointmap：

```bash
zsh streaming_couping/commands_prepare_multiscene_geometry_cache.txt
```

缓存使用每个场景 2 个分离的 30 帧 episode，candidate generation 阶段只读取 RGB 和冻结 StreamVGGT。
GT 在 candidate features/raw pointmap 生成后才打开，用于固定 reference-frame Sim(3) 和监督数据准备。

缓存完成后训练 residual baseline：

```bash
zsh streaming_couping/commands_train_multiscene_geometry.txt
```

第一版只训练：

```text
frozen StreamVGGT backbone
frozen original PointHead
trainable residual_no_gate
SAM = 0
```

checkpoint 只能由 validation scene 的 RMSE/P90 选择，test scene 只做一次冻结评价。

### 当前尚未完成

- multi-scene trainable original PointHead 对比；
- SAM object pooling + boundary geometry decoder；
- shuffled-ID 和 shifted-mask controls 的新 protocol；
- 多 fold 的稳定性统计；
- RetrieveVGGT-style medium-budget Segment Sampling baseline。

## 7. 当前有效/无效结论总表

| 方法/组件 | 当前结论 | 证据范围 |
|---|---|---|
| StreamVGGT full-history raw pointmap | 有效的正式 geometry baseline | V0 及多场景数据 |
| QK Top-4 → CameraHead | 当前序列 pose 有效 | 单序列 V0，center -10.93%、rotation -6.15% |
| QK Top-4 → PointHead geometry | 未证明有效，早期结果偏弱 | 小预算 retrieval 对比 |
| Recent / geometry-diverse retrieval | 未超过 full-history | 早期 retrieval ablation |
| SAM persistent masks/IDs → semantic lifting | 有效作为 semantic mapping 系统组件 | V0 pipeline |
| SAM-guided history retrieval | 未得到稳定 geometry gain | 多个 SAM retrieval/fusion ablation |
| R3 frozen residual（单场景训练后跨场景评价） | 有 development signal，但不泛化 | 0a GO、1a NO_GO |
| Multi-scene residual | 尚未运行 | 当前 cache/training pilot |
| SAM object-aware geometry decoder | 尚未实现/验证 | 下一阶段 |
| RetrieveVGGT Segment Sampling | 尚未完成正式复现 | 不能下结论 |

## 8. 当前执行顺序

1. 完成 multi-scene frozen geometry cache。
2. 运行 residual-no-gate scene-disjoint pilot。
3. 如果 test scene 稳定超过 raw，再增加 trainable PointHead baseline。
4. 只有 geometry-only baseline 稳定后，才加入 SAM object pooling、boundary 和 persistent ID。
5. 如果 geometry enhancement 仍不稳定，项目仍以 V0 semantic mapping system 作为可交付结果，
   不继续通过增加 gate、UQ、BA、ICP 或复杂 fusion 强行挽救 pointmap。
