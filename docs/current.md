# 当前模型结构梳理

本文档用于 PPT 对齐当前代码状态、已实现模块和下一步需要补齐的模块。当前实现目标是验证：

```text
SAM3 semantic tokens + StreamVGGT geometry tokens
  -> latent fusion
  -> semantic / 3D point / cross-frame correspondence / instance mask
```

当前代码已经加入 token-level instance mask decoder；全局语义点云融合导出模块还没有实现。

## 1. 输入数据

每次训练采样一个 ScanNet++ 连续帧 clip：

```text
RGB sequence:
  T = 4
  resized RGB image paths

GT supervision:
  semantic_masks
  instance_masks
  GT pointmaps from mesh + COLMAP camera rasterization
  object metadata labels
```

默认过滤大结构和不需要追踪的类别：

```text
wall, floor, ceiling, window, door, door frame, pillar,
table, cabinet, storage cabinet, whiteboard,
electrical duct, electric duct
```

默认 prompt 策略：

```text
prompt_mode = random_instance
```

每个 clip 随机选择一个大小合理、跨帧可见、未被 blacklist 过滤的 instance，并使用它的类别名作为 SAM3 text prompt。若当前 clip 中存在同类别多个可见 instance，会优先采样这些类别，让 correspondence loss 具备同类负例。

## 2. SAM3 分支

代码位置：

```text
src/vggtsam/adapters/sam3_intermediate.py
```

当前使用：

```text
SAM3 image backbone:
  model.backbone.forward_image(images)

SAM3 text backbone:
  model.backbone.forward_text([prompt])
```

当前选择的视觉层：

```text
feature_source = detector_fpn2
source tensor = backbone_out["backbone_fpn"][-1]
```

当前 token 处理：

```text
SAM3 输入分辨率:
  1008 x 1008
  实际输入 tensor: [T, 3, 1008, 1008]
  其中 3 是 RGB 通道

SAM3 spatial feature:
  detector_fpn2
  resized to 72 x 72

text conditioning:
  concat pooled language feature to every spatial token
```

选择 `detector_fpn2` 的原因：

```text
1. SAM3 1008 输入下，ViT patch size = 14，因此 1008 / 14 = 72。
2. detector_fpn2 对应较低分辨率、语义更强的 FPN 层，空间大小与 72 x 72 token grid 对齐。
3. 当前阶段希望先验证 latent-level fusion 和 correspondence，72 x 72 计算量较可控。
4. 这不是最终结论；detector_fpn1 / detector_fpn0 应作为后续高分辨率 ablation。
```

`resized to 72 x 72` 的实现：

```text
spatial = select_sam3_spatial_feature(backbone_out, source="detector_fpn2")
spatial = ensure_bchw_tensor(spatial).float()
spatial = F.interpolate(spatial, size=(72, 72), mode="bilinear")
```

如果原始 spatial feature 已经是 72 x 72，则不会发生实际 resize。

Prompt 编码：

```text
text_out = model.backbone.forward_text([prompt])
language_features = text_out["language_features"]
pooled language feature -> [1, 256]
```

当前 `text_conditioning=concat`：

```text
SAM3 visual token:
  [1, T * 72 * 72, 256]

pooled prompt token:
  [1, 256]
  expand to [1, T * 72 * 72, 256]

concat:
  [1, T * 72 * 72, 256 + 256]
  = [1, T * 72 * 72, 512]
```

实测 shape：

```text
SAM3 tokens:
  [1, T * 72 * 72, 512]

per frame:
  [1, 5184, 512]
```

其中：

```text
72 * 72 = 5184
512 = 256 detector_fpn2 channels + 256 pooled prompt feature
```

## 3. StreamVGGT 分支

代码位置：

```text
src/vggtsam/adapters/streamvggt_latent.py
```

当前使用：

```text
StreamVGGT aggregator:
  aggregated_tokens_list, patch_start_idx = model.aggregator(batch_images)

selected layer:
  aggregated_tokens_list[layer_index]
  layer_index = -1
```

当前 token 处理：

```text
patch tokens:
  remove prefix tokens before patch_start_idx

original patch grid:
  depends on StreamVGGT image preprocessing
  example observed: 25 x 37

geometry context grid:
  resized to 12 x 12

dense geometry grid:
  resized to 72 x 72
  used as aux / point target alignment reference
```

实测 shape：

```text
geometry tokens:
  [1, T * 12 * 12, 2048]

per frame:
  [1, 144, 2048]

pointmap_grid:
  [T, 72, 72, 3]

camera_tokens:
  observed [1, T, 9]
  当前默认 use_camera_tokens = false
```

## 4. Fusion 模型

代码位置：

```text
src/vggtsam/models/fusion.py
src/vggtsam/models/latent_fusion.py
```

每一帧流式处理：

```text
SAM3 semantic tokens:
  [1, 5184, 512]

StreamVGGT geometry tokens:
  [1, 144, 2048]

projection:
  SAM3 512  -> d_fuse 256
  VGGT 2048 -> d_fuse 256

cross attention:
  query = SAM3 tokens
  key/value = StreamVGGT geometry tokens

fused tokens:
  [1, 5184, 256]
```

当前不是直接把 RGB 输入到一个端到端模型中，而是先用 frozen SAM3 / StreamVGGT 抽中间特征，再训练 fusion heads。

## 5. Fused Tokens 的输出头

当前 fused tokens 进入四类输出：

```text
semantic_head:
  input  [1, 5184, 256]
  output [1, 5184, 1024]
  meaning: per-token semantic logits

point_head:
  input  [1, 5184, 256]
  output [1, 5184, 3]
  meaning: per-token 3D point prediction

match_head:
  input  [1, 5184, 256]
  output [1, 5184, 256]
  meaning: normalized token embedding for cross-frame matching

mask decoder:
  ref GT instance mask + ref fused tokens
    -> instance prototype [1, 256]

  current/all fused tokens
    -> mask logits [T, 5184]
    -> reshape to [T, 72, 72]
```

mask decoder 的实现位置：

```text
src/vggtsam/models/latent_fusion.py
  build_mask_prototype()
  decode_mask_from_prototype()
```

另外模型仍然提供：

```text
compute_mask_correspondence(curr_embeddings, hist_embeddings)
  -> [N_curr, N_hist] correspondence logits
```

这里的 `mask correspondence` 不是最终 binary mask，而是 token-to-token 跨帧对应关系矩阵，用来保持同一个 instance 在历史帧和当前帧中的 embedding 一致。

## 6. 当前训练方式

当前训练已经改成逐帧 streaming-style fusion：

```text
for t in range(T):
  1. 取第 t 帧 SAM3 tokens
  2. 取第 t 帧 StreamVGGT geometry tokens
  3. fusion 得到 fused tokens
  4. 预测 semantic logits / pointmap / match embeddings
  5. 从历史帧随机选 hist_t
  6. 计算 current-to-history correspondence matrix
  7. 用 instance id 构造 GT match matrix
  8. 将当前 embedding 写入 history buffer

clip-level mask decoder:
  1. 优先使用 prompt 采样到的 instance id 作为目标实例。
  2. 选择该 instance 在 token 数最多的一帧作为 ref frame。
  3. 用 ref frame 的 GT instance mask 提取 ref fused token prototype。
  4. 用 prototype 对 T 帧 fused tokens 解码 mask logits。
  5. 用 GT instance mask 的 token grid 监督 decoder 输出。
  6. mask decoder 的负样本来自 mask_supervision_tokens，
     包含同一 clip 中其它经过清洗的有效物体 token。
```

Loss：

```text
semantic_loss:
  pred semantic logits -> semantic mask token label

point_loss:
  pred pointmap -> GT pointmap token

match_loss:
  current token 与 history token 的 correspondence BCE

  positive:
    same instance id across frames

  negative:
    different instance id across frames

mask_loss:
  decoder mask logits vs GT instance mask token label
  supervision tokens include other valid objects as negatives

mask_dice_loss:
  decoder mask probability vs GT instance mask token label
```

## 7. 当前 Mask Decoder 与可视化

当前可视化已经改成 mask decoder 输出，而不是旧的 top-k 相似度传播。

当前流程：

```text
1. 选一个 target instance。
2. 选择 target instance token 数最多的一帧作为 ref frame。
3. 用 ref frame 的 GT instance mask 和 ref fused tokens 构造 prototype。
4. 对 clip 内每一帧 fused tokens 解码 mask logits。
5. 将 logits reshape 为 [T, 72, 72] 并上采样到原 RGB 分辨率可视化。
```

训练监督：

```text
GT instance mask:
  original resized image space
  -> majority pooling to 72 x 72 token grid

mask supervision tokens:
  all visible / non-excluded / size-filtered object tokens
  target instance = positive
  other valid object tokens = negative

decoder output:
  mask_logits [T, 5184]
  -> [T, 72, 72]

loss:
  BCEWithLogits + Dice
```

可视化输出：

```text
outputs/latent_fusion_debug/visualizations/step_xxxxxx.png
  RGB / GT instance / Decoder mask

outputs/latent_fusion_debug/visualizations/step_xxxxxx_crops.png
  RGB crop / GT crop / Pred crop
```

## 8. 当前 Mask Decoder 的边界

```text
1. 当前 mask decoder 是 token-level，输出分辨率仍然是 72 x 72。
2. 可视化会上采样到原 RGB 尺寸，所以边界不会像原始 mask 一样精细。
3. decoder 使用 ref mask prototype，但还没有接入 SAM3/SAM2 原生高分辨率 mask decoder。
4. 当前是每个 clip 训练一个 target instance 的 decoder mask；后续可扩展到多实例并行。
5. correspondence loss 已经使用历史帧，但 mask decoder 目前是 ref prototype -> all frames，还不是显式 recurrent memory decoder。
```

后续更贴近最终 idea 的方向：

```text
1. 引入更高分辨率特征，例如 detector_fpn1 / detector_fpn0。
2. 将 72 x 72 mask logits 送入 refinement decoder，恢复更清晰边界。
3. 让 mask decoder 显式读 history memory / correspondence matrix。
4. 支持一个 clip 内多个 instance 同时解码。
5. 输出 pred semantic point cloud 与 GT semantic point cloud 用于对比。
```

## 9. 语义点云与 GT 点云

当前已经有 per-token point prediction：

```text
pred_pointmap:
  [T, 72, 72, 3]
```

也有 per-token semantic prediction：

```text
pred_semantic_logits:
  [T, 72, 72, 1024]
```

也有 GT pointmap：

```text
gt_pointmap:
  generated from ScanNet++ mesh + COLMAP camera rasterization
  [T, 72, 72, 3] after pooling
```

下一步需要实现全局融合导出：

```text
pred semantic point cloud:
  pred_pointmap + pred semantic logits + predicted/refined masks

GT semantic point cloud:
  gt_pointmap + GT semantic masks + GT instance masks
```

这样后续可以做：

```text
1. 语义点云可视化对比
2. instance tracking 对比
3. point/semantic/mask 指标评估
```

当前新增了 per-clip pointmap 可视化脚本：

```text
scripts/export_latent_fusion_pointclouds.py
```

它会导出：

```text
gt_*:
  ScanNet++ mesh + COLMAP camera rasterization 得到的 pointmap

streamvggt_*:
  frozen StreamVGGT 原始 pointmap 输出

pred_*:
  当前 latent fusion 模型 point head 输出
```

这些 `.ply` 用于检查 pointmap 监督和模型预测质量；它还不是跨帧去重、融合后的全局语义点云。

## 10. 当前边界

```text
1. SAM3 和 StreamVGGT 仍然 frozen。
2. 当前 fusion 训练是逐帧 streaming-style，但 SAM3 / StreamVGGT adapter 仍然是先一次性抽 clip 特征。
3. StreamVGGT camera tokens 默认关闭。
4. 当前 mask decoder 是 72 x 72 token-level decoder，不是高分辨率边界 decoder。
5. 当前还没有全局语义点云融合导出。
```
