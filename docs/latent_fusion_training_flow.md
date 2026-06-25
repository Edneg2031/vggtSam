# Latent Fusion 训练流

这份文档记录当前 `train_latent_fusion.py` 的数据流和训练过程。它对应的是当前主线 idea：

```text
SAM3 中间语义 / tracking tokens
  as query

StreamVGGT latent geometry tokens + camera tokens
  as key/value context

cross-attention fusion
  -> semantic logits
  -> pointmap prediction
  -> cross-frame match embeddings
```

注意：这条训练流不依赖 SAM3 final mask。SAM3 最终是否检测到物体，不决定训练是否有输入。

## 入口

训练入口：

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 200 \
  --device cuda
```

特征检查入口：

```bash
PYTHONPATH=src python scripts/inspect_latent_fusion_features.py \
  --config configs/latent_fusion_train.yaml \
  --device cuda
```

主要配置文件：

```text
configs/latent_fusion_train.yaml
```

主要代码文件：

```text
scripts/train_latent_fusion.py
src/vggtsam/training/latent_fusion.py
src/vggtsam/adapters/sam3_intermediate.py
src/vggtsam/adapters/streamvggt_latent.py
src/vggtsam/models/latent_fusion.py
src/vggtsam/models/fusion.py
```

## 数据来源

训练从已处理好的 ScanNet++ manifest 读取数据：

```text
data/processed/scannetpp_2d/manifest.json
```

每个 scene 下的 frame 需要包含：

```text
image_path       RGB 图片
instance_mask    由 3D instance annotation 投影得到的 2D instance id
semantic_mask    由 3D semantic annotation 投影得到的 2D semantic id
```

当前默认 scene：

```yaml
dataset:
  scene_id: 0a5c013435
  sequence_length: 4
  frame_stride: 1
```

也就是说，每次训练随机采样一个连续 4 帧窗口。

## Clip 采样与实例过滤

采样逻辑在：

```text
ScanNetPPObjectSequenceDataset
```

每次得到：

```text
ObjectSequence:
  scene_id
  frame_indices
  image_paths
  instance_masks
  semantic_masks
  visible_instance_ids
```

在读取每帧 `instance_mask` 时先做粗过滤：

```yaml
objects:
  min_pixels: 128
  max_area_ratio: 0.25
  min_visible_frames: 2
  max_objects_per_frame: 32
  ignore_instance_id: 0
```

含义：

```text
instance id = 0:
  忽略背景 / 无效区域

min_pixels:
  去掉极小噪声 instance

max_area_ratio:
  去掉占画面比例过大的结构面或大物体

min_visible_frames:
  只保留至少跨多帧可见的 instance，用于跨帧 matching 监督

max_objects_per_frame:
  限制每帧最多保留的 instance 数量，控制训练开销
```

后续如果确认了 ScanNet++ semantic id 映射，可以在配置里加入墙、地板、天花板等大结构黑名单：

```yaml
excluded_semantic_labels: []
```

## SAM3 分支

SAM3 分支在：

```text
src/vggtsam/adapters/sam3_intermediate.py
```

当前使用 `build_sam3_image_model` 加载 image model，并冻结参数：

```text
sam3_model.requires_grad_(False)
```

每个 clip 的 RGB 会被 SAM3 adapter 处理成：

```text
[T, 3, 1008, 1008]
```

归一化方式与 SAM3 image processor 一致：

```text
Resize(1008, 1008)
Normalize(mean=0.5, std=0.5)
```

然后调用：

```text
model.backbone.forward_image(images)
model.backbone.forward_text([prompt])
```

当前默认取：

```yaml
sam3:
  feature_source: detector_fpn2
  prompt: object
  text_conditioning: concat
```

`detector_fpn2` 对应 SAM3 detector neck 的低分辨率语义特征：

```text
detector_fpn2: [T, 256, 72, 72]
```

adapter 会把它整理成 token：

```text
[1, T * 72 * 72, 256]
```

如果 `text_conditioning: concat`，会额外把 pooled text feature 拼到每个 spatial token 上：

```text
language_features -> pooled text vector
sam_token = concat(spatial_token, text_vector)
```

所以最终 `sam_tokens` 的最后一维会大于 256。具体维度由服务器实际 SAM3 输出决定，训练时会自动初始化模型。

可选的 SAM3 feature source：

```text
detector_fpn2
detector_fpn1
detector_fpn0
vision_features
tracker_fpn2
```

第一版优先使用 `detector_fpn2`，因为它天然是 `72 x 72`，适合作为融合 query grid。

## StreamVGGT 分支

StreamVGGT 分支在：

```text
src/vggtsam/adapters/streamvggt_latent.py
```

当前使用 frozen StreamVGGT：

```text
streamvggt_model.requires_grad_(False)
```

RGB 图片先走 StreamVGGT 自己的 preprocessing：

```yaml
geometry:
  image_mode: crop
```

之后调用：

```text
model.aggregator(images)
```

得到：

```text
aggregated_tokens_list
patch_start_idx
```

adapter 会取指定层：

```yaml
geometry:
  layer_index: -1
```

然后切出 patch tokens：

```text
patch_tokens = aggregated_tokens[:, :, patch_start_idx:, :]
```

这些 token 先 reshape 回 StreamVGGT 自己的 patch grid，再 resize 成两个网格：

```yaml
geometry:
  token_grid: [72, 72]
  context_grid: [12, 12]
```

两者用途不同：

```text
token_grid = 72 x 72:
  和 SAM3 token / ScanNet++ mask supervision 对齐
  用于 point target、semantic target、instance target

context_grid = 12 x 12:
  作为 cross-attention 的 geometry key/value
  控制显存和注意力计算量
```

StreamVGGT adapter 同时会调用：

```text
camera_head -> camera_tokens
point_head  -> pointmap_grid
```

输出主要包括：

```text
geometry_tokens: [1, T * 12 * 12, D_geo]
camera_tokens:   [1, T, 9]
pointmap_grid:   [T, 72, 72, 3]
```

其中 `pointmap_grid` 当前作为 3D pseudo target 使用。

## Token-Grid 监督构造

监督构造在：

```text
build_latent_batch()
```

输入：

```text
instance_masks
semantic_masks
visible_instance_ids
pointmap_grid
```

首先把 ScanNet++ 的原始 mask 下采样到 `72 x 72`：

```text
instance_mask -> majority_pool -> instance_grid [T, 72, 72]
semantic_mask -> majority_pool -> semantic_grid [T, 72, 72]
```

这里不是简单 nearest resize，而是对每个 token cell 取 majority label，并记录 majority ratio。

一个 token 会被用于训练，需要同时满足：

```text
instance_id != 0
instance majority ratio >= min_token_majority
semantic majority ratio >= min_token_majority
semantic label != semantic_ignore_label
semantic label 在 [0, num_classes)
semantic label 不在 excluded_semantic_labels
instance 至少跨 min_visible_frames 可见
instance token 数 >= min_tokens_per_instance
instance token 面积比例 <= max_area_ratio
point target 是 finite 数值
```

这些过滤是为了解决两个问题：

```text
1. mask 边界混合或噪声 token 不参与监督
2. 墙、地板、天花板、大结构或超大区域不主导训练
```

当前 batch target：

```text
semantic_labels: [T * 72 * 72]
instance_ids:    [T * 72 * 72]
frame_ids:       [T * 72 * 72]
valid_tokens:    [T * 72 * 72]
point_targets:   [T * 72 * 72, 3]
```

## 模型 Forward

模型在：

```text
src/vggtsam/models/latent_fusion.py
src/vggtsam/models/fusion.py
```

输入：

```text
sam_tokens:
  [1, T * 72 * 72, D_sam]

geometry_tokens:
  [1, T * 12 * 12, D_geo]

camera_tokens:
  [1, T, 9]
```

核心 fusion：

```text
query = proj_sam(sam_tokens)
context = concat(
  proj_geometry(geometry_tokens),
  proj_camera(camera_tokens)
)

fused = MultiHeadCrossAttention(
  query=query,
  key=context,
  value=context
)

fused = LayerNorm(fused + query)
```

输出：

```text
logits:
  [1, T * 72 * 72, num_classes]

pointmap:
  [1, T * 72 * 72, 3]

embeddings:
  [1, T * 72 * 72, d_fuse]
```

这里的 `embeddings` 是跨帧 matching 使用的 token embedding。

## Loss

当前有三个 loss：

```text
semantic_loss
point_loss
match_loss
```

### Semantic Loss

只在 `valid_tokens` 上计算：

```text
cross_entropy(logits[valid_tokens], semantic_labels[valid_tokens])
```

目标是让 fused token 学到 ScanNet++ semantic label。

### Point Loss

只在 `valid_tokens` 上计算：

```text
smooth_l1_loss(pointmap[valid_tokens], point_targets[valid_tokens])
```

当前 `point_targets` 来自 frozen StreamVGGT point_head 的输出，所以这是第一版 pseudo 3D supervision。后续如果引入 ScanNet++ 真 3D 投影点，也可以替换这里。

### Match Loss

matching 不对所有 `T * 72 * 72` token 做全量两两计算，而是先采样：

```yaml
max_match_tokens: 2048
```

采样条件：

```text
token 有效
token 所属 instance 至少出现在两个不同 frame
```

正样本定义：

```text
instance_id 相同
frame_id 不同
```

负样本自然来自同 batch 中其他 instance 的 token。

当前实现是 supervised contrastive loss：

```text
normalize(embeddings)
logits = embeddings @ embeddings.T / temperature
同 instance 跨帧 token 拉近
不同 instance token 拉远
```

## 参数更新

当前 frozen：

```text
SAM3 image backbone
StreamVGGT backbone / heads
```

当前训练：

```text
LatentSAMVGGTModel
  projection layers
  cross-attention fusion
  point head
  semantic head
  match head
```

优化器：

```text
AdamW
lr = 0.0003
grad clip = 1.0
```

总 loss：

```text
loss =
  semantic_weight * semantic_loss
  + point_weight * point_loss
  + match_weight * match_loss
```

默认权重：

```yaml
loss:
  semantic_weight: 1.0
  point_weight: 1.0
  match_weight: 0.5
```

## 输出

默认输出目录：

```text
outputs/latent_fusion_debug
```

训练会写：

```text
training_history.csv
training_curves.png
ckpt_stepXXXXXX.pt
ckpt_last.pt
```

CSV 字段：

```text
step
loss
semantic_loss
point_loss
match_loss
num_tokens
num_match_tokens
num_instances
```

`training_curves.png` 会画 loss 曲线和 token/instance 数量曲线。

## 当前限制

1. SAM3 只使用中间 backbone/FPN/text feature，没有使用 final mask，也没有训练 mask decoder。
2. `point_loss` 当前对齐的是 StreamVGGT pseudo pointmap，不是 ScanNet++ 真值 3D pointmap。
3. `excluded_semantic_labels` 还没有填 ScanNet++ 的墙、地板、天花板语义 id。
4. `context_grid` 默认压到 `12 x 12` 是为了先控制显存，后续可以逐步增大。
5. 当前是 single batch / single clip 训练，还没有 DataLoader 多 worker 或特征缓存。

## 建议调试顺序

第一步先检查 adapter 输出：

```bash
PYTHONPATH=src python scripts/inspect_latent_fusion_features.py \
  --config configs/latent_fusion_train.yaml \
  --device cuda
```

需要重点看：

```text
SAM3 tokens shape
StreamVGGT geometry_tokens shape
camera_tokens shape
pointmap_grid shape
```

第二步小步训练：

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 20 \
  --device cuda
```

如果能输出类似：

```text
initialized LatentSAMVGGTModel sam_dim=... geometry_dim=... camera_dim=...
step=1 loss=...
```

说明主数据流已经跑通。

第三步再扩大：

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 200 \
  --device cuda
```
