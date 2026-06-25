# Latent Fusion 技术设计与训练流程

## 1. 文档目的

本文档说明当前代码如何把 `idea/ours_model.py` 与 `idea/ours_training.py` 中的伪代码落实为可运行训练流程。

两份伪代码已经定义了核心方向：

```text
SAM3 semantic/tracking tokens
  as query

StreamVGGT geometry/camera tokens
  as key/value context

cross-attention fusion
  -> pointmap prediction
  -> semantic logits
  -> cross-frame matching embeddings
```

但伪代码没有明确以下工程选择：

```text
1. SAM3 具体取哪一层作为 sam_feat
2. StreamVGGT 具体取哪一层作为 vggt_feat / cam_token
3. ScanNet++ mask 如何对齐到 token grid
4. 跨帧 matching 的真值矩阵如何构造
5. 当前哪些模块 frozen，哪些模块训练
6. 当前实现与伪代码相比有哪些简化
```

本文档逐项明确这些选择。

## 2. 对应代码

入口脚本：

```text
scripts/train_latent_fusion.py
scripts/inspect_latent_fusion_features.py
```

配置文件：

```text
configs/latent_fusion_train.yaml
```

核心实现：

```text
src/vggtsam/adapters/sam3_intermediate.py
src/vggtsam/adapters/streamvggt_latent.py
src/vggtsam/models/latent_fusion.py
src/vggtsam/models/fusion.py
src/vggtsam/training/latent_fusion.py
```

## 3. 总体数据流

当前训练每一步随机采样一个连续 ScanNet++ clip：

```text
processed ScanNet++ manifest
  -> image_paths
  -> instance_masks
  -> semantic_masks
```

同一个 clip 同时送入两个 frozen foundation model adapter：

```text
image_paths
  -> SAM3IntermediateAdapter
     -> sam_tokens

image_paths
  -> StreamVGGTLatentAdapter
     -> geometry_tokens
     -> camera_tokens
     -> pointmap_grid
```

ScanNet++ 标注被下采样到和 SAM3 token 对齐的监督网格：

```text
instance_masks -> instance_grid [T, 72, 72]
semantic_masks -> semantic_grid [T, 72, 72]
```

模型 forward：

```text
sam_tokens      [1, T * 72 * 72, D_sam]
geometry_tokens [1, T * 12 * 12, D_geo]
camera_tokens   [1, T, 9]

  -> LatentSAMVGGTModel

logits      [1, T * 72 * 72, num_classes]
pointmap    [1, T * 72 * 72, 3]
embeddings  [1, T * 72 * 72, d_fuse]
```

训练损失：

```text
semantic_loss:
  fused token -> semantic label

point_loss:
  fused token -> StreamVGGT pointmap pseudo target

match_loss:
  same ScanNet++ instance id across frames -> close embeddings
  different instance id -> separated embeddings
```

## 4. 关键设计澄清

### 4.1 当前数据集是否缺少 `gt_pointmaps`

是的，当前 `prepare_scannetpp_2d.py` 生成的 processed ScanNet++ 数据集没有显式 `gt_pointmaps`。

当前 processed 数据包含：

```text
RGB image_path
semantic_mask
instance_mask
raster metadata / visualization
```

它不包含：

```text
gt_pointmaps: [T, 72, 72, 3] 或 [T, H, W, 3]
```

因此当前代码中的 `point_loss` 使用的是：

```text
frozen StreamVGGT point_head output
  -> pointmap_grid [T, 72, 72, 3]
  -> pseudo point target
```

这和 `ours_training.py` 里的 `gt_pointmaps` 不是完全等价的真值监督。当前版本先用 StreamVGGT 自己的 3D 输出作为 pseudo target，目的是验证：

```text
SAM3 token 是否能通过 cross-attention 吸收 StreamVGGT geometry context
fusion head 是否能回归 3D-like point representation
semantic / instance supervision 是否能和 geometry token 对齐
```

如果要严格实现伪代码中的 `gt_pointmaps`，后续需要在数据预处理阶段额外生成 per-pixel 或 per-token 3D 真值，例如：

```text
ScanNet++ mesh / depth / camera pose / intrinsics
  -> rasterize visible 3D coordinate per pixel
  -> downsample / pool to 72 x 72 token grid
  -> gt_pointmaps
```

这一部分当前没有实现。

### 4.2 为什么把 instance masks 下采样到 token grid

ScanNet++ 的 `instance_masks` 和 `semantic_masks` 仍然是当前训练的真值来源。它们被下采样，不是因为它们不是真值，而是因为当前模型的输出还不是高分辨率 mask。

当前模型输出是 token-level：

```text
logits:     [1, T * 72 * 72, num_classes]
pointmap:   [1, T * 72 * 72, 3]
embeddings: [1, T * 72 * 72, d_fuse]
```

也就是说，当前版本没有输出：

```text
final mask: [T, H, W] 或 [T, num_objects, H, W]
```

因此监督必须先对齐到当前输出空间：

```text
semantic_mask [H, W]
  -> majority pooling
  -> semantic_grid [72, 72]

instance_mask [H, W]
  -> majority pooling
  -> instance_grid [72, 72]
```

这一步的意义是把原始 2D GT mask 转换成 token-level GT：

```text
每个 SAM3 FPN-2 token 对应一个 2D patch 区域
该 patch 内的 majority semantic label 作为 token semantic target
该 patch 内的 majority instance id 作为 token identity target
```

只有 majority ratio 足够高的 token 才参与训练，边界混合 token 会被忽略。

后续如果加入 mask decoder 或 high-resolution refinement，那么原始 `instance_masks` 应该直接用于最终 mask 输出的监督：

```text
mask_logits [T, num_objects, H, W]
  vs.
instance_masks [T, H, W]

loss:
  BCE / Dice / focal
```

当前版本还没有这个 final mask head，所以现在只监督 token-level 输出。

### 4.3 point/semantic/match heads 是如何构建的

这些 head 在当前代码里由 MLP 构成，位置是：

```text
src/vggtsam/models/fusion.py
LatentGeometrySemanticFusion
```

当前结构：

```text
point_head:
  Linear(d_fuse, d_fuse)
  GELU
  Linear(d_fuse, 3)

semantic_head:
  Linear(d_fuse, d_fuse)
  GELU
  Linear(d_fuse, num_classes)

match_head:
  Linear(d_fuse, d_fuse)
  GELU
  Linear(d_fuse, d_fuse)
  normalize
```

它们对应 `ours_model.py` 中的抽象 heads：

```text
pointmap_head:
  当前拆成 point_head + semantic_head

classifier_head:
  当前对应 semantic_head

match_query_proj / match_key_proj:
  当前简化为共享 match_head，然后用 embedding similarity 做 matching
```

这些 loss 的方向来自 `ours_training.py`：

```text
3D loss
semantic classification loss
cross-frame matching loss
```

但当前实现不是逐字照搬伪代码：

```text
ours_training.py:
  loss_3d = L1(pred_pointmap, gt_pointmaps)
  loss_cls = CE(pred_logits, gt_semantics)
  loss_match = BCE(pred_match_matrix, gt_match_matrix)

current implementation:
  point_loss = SmoothL1(pred_pointmap, StreamVGGT pseudo pointmap)
  semantic_loss = CE(pred_logits, downsampled semantic_grid)
  match_loss = supervised contrastive loss over sampled token embeddings
```

这里把 matching BCE 换成 supervised contrastive，是为了避免构造完整的 `[N, N]` token correspondence matrix。`T=4, 72x72` 时 token 数已经超过 2 万，完整矩阵显存开销过大。

### 4.4 `sam3.prompt: object` 是否表示随机选一个 object 训练

不是。

当前训练不是随机选一个 instance，也不是只训练 SAM3 检测到的某个 object。

当前 `prompt: object` 的作用是：

```text
让 SAM3 text encoder 产生一个通用 object 文本特征
把这个 pooled text feature concat 到每个 SAM3 spatial token
```

也就是说，它是全局 text conditioning，不是 instance sampler。

当前每个训练 step 的监督对象来自 ScanNet++ mask：

```text
一个连续 clip 内所有通过过滤的 valid tokens / valid instances
```

训练使用的是多 object / 多 frame token supervision：

```text
same instance id across frames
  -> positive matching pairs

different instance ids
  -> negative matching pairs
```

因此当前 `prompt: object` 代表的是“泛物体”训练。它不会导致模型随机只关注一个 object。

后续可以升级为 category-conditioned training：

```text
从 semantic_grid 中随机选一个 category
prompt = category name
只把该 category 的 instances 当作强正样本
其他 category 作为负样本或 ignore
```

这会更接近开放词汇推理，但当前第一版还没有做这个 category prompt sampling。

## 5. `ours_model.py` 的当前落地

### 5.1 Backbone 加载

伪代码：

```python
self.stream_vggt_backbone = self._load_vggt_backbone()
self.sam3_backbone = self._load_sam3_backbone()
```

当前实现：

```text
SAM3:
  load_sam3_image_model()
  SAM3IntermediateAdapter

StreamVGGT:
  load_streamvggt_latent_model()
  StreamVGGTLatentAdapter
```

当前二者均 frozen：

```text
sam3_model.requires_grad_(False)
streamvggt_model.requires_grad_(False)
```

当前不做 LoRA，也不 fine-tune SAM3 / StreamVGGT 本体。训练只更新 fusion model。

### 5.2 SAM3 中间层选择

伪代码：

```python
sam_feat = self.sam3_backbone(image, text_prompts)
```

当前明确选择：

```text
SAM3 detector neck FPN-2
```

代码位置：

```text
src/vggtsam/adapters/sam3_intermediate.py
select_sam3_spatial_feature()
```

具体取法：

```python
backbone_out = model.backbone.forward_image(images)
spatial = backbone_out["backbone_fpn"][-1]
```

配置项：

```yaml
sam3:
  feature_source: detector_fpn2
```

在 SAM3 image model 中，输入被 resize 到 `1008 x 1008`，ViT patch size 为 `14`：

```text
1008 / 14 = 72
```

SAM3 detector neck 产生多尺度 FPN。当前取 detector 分支里与 patch grid 对齐的低分辨率层：

```text
detector_fpn2 ~= [T, 256, 72, 72]
```

adapter 会保证输出 resize 到配置中的 token grid：

```yaml
geometry:
  token_grid: [72, 72]
```

因此当前 `sam_tokens` 的空间结构是：

```text
[T, 72, 72, C_sam]
```

再展平成：

```text
[1, T * 72 * 72, C_sam]
```

### 5.3 为什么选择 detector_fpn2

当前选择 `detector_fpn2`，而不是 final mask、FPN-0/FPN-1 或 tracker memory，原因如下：

```text
1. 不依赖 SAM3 final detection
   SAM3 final mask 可能因为 prompt 或阈值没有检测到目标。
   当前 idea 的核心是中间特征融合，不应该被 final mask 是否存在阻断。

2. 与 ViT patch grid 对齐
   1008 输入、14 patch size 自然得到 72 x 72。
   ScanNet++ semantic/instance mask 可以稳定下采样到这个 grid。

3. 保留开放词汇语义分支信息
   detector neck 是 SAM3 开放词汇检测 / 文本分割分支的一部分，
   比纯视觉底层特征更接近当前任务。

4. 计算量可控
   FPN-0/FPN-1 分辨率更高，适合后续 mask refinement，
   不适合第一版直接做全 token fusion。
```

当前没有选择：

```text
SAM3 final masks:
  不稳定；检测不到时没有训练输入；不符合当前 latent fusion idea。

Detector Transformer output:
  理论上很适合 text-conditioned object query，
  但当前代码先选择更稳定、shape 更明确的 FPN-2。

SAM2 tracker memory tokens:
  更适合后续时序 memory 版本。
  当前第一版先用 ScanNet++ instance id 监督跨帧一致性。
```

### 5.4 SAM3 文本特征使用方式

伪代码中 `sam3_backbone(image, text_prompts)` 表示 SAM3 应该是 text-conditioned。

当前实现：

```python
text_out = model.backbone.forward_text([prompt])
language = pool_language_features(text_out)
sam_token = concat(spatial_token, language)
```

配置项：

```yaml
sam3:
  prompt: object
  text_conditioning: concat
```

也就是说，当前不是直接取 SAM3 detector transformer 的 text-conditioned query，而是：

```text
SAM3 detector FPN-2 spatial feature
  + pooled SAM3 language feature
  -> sam_tokens
```

这是一个工程上更稳定的第一版实现。后续如果能稳定 hook detector transformer output，可以把 `feature_source` 扩展到 transformer query token。

### 5.5 StreamVGGT 层选择

伪代码：

```python
vggt_feat, cam_token, updated_kv_cache = self.stream_vggt_backbone(image, kv_cache)
```

当前明确选择：

```text
StreamVGGT aggregator 最后一层 patch tokens
StreamVGGT camera_head 输出 pose encoding
```

代码位置：

```text
src/vggtsam/adapters/streamvggt_latent.py
```

具体取法：

```python
aggregated_tokens_list, patch_start_idx = model.aggregator(images)
tokens = aggregated_tokens_list[layer_index]
patch_tokens = tokens[:, :, patch_start_idx:, :]
```

配置项：

```yaml
geometry:
  layer_index: -1
  context_grid: [12, 12]
```

`patch_start_idx` 之前是 camera/register tokens，之后是 patch tokens。当前只取 patch tokens 作为几何上下文 token，并单独调用 camera head：

```python
pose_enc_list = model.camera_head(aggregated_tokens_list)
camera_tokens = pose_enc_list[-1]
```

当前输出：

```text
geometry_tokens: [1, T * 12 * 12, D_geo]
camera_tokens:   [1, T, 9]
```

### 5.6 为什么 geometry context 用 12x12

SAM3 query 是 `T * 72 * 72`，如果 geometry key/value 也用 `T * 72 * 72`，cross-attention 显存和计算量会很大。

因此当前实现区分两个 grid：

```yaml
token_grid: [72, 72]
context_grid: [12, 12]
```

含义：

```text
token_grid:
  SAM3 query grid
  ScanNet++ supervision grid
  pointmap target grid

context_grid:
  StreamVGGT geometry key/value grid
  用于控制 cross-attention 开销
```

### 5.7 Fusion 模块

伪代码：

```python
f_vggt = self.proj_vggt(vggt_feat)
f_cam = self.proj_cam(cam_token)
f_sam = self.proj_sam(sam_feat)

geometry_context = torch.cat([f_vggt, f_cam], dim=1)
f_fused, _ = self.cross_attention_fusion(
    query=f_sam,
    key=geometry_context,
    value=geometry_context,
)
```

当前实现：

```text
src/vggtsam/models/fusion.py
LatentGeometrySemanticFusion
```

核心等价关系：

```text
proj_semantic = proj_sam
proj_geometry = proj_vggt
proj_camera   = proj_cam
cross_attention = cross_attention_fusion
```

当前 forward：

```text
query = proj_semantic(sam_tokens)
context = concat(
  proj_geometry(geometry_tokens),
  proj_camera(camera_tokens)
)

fused = MultiheadAttention(
  query=query,
  key=context,
  value=context
)

fused = LayerNorm(fused + query)
```

### 5.8 多任务 Heads

伪代码：

```python
pred_pointmap = geo_outputs[..., :3]
pred_logits = classifier_head(semantic_embeddings)
match_matrix = q @ k.T
```

当前实现：

```text
point_head:
  fused token -> [x, y, z]

semantic_head:
  fused token -> semantic logits

match_head:
  fused token -> normalized match embedding
```

当前没有直接输出 dense `match_matrix`，而是输出 embedding，并在 loss 里对采样 token 计算相似度矩阵：

```python
logits = embeddings @ embeddings.T / temperature
```

这样比完整 `[N, N]` correspondence matrix 更省显存。

## 6. `ours_training.py` 的当前落地

### 6.1 Dataset 与 sequence

伪代码：

```python
video_sequence 包含:
  frames
  text_prompts
  gt_pointmaps
  gt_semantic_masks
  gt_instance_ids
```

当前实现的数据来自 processed ScanNet++：

```text
image_paths
semantic_masks
instance_masks
```

当前没有直接读取真实 `gt_pointmaps`。第一版使用 StreamVGGT frozen point head 生成 pseudo pointmap target：

```text
pointmap_grid = StreamVGGT point_head output
```

因此当前对应关系是：

```text
frames:
  image_paths 加载得到

text_prompts:
  config.sam3.prompt，默认 "object"

gt_semantic_masks:
  ScanNet++ semantic_masks 下采样到 72x72

gt_instance_ids:
  ScanNet++ instance_masks 下采样到 72x72

gt_pointmaps:
  当前暂用 frozen StreamVGGT pointmap_grid
```

### 6.2 Clip 采样

当前每 step 随机采样一个连续窗口：

```yaml
dataset:
  sequence_length: 4
  frame_stride: 1
```

输出：

```text
T = 4
image_paths:     List[Path]
instance_masks:  List[np.ndarray]
semantic_masks:  List[np.ndarray]
```

### 6.3 Mask 到 token grid 的监督构造

伪代码假设已经有：

```python
gt_semantic_masks: [B, T, N]
gt_instance_ids: [B, T, N]
```

当前代码通过 majority pooling 构造：

```text
semantic_mask [H, W]
  -> semantic_grid [72, 72]

instance_mask [H, W]
  -> instance_grid [72, 72]
```

每个 `72x72` token cell 取对应像素区域的 majority label，同时记录 majority ratio。

token 参与训练需要满足：

```text
instance_id != 0
semantic_id != semantic_ignore_label
semantic_id 在 [0, num_classes)
majority ratio >= min_token_majority
instance 至少跨 min_visible_frames 可见
instance token 数 >= min_tokens_per_instance
instance 面积比例 <= max_area_ratio
semantic_id 不在 excluded_semantic_labels
point target 是 finite
```

这一步解决了当前数据里的两个噪声问题：

```text
1. 极小噪声 mask 不参与训练
2. 大面积墙面 / 地板 / 天花板可以通过面积阈值与 semantic blacklist 过滤
```

### 6.4 当前 training loop 与伪代码差异

伪代码是逐帧 streaming：

```python
for t in range(T):
    outputs = model(current_frame, text_prompts, kv_cache=kv_cache)
    historical_token_buffer.append(fused_tokens.detach())
```

当前实现是 clip-level batched token fusion：

```text
先对整个 T 帧 clip 提取 SAM3 tokens
先对整个 T 帧 clip 提取 StreamVGGT tokens
一次 forward 得到所有 T 帧 token 输出
```

当前没有显式维护 `kv_cache`。原因：

```text
1. 第一版目标是先验证 latent fusion 数据流和监督闭环
2. StreamVGGT inference / aggregator 已经在 clip 内处理多帧
3. SAM3 tracker memory 接入会放到后续版本
```

因此当前是：

```text
sequence-level training
not frame-by-frame streaming training
```

但监督目标保持与伪代码一致：

```text
semantic supervision
3D point supervision
cross-frame instance matching supervision
```

### 6.5 Loss 实现

伪代码：

```python
loss_3d = F.l1_loss(pred_pointmap, gt_pointmaps[:, t, ...])
loss_cls = F.cross_entropy(pred_logits.transpose(1, 2), gt_semantics[:, t, ...])
loss_match = F.binary_cross_entropy(pred_match_matrix, gt_match_matrix)
```

当前实现：

```text
semantic_loss:
  cross_entropy(logits[valid_tokens], semantic_labels[valid_tokens])

point_loss:
  smooth_l1_loss(pointmap[valid_tokens], point_targets[valid_tokens])

match_loss:
  supervised contrastive loss over cross-frame same-instance tokens
```

当前 `match_loss` 没有使用 BCE，是为了避免完整 `[N, N]` match matrix 显存过大。实现上先筛选：

```text
valid tokens
instance 至少出现在两个 frame
最多 max_match_tokens 个 token
```

正样本：

```text
same instance_id
different frame_id
```

负样本：

```text
different instance_id
```

最终：

```python
logits = normalize(embeddings) @ normalize(embeddings).T / temperature
```

### 6.6 参数更新

当前训练参数：

```text
LatentSAMVGGTModel:
  projection layers
  cross-attention
  point head
  semantic head
  match head
```

当前 frozen：

```text
SAM3
StreamVGGT
```

优化器：

```text
AdamW
lr = 3e-4
grad clip = 1.0
```

总 loss：

```text
loss =
  semantic_weight * semantic_loss
  + point_weight * point_loss
  + match_weight * match_loss
```

默认：

```yaml
loss:
  semantic_weight: 1.0
  point_weight: 1.0
  match_weight: 0.5
```

## 7. 当前实现明确没有做的事情

当前第一版没有实现以下内容：

```text
1. 不使用 SAM3 final masks 作为训练输入
2. 不训练 SAM3 或 StreamVGGT
3. 不接 LoRA
4. 不使用 SAM3 tracker memory / KV cache
5. 不从 detector transformer 中 hook object query
6. 不使用真实 ScanNet++ 3D pointmap 作为 point loss target
7. 不训练 mask decoder
```

这些不是被忘掉，而是当前版本为了先验证核心 latent fusion 闭环而做的边界收缩。

## 8. 当前配置的关键决策

```yaml
sam3:
  feature_source: detector_fpn2
  prompt: object
  text_conditioning: concat

geometry:
  token_grid: [72, 72]
  context_grid: [12, 12]
  layer_index: -1

objects:
  min_token_majority: 0.55
  min_tokens_per_instance: 2
  max_match_tokens: 2048
```

解释：

```text
detector_fpn2:
  当前 SAM3 中间特征层选择，负责提供 semantic query。

prompt: object:
  第一版使用通用 object prompt。
  后续可以换成随机 semantic category prompt。

text_conditioning: concat:
  把 pooled language feature 拼到每个 SAM3 spatial token 上。

token_grid [72,72]:
  对齐 SAM3 1008 输入 / 14 patch grid。

context_grid [12,12]:
  压缩 StreamVGGT geometry context，控制 attention 开销。

max_match_tokens 2048:
  防止 cross-frame matching 矩阵过大。
```

## 9. 调试命令

先检查两侧 adapter 输出：

```bash
PYTHONPATH=src python scripts/inspect_latent_fusion_features.py \
  --config configs/latent_fusion_train.yaml \
  --device cuda
```

预期重点输出：

```text
SAM3:
  tokens=(1, T * 72 * 72, D_sam)

StreamVGGT:
  geometry_tokens=(1, T * 12 * 12, D_geo)
  camera_tokens=(1, T, 9)
  pointmap_grid=(T, 72, 72, 3)
```

小步训练：

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 20 \
  --device cuda
```

正常启动时应出现：

```text
initialized LatentSAMVGGTModel sam_dim=... geometry_dim=... camera_dim=...
step=1 loss=...
```

## 10. 输出文件

默认输出目录：

```text
outputs/latent_fusion_debug
```

训练输出：

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

## 11. 后续升级方向

建议后续按以下顺序升级：

```text
1. 跑通 detector_fpn2 + StreamVGGT latent fusion baseline
2. 填入 ScanNet++ semantic blacklist，明确过滤墙 / 地板 / 天花板
3. 尝试 category-specific prompt，而不是固定 "object"
4. hook SAM3 Detector Transformer + Language Features，替换或补充 detector_fpn2
5. 接入 SAM3 tracker memory tokens，实现真正 streaming KV / memory training
6. 用 ScanNet++ 几何或重建结果替代 StreamVGGT pseudo point target
7. 加 mask decoder / high-resolution refinement
```
