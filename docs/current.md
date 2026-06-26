# 当前模型结构梳理

本文档用于 PPT 对齐当前代码状态、已实现模块和下一步需要补齐的模块。当前实现目标是验证：

```text
SAM3 semantic tokens + StreamVGGT geometry tokens
  -> latent fusion
  -> semantic / 3D point / cross-frame correspondence
```

当前代码还没有最终的高质量 mask decoder，也还没有全局语义点云融合导出模块。

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

SAM3 spatial feature:
  detector_fpn2
  resized to 72 x 72

text conditioning:
  concat pooled language feature to every spatial token
```

实测 shape：

```text
SAM3 tokens:
  [1, T * 72 * 72, 512]

per frame:
  [1, 5184, 512]
```

其中 `512 = 256 detector_fpn2 channels + 256 pooled text feature`。

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

当前 fused tokens 进入三个 head：

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
```

另外模型提供：

```text
compute_mask_correspondence(curr_embeddings, hist_embeddings)
  -> [N_curr, N_hist] correspondence logits
```

这里的 `mask correspondence` 不是最终 binary mask，而是 token-to-token 跨帧对应关系矩阵。

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
```

## 7. 当前 Mask 可视化

当前可视化不是最终 mask decoder 输出。

当前流程：

```text
1. 选一个 ref instance。
2. 用 ref frame 的 GT instance mask 找到 ref object tokens。
3. 对 ref object tokens 的 match embedding 求 prototype。
4. 每帧计算 token embedding 与 prototype 的相似度。
5. 每帧取 top-k token 作为 propagated mask。
```

这个 top-k 方法只是为了诊断 correspondence 是否能追踪同一个 instance。它会有明显问题：

```text
1. k 来自 ref mask 面积，目标尺度变化时会不准。
2. 同类物体 embedding 接近时，容易选到其他同类物体。
3. 它没有利用真实 instance mask 训练一个高分辨率 decoder。
4. 它不应该作为最终 mask 输出方式。
```

## 8. 关于 Mask Decoder

你的判断是对的：如果最终任务需要输出 mask，就应该使用真实 instance mask 训练 mask decoder。

之前 object-query mask decoder 效果差，不代表 mask decoder 方向错，主要问题可能是：

```text
1. query slot 分配不稳定。
2. 当时 prompt 还是 fixed "object"，语义条件太弱。
3. BCE 背景占比太大，loss 下降不代表 mask 形状好。
4. decoder 没有显式使用 ref mask / history correspondence。
5. 直接从 72x72 token 上采样到高分辨率，边界能力有限。
```

更合理的下一版 mask decoder 应该是：

```text
ref GT mask
  -> ref object token prototype / object query

current fused tokens
  -> mask decoder
  -> mask logits [T, 72, 72] or higher resolution

supervision:
  GT instance mask pooled/resized to token grid
  BCE + Dice / Focal

additional supervision:
  correspondence loss keeps same instance embedding close
```

也就是说，mask decoder 应该和 correspondence/history 结合，而不是单独靠固定 object query。

## 9. 语义点云与 GT 点云

当前已经有 per-token point prediction：

```text
pred_pointmap:
  [T, 72, 72, 3]
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

## 10. 当前边界

```text
1. SAM3 和 StreamVGGT 仍然 frozen。
2. 当前 fusion 训练是逐帧 streaming-style，但 SAM3 / StreamVGGT adapter 仍然是先一次性抽 clip 特征。
3. StreamVGGT camera tokens 默认关闭。
4. 当前 mask 是 correspondence top-k 可视化，不是最终 mask decoder。
5. 当前还没有全局语义点云融合导出。
```
