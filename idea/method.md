# 当前 Dense Fusion Baseline

## 1. 当前任务
2026 6 30
当前代码实现的是一个 dense baseline：

```text
输入：
  连续 RGB clip
  一个 text prompt

输出：
  pred_mask_logits: [T, H, W]
  pred_pointmap: [T, H, W, 3]
  pred_point_conf: [T, H, W]  # 仅 stream_dpt point decoder 输出
  prompt_score: [T, H, W]
  instance_embedding: [T, H, W, D]
```

目标是先让模型在图像分辨率上预测 prompt 相关物体的 mask，并在该 mask 区域输出 object pointmap，之后再把 mask 内 pointmap lift 成 3D object point cloud。

## 2. 数据输入

训练使用预处理后的 ScanNet++ pinhole 数据：

```text
RGB image paths
semantic_masks: [T, H_raw, W_raw]
instance_masks: [T, H_raw, W_raw]
pointmaps: [T, H_raw, W_raw, 3]
```

训练时会把 mask 和 pointmap resize 到配置里的 dense 输出分辨率，例如：

```yaml
model.output_size: [256, 384]
```

注意：这里的 loss 是在 `256 x 384` 这种 dense image grid 上算的，不是在 `72 x 72` token grid 上算的。`72 x 72` 只作为 SAM3 / StreamVGGT 的内部特征融合网格。

## 3. Prompt 构造

当前 prompt 的文本部分仍然是纯文本类别名；为了做 overfit sanity check，训练还会使用一个 `reference_mask`，但它只来自训练 GT，不是 SAM3 的交互提示。

默认配置：

```yaml
sam3.prompt_mode: random_instance
```

当前默认是 overfit-instance 模式：

1. 从 clip 中找出跨至少 `min_visible_frames` 可见的有效 instance。
2. 过滤掉太小、太大、以及 wall / floor / ceiling 等类别。
3. 固定一个窗口重复训练。
4. 选择一个有效 instance 作为 `sampled_instance_id`。
5. 读取该 instance 的类别名作为 prompt。
6. 使用该 instance 在第一个可见帧的 GT mask 作为 `reference_mask`，池化 fused feature 得到 object query。

例如日志里的：

```text
prompt='dustbin'
prompt='satchel'
prompt='picture'
```

这些 prompt 都是从 ScanNet++ instance metadata 里采样出来的类别名称。`object / objects / unknown` 这类泛化标签已经默认排除。

如果使用命令行：

```bash
--prompt chair
```

则会切到 fixed prompt；如果仍使用 instance target，需要同时指定具体 `--instance-id`，否则更适合配合 `--target-mode class` 使用。

## 4. GT 构造

对每个 frame，当前代码构造：

```text
prompt_mask:
  当前 sampled_instance_id 的区域

mask_supervision:
  有效 semantic 区域 union prompt_mask

point_valid:
  pointmap 有效区域

semantic_valid:
  semantic label 有效区域

instance_valid:
  sampled_instance_id 且 pointmap 有效的区域
```

因此当前 `prompt_mask` 是“实例级”的，不再是“同类别所有物体”。例如 `prompt='picture'` 时，只监督被采样的那个 picture instance。

## 5. 模型流程

每个 step 的前向流程：

```text
RGB clip + text prompt
  -> SAM3 image model
  -> detector_fpn2 tokens + pooled text feature

RGB clip
  -> StreamVGGT
  -> geometry tokens

SAM3 tokens as query
StreamVGGT geometry tokens as key/value
  -> cross-attention fusion
  -> fused tokens

fused tokens
  + reference_mask pooled object query
  -> dense decoder upsample 到 output_size
  -> mask head
  -> pointmap decoder
  -> semantic/text alignment head
  -> instance embedding head
  -> auxiliary closed-set semantic head
```

当前 StreamVGGT 和 SAM3 都是 frozen，只训练 fusion module 和 dense heads。

当前 StreamVGGT 默认使用 streaming KV cache：

```yaml
geometry.streaming_cache: true
```

### StreamVGGT Memory

当 `geometry.streaming_cache=true` 时，StreamVGGT 不是一次性把整段 clip 当普通 batch 处理，而是在 adapter 里逐帧调用 aggregator：

```text
past_key_values = [None] * aggregator.depth

for frame_idx in range(T):
  current_frame
    + past_key_values
    -> StreamVGGT aggregator(use_cache=True, past_frame_idx=frame_idx)
    -> aggregated_tokens_list
    -> updated past_key_values
```

因此第 `t` 帧输出的 StreamVGGT tokens 已经包含前面帧通过 KV cache 带来的历史信息。当前代码会从这些带历史的 tokens 中取两类特征：

```text
1. layer_index 对应 tokens
   -> resize 到 context_grid
   -> cross-attention 里的 geometry context

2. layers [4, 11, 17, 23]
   -> stream_dpt point decoder 的 DPT tokens
```

所以 StreamVGGT memory 会影响两条路径：

```text
StreamVGGT KV cache
  -> geometry tokens
  -> fused tokens
  -> mask / simple point head

StreamVGGT KV cache
  -> DPT layer tokens
  -> stream_dpt point decoder
```

如果设置：

```bash
--no-geometry-streaming-cache
```

则不使用逐帧 KV cache，而是走普通 clip-level aggregator 前向。这个设置可用于消融 StreamVGGT 原生历史信息。

SAM3 使用 image / detector intermediate feature，不是 video tracker memory。

### Pointmap Decoder

当前支持两种 point decoder：

```yaml
model.point_decoder: simple | stream_dpt
```

`simple` 是早期 baseline：

```text
dense fused feature
  -> shallow Conv point head
  -> pred_pointmap
```

`stream_dpt` 是当前默认配置：

```text
StreamVGGT aggregator layers [4, 11, 17, 23]
  -> StreamVGGT DPTHead
  -> pred_pointmap + pred_point_conf
```

`stream_dpt` 会加载原 StreamVGGT `point_head` 权重。融合方式不是替换 StreamVGGT token，而是把 `fused_tokens + object_query` 投影到 StreamVGGT token 维度后，作为 residual condition 加到 DPT patch tokens 上：

```text
condition = Linear(interp(fused_tokens + object_query))
stream_patch_tokens = stream_patch_tokens + scale * condition
```

其中 `scale` 是可学习参数，初始化为 `0.1`。

### Object Query Memory

当前还有一个轻量 object query memory，它不是 SAM3 原生 tracker memory。

初始化：

```text
reference_mask + fused tokens
  -> masked pooling
  -> object_query
```

逐帧更新：

```text
update_mask + fused tokens
  -> candidate object query
  -> GRUCell(object_query, candidate)
```

更新 mask 来源由配置控制：

```yaml
history.update_source: gt | pred | gt_or_pred
```

`gt` 是训练阶段 oracle / teacher forcing，`pred` 更接近推理时设置。

## 6. Loss 设计

每一帧分别计算 loss，再对整个 clip 求平均。

### Mask Loss

```text
L_mask = BCEWithLogits(pred_mask_logits, prompt_mask)
L_dice = Dice(pred_mask_logits, prompt_mask)
```

这是当前最重要的可视化指标，用来判断 prompt object mask 是否学出来。

### Pointmap Loss

默认只在 prompt foreground 且 pointmap 有效的位置计算：

```text
valid = prompt_mask & point_valid
L_point = SmoothL1(pred_pointmap[valid], gt_pointmap[valid])
```

也可以切到 pred mask 筛选：

```yaml
loss.point_valid_source: gt | pred
loss.point_valid_threshold: 0.5
```

当 `point_valid_source=pred` 时：

```text
valid = point_valid & (sigmoid(pred_mask_logits) > threshold)
```

这个设置用于消融 hard pred-mask point supervision。当前观察是它容易受 mask 错选区域影响，因此不能直接替代 GT valid。

所以当前模型不会被要求重建全场景点云，只监督 prompt 相关物体区域或 pred mask 选中的区域。

### Chamfer / Reprojection Loss

当前代码保留两个几何辅助项：

```text
L_chamfer:
  在 point valid 区域采样 pred / GT points，计算双向 Chamfer

L_reprojection:
  用 pred mask 概率作为 soft weight，把 pred pointmap 投影回当前帧，
  与 GT prompt mask 做 BCE + Dice
```

当前消融中经常把它们设为 0，只看 L1 point loss：

```bash
--chamfer-weight 0
--reprojection-weight 0
```

### Text Alignment Loss

SAM3 text encoder 输出 text embedding。模型 dense semantic head 输出每个像素的 semantic embedding。

```text
prompt_score = cosine(semantic_embedding, text_embedding) * learnable_scale
L_text = BCEWithLogits(prompt_score, prompt_mask)
```

这个 loss 让 foreground pixel 与 prompt text embedding 对齐。

### Auxiliary Semantic Loss

辅助 closed-set 分类头：

```text
L_aux_cls = CrossEntropy(aux_cls_logits[semantic_valid], gt_semantic[semantic_valid])
```

它只作为训练稳定项，不是主输出。

### Instance Match Loss

从当前帧和历史帧采样目标 instance 像素对：

```text
match = 1 if instance_id_curr == instance_id_hist
match = 0 otherwise

L_match = BCEWithLogits(sim(instance_embedding_curr, instance_embedding_hist), match)
```

该 loss 用来让同一个 ScanNet++ instance 在不同帧中的 embedding 保持一致。

### 总损失

当前配置大致是：

```text
L = 1.0 * L_mask
  + 1.0 * L_dice
  + 1.0 * L_point
  + 0.1 * L_chamfer
  + 0.1 * L_reprojection
  + 0.5 * L_text
  + 0.1 * L_aux_cls
  + 0.25 * L_match
```

## 7. 当前输出文件

训练会输出：

```text
outputs/dense_fusion_debug/training_history.csv
outputs/dense_fusion_debug/training_curves.png
outputs/dense_fusion_debug/visualizations/step_xxxxxx.png
outputs/dense_fusion_debug/pointclouds/step_xxxxxx_gt_object.ply
outputs/dense_fusion_debug/pointclouds/step_xxxxxx_pred_object.ply
```

可视化图包含：

```text
RGB
GT prompt mask
Pred mask
Pred prompt score heatmap
```

点云文件用于比较：

```text
GT prompt object point cloud
Pred prompt object point cloud
```

## 8. 当前 baseline 的边界

当前 baseline 已经是 dense image-grid 训练，但仍然有几个限制：

```text
1. prompt 的文本部分仍是类别名；reference_mask 来自训练 GT，不是 SAM3 交互 prompt。
2. SAM3 没有使用 video tracker memory。
3. object query memory 是当前代码自定义实现，不是 SAM3 原生 memory。
4. hard pred-mask point supervision 容易受 mask 错选区域影响。
5. stream_dpt 当前只是 residual token conditioning，mask / occupancy 还没有进入 DPT 解码过程。
6. 当前输出的是 visible object pointmap，还没有 canonical / amodal object memory。
```

下一步如果要做更稳定的 object geometry，优先考虑：

```text
1. 先做 simple / stream_dpt 与 GT valid / pred valid 的消融。
2. 再把 pred mask 从 hard selector 改成 soft weighting。
3. 最后把 soft occupancy 作为 point decoder 的条件，而不是只用于 loss / export。
```
