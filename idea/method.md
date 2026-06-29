# 当前 Dense Fusion Baseline

## 1. 当前任务

当前代码实现的是一个 dense baseline：

```text
输入：
  连续 RGB clip
  一个 text prompt

输出：
  pred_mask_logits: [T, H, W]
  pred_pointmap: [T, H, W, 3]
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

当前 prompt 是纯文本，没有 box、point、click、reference mask 等空间提示。

默认配置：

```yaml
sam3.prompt_mode: random_instance
```

每个训练 step：

1. 从 clip 中找出跨至少 `min_visible_frames` 可见的有效 instance。
2. 过滤掉太小、太大、以及 wall / floor / ceiling 等类别。
3. 随机选择一个有效 instance。
4. 读取该 instance 的类别名作为 prompt。

例如日志里的：

```text
prompt='dustbin'
prompt='satchel'
prompt='picture'
```

这些 prompt 都是从 ScanNet++ instance metadata 里采样出来的类别名称。

如果使用命令行：

```bash
--prompt chair
```

则会切到 fixed prompt，并只训练 label 匹配 `chair` 的 instance 区域。

## 4. GT 构造

对每个 frame，当前代码构造：

```text
prompt_mask:
  当前 prompt 对应类别的所有有效 instance 区域

mask_supervision:
  有效 object 区域 union prompt_mask

point_valid:
  pointmap 有效区域

semantic_valid:
  semantic label 有效且没有被过滤的 object 区域

instance_valid:
  有效 object 且 pointmap 有效的区域
```

当前 `prompt_mask` 是“类别级”的：

```text
prompt='picture'
-> 所有有效 picture instances 都是 foreground
```

不是“只追踪被随机选中的那个具体 instance”。具体 instance identity 目前由 `instance_embedding` 和 match loss 监督。

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
  -> dense decoder upsample 到 output_size
  -> mask head
  -> pointmap head
  -> semantic/text alignment head
  -> instance embedding head
  -> auxiliary closed-set semantic head
```

当前 StreamVGGT 和 SAM3 都是 frozen，只训练 fusion module 和 dense heads。

当前 StreamVGGT 使用的是 clip-level 多帧输入，不是 streaming KV cache。SAM3 使用 image backbone intermediate feature，不是 video tracker memory。

## 6. Loss 设计

每一帧分别计算 loss，再对整个 clip 求平均。

### Mask Loss

```text
L_mask = BCEWithLogits(pred_mask_logits, prompt_mask)
L_dice = Dice(pred_mask_logits, prompt_mask)
```

这是当前最重要的可视化指标，用来判断 prompt object mask 是否学出来。

### Pointmap Loss

只在 prompt foreground 且 pointmap 有效的位置计算：

```text
valid = prompt_mask & point_valid
L_point = SmoothL1(pred_pointmap[valid], gt_pointmap[valid])
```

所以当前模型不会被要求重建全场景点云，只监督 prompt 相关物体区域的 3D 点。

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

从当前帧和历史帧采样有效 object 像素对：

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
1. prompt 只有 text，没有空间 reference。
2. random_instance 采样的是一个 instance，但 GT foreground 是该类别所有有效 instances。
3. SAM3 没有使用 video tracker memory。
4. StreamVGGT 没有使用 streaming KV cache。
5. object memory 目前只是通过导出 point cloud 可视化，还没有实现长期在线维护。
```

下一步如果要做“具体 instance 追踪”，需要加入 reference mask / box / point prompt，或者用第一帧 GT instance mask 构造 instance prototype。
