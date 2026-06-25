# 当前进展

当前目标是搭建一个面向室内小物体追踪的多模态模型：

```text
RGB 连续帧 + 语义文本 prompt
  -> 融合 SAM3 语义特征与 StreamVGGT 几何特征
  -> 输出语义点云 token、3D pointmap token、跨帧对应 embedding
  -> 通过 ref mask + token correspondence 推出跨帧 mask 追踪结果
```

## 数据输入

训练使用已经处理好的 ScanNet++ 数据：

```text
1. resized RGB 连续帧
2. instance masks / semantic masks
3. mesh + COLMAP pose/intrinsics 光栅化得到的 GT pointmap
4. object metadata label，例如 office chair / monitor / cabinet
```

当前默认过滤：

```text
wall / floor / ceiling
```

默认 `prompt_mode=random_instance`：每个 clip 随机选择一个大小合理、跨帧可见、未被 blacklist 过滤的 instance，并使用它的类别名作为 SAM3 text prompt。如果传入具体 prompt，例如 `--prompt chair`，则切换为固定 prompt 并只训练对应类别过滤后的 token。

## 模型结构

当前代码已经回到更接近 `idea/ours_model.py` 的结构：

```text
RGB sequence
  -> frozen SAM3 image/text backbone
     -> detector_fpn2 semantic tokens

RGB sequence
  -> frozen StreamVGGT aggregator
     -> geometry tokens

SAM3 tokens as query
StreamVGGT tokens as key/value
  -> cross-attention fusion
  -> fused tokens
```

当前模型显式输出：

```text
pred_logits:
  每个 token 的 semantic logits

pred_pointmap:
  每个 token 的 3D 坐标

fused_tokens:
  融合后的语义-几何 token

embeddings:
  用于跨帧 correspondence 的 token embedding
```

当前模型 **不再显式输出 `mask_logits`**。mask 追踪结果不是直接 decode 出来的，而是用 ref frame 的 GT instance mask 选出 object tokens，再用这些 token 的 embedding 和后续帧 token 做相似度传播得到。

## 训练监督

当前训练 loss 为三项：

```text
semantic_loss:
  pred_logits -> ScanNet++ semantic mask token label

point_loss:
  pred_pointmap -> GT pointmap token

match_loss:
  计算跨帧 token correspondence matrix。
  同一 instance id 的跨帧 token pair 为正样本，不同 instance id 为负样本，
  使用 BCEWithLogits 监督。
```

这对应 `idea/ours_training.py` 里的核心逻辑：

```text
gt_match_matrix[i, j] = instance_id_curr[i] == instance_id_hist[j]
```

## Mask 可视化

训练过程会保存 correspondence mask propagation 可视化：

```text
outputs/latent_fusion_debug/visualizations/step_XXXXXX.png
outputs/latent_fusion_debug/visualizations/step_XXXXXX_crops.png
```

可视化含义：

```text
1. 从一个可见 instance 中选 ref frame。
2. 用该 instance 的 GT mask 取 ref object tokens。
3. 求 ref object token embedding 的 prototype。
4. 与每一帧所有 token embedding 计算相似度。
5. 每一帧选取与 ref prototype 最相似的 top-k token，k 等于 ref mask 覆盖的 token 数。
6. top-k token map 上采样回 RGB 尺寸，作为 propagated mask 显示。
```

因此现在的 mask 可视化是在验证“跨帧追踪/对应关系”，不是验证一个单帧 mask decoder。

## 当前边界

```text
1. SAM3 和 StreamVGGT 仍然 frozen。
2. 当前是单卡训练，没有做 DDP。
3. StreamVGGT camera tokens 默认关闭，用于先验证普通 geometry tokens。
4. 当前还没有最终的全局语义地图点云聚合模块。
5. 当前 mask 是 correspondence propagation 结果，还不是独立高分辨率分割 decoder。
```

## 推荐运行

```bash
PYTHONPATH=src python scripts/train_latent_fusion.py \
  --config configs/latent_fusion_train.yaml \
  --iterations 200 \
  --device cuda
```
