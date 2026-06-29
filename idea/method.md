# Method 草案：完整第一版 Baseline

## 1. 任务定义

第一版先做一个完整、可视化清楚、能稳定训练的 baseline：

```text
输入：
  单目 RGB 连续帧 / 视频片段
  一个开放词汇 text prompt，例如 "chair"、"bed"、"monitor"

输出：
  prompt 对应类别在每一帧的 2D mask
  prompt 相关物体的 full-resolution object pointmap
  跨帧一致的 instance 对应关系
  由 mask + pointmap lift 得到的 3D object point cloud / object memory
```

这里的重点不是先做全场景 closed-set semantic reconstruction，而是先做：

```text
text prompt -> 2D object mask -> object pointmap -> 3D object memory
```

也就是说，模型只需要为 prompt 相关的物体区域输出可靠结果；墙、地板、天花板等大结构不是第一版目标。

## 2. 数据准备

第一版使用 ScanNet++ pinhole 数据构造监督。

每个 scene 预处理得到：

```text
RGB:
  [T, H, W, 3]

semantic_masks:
  [T, H, W]
  每个像素的类别 id，ignore label = 65535

instance_masks:
  [T, H, W]
  同一个 instance id 在不同帧中表示同一个 3D 物体

pointmaps:
  [T, H, W, 3]
  每个可见像素对应的 world-space XYZ

point_valid:
  [T, H, W]
  pointmap 是否有效
```

生成方式：

1. 使用 pinhole RGB 和 pinhole COLMAP 位姿作为投影相机。
2. 使用原始 ScanNet++ aligned 3D mesh、semantic annotation 和 `segments_anno.json` 作为 3D 标注来源。
3. 将 3D semantic / instance / XYZ 投影到 pinhole RGB frame。
4. 保存 full-resolution `semantic_masks`、`instance_masks`、`pointmaps` 和可视化 summary。

训练采样：

```text
按时间顺序采样连续 clip
sequence_length = T
frame_step 控制帧间隔
```

实例过滤：

```text
去掉 instance id = 0
去掉 semantic ignore
去掉小面积噪声 instance
去掉面积比例过大的结构 instance
去掉 wall / floor / ceiling 等大结构类别
只保留至少跨 min_visible_frames 可见的 instance
```

训练时从有效 instance 中随机选一个 instance，读取它的类别名称作为 text prompt。若 prompt 是类别词，例如 `chair`，则该类别下所有有效 instances 都可以作为 prompt foreground；不同 instance 之间仍然用 `instance_id` 做区分和跨帧匹配。

## 3. 为什么不把训练目标放在 token grid 上

`72 x 72 token grid` 可以作为模型内部 latent feature，但不适合作为第一版完整 baseline 的最终训练/可视化目标：

```text
1. mask 边界过粗，看不出分割是否真的成功
2. 小物体容易只剩几个 token，监督不稳定
3. point cloud 太稀疏，不利于判断 3D 输出质量
4. 和最终任务的 2D masklet / 3D object point cloud 不直观对应
```

因此第一版 baseline 采用：

```text
内部可以用 token / feature map 做融合
输出和 loss 必须回到图像分辨率 H x W
```

如果显存压力大，可以把输入统一 resize 到训练分辨率 `H_train x W_train`，但 mask、pointmap、loss 仍然在这个 dense image grid 上计算，而不是在 `72 x 72` token grid 上计算。

## 4. 模型结构

整体结构：

```text
RGB sequence
  -> StreamVGGT
     -> geometry tokens
     -> pointmap / depth / confidence
     -> camera tokens
     -> 后续接入 streaming KV cache

RGB sequence + text prompt
  -> SAM3
     -> prompt-conditioned semantic / tracking features
     -> optional SAM3 mask proposals

SAM3 semantic features + StreamVGGT geometry features
  -> cross-attention / feature fusion
  -> dense decoder
  -> full-resolution outputs
```

内部融合可以仍然发生在 latent feature 上：

```text
sam_tokens / sam_feature_map
geometry_tokens / geometry_feature_map
camera tokens
```

但融合后需要经过 dense decoder 上采样回 `H x W`：

```text
fused_features
  -> mask decoder
  -> pointmap decoder
  -> semantic / text alignment decoder
  -> instance embedding decoder
```

第一版可以先冻结 SAM3 和 StreamVGGT，只训练：

```text
projection layers
fusion module
dense decoder
mask / point / semantic / instance heads
```

## 5. 模型输出

第一版模型输出应是 full-resolution 的：

```text
pred_mask_logits:
  [T, H, W]
  prompt foreground mask logits

pred_pointmap:
  [T, H, W, 3]
  每个像素预测 world-space XYZ
  只要求 prompt foreground 区域准确

semantic_embedding:
  [T, H, W, D]
  每个像素的开放词汇语义特征

prompt_score:
  [T, H, W]
  semantic_embedding 与 text embedding 的相似度

instance_embedding:
  [T, H, W, D]
  用于跨帧同一 instance 匹配

object_memory:
  对每个 persistent object 维护：
    instance identity
    text / semantic prototype
    geometry prototype
    accumulated 3D points
    confidence
    last seen frame
```

可以保留一个 closed-set `aux_cls_head` 做辅助训练，但它不是主输出。主语义输出应该是 `semantic_embedding` 与 text prompt embedding 的相似度。

## 6. 训练流程

每次训练采样一个连续 clip：

1. 读取 RGB、semantic mask、instance mask、pointmap。
2. 根据面积、类别、可见帧数过滤无效 instance。
3. 随机选择一个有效 instance 的类别作为 text prompt。
4. 构造 prompt foreground：

```text
prompt_foreground = semantic label == prompt label
                 或 instance label 属于 prompt 类别
```

5. SAM3 提取 prompt-conditioned semantic / tracking features。
6. StreamVGGT 提取 geometry / camera features。
7. Fusion module 融合语义与几何。
8. Dense decoder 输出 full-resolution mask、pointmap、semantic embedding、instance embedding。
9. 用 full-resolution GT mask / pointmap / instance id 计算 loss。
10. 将预测 mask 内的 pointmap lift 到 3D，导出 object point cloud 可视化。

## 7. Loss 设计

### 7.1 Prompt Mask Loss

监督 prompt 对应类别的 full-resolution foreground mask：

```text
gt_prompt_mask[t, y, x] = 1
  if pixel belongs to a valid instance whose label matches prompt
  else 0
```

```text
L_mask = BCEWithLogits(pred_mask_logits, gt_prompt_mask)
       + Dice(pred_mask_logits, gt_prompt_mask)
```

该 loss 直接决定模型是否能在图像上分出 prompt 物体，是第一版最重要的可视化指标。

### 7.2 Object Pointmap Loss

只在 prompt foreground 且 pointmap 有效的像素上监督 3D 坐标：

```text
valid_point = gt_prompt_mask
            & point_valid
            & object_filter_valid

L_point = SmoothL1(pred_pointmap[valid_point], gt_pointmap[valid_point])
```

这表示模型不需要重建所有背景点，只需要把 prompt 相关物体区域变成准确的 3D object points。

### 7.3 Text Alignment Loss

主语义监督使用开放词汇 text alignment，而不是固定类别 softmax 作为最终定义：

```text
text_feat = TextEncoder(prompt)
score = cosine(semantic_embedding, text_feat)
```

前景像素应该与 prompt embedding 相似，背景像素应该不相似：

```text
L_text = BCEWithLogits(score, gt_prompt_mask)
```

如果有多个 prompt 或 batch 内多个类别，可以扩展为 contrastive loss。

### 7.4 Auxiliary Semantic Classification Loss

为了让训练更稳定，可以保留 closed-set 辅助分类：

```text
L_cls = CrossEntropy(aux_cls_logits[valid_semantic], gt_semantic_label[valid_semantic])
```

但该 loss 是辅助项，不应替代 text alignment。否则方法会退化成 closed-set semantic segmentation。

### 7.5 Cross-frame Instance Matching Loss

用 full-resolution instance id 监督跨帧一致性。为了节省显存，可以从有效 object 像素中采样点对，而不是计算全图所有像素对。

```text
match_ij = 1 if instance_id_curr[i] == instance_id_hist[j]
         = 0 otherwise
```

```text
L_match = BCEWithLogits(
  sim(instance_embedding_curr[i], instance_embedding_hist[j]),
  match_ij
)
```

这个 loss 用来保证同一个 3D 物体在不同视角下的 embedding 稳定。

### 7.6 Reprojection Consistency Loss

后续增强项。利用 pointmap 将预测 mask lift 到 3D，再投影到相邻帧：

```text
pred_mask_t + pred_pointmap_t
  -> 3D object points
  -> project to frame t+k
  -> compare with gt / pred mask at t+k
```

```text
L_reproj = mask reprojection consistency
         + depth / visibility weighted consistency
```

这个 loss 用来减少跨视角 mask 漂移，是后续提升 tracking consistency 的重点。

## 8. 第一版 Baseline 必须展示什么

第一版结果应该能直接可视化：

```text
1. RGB + GT prompt mask + pred prompt mask
2. RGB + pred mask 边界
3. pred mask 内导出的 3D object point cloud
4. GT pointmap object cloud 与 pred object cloud 对比
5. 同一 instance 跨帧 mask / point cloud 的对应关系
```

第一版成功标准：

```text
mask 能在 full-resolution 图像上看出来
pred point cloud 能形成目标物体的大致形状
同一物体跨帧 embedding / memory 不乱跳
墙、地板、天花板等大结构不会主导训练
```

第一版暂不追求：

```text
全场景 closed-set 语义地图
所有物体一次性完整分割
端到端微调 SAM3 / StreamVGGT
长期大规模 object memory
复杂自然语言描述
```

## 9. 后续扩展方向

baseline 跑通后再逐步加入：

```text
StreamVGGT streaming KV cache
SAM3 video memory / tracker features
多 prompt / 多 object 同时查询
更高分辨率 mask refinement
object memory 在线合并与更新
reprojection consistency loss
text embedding contrastive training
```

当前最重要的是先得到一个完整、可解释、可视化明确的 baseline，而不是在低分辨率 token grid 上证明一个看不清的中间结果。
