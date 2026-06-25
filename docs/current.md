# 当前进展

当前目标是搭建一个面向室内小物体追踪的多模态模型：

```text
RGB 连续帧 + 语义文本 prompt
  -> 融合 SAM3 语义特征与 StreamVGGT 几何特征
  -> 输出物体 mask、3D 位置、语义类别和跨帧对应关系
```

## 数据流动

输入：

```text
1. ScanNet++ resized RGB 连续帧
2. ScanNet++ instance / semantic masks
3. ScanNet++ mesh + COLMAP pose/intrinsics 光栅化得到的 GT pointmap
4. 每个 scene 的 object metadata label，例如 office chair / monitor / cabinet
```

训练时每个 clip 会随机采样一个大小合理、跨帧可见、未被 blacklist 过滤的 instance，并使用它的类别名作为 SAM3 text prompt。

当前默认过滤：

```text
wall / floor / ceiling
```

示例 prompt：

```text
power socket
monitor
computer tower
office chair
cabinet
```

## 模型流程

```text
RGB sequence
  -> frozen SAM3 image/text backbone
     -> detector_fpn2 tokens + text feature

RGB sequence
  -> frozen StreamVGGT aggregator
     -> geometry tokens

当前默认先不使用 StreamVGGT camera tokens，用于隔离验证普通 geometry patch tokens 是否足够训练 mask。

SAM3 tokens as query
StreamVGGT tokens as key/value
  -> cross-attention fusion
```

关键模型输出：

```text
token semantic logits:
  每个 72x72 token 的语义分类

token pointmap:
  每个 72x72 token 的 3D 坐标预测

token matching embedding:
  token 级跨帧 instance 对齐特征

object mask logits:
  [T, num_queries, 144, 144]
  每一帧、每个 object query 的 mask

object semantic logits:
  object query 的语义分类

object centroid:
  object query 对应的 3D 中心点

object association embedding:
  object query 的跨帧追踪特征
```

输出：

```text
1. per-frame object masks
2. per-frame object 3D centroids
3. per-frame semantic / point tokens
4. cross-frame object association embeddings
```

当前还没有把多帧 mask + pointmap + association 显式融合成最终的全局语义地图点云；但模型输出已经具备构建该语义地图点云所需的中间量。

## 当前训练监督

```text
semantic_loss:
  token -> ScanNet++ semantic label

point_loss:
  token -> COLMAP/mesh rasterized GT pointmap

match_loss:
  同一 instance id 的 token embedding 跨帧拉近

object_mask_loss / object_dice_loss:
  object query mask -> ScanNet++ instance mask

object_point_loss:
  object query centroid -> instance 内 GT pointmap 均值

object_semantic_loss:
  object query -> instance semantic label

object_match_loss:
  同一 instance id 的 object query embedding 跨帧拉近
```

## 当前实验结果

Inspect 已确认：

```text
pointmaps_available=True
SAM3 tokens=(1, 20736, 512)
StreamVGGT geometry_tokens=(1, 576, 2048)
camera_tokens=(1, 4, 9)
sampled_prompt='monitor'
```

20 step 小训练可以跑通，loss 有下降：

```text
step=1  loss=25.3288 obj_mask=0.7116 obj_point=2.4610 prompt='power socket'
step=10 loss=19.7917 obj_mask=0.1490 obj_point=1.4021 prompt='monitor'
step=20 loss=17.0002 obj_mask=0.0037 obj_point=0.9013 prompt='computer tower'
```

这说明当前动态 prompt、多类 object supervision、GT pointmap、object-query mask 分支已经接入并可训练。

训练过程会保存 mask 可视化到：

```text
outputs/latent_fusion_debug/visualizations/step_XXXXXX.png
```

## 当前边界

```text
1. SAM3 和 StreamVGGT 仍然 frozen。
2. 当前是单卡训练，没有做 DDP。
3. mask 输出先在 144x144 上训练，后续可提升到 288x288 或加入高分辨率 refinement。
4. object query 当前按 ScanNet++ instance id 固定 slot，还没有使用 Hungarian matching。
5. 当前输出的是 per-frame object 结果和 association embedding，还没有实现最终语义地图点云聚合模块。
```

## 下一步建议

```text
1. 保存 object mask / point / association 的可视化结果。
2. 加 inference/export，把多帧 object masks + pointmap + association 融合成语义地图点云。
3. 将 object query 分配从固定 slot 升级为 Hungarian matching。
4. 尝试更高 mask_grid，例如 288x288。
5. 对 SAM3 不同中间层和 tracker memory 做 ablation。
```
