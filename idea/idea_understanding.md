# 对当前想法的理解

## 任务需求

目标不是单纯做 2D 分割，也不是直接复现 3AM 的随机 reference object 跟踪流程，而是做一个面向室内场景的几何感知开放词汇物体追踪系统。

输入：

```text
RGB 序列 / 视频
开放词汇 text prompt
```

期望输出：

```text
房间中小物体的跨帧追踪结果
语义地图 / 语义点云
2D mask 与 3D object identity 的对应关系
```

重点对象是房间里的具体小物体，例如椅子、桌子、柜子、箱子、杯子等。墙、地板、天花板、大面积结构面不是主要目标，训练和采样时应尽量过滤掉。

## 核心思想

核心思想是把三类信息结合起来：

```text
ScanNet++ 3D 标注
  -> 提供稳定的跨帧 instance 监督

StreamVGGT
  -> 提供几何、深度、相机、点云先验

SAM3
  -> 提供开放词汇 text prompt 到 object mask / object candidate 的入口
```

训练阶段不依赖 SAM3 text prompt 是否稳定，因为 SAM3 是开放词汇推理入口，不适合作为稳定 GT。训练阶段应优先使用 ScanNet++ 由 3D instance annotation 投影得到的跨帧一致 `instance_masks` 作为监督。

推理阶段再使用 SAM3 根据 text prompt 产生候选 object mask，然后用训练好的几何语义融合模型把这些 mask 绑定到稳定的 3D object identity 上，并跨帧追踪。

## Model 理解

模型应该是 object-level 的，而不是 dense pixel-level 的。

整体结构：

```text
RGB sequence
  -> StreamVGGT
     -> pointmap / depth / confidence / camera pose / geometry tokens

object masks or SAM3 masks
  -> object queries

object queries + geometry context
  -> cross-attention fusion
  -> object semantic logits
  -> object 3D centroid / object 3D feature
  -> cross-frame match embedding
```

融合方式应该接近：

```text
object / semantic tokens as query
geometry / camera tokens as key-value
cross-attention
```

也就是说，语义或物体 token 主动从几何上下文中读取信息。这样可以做到：

- 用几何稳定语义；
- 用语义锚定几何；
- 用 3D object identity 约束跨帧 mask 对应关系。

## Training 理解

训练时不应该只随机选一个 instance 作为 reference。更合理的是从一个连续 clip 中选出多个有效 object instances。

当前更合适的训练流程：

```text
1. 从 processed ScanNet++ 中随机采样一个连续 clip
2. 读取 RGB、instance_mask、semantic_mask
3. 根据面积、可见帧数、类别过滤无效 object
4. 对每个有效 instance 形成 object query
5. 用 StreamVGGT frozen 输出提供几何上下文
6. 训练 fusion model
```

监督信号：

```text
semantic loss:
  object query 分类到 ScanNet++ semantic label

3D loss:
  object query 预测的 3D centroid / feature 与 mask 区域内的 StreamVGGT pointmap 聚合结果一致

cross-frame matching loss:
  同一个 instance id 的 object embeddings 拉近
  不同 instance id 的 object embeddings 拉远

可选 mask loss:
  如果后续加 mask decoder，可用 instance_mask 做 BCE / Dice
```

ScanNet++ 的 `instance_masks` 来自同一个 3D annotation 投影，因此同一个 instance id 在不同帧中表示同一个 3D 物体。这一点是训练跨帧 object identity 的关键。

## 当前 v0 实现

当前代码中的 v0 训练遵循这个方向：

```text
随机采样连续 clip
不是随机采样单个 instance
每个 clip 内保留多个有效 instances
用 semantic_mask 的众数作为 object semantic label
用 instance id 做跨帧 contrastive / matching 监督
用 StreamVGGT point/depth/conf/camera_pose 做几何上下文
```

当前 v0 暂时没有使用 SAM3 输出，因为 inspect 结果显示 `chair` prompt 在当前测试帧中没有产生 mask。SAM3 后续应作为推理阶段或 noisy query augmentation 接入。

## 当前代码训练流程

当前训练入口是：

```bash
PYTHONPATH=src python scripts/train_object_fusion.py \
  --config configs/object_fusion_train.yaml \
  --iterations 200 \
  --device cuda
```

配置文件是 `configs/object_fusion_train.yaml`。当前训练流程更具体地说是：

```text
1. 从 data/processed/scannetpp_2d/manifest.json 读取已处理的 ScanNet++ 场景。
2. 在一个场景内随机采样连续帧窗口，例如 sequence_length=4。
3. 对每个窗口读取：
   - 原始 RGB 图片路径；
   - instance_masks；
   - semantic_masks。
4. 运行 frozen StreamVGGT，得到每帧：
   - pts3d_in_other_view；
   - depth；
   - conf；
   - depth_conf；
   - camera_pose。
5. 对每帧 instance mask 做过滤：
   - 去掉 instance id 0；
   - 去掉小于 min_pixels 的噪声；
   - 去掉面积比例大于 max_area_ratio 的大结构；
   - 只保留至少跨 min_visible_frames 可见的实例；
   - 每帧最多保留 max_objects_per_frame 个对象。
6. 对每个保留 instance 构造 object query：
   - mask 区域内 xyz/depth/conf/depth_conf 的均值；
   - 2D centroid；
   - area ratio。
7. 对 StreamVGGT 几何输出做 adaptive pooling，形成 geometry tokens。
8. 将 object query 作为 query，geometry tokens + camera tokens 作为 context，送入 ObjectFusionModel。
9. 计算三个损失：
   - semantic_loss：object query 分类到 semantic_mask 的众数 label；
   - centroid_loss：预测 3D centroid 对齐 mask 区域内 pointmap 的均值；
   - contrastive_loss：同 instance id 的 object embeddings 拉近，不同 id 拉远。
10. 写出训练日志和 checkpoint。
```

当前输出：

```text
outputs/object_fusion_debug/training_history.csv
outputs/object_fusion_debug/training_curves.png
outputs/object_fusion_debug/ckpt_last.pt
```

也可以用已有的 CSV 重新画曲线：

```bash
PYTHONPATH=src python scripts/plot_training_curves.py \
  --metrics outputs/object_fusion_debug/training_history.csv \
  --output outputs/object_fusion_debug/training_curves.png
```

## 当前 200 Step Debug 结果

一次简单 debug 训练已经可以跑完 200 step。关键现象：

```text
step=1   loss=9.3520 semantic=6.8605 centroid=0.1609 contrastive=4.6612 objects=108
step=100 loss=4.4040 semantic=2.8750 centroid=0.0220 contrastive=3.0140 objects=99
step=200 loss=4.0571 semantic=2.5515 centroid=0.0107 contrastive=2.9898 objects=110
```

初步观察：

- 总 loss 从约 9.35 降到约 4.06，说明 v0 训练链路是通的；
- semantic loss 下降明显，说明 object query 与 semantic label 的监督能被模型学习；
- centroid loss 很快下降到较低水平，说明基于 mask 区域 pointmap 均值的 3D centroid 监督比较容易；
- contrastive loss 有下降但仍然较高，说明跨帧 identity embedding 是后续重点，需要更好的 object token、更多负样本设计或更稳定的 geometry tokens；
- 每步对象数量在几十到一百多个之间波动，说明当前不是随机单 instance 训练，而是多 object / 多 frame 训练。

## 后续可以继续确认的问题

1. 是否需要把训练形式改成 text-conditioned？

   例如随机选择一个 semantic category，将该类别对应的所有 instances 作为正样本，用类别 text embedding 作为 query。这会更接近开放词汇推理形式。

2. 是否需要从 StreamVGGT 内部抓更强的 geometry tokens？

   当前 v0 使用的是 StreamVGGT 最终输出的 point/depth/conf/camera_pose。后续可以 hook `aggregator` 输出的 `aggregated_tokens_list`，作为更接近 transformer latent space 的 geometry tokens。

3. 如何过滤墙、地板、天花板等大结构？

   当前主要使用面积阈值和可见帧阈值。后续如果 semantic label id 映射明确，应加入类别黑名单过滤。

4. SAM3 接入方式是什么？

   训练阶段可先不用 SAM3。推理阶段可用 SAM3 text prompt 产生 mask，再用 fusion model 判断这些 mask 的 3D identity 和跨帧一致性。
