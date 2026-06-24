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
  -> 提供开放词汇、视频记忆、mask decoder 之前/附近的中间语义追踪特征
```

一个关键修正是：这里不应该把 SAM3 理解成“先检测出 object mask，再把 mask 送进后续模型”。如果 SAM3 最终输出没有检测到物体，那只是它的最终 mask/prompt 解码器没有给出有效实例；这不等价于 SAM3 backbone、memory、prompt-conditioned tokens 中没有可用的语义和跟踪特征。

真正想做的是在 SAM3 的中间特征层面和 StreamVGGT 融合：

```text
SAM3 intermediate semantic/tracking tokens
  as query

StreamVGGT geometry tokens + camera tokens
  as key/value context

cross-attention fusion
  -> 3D-aware semantic tokens
  -> pointmap / semantic logits / match embeddings / downstream decoder input
```

ScanNet++ 的 `instance_masks` 和 `semantic_masks` 是训练监督，而不是最终架构里唯一的输入。SAM3 最终 mask 可以作为推理时的辅助输出或下游 decoder，但不应该是整个融合模型成立的前提。

## SAM3 输出层级理解

已有一次 SAM3 inspection 的结果大致是：

```text
输入 RGB:
  [B, 3, 1008, 1008]

Patch Embedding:
  patch size = 14
  1008 / 14 = 72

32 层 ViT 主干:
  hidden dim = 1024
  final map ~= [B, 72, 72, 1024]

SAM3 Detector Neck:
  开放词汇检测 / 文本分割分支
  FPN-0 [B, 256, 288, 288]
  FPN-1 [B, 256, 144, 144]
  FPN-2 [B, 256,  72,  72]
  Detector Transformer + Language Features

SAM2 Tracker Neck:
  视频传播 / 交互分割分支
  FPN-0 [B, 256, 288, 288]
  FPN-1 [B, 256, 144, 144]
  FPN-2 [B, 256,  72,  72]
  Memory Attention + Mask Decoder
```

这个结果很关键，因为 `72 x 72` 是最自然的第一版融合网格：

```text
SAM3 ViT / FPN-2 tokens:
  semantic + open-vocabulary + tracking prior

StreamVGGT patch / aggregator tokens:
  geometry + camera + 3D prior

ScanNet++ projected masks:
  downsample 到 72 x 72 做 semantic / instance / matching 监督
```

因此第一版真正模型不建议先从高分辨率 mask decoder 做起，而是先在 `72 x 72` latent grid 上做：

```text
sam_tokens_72 = project(SAM3 detector/tracker FPN-2 or ViT tokens)
geo_tokens_72 = project(StreamVGGT latent geometry tokens)
cam_tokens    = project(StreamVGGT camera tokens)

fused_tokens = CrossAttention(
  query = sam_tokens_72,
  key/value = concat(geo_tokens_72, cam_tokens)
)
```

不同 SAM3 层的用途可以这样分工：

```text
ViT final map [B, 72, 72, 1024]:
  更底层、更通用，适合作为稳健 spatial semantic tokens。

Detector FPN-2 [B, 256, 72, 72]:
  最适合第一版 text/open-vocabulary fusion，分辨率和 VGGT patch grid 对齐。

Tracker FPN-2 [B, 256, 72, 72]:
  更适合加入时序传播和 memory consistency。

Detector Transformer + Language Features:
  如果能 hook 出来，最适合作为 text-conditioned query 或 prompt-conditioned token。

FPN-1 / FPN-0:
  暂时不作为主融合层，后续用于 mask decoder / 高分辨率 refinement。
```

训练标签也应该先对齐到这个 token grid：

```text
semantic_mask -> nearest/majority downsample -> [B, 72, 72]
instance_mask -> nearest/majority downsample -> [B, 72, 72]
pointmap/depth/conf -> average/valid pooling -> [B, 72, 72, ...]
```

对每个 72x72 token，只在 majority ratio 足够高时计算监督；混合像素太多、instance id 为 0、墙地板天花板、面积过大的结构 token 都应该 ignore。这样可以避免 final mask 没检测到时训练断掉，也避免噪声小块和大结构面主导 loss。

## Model 理解

模型不应该只做“SAM3 final mask -> object query -> 分类/追踪”。更接近 `idea/ours_model.py` 的设计是 latent token fusion：SAM3 提供语义/追踪 token，StreamVGGT 提供几何/相机 token，二者先在隐空间里融合，再接不同任务 head。

整体结构：

```text
RGB sequence
  -> StreamVGGT
     -> latent geometry tokens / camera tokens / pointmap / depth / confidence

RGB sequence + text prompt
  -> SAM3
     -> intermediate text-conditioned semantic/tracking tokens
     -> optional final masks

SAM3 tokens + StreamVGGT tokens
  -> cross-attention fusion
  -> fused semantic-geometry tokens
  -> pointmap head
  -> semantic logits head
  -> cross-frame match embedding
  -> optional mask / downstream decoder head
```

融合方式应该接近：

```text
SAM3 semantic/tracking tokens as query
geometry / camera tokens as key-value
cross-attention
```

也就是说，语义或物体 token 主动从几何上下文中读取信息。这样可以做到：

- 用几何稳定语义；
- 用语义锚定几何；
- 用 3D object identity 约束跨帧 token / mask 对应关系；
- 在 SAM3 final mask 失败时，仍然可以让中间 token 和几何 token 产生可训练信号。

## Training 理解

训练时不应该只随机选一个 instance 作为 reference。更合理的是从一个连续 clip 中选出多个有效 object instances，并用这些 instance/semantic 标注去监督融合后的 token。

最终想要的训练流程更接近 `idea/ours_training.py`：

```text
1. 从 processed ScanNet++ 中随机采样一个连续 clip
2. 读取 RGB、instance_mask、semantic_mask
3. 根据面积、可见帧数、类别过滤无效 object / 大结构
4. SAM3 对 RGB + text prompt 输出中间语义/追踪 tokens
5. StreamVGGT 对 RGB 序列输出 geometry/camera tokens，并保留流式 KV/cache
6. SAM3 tokens 作为 query，StreamVGGT tokens 作为 context 做 cross-attention
7. fused tokens 接 pointmap、semantic、matching 等 head
8. 用 ScanNet++ 的 projected 2D/3D 标注监督这些 head
```

监督信号：

```text
semantic loss:
  fused token / token-pooled object 分类到 ScanNet++ semantic label

3D loss:
  fused token 预测 pointmap / centroid，对齐 GT pointmap 或 mask 区域内的 3D 聚合

cross-frame matching loss:
  同一个 ScanNet++ instance id 的 fused embeddings 拉近 / 匹配为 1
  不同 instance id 的 fused embeddings 拉远 / 匹配为 0

可选 mask loss:
  如果后续加 mask decoder，可用 instance_mask 做 BCE / Dice
```

ScanNet++ 的 `instance_masks` 来自同一个 3D annotation 投影，因此同一个 instance id 在不同帧中表示同一个 3D 物体。这一点是训练跨帧 object identity 的关键。

## 当前 v0 实现

当前代码中的 v0 训练只打通了数据、StreamVGGT 几何和 ScanNet++ 监督，还没有实现真正的 SAM3 中间特征融合。它更像一个 sanity baseline：

```text
随机采样连续 clip
不是随机采样单个 instance
每个 clip 内保留多个有效 instances
用 semantic_mask 的众数作为 object semantic label
用 instance id 做跨帧 contrastive / matching 监督
用 StreamVGGT point/depth/conf/camera_pose 做几何上下文
```

当前 v0 暂时没有使用 SAM3。之前说“因为 `chair` prompt 没有产生 mask，所以训练先不用 SAM3”这个说法只适合作为 v0 的工程原因，不应该作为最终架构判断。最终版本需要绕开“final mask 有没有检测到”的限制，直接抓 SAM3 的中间 token / memory / prompt-conditioned features 来融合。

因此，v0 的价值是：

```text
验证 ScanNet++ projected masks 是否可训练
验证 StreamVGGT 输出是否能提供 3D/semantic 监督信号
验证 continuous clip + instance id matching 的训练闭环
```

但 v0 不是最终模型。

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

   下一步应该优先写 SAM3 intermediate feature inspector / adapter，而不是继续依赖 `out_binary_masks`。需要确认 SAM3 哪些模块能稳定吐出：

   ```text
   image encoder tokens
   prompt/text-conditioned tokens
   video memory tokens
   mask decoder tokens before final threshold
   object pointer / tracking tokens
   ```

   拿到这些 tensor shape 后，再把 `ObjectFusionModel` 升级成 `LatentGeometricSemanticModel` 风格：SAM3 tokens 做 query，StreamVGGT latent geometry/camera tokens 做 key-value。
