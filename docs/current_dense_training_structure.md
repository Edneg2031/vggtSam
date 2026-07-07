# 当前计划：3D-aware SAM3 Tracker Adapter

## 目标

原版 SAM3 在长视角变化、遮挡、目标短暂消失后，可能无法重新判断同一物体是否存在。

当前计划是：

```text
把 StreamVGGT 的 3D 几何信息融合进 SAM3 中间特征，
再让融合后的特征继续走 SAM3 原本的视频 tracker 流程，
从而提升跨视角实例 mask 追踪。
```

## 结构图

```text
RGB sequence + GT instance mask
        │
        ├──────────────────────────────────────┐
        │                                      │
        ▼                                      ▼
Frozen SAM3 tracker encoder              Frozen StreamVGGT
tracker_fpn2                              aggregator last layer
[T, 256, 72, 72]                          geometry [T, 144, 2048]
        │                                 camera   [T, 1, 9]
        │                                      │
        ▼                                      ▼
SAM tokens                            3D / camera tokens
[T, 5184, 512]                              │
        │                                      │
        └───────────────┬──────────────────────┘
                        ▼
        Camera-guided cross attention
        query: SAM tokens
        key/value: StreamVGGT geometry + camera
                        │
                        ▼
        fused tokens
        [T, 5184, 256]
                        │
                        ▼
        SAM3 FPN adapter
        fused tokens -> 3D-aware FPN residual
                        │
                        ▼
        SAM3 original FPN + residual
        fpn0 + residual_fpn0  [T, 32, 288, 288]
        fpn1 + residual_fpn1  [T, 64, 144, 144]
        fpn2 + residual_fpn2  [T,256,  72,  72]
                        │
                        ▼
        SAM3 original tracker flow
        memory + object score gate + mask decoder
                        │
                        ▼
        Pred instance mask [T, 256, 384]
                        │
                        ▼
        GT instance mask supervision
        BCE + Dice
```

## Token 含义

### SAM tokens

SAM tokens 来自 SAM3 tracker 分支的 FPN 特征：

```text
sam2_backbone_out["backbone_fpn"][-1]
```

也就是：

```text
tracker_fpn2: [T, 256, 72, 72]
```

这里不是把原图 mask resize 成 token，而是取 SAM3 backbone/tracker 已经算出的中间特征。

处理方式：

```text
[T, 256, 72, 72]
  -> 展平空间维度
  -> [T, 5184, 256]
  -> concat text feature [256]
  -> [T, 5184, 512]
```

所以图里的 SAM tokens 是：

```text
[T, 5184, 512]
```

其中：

```text
5184 = 72 * 72
512 = 256 visual feature + 256 text feature
```

### 3D / geometry tokens

3D tokens 来自 StreamVGGT aggregator 的最后一层：

```text
layer_index = -1
```

它表示当前帧的几何上下文特征，不是显式点云坐标本身。

当前把 StreamVGGT patch tokens resize 到：

```text
context_grid = 12 x 12
```

所以每帧 geometry tokens 是：

```text
[T, 144, 2048]
```

其中：

```text
144 = 12 * 12
2048 = StreamVGGT aggregator token dim
```

### Camera tokens

camera tokens 来自 StreamVGGT camera head 的 pose encoding：

```text
[T, 1, 9]
```

它不是相机内参矩阵本身，而是 StreamVGGT 内部预测/编码出的相机位姿表示。当前把它作为全局相机上下文，参与 geometry-to-SAM 的融合。

## 融合方式

当前使用 `camera_guided` fusion。它的形式参考 SpaceMind 里“用相机/几何信息调制视觉 token”的思路，但不是直接复制 SpaceMind 代码。

当前实现是：

```text
query:
  SAM tokens -> [T, 5184, 256]

key / value:
  StreamVGGT geometry tokens -> [T, 144, 256]
  camera token              -> [T,   1, 256]
```

然后做 cross attention：

```text
SAM query attends to geometry/camera context
  -> fused tokens [T, 5184, 256]
```

## fused tokens 如何回到 SAM3 流程

可以简化理解为：

```text
SAM token + 3D/camera token
  -> cross attention
  -> fused token
  -> adapter
  -> 继续走 SAM3 tracker
```

但具体实现上，`fused token` 不是直接替换 SAM3 token，而是先生成 FPN residual：

```text
fused tokens [T, 5184, 256]
  -> reshape [T, 256, 72, 72]
  -> FPN adapter
  -> residual_fpn0 / residual_fpn1 / residual_fpn2
```

然后加回 SAM3 原始 FPN：

```text
fpn0 + residual_fpn0
fpn1 + residual_fpn1
fpn2 + residual_fpn2
```

再进入 SAM3 原始 tracker flow：

```text
memory
object score gate
mask decoder
```

## 训练与当前结论

冻结：

```text
SAM3 主体
StreamVGGT 主体
```

训练：

```text
projection
camera-guided fusion
SAM3 FPN adapter
```

当前主要监督：

```text
GT instance mask -> BCE + Dice
```

我们也尝试了 objectness loss：

```text
GT mask 非空 -> object_score_logits positive
GT mask 为空 -> object_score_logits negative
```

这个 loss 能快速打开 SAM3 的 object gate，但当前实验里：

```text
有 3D 和无 3D 都能快速过拟合
```

因此它更像诊断工具，说明 SAM3 的 failure point 在 objectness gate；但还不能证明 3D 信息真正起作用。

下一步讨论重点应该是：

```text
如何让 3D 信息更直接参与跨视角 instance correspondence，
而不是只通过 objectness gate 过拟合。
```
