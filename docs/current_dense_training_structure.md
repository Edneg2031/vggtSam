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
        BCE + Dice + objectness loss
```

## 关键理解

可以概括成：

```text
SAM token + StreamVGGT 3D token
  -> cross attention fusion
  -> fused token
  -> adapter
  -> 回到 SAM3 原 tracker 流程
```

但实现上不是把 `fused token` 直接替换 SAM3 token，而是：

```text
fused token 生成 residual，
residual 加到 SAM3 原始 FPN 上，
然后继续走 SAM3 memory / object gate / mask decoder。
```

`residual` 的作用就是给 SAM3 原始特征加一个 3D-aware 修正量：

```text
SAM3 原始特征 + 3D 修正量 -> 更容易判断目标是否重新出现
```

## 训练方式

冻结：

```text
SAM3 主体
StreamVGGT 主体
```

训练：

```text
SAM/3D projection
camera-guided fusion
SAM3 FPN adapter
```

监督：

```text
1. mask BCE loss
2. mask Dice loss
3. objectness loss
```

其中 objectness loss 用来直接训练 SAM3 的目标存在判断：

```text
GT mask 非空 -> object_score_logits = positive
GT mask 为空 -> object_score_logits = negative
```

## 当前实验判断标准

做两组消融：

```text
geometry_ablation=none  使用 StreamVGGT 3D + camera
geometry_ablation=zero  不使用 3D，只保留同样 adapter 结构
```

如果 3D 版本更早做到：

```text
sam3_source_present_frames = GT 可见帧数量
sam3_source_iou > 0.9
object_score_logits > 0
```

说明 StreamVGGT 3D 信息确实帮助 SAM3 重新发现目标。

如果两组差不多，说明当前提升主要来自 adapter 过拟合，而不是 3D 信息。
