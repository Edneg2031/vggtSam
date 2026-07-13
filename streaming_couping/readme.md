# Streaming Coupling Baseline

这是一个用于验证 **StreamVGGT 几何是否能帮助 SAM3 在大视角重访时找回实例** 的最小实验，不是完整训练框架。

## 数据流

```text
RGB 序列 + reference frame GT instance mask
        |                         |
        v                         v
Frozen SAM3 video tracker   Frozen StreamVGGT (causal cache)
原始跨帧 mask              world pointmap + camera + confidence
        |                         |
        |                    reference mask 内点
        |                         v
        |                  object world-point map
        |                         |
        |             投影到当前帧 + 当前 pointmap 支持检查
        |                         v
        |                 coarse geometry box
        |                         |
        +---- tracker 丢失/低分时---+
                                  v
                    SAM3 text + geometry-box re-segmentation
                                  |
                                  v
                         bridge final instance mask
```

只有 reference frame 的 GT mask 参与初始化；其他帧 GT 只用于评估。几何投影不会直接作为最终 mask，而是作为 SAM3 的候选框。

## 三组控制

- `zero`：不使用 StreamVGGT，结果就是原始 SAM3。
- `aligned`：使用与当前 RGB 帧对齐的 StreamVGGT 输出。
- `shuffled`：保留 reference 几何，打乱其余帧几何。它用于检查提升是否真的依赖时序对齐。

有效的几何贡献至少应满足：

```text
aligned > zero
aligned > shuffled
```

仅有 `aligned > zero` 不够，因为错误几何也可能提供一个宽松的 box。

## 运行

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_bridge \
  --config streaming_couping/configs/default.yaml
```

覆盖实验对象：

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_bridge \
  --config streaming_couping/configs/default.yaml \
  --scene-id 00a231a370 \
  --instance-id 37 \
  --frame-indices 133 162 520 566 477 \
  --sam3-device cuda:3 \
  --geometry-device cuda:1 \
  --output-dir outputs/streaming_couping_inst37
```

## 输出

- `summary.csv`：SAM3 与 bridge 的跨视角 IoU、召回率和 absent false-positive ratio。
- `<mode>/frame_metrics.csv`：每帧投影点数、3D 支持率、candidate/fallback/final IoU 和门控原因。
- `<mode>/tracking_report.png`：`RGB | GT | SAM3 original | geometry candidate | bridge final`。候选图中黄色是原始投影点，绿色是当前 pointmap 支持的投影点，矩形是送给 SAM3 的候选框。

## 当前边界

- SAM3 和 StreamVGGT 均冻结，没有训练 adapter。
- fallback 是当前帧的 SAM3 文本 + 几何框重新分割，不会伪装成原视频 memory 的连续更新。
- 当前只验证 `geometry -> tracking`。点云聚合、相机回环优化、双向联合训练均未宣称完成。
