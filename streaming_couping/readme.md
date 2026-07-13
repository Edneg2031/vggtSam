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
                    SAM3 geometry-prompt re-segmentation
                                  |
                                  v
                         bridge final instance mask
```

只有 reference frame 的 GT mask 参与初始化；其他帧 GT 只用于评估。几何投影不会直接作为最终 mask，而是作为 SAM3 的当前帧提示。
这里的当前帧 pointmap 是 frozen StreamVGGT 从当前 RGB 预测的输出，不是预处理数据中的 GT pointmap。历史点与当前点的距离检查完全在 StreamVGGT 的共享重建坐标系中完成。
默认不会把 SAM3 输出硬裁剪到候选框内，因为 reference frame 可能只观察到物体的一部分。CSV 同时记录原始 refinement 与裁剪版本的 IoU，供消融比较。
reference frame 用于初始化，不会为自己生成 geometry candidate。

## 实现对应

- SAM3 原始视频追踪：`src/backbones/sam3_wrapper.py`
- StreamVGGT 流式几何：`src/backbones/streamvggt_wrapper.py`
- reference 物体点缓存：`src/aggregation/point_map_fusion.py`
- 重访候选框生成：`src/aggregation/mine_revisit_segments.py`
- fallback 门控：`src/bridge/gating.py`
- 三组控制、指标与可视化：`src/pipeline.py`

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

## SAM3 提示消融

候选通过几何门控后，可以用三种方式提示当前帧 SAM3：

- `box`：文本 + 几何候选框，当前默认基线。
- `point`：只用几何支持区域中的 3 个正点，绕开候选框边界。
- `box_point`：先用文本 + 框选择实例，再用几何正点细化同一实例。

三者都只使用 StreamVGGT 支持点，不读取当前帧 GT。可通过
`--fallback-prompt-mode` 选择；完整命令见 `commands.txt`。

## Memory 写回消融

几何定位有效后，按三个阶段比较：

- `B0 / sam3_*`：原始 SAM3 视频追踪。
- `B1 / bridge_*`：SAM3 丢失时，用几何提示做当前帧无状态恢复；不更新原会话。
- `B2 / memory_*`：只追踪到 B1 的第一次有效恢复，用同一 `obj_id` 写回同一个 SAM3 session；该交互路径运行 SAM3 memory encoder，再继续处理未来帧。后续帧只由 SAM3 memory 追踪，不再使用几何 fallback，也不会提前读取未来帧状态。

运行时加 `--memory-writeback`。判断 memory 是否有效，应重点查看
`sam3_post_recovery_iou`、`bridge_post_recovery_iou` 和
`memory_post_recovery_iou`。这些指标只统计恢复帧之后目标真实可见的帧，
不会把恢复帧本身计入收益。

## 运行

```bash
PYTHONPATH=src:. python -m streaming_couping.scripts.run_bridge \
  --config streaming_couping/configs/default.yaml
```

旧版预处理 manifest 的 RGB 仍指向原始 NAS。若原图权限不可读，先用
`scripts/cache_scannetpp_rgb.py` 生成 processed RGB cache 和新 manifest，
再通过 `--manifest` 传入。`--allow-summary-fallback` 只适合调试：它从
已有 summary 的左侧 RGB 面板恢复图像，包含顶部标题条，不能作为最终定量输入。

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
- `<mode>/memory_tracking_report.png`：`B2` 写回后同一实例 ID 的 SAM3 memory 追踪结果。

## 当前边界

- SAM3 和 StreamVGGT 均冻结，没有训练 adapter。
- 默认 fallback 是当前帧无状态重分割；只有显式传入 `--memory-writeback` 才运行一次同会话 memory 写回。
- 当前只验证 `geometry -> tracking`。点云聚合、相机回环优化、双向联合训练均未宣称完成。
