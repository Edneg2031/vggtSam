# Geometry-Specific History Selection

## 目的

正式 V0 保持冻结：

- pose：第一帧 anchor + QK Top-4；
- pointmap：full-history；
- semantics：SAM3.1 persistent masks / IDs 投影到 pointmap。

本实验只验证一个假设：QK Top-4 更适合定位，而 point head 可能需要既有图像重叠、历史帧之间又有视角多样性的上下文。

## 四个对比分支

1. `full_history`：冻结 V0 pointmap，作为基线；
2. `qk_top4`：anchor + QK Top-4；
3. `recent_top4`：anchor + 最近 4 帧；
4. `geometry_diverse_top4`：anchor + 几何多样性 Top-4。

`geometry_diverse_top4` 先取 QK Top-12 作为固定 overlap pool。它首先选择 QK score 最高帧，之后逐帧 greedy 选择与已选集合平均相机中心 baseline 和平均相对旋转最大者。两项在每个 greedy step 内分别转成归一化 rank 后等权相加，因此不需要为 native translation 和旋转角度调 `lambda`。

## 实验隔离

- 不训练；
- 不使用 SAM；
- 不运行 camera head 或 depth head；
- 候选生成只读取 RGB、帧编号、冻结 QK pose 和 StreamVGGT 原生 QK score；
- GT 只在候选保存后用于 reference-frame Sim(3) 对齐和评分；
- 结果写入独立目录，不覆盖 V0。

## 运行

```bash
zsh streaming_couping/commands_geometry_history_selection.txt
```

## 判断

只有 `geometry_diverse_top4` 同时满足以下条件，才认为该方向在当前序列成立：

- paired RMSE 优于 `full_history`；
- sampled symmetric mean 优于 `full_history`；
- 少于一半非参考帧变差，并记录 improved-frame ratio；
- 与冻结 QK pose 的重投影 compatibility 不差于 `full_history`；
- 第一参考帧与 V0 pointmap 保持等价。

若通过，说明 camera head 和 point head 确实需要不同 history policy，下一步才测试 SAM co-visibility。若仍明显差于 full-history，则 V0 继续使用 full-history pointmap，后续转向 `2D correspondence → multi-view triangulation → independent sparse 3D anchors`，不继续扫描 history 参数。
