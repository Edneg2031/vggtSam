# Geometry-Specific History Selection

## 目的

正式 V0 保持冻结：

- pose：第一帧 anchor + QK Top-4；
- pointmap：full-history；
- semantics：SAM3.1 persistent masks / IDs 投影到 pointmap。

本实验只验证一个假设：QK Top-4 更适合定位，而 point head 可能需要既有图像重叠、又有视角互补的历史帧。

## 三个对比分支

1. `full_history`：冻结 V0 pointmap，作为下限基线；
2. `qk_top4`：anchor + QK Top-4，复现已知 pointmap 下降分支；
3. `geometry_top4`：anchor + 几何历史 Top-4。

`geometry_top4` 先取 QK Top-8 作为 overlap pool，再使用冻结 QK pose 计算相机中心 baseline 和相对旋转角；两项分别转换为归一化 rank，等权排序后选择 4 帧。这里不用 metric threshold，也不调参。

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

只有 `geometry_top4` 同时满足以下条件，才认为该方向在当前序列成立：

- paired RMSE 优于 `full_history`；
- sampled symmetric mean 优于 `full_history`；
- 少于一半非参考帧变差；
- 与冻结 QK pose 的重投影 compatibility 不差于 `full_history`；
- 第一参考帧与 V0 pointmap 保持等价。

若通过，下一步只测试 SAM co-visibility 是否能改善 overlap pool；若不通过，V0 继续使用 full-history pointmap，不再增加复杂度。
