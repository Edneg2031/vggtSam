# T0：SAM-Indexed Independent 3D Anchor Probe

## 问题

本实验只回答：冻结的 2D correspondence 与 QK camera rays 能否生成比相同像素 raw pointmap 更准确的独立 3D anchors，以及 SAM persistent ID 是否能提高这些 anchors 的可靠性。

## 候选生成

候选生成只读取：

- V0 cache 中第一 DPT level 的 frame-attention patch tokens；
- SAM persistent masks、scores、slots；
- 冻结 QK pose、intrinsics 和完整 causal QK ranking。

它不读取 raw pointmap、GT pointmap 或 GT pose。每个 current instance 先从 co-visible history 中按 QK ranking 取最多 8 帧，再用 cosine mutual-nearest-neighbor 建立 patch correspondence，最后用最多 4 个历史观测和 current ray 做线性多射线交会。

所有 validity gate 都在配置中预先固定：descriptor similarity/margin、正深度、ray angle、condition number 和按 patch size 缩放的 reprojection threshold。候选保存后才读取 raw/GT 评分。

## 分支

- `foreground_union`：历史搜索限制在所有目标前景，不区分 ID；
- `correct_persistent_id`：current slot 只匹配相同历史 slot；
- `shuffled_persistent_id`：使用固定循环错配的历史 slot；
- `shifted_mask_control`：历史 slot mask 做面积严格不变的固定空间平移。

四个分支对 correct slot 使用相同 current queries 和相同 causal history，避免 query/history 数量混淆身份效果。`wardrobe` 仅保留在正式 V0 semantic map，本诊断因其场景占比过大而预先排除。

## 评分

QK triangulated anchors 和 raw pointmap 分别通过其完整相机轨迹拟合一次固定 scoring-only Sim(3)；不会针对 instance、branch 或 subset 单独对齐。每个 anchor 与相同 current patch-center pixel 的 raw XYZ 和 GT XYZ 配对。

主指标：

- valid anchor rate；
- triangulated/raw GT RMSE、median、P90；
- improved anchor ratio；
- mean/median `raw_error - tri_error`；
- correct/shuffled/shifted 的非 GT quality 排序 equal-count 比较。

## 决策

- `GO`：correct ID anchors 优于 raw，improved ratio 大于 50%，且在 equal-count、valid rate、reprojection 三方面优于 shuffled/shifted；
- `PARTIAL_GO`：triangulation 优于 raw，但 SAM identity 独特贡献未成立；
- `NO_GO`：冻结 correspondence + QK camera 不能优于 raw pointmap。

本轮不修改 pose、pointmap 或 semantic map，不做 BA、ICP、surfel/Gaussian update，也不训练模型。

## 运行

```bash
zsh streaming_couping/commands_t0_sam_indexed_anchor_probe.txt
```
