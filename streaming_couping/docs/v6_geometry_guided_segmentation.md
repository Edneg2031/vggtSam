# V6：轻量几何辅助 SAM3.1 分割

V6 当前只研究一个问题：几何模型给出的粗空间线索，能否纠正 SAM3.1 的跨帧漏检和错检。
它不训练 camera/pointmap head，也不把 StreamVGGT hidden token 接入 SAM3.1。V4/V5 和旧
V6 camera-token 消融均不修改。

## 1. 模块边界

分割核心只接受每帧三张与输出图像同尺寸的布尔 mask：

```text
GeometrySegmentationPrompt
├── box_mask       粗定位范围
├── positive_mask  确认属于目标的稀疏像素
└── negative_mask  确认不属于目标的稀疏像素，可为空
```

当前用一个独立 adapter 将 StreamVGGT 输出转为这三张 mask：

```text
参考帧实例 mask
  → 在参考帧 StreamVGGT pointmap 中取可信 3D 点
  → 用后续帧 pose + intrinsics 投影
  → 用后续帧 pointmap 做距离一致性检查
  → box / positive / negative 三张二维提示
```

StreamVGGT adapter 位于 `src/streamvggt_geometry_prompt.py`；SAM3.1 细化和选择策略位于
`src/v6_geometry_segmentation.py`。后续替换几何模型时，只需要生成相同的三张 mask，
不需要修改 SAM3.1 或自适应选择代码。

这里没有 ICP、当前候选 mask 的 3D 注册、多帧 object-map 更新或几何模型专属 hidden
feature。参考帧点只建立一次，避免错误预测被写回几何记忆。

## 2. SAM3.1 如何使用几何

SAM3.1 的交互协议不允许在同一次调用里混用 text/box 和 point，因此每个提示分支分两步：

1. 用 `text + box_mask` 找到候选实例；
2. 在同一 session 内对该实例加入 positive points，可选加入 negative points。

候选排序不使用 GT：

```text
geometry score =
0.55 × positive-support recall
+ 0.30 × candidate-inside-box precision
+ 0.15 × SAM3.1 score
```

较高的 box precision 权重用于抑制“覆盖了几何点、但 mask 扩张到大面积相似物体”的候选。

## 3. 保守自适应纠错

几何提示不是无条件替换 raw SAM3.1。只有 raw mask 满足以下至少一项才进入纠错判断：

- raw mask 为空；
- raw 对 positive support 的覆盖过低；
- raw mask 大部分落在 geometry box 外。

prompted mask 还必须同时满足：

- 面积相对 raw 不过小或过大；
- positive-support recall 至少提高一个 margin；
- 总 geometry score 至少提高一个 margin。

否则逐帧保留 raw SAM3.1。几何提示缺失或被当前 adapter 拒绝时也直接回退 raw。

## 4. 一次运行的六个对照

| variant | 含义 |
|---|---|
| `raw_sam31` | SAM3.1 标准 tracking |
| `current_sam3_late_geometry` | 旧 SAM3 + 后验几何门控，仅作历史对照 |
| `v6_sam31_points_positive` | 每帧都采用 box + 正点细化，失败则回退 raw |
| `v6_sam31_points_posneg` | 每帧都采用 box + 正负点细化，失败则回退 raw |
| `v6_sam31_adaptive_positive` | 仅在 raw 明显不可靠且正点候选改善时替换 |
| `v6_sam31_adaptive_posneg` | 仅在 raw 明显不可靠且正负点候选改善时替换 |

前两个 point variant 用于判断提示本身有没有能力；两个 adaptive variant 才是部署策略候选。
实验命令为了同时完成消融，会分别运行 positive 与 posneg session。最终确定一种提示后，部署
只需运行选中的一个分支。

## 5. 运行与输出

```bash
zsh streaming_couping/commands_v6_geometry_segmentation.txt
```

命令会复用已有 StreamVGGT feature cache，不训练模型。输出：

```text
outputs/streaming_couping_v6_geometry_segmentation/
├── v6_segmentation_summary.csv
├── v6_segmentation_frames.csv
├── v6_geometry_prompt_diagnostics.csv
├── masks/<clip>.pt
└── visualizations/<clip>/<frame>.png
```

短表主要看：

- `mean_iou_delta_from_raw` 是否在 train、validation、test 都非负；
- `improved_frames` 是否多于 `worse_frames`；
- recall 是否改善且没有引入更多 absent-frame false positive；
- adaptive 是否保住 raw 本来正确的帧。

逐帧 diagnostics 会记录当前 StreamVGGT adapter 的接受状态、三张提示的像素数，以及
adaptive 最终采用或拒绝 prompted mask 的原因。GT 只用于离线 CSV 指标和可视化，不参与
prompt、候选排序或回退决策。

## 6. 当前实验依据

上一轮无条件 point prompt 的结果已经表明该接口具备作用能力：validation 50–80 的
`points_3d` IoU 从 raw SAM3.1 的 `0.6560` 提高到 `0.8657`；但 train 和 cross-clip
没有稳定提高。box-only 在三个 split 都明显降低 recall。

因此本轮删除 box-only 输出和 3D registration gate，保留 point prompt，并用保守
adaptive policy 解决“部分序列有效、部分序列退化”的问题。是否成为最终 V6，必须由三个
split 的新结果共同决定。
