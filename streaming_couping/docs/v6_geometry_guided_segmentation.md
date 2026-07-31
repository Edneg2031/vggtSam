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

SAM3.1 的交互协议不允许在同一次调用里混用 text/box 和 point，因此当前正点分支分两步：

1. 用 `text + box_mask` 找到候选实例；
2. 在同一 session 内对该实例加入 positive points。

第一轮结果中负点在 test 与正点相同、在 train 更差，而且依赖“未被当前点图支持的投影点
一定是背景”这一不可靠假设。因此本轮不运行负点 session；接口仍保留可选 negative mask，
但不增加推理开销。

候选排序不使用 GT：

```text
geometry score =
0.55 × positive-support recall
+ 0.30 × candidate-inside-box precision
+ 0.15 × SAM3.1 score
```

较高的 box precision 权重用于抑制“覆盖了几何点、但 mask 扩张到大面积相似物体”的候选。

## 3. 保守自适应纠错

`safe` 门控中，只有 raw mask 满足以下至少一项才进入纠错判断：

- raw mask 为空；
- raw 对 positive support 的覆盖过低；
- raw mask 大部分落在 geometry box 外。

prompted mask 还必须同时满足：

- 面积相对 raw 不过小或过大；
- positive-support recall 至少提高一个 margin；
- 总 geometry score 至少提高一个 margin。

否则逐帧保留 raw SAM3.1。几何提示缺失或被当前 adapter 拒绝时也直接回退 raw。

为了检验该门控是否过于保守，同一个 positive candidate 还免费派生三档 competitive
门控。当 raw 本来看起来可靠时，candidate 的 support gain 和 geometry-score gain 必须
同时超过 `0.05 / 0.10 / 0.15` 才能替换。它们不使用 GT，也不重新运行 SAM3.1。

## 4. 一次运行的七个对照

| variant | 含义 |
|---|---|
| `raw_sam31` | SAM3.1 标准 tracking |
| `current_sam3_late_geometry` | 旧 SAM3 + 后验几何门控，仅作历史对照 |
| `v6_sam31_points_positive` | 每帧都采用 box + 正点细化，失败则回退 raw |
| `v6_sam31_adaptive_positive_safe` | 仅在 raw 明显不可靠且候选改善时替换 |
| `v6_sam31_adaptive_positive_compete_005` | reliable raw 也允许被领先 0.05 的候选替换 |
| `v6_sam31_adaptive_positive_compete_010` | reliable raw 需要被候选领先 0.10 |
| `v6_sam31_adaptive_positive_compete_015` | reliable raw 需要被候选领先 0.15 |

`points_positive` 只表示提示能力上限；四个 adaptive variant 才是部署策略候选。七个
variant 中只有 raw tracking 和一份 positive candidate 需要模型前向，其余门控都是张量比较。

## 5. 运行与输出

```bash
zsh streaming_couping/commands_v6_geometry_segmentation.txt
```

命令会复用已有 StreamVGGT feature cache，不训练模型。输出：

```text
outputs/streaming_couping_v6_geometry_segmentation/
├── v6_segmentation_summary.csv
├── v6_policy_selection.csv
├── v6_segmentation_frames.csv
├── v6_geometry_prompt_diagnostics.csv
├── masks/<clip>.pt
└── visualizations/<clip>/<frame>.png
```

`v6_policy_selection.csv` 是默认上传短表。`development_best` 只使用 train + validation：

- 先要求 `development_worse_frames == 0`；
- 再最大化按 instance-frame 加权的 `development_weighted_delta`；
- test 指标标为 `report_only`，不参与策略选择。

逐帧 diagnostics 会记录当前 StreamVGGT adapter 的接受状态、三张提示的像素数，以及
adaptive 最终采用或拒绝 prompted mask 的原因。GT 只用于离线 CSV 指标和可视化，不参与
prompt、候选排序或回退决策。

## 6. 当前实验依据

上一轮 positive point prompt 在 test 将可见帧 IoU 从 `0.3665` 提高到 `0.6301`，
validation 从 `0.6560` 提高到 `0.8076`，但 train 从 `0.6567` 降到 `0.4973`。当前 safe
门控在 train 和 validation 分别提高 `0.0213` 与 `0.0995`，且三个 split 都是零
`worse_frames`；但它在 test 完全没有触发。

因此本轮只回答一个问题：在不破坏 development safety 的前提下，competitive margin 能否
释放一部分被 safe 门控挡住的候选。test 已经看过，仍只作报告，不能用它反向选择 margin。
