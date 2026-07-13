# Single-pair 可学习性诊断

## 目的

在增加新的 geometry fusion 前，验证下面这条最短链路是否能对固定目标过拟合：

```text
query RGB + query GT mask
          |
          v
frozen SAM3 reference memory
          |
target RGB + trainable FPN2 residual
          |
SAM3 memory attention + mask decoder
          |
target raw mask logits
          |
BCEWithLogits + Dice + presence
```

固定样本为 ScanNet++ scene `00a231a370`、instance `37`、query frame `133`、target frame `520`。两帧中目标都必须可见；query mask 只作为 reference prompt，target GT 只用于监督。

## 已确认的实现事实

- SAM3 encoder 和 StreamVGGT 在 feature extraction 中使用 `no_grad` 并缓存到 CPU，这是预期行为。
- trainable adapter forward 和 SAM3 source tracker forward 不在 `no_grad` 中。
- reference 使用 mask prompt 时，SAM3 会直接把 reference GT mask编码为 conditioning memory，因此 reference mask 输出不提供有效的 decoder 梯度。
- target frame 必须通过 memory attention、object score 和 mask decoder将梯度传回 FPN2 residual。
- mask loss直接接收 SAM3 raw logits，只调用一次 sigmoid（Dice 内部）或直接使用 `binary_cross_entropy_with_logits`。
- SAM3 将原图拉伸为 `1008x1008`；StreamVGGT `crop` 模式固定宽度 518、保持宽高比、把高度取为 14 的倍数并按需中心裁剪。二者不是同一图像变换。

## 三种基础模式

- `sam_only`：只使用 SAM3 FPN2，训练 2D residual adapter。
- `constant_prompt`：在 FPN2 residual adapter 内加入固定可学习 feature prompt，不加载 StreamVGGT。
- `random_geometry`：加载 StreamVGGT 仅用于估计 feature 均值和标准差，随后使用固定随机 feature。
- `real_geometry`：可选对照，不属于基础通过条件。

## 输出与判定

- `parameter_audit.csv`：参数名、shape、数量、`requires_grad`、optimizer group 和学习率。
- `module_diagnostics.csv`：每一步各模块 parameter/gradient/update norm 和 NaN/Inf。
- `tensor_audit.json`：tensor shape、dtype、范围、可导状态及三套坐标变换。
- `training_history.csv`：teacher-forced train IoU 与无 GT full-flow eval IoU。
- `visualizations/`：query prompt、target GT、teacher-forced mask、full-flow mask。

训练 IoU `0.95` 只是一项两帧可学习性诊断，不是 geometry 实验的准入条件。默认会完整运行所有基础模式并报告是否达到该值；只有设置 `STRICT_IOU_GATE=1` 时才会因基线未达到阈值而终止。若 train IoU 高但 eval IoU 低，说明 mask decoder 可学习但 object gate 或 memory inference 仍有问题；若 `residual_gradient_norm=0`，说明梯度在 SAM3 source flow 内被截断；若 residual gradient 非零而 parameter update 为零，则检查 residual head 和 optimizer。
