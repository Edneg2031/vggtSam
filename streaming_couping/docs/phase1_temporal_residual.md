# Phase 1：Trust-Aware Pointmap Residual（单场景开发实验）

## 目的

当前挂载盘不可用，暂时只有正式 V0 的一个 30 帧 clip。因此先做固定 temporal split，验证训练管线和
“局部 residual 是否有学习信号”，不声明跨场景泛化。

正式 V0 始终保持：

```text
Pose     = first-frame anchor + QK Top-4
Pointmap = full-history raw StreamVGGT pointmap
Semantic = SAM persistent mask / ID
```

本实验不覆盖任何 V0 artifact。

## 固定 protocol

```text
train:  sequence index  0–17（18帧）
val:    sequence index 18–23（ 6帧）
test:   sequence index 24–29（ 6帧）
```

四个对照：

```text
A. raw_full_history
B. residual_no_gate
C. gated_residual
D. gated_residual_uncertainty
```

模型读取冻结 V0 cache 中 PointHead 实际使用的四层完整 DPT patch token（layers 4/11/17/23，
frame half + global-history half），不重新运行 StreamVGGT。外接小头输出：

```text
geometry feature → Residual Head → delta
                 → Gate Head     → gate
                 → UQ Head       → log variance

X_new_native = X_raw_native + gate * delta_native
```

StreamVGGT backbone、原 PointHead、CameraHead 全部冻结；第一阶段不读取 SAM feature。Residual 最后一层
零初始化，gate bias 固定为 `-2`，所以优化前输出严格等于 raw pointmap。

## 坐标与 GT 使用

V0 cache 的 raw pointmap 位于 StreamVGGT native gauge，GT 位于 metric world gauge。训练时用 cache 已有的
reference-frame Sim(3) 将 GT 逆变换到 native gauge，再监督 residual；评价时用同一个固定 Sim(3) 将候选
变回 metric gauge。这样网络学习局部几何误差，不会把全局坐标对齐误当成 residual。

GT 只用于 train label、validation 监控和最终 test 评价。Epoch 数和所有阈值在运行前固定，test 的 6 帧
不参与参数更新或模型选择。

## 运行

要求下列两个只读 artifact 已存在：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_v0/cache/00a231a370_90_525_step15_37_68_54.pt
/data184/open_source/vggtSam/outputs/streaming_couping_v0/qk_pose_retrieval/qk_pose_output.pt
```

运行：

```bash
zsh streaming_couping/commands_temporal_residual_training.txt
```

默认只用物理 GPU 1。可改为：

```bash
TEMPORAL_RESIDUAL_GPU=2 zsh streaming_couping/commands_temporal_residual_training.txt
```

输出目录：

```text
/data184/open_source/vggtSam/outputs/streaming_couping_phase1_temporal_residual
```

其中包括 `summary.json`、`branch_summary.csv`、`frame_metrics.csv`、`training_curve.csv`、
`risk_coverage.csv`、`models.pt` 和可直接复制的 `copyable_result.txt`。

## 判断边界

只有 D 分支在 test 上同时满足 RMSE 降低、P90 不恶化、至少 4/6 帧改善，且 uncertainty-error Spearman
不低于 0.20、低 uncertainty 的 50% coverage 风险更低，才记为 temporal development `GO`。

即使得到 `GO`，下一步也只能是在存储恢复后用完全冻结的 protocol 做 held-out scene 验证；不能据此加入
SAM 或宣称跨场景 pointmap 已提升。若为 `NO_GO`，先检查 residual 学习信号，不继续堆 SAM、ICP、BA、
triangulation 或 Gaussian。
