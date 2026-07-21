# Persistent instance token 引导的 StreamVGGT 位姿修复

## 研究问题

在不改写 SAM3 feature/memory、不微调 StreamVGGT 主干、也不允许每实例独立修改
相机的前提下，验证可靠 persistent instance 是否能够改善共享相机位姿。

该分支与最终 V3 并行存在：V3 是已经冻结的显式 pointmap/ray 后处理方法；这里是
一个新的可学习 CameraHead adapter。

## 为什么不重复旧 SAM token fusion

旧实验把 StreamVGGT layer-17 dense token 融入 SAM3 FPN2：aligned cross-view IoU
为 `0.9262`，zero 为 `0.8930`，但 shuffled 仍为 `0.9238`。它证明额外 residual
可以帮助 mask 拟合，却没有证明正确逐帧几何对应是原因；分别训练的固定 shuffled
分支还可能记住错误排列。

因此新分支遵循：

- 不把任何 VGGT/fused token 写入 SAM3；
- SAM3 只提供 recovered mask、persistent ID、score 和 frozen mask-pooled FPN2；
- 主要信号是显式 current/history instance geometry 与二者残差；
- zero/shuffle 控制全部在同一个 aligned checkpoint 推理时执行。

## 模型

每帧、每实例 observation 包含：

```text
SAM3 FPN2 masked mean/std
normalized point center
covariance eigenvalues and robust extent
current-to-map ICP translation / fitness / RMSE
point confidence and mask area
multi-instance consensus residual
tracker / geometry / deterministic static confidence
```

因果 memory 在处理当前帧前保留上一状态：

```text
token_t = MLP(current_t, memory_{t-1}, current_t - memory_{t-1})
camera_t queries valid instance tokens
memory_t = EMA(memory_{t-1}, current_t) only when trusted
```

reference 帧只初始化 memory，不产生有效 token。没有历史、点数不足、tracker/geometry
不可靠或 static score 不足时，不修改当前 camera token。

当前真正进入 CameraHead 的 hidden token 为最终 aggregator token 0，维度 2048；不是
已有 adapter 暴露的 9D pose encoding。主分支为：

```text
camera_hidden = camera_hidden
              + sigmoid(gate) * zero_proj(cross_attn(camera_hidden, instances))
```

`zero_proj` 无 bias且权重全零，因此初始化逐元素等于 baseline；无有效实例的帧在
训练后也保持严格不变。CameraHead 参数冻结，
但 forward 不处于 `no_grad`，pose loss仍能反传到 adapter。

现有 recovery 配置启用了 StreamVGGT streaming cache，因此 cache/train/eval 均逐帧
重放 CameraHead KV-cache路径，而不是把七帧 token 改成一次非缓存解码。缓存保存
CameraHead真正读取的 final aggregator camera token，并用
`cached_raw_pose + frozen_head(refined_token) - frozen_head(raw_token)`锚定原始输出；因此
即使独立重放 frozen head存在实现差异，zero-init/module-off仍严格恢复原始 pose，同时
保留对 adapter 的梯度。四层 DPT token和 StreamVGGT image tensor同样以 FP32缓存，
`all_token_fusion`还逐元素检查原始 depth与pointmap。

## 完整消融

| mode | 输入/作用 | 判定问题 |
|---|---|---|
| `baseline` | 原始 CameraHead | raw StreamVGGT |
| `camera_geometry_only` | 结构化 geometry，无 SAM appearance | 显式实例几何是否有效 |
| `camera_sam_only` | SAM appearance + tracker gate | 提升是否只是外观/容量 |
| `camera_token_fusion` | geometry + SAM appearance | 推荐主方法 |
| `all_token_fusion` | DPT 4/11/17/23 全 token查询实例 | 广泛融合是否污染深度/pointmap |

每个训练 checkpoint 都执行：

```text
aligned
module_off
zero_appearance
zero_geometry
shuffle_instance_ids after reference
shuffle_time after reference
```

这些是 inference-time perturbation，不会为 shuffled/zero 重新训练模型。主方法只有在
aligned 优于 module-off/zero/shuffle 时，才能归因于正确的 persistent correspondence。
其中 `zero_geometry`沿用 aligned 分支的可信实例 mask，只清空 token里的结构化几何与
geometry/static score输入，因此不会退化成另一个 module-off，也不会放进原本被拒绝的
实例。

## 损失

Camera-only 三组：

```text
L = 20 Lcamera
  + 2 Lrelative_rotation
  + 2 Ltranslation_direction
  + 1 Ltrimmed_rigid
  + 0.25 Lmatched_centroid
  + 1e-4 Lcamera_residual
```

刚体损失使用 frozen predicted depth 和 mask pixel构建 camera-local points，再通过
refined pose 转到公共坐标；depth、mask、nearest-neighbor selection和置信度全部
detach。只比较相邻可信 observation，并用对称 trimmed Chamfer降低局部可见表面
差异的影响；评估同时记录归一化值与 fixed-reference Sim(3) 下的米制值。

`all_token_fusion` 额外训练 fixed-reference Sim(3) pointmap loss和 scale-invariant
depth loss。Camera-only 中 depth/pointmap只记录，不伪装成有梯度的监督。

GT relative pose translation会除以 reference pointmap Sim(3) 的 native-to-metric scale，
与 StreamVGGT depth/pose 的原生尺度一致；否则直接使用米制 translation会破坏最终固定
reference alignment。

## 数据与结论边界

当前配置中的 7 帧 bed/cabinet/wardrobe 序列只是压力测试和代码可学习性检查。如果
只有该 clip，日志会明确警告评估发生在训练数据上，不能声称泛化。

正式结论前，应在 `dataset.clips` 增加：

- 多个 `split: train` 场景；
- 至少一个不同场景 `split: val`；
- 最终不同场景 `split: test`。

reference 后的 GT mask从不进入 observation/gate/memory；GT pose、pointmap和由其生成
的 depth只进入训练监督及评估。

## 当前单 clip 压力测试记录

以下结果来自同一个 `00a231a370_90_240_37_68_54` clip上的训练后诊断，不能作为
held-out泛化结论：

| mode | ATE RMSE | adjacent RPE translation | all-pairs translation direction | joint@5deg |
|---|---:|---:|---:|---:|
| baseline | 0.2280 | 0.1522 | 14.56° | 19.0% |
| camera geometry only | 0.1529 | 0.1023 | 3.60° | 81.0% |
| camera SAM only | **0.1488** | **0.0744** | **1.34°** | **100%** |
| camera geometry + SAM | 0.1668 | 0.1081 | 4.62° | 61.9% |
| all token fusion | 0.1630 | 0.1104 | 4.46° | 61.9% |

当前最佳是 `camera_sam_only`，不是 geometry+SAM，也不是 all-token。same-checkpoint
因果控制中，SAM-only的 all-pairs translation direction由 aligned `1.34°`退化到
shuffle-ID `3.98°`、shuffle-time `15.20°`；geometry-only由 `3.60°`退化到
`11.66°/12.09°`。这与旧 VGGT token写入SAM实验中 aligned与 shuffled近乎相同的
结果不同。

实例刚体诊断同样支持这一点。相对 module-off，SAM-only将米制 instance Chamfer从
`0.07639 m`降至`0.06403 m`（16.2%），matched centroid从`0.03032 m`降至
`0.01548 m`（49.0%）；shuffle-time后分别恶化到`0.09137 m/0.04811 m`。
geometry-only的对应改善为10.8%/37.3%。

Full point-head评估使用所有有限GT对应像素时，all-token aligned将 mean paired distance
从baseline `0.11606 m`降至`0.06805 m`（41.4%），RMSE从`0.15032 m`降至
`0.11716 m`（22.1%），P90从`0.16583 m`降至`0.09995 m`（39.7%）。同一checkpoint
的zero-appearance仅改善6.8%/4.5%，shuffle-ID与shuffle-time相对aligned分别使mean
误差恶化26.6%和44.2%。Depth的median-scaled AbsRel由`0.03773`降至`0.02295`
（39.2%），scale-invariant log RMSE由`0.06254`降至`0.04338`（30.6%）。因此
all-token确实改善pointmap/depth形状，而不只是改变相机位姿；appearance是主要贡献，
geometry提供较小但可测的额外改善。

所有 checkpoint的 module-off严格回退检查均通过：camera token、pose、depth和pointmap
的最大绝对差均为`0.0`，`strict_equal=1`。当前 gate约为`0.500`，说明有效更新主要由
从零开始学习的 projection承担，sigmoid gate本身几乎没有离开初始化值。geometry相关
模式只有4/7帧 active，而SAM-only有6/7帧 active；这也是联合模式弱于SAM-only时需要
与“特征负干扰”分开考虑的覆盖率因素。

## 运行

从仓库根目录运行一次：

```bash
zsh streaming_couping/commands_instance_token_pose.txt
```

流程顺序为 frozen feature cache、四组训练、同 checkpoint perturbation评估。已有
tracking cache会复用，缓存阶段不会重复运行 SAM3 tracker。

优先回传：

```text
evaluation/pose_summary.csv
evaluation/pose_pair_summary.csv
evaluation/pose_rpe.csv
evaluation/instance_diagnostics.csv
evaluation/baseline_equivalence.csv
evaluation/pointmap_summary.csv
evaluation/depth_summary.csv
完整 log
```

`baseline_equivalence.csv`中所有 `strict_equal` 必须为 `1`。如果 feature cache已经完整，
修复代码后可以直接复用，不需要重新运行 SAM3 pooling。

## 点云与深度评估

`--stage eval`还会生成：

```text
evaluation/pointmap_frame_metrics.csv
evaluation/pointmap_summary.csv
evaluation/depth_frame_metrics.csv
evaluation/depth_summary.csv
```

`pointmap_summary.csv`区分两种几何来源：

- `point_head`：直接评估 StreamVGGT DPT point head输出；camera-only模式应与 baseline
  完全相同，只有 `all_token_fusion`会修改它；
- `baseline_point_head_refined_pose`：先用 baseline pose将 frozen baseline point-head世界点
  还原到每帧相机局部坐标，再用 refined pose放回世界坐标，只衡量位姿修复对完整点云
  的影响，不混入另一个尺度的depth head。

所有模式严格复用 cache中由 baseline reference frame拟合的一次 Sim(3)，不会用 refined
结果重新拟合。评估mask是所有有限GT对应点，不使用预测置信度，因此点数会多于旧的
`confidence >= 0.3` pointmap报告。StreamVGGT depth head与point head并不共享绝对尺度；
`depth_summary.csv`因此使用 baseline reference depth的独立 median scale，并同时记录
米制 MAE/RMSE/AbsRel、每帧 median-scale AbsRel与scale-invariant log RMSE。

`instance_diagnostics.csv`中的 instance Chamfer是 frozen depth经过 refined pose后的跨帧
刚体自一致性，不是 pointmap对GT的精度；它与上述 full-scene paired pointmap指标不能
互相替代。
