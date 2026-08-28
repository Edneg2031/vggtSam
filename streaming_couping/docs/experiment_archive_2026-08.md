# 2026-08 实验归档：SAM3.1—StreamVGGT 闭环路线

本记录保存本轮已经实际运行的六项诊断实验及其结论。实验输出目录位于
`/data184/open_source/vggtSam/outputs/`；本文件只记录结果，不复制大体积模型、cache
或 GT 文件。

## 归档与基线

- 实验状态备份：Git `3a4ceea`，本地分支 `archive/experiments-before-v0-restore`。
- 回滚后的代码基线：Git `9852343`，精确 V0 baseline snapshot。
- 回滚不删除服务器输出，也不将任何实验分支写回 V0。
- `candidate_generation_gt_fields=0` 的诊断均在候选冻结后才打开 GT；实际 SAM
  重分割实验另行标注。

## 实验结果

### 1. SAM3.1 class-agnostic auto-proposal

命令：`commands_run_sam31_auto_proposal.txt`。4 个 clip，每个冻结 16 条视觉点
track；StreamVGGT 不重跑，SAM 用 RGB + 固定视觉点网格生成候选。

- 历史 prompt scope：`F5cm=0.12607`、`voxelIoU@5cm=0.03728`。
- auto visual points：`F5cm=0.03167`、`voxelIoU@5cm=0.00912`，frame IDF1
  相对 prompt 下降 `0.17396`，4 个 clip 没有一个改善。
- all-instance scope 的 auto recall 诊断略有改善，但不是 GO gate，且 generic
  object identity 没有解决类别识别。

结论：盲目全图 proposal 不足以替代预先给定 prompt；该分支不推进。

### 2. V0 explicit bidirectional feedback

命令：`commands_run_v0_bidirectional_feedback.txt`。单个 V0 场景、30 帧、16 slots，
复用冻结 cache，未将 temporal points 真正反馈给 SAM。

| variant | tracking IoU | frame IDF1 | voxelIoU@5cm | F5cm |
|---|---:|---:|---:|---:|
| raw V0 | 0.41345 | 0.50510 | 0.09964 | 0.34401 |
| depth refined | 0.41370 | 0.50510 | 0.10014 | 0.34447 |
| GT-mask oracle | 0.81773 | 0.89973 | 0.18513 | 0.50325 |

Depth refinement 只删除 `256/772643` 个像素，比例 `0.000331`；temporal projection
的 GT-mask hit 为 raw pose `0.4565`、selected pose `0.5111`。提升不足以证明闭环有效。

结论：oracle gap 确实存在，但当前两个反馈模块没有形成有效收益；仅作 diagnostic。

### 3. Temporal prompt A–E geometry matrix

命令：`commands_run_v0_temporal_prompt_matrix.txt`。单场景、30 帧、325 queries，
不重跑 SAM，raw StreamVGGT pose，候选冻结后评估 precision/coverage。

| branch | precision | coverage | coverage F1 |
|---|---:|---:|---:|
| A center | 0.4343 | 0.4216 | 0.4279 |
| B surface-3 | 0.4063 | 0.5980 | 0.4839 |
| C surface-5 | 0.4021 | 0.6373 | 0.4931 |
| E depth gate 0.15 | 0.4891 | 0.5784 | 0.5300 |
| E depth gate 0.20 | 0.4507 | 0.6078 | 0.5176 |

最佳 F1 仍只有 `0.5300`，没有达到高 precision 与高 coverage 同时满足的条件，
因此才进行下一步小规模真实 SAM A/B，而不是直接部署 prompt。

### 4. Real SAM temporal prompt A/B

命令：`commands_run_v0_temporal_prompt_sam_ab.txt`。单场景、96 个 query events、
实际 SAM 单帧 points-only resegmentation；pose、pointmap、slot assignment 和未查询
帧均冻结，没有完整 temporal propagation。

| variant | mean IoU | frame IDF1 | voxelIoU@5cm | F5cm | worsened-frame ratio |
|---|---:|---:|---:|---:|---:|
| raw V0 | 0.41345 | 0.50510 | 0.09964 | 0.34401 | 0.000 |
| A center | 0.35981 | 0.42025 | 0.06911 | 0.23774 | 0.533 |
| C surface-5 | 0.37241 | 0.43284 | 0.05147 | 0.18872 | 0.433 |
| E depth gate 0.15 | 0.38218 | 0.45113 | 0.05204 | 0.19121 | 0.333 |
| E depth gate 0.20 | 0.37619 | 0.44000 | 0.05176 | 0.19134 | 0.400 |

Prompt fallback rate 为 `0.548–0.882`，但所有分支的最终 tracking/map 指标仍低于
raw；15% review threshold 也全部被超过。输出中 score-fallback 的 smoke 计数与逐分支
统计不一致，因此没有把它作为正面证据。

结论：离散时序 point prompt 的实际 SAM 实现被否决，不推进。

### 5. Historical-depth Veto 与 joint geometry feedback

命令：`commands_analyze_v0_geometry_feedback.txt`。单场景、30 帧，未重跑模型。

- current-depth control 基本保持 raw，但 map VoxelIoU 从 `0.09964` 降到 `0.09923`。
- history median + MAD：tracking IoU `0.41345→0.35939`，frame IDF1
  `0.50510→0.45408`，map VoxelIoU `0.09964→0.10324`，但 F5cm
  `0.34401→0.31340`，worsened-frame ratio `0.400`。
- history quantile interval：tracking IoU `0.35536`、frame IDF1 `0.44388`、
  map VoxelIoU `0.09699`，同样退化。
- causal self-consistency affine 的 holdout RMSE 从 `0.12161` 降到 `0.09138`，
  但这是历史 pointmap 一致性，不是 GT 几何质量证明。

结论：历史 pointmap 作为 veto/reference 不够可靠，容易误删前景；Veto 和 pose/depth
闭环均不自动推广。

### 6. Multi-clip causal affine + depth-head ray reconstruction

命令：`commands_analyze_multiclip_affine_geometry.txt`。4 个 scene-disjoint clip、
共 120 帧；每个 clip 为 21 帧 calibration + 9 帧 holdout；不重跑 StreamVGGT/SAM。

Depth-head affine 在 4 个 clip 中只有 1 个同时改善 holdout median 与 P90（要求至少
3/4）。主分支 holdout aggregate 为：

| variant | voxelIoU@5cm | F5cm |
|---|---:|---:|
| raw V0 | 0.00554 | 0.02582 |
| depth-head self-affine | 0.00428 | 0.01763 |

分场景只有 `00a...` 和 `1eacc...` 获益；`1a8e...` 明显崩溃，`0a184...` 没有有效
map。geometry gate 仅 `2/4` 通过，affine evidence gate 仅 `1/4` 通过，最终
`gate=NO_GO`。

结论：全局 per-scene scale-shift 不能稳定修复 depth-head；ray reconstruction 在
个别场景有效，但跨场景不可推广。

## 总结决策

本轮所有待验证路线均未通过稳定性或跨场景 gate：

1. SAM3.1 class-agnostic auto-proposal：NO-GO。
2. 离散 temporal point prompt：NO-GO。
3. 历史深度 Veto：NO-GO。
4. depth-head affine / ray reconstruction：NO-GO。

因此仓库代码恢复到 `9852343` 的 V0 baseline；本轮实验只作为负结果与路线选择依据，
不声称任何闭环或几何改进。
