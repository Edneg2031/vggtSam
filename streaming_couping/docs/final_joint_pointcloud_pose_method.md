# StreamVGGT + SAM3：保留的 V4 与 V5

当前代码只保留两条可运行方法：

| 版本 | 用途 | 核心选择 |
|---|---|---|
| V4 coverage-first | 后续解决错检的研究主线 | 保留关联 mask 覆盖率，使用 appearance + additive pose |
| V5 adaptive-best | 点云安全的最终对照 | residual + bounded SO(3)，按 ray support 选择 raw/learned pointmap |

旧结构消融、historical-anchor、current-raw solver、joint BA 和 shared-SE(3) 已从代码删除。
历史实验结论体现在这里的版本选择中，不再保留不可运行的实验实现。

## 1. 公共数据流

```text
RGB 序列 + 参考帧实例 mask
  ├─ StreamVGGT
  │    ├─ aggregator camera/patch tokens
  │    ├─ raw world pointmap
  │    └─ raw camera pose
  └─ SAM3 persistent tracking
       ├─ raw masks
       ├─ geometry registration → MATCH / UNKNOWN / MISMATCH
       └─ causal appearance/geometry memory
                    ↓
          persistent instance tokens
             ┌──────┴──────┐
             │             │
      camera-token fusion  patch-token fusion
             │             │
      refined rotation     refined pointmap
             └──────┬──────┘
          angular-Huber point-to-ray
                    ↓
          refined camera translation
```

SAM3 外观来自冻结 detector backbone 的 `detector_fpn2` 特征。每个实例在 mask 内池化
feature mean/std，并和因果 memory、三维统计及质量分数一起编码为 512 维 persistent
instance token。token 只在可靠观测后更新；开始新 clip 时重新初始化。

StreamVGGT point head 使用 aggregator 的第 `4/11/17/23` 层 patch tokens。四层分别做
实例 cross-attention，再送回冻结 DPT point head。camera branch 只修改 camera token，
最终由冻结 CameraHead 解码。

所有可学习写入均为门控残差：

```text
output = frozen_token + sigmoid(gate) * zero_initialized_projection(attention)
```

projection 初始为零，因此 `module_off` 和零初始化都能精确恢复冻结 StreamVGGT。

## 2. 实例身份与三套 mask

当前观测与参考帧/历史可靠实例点云做有界 3D registration：

- `MATCH`：允许进入 camera、geometry、pointmap 和 ray solver；
- `UNKNOWN`：几何证据不足；可保留在关联输出，是否进入 camera 由版本权重决定；
- `MISMATCH`：明确空间冲突；不进入学习或解析几何分支。

两个版本都完整导出：

```text
segmentation_masks/       MATCH+UNKNOWN 关联结果，便于观察覆盖率
geometry_trusted_masks/   MATCH-only，实际几何支持
raw_tracking_masks/       未经过身份门控的 SAM3 原始追踪
```

当前的新序列问题属于“外观相似、空间不一致的错检”。V4 暂时优先保留覆盖率，后续应改进
UNKNOWN/MISMATCH 判定，而不是继续收紧阈值造成漏检。

## 3. V4 coverage-first

固定配置：

```text
pose feature             = appearance_only
rotation update          = additive_encoding
unknown camera weight    = 0
patch attention          = dilated union mask
ray solver               = current refined pointmap
```

V4 使用 `v4_match_additive_union` 的已训练权重。关联 mask 仍保留 MATCH+UNKNOWN，但 camera、
geometry 和 ray support 只使用 MATCH。这样能够保留可视化覆盖率，同时避免 UNKNOWN 直接
修改位姿。

已记录结果：

| clip | Raw ATE | V4 ATE | Raw pointmap | V4 pointmap |
|---|---:|---:|---:|---:|
| 90–240 | 0.36879 | 0.20464 | 0.15258 | 0.14037 |
| 492–589 | 0.36079 | 0.33316 | 0.22446 | 0.22644 |

结论：

- 两段序列位姿都改善；
- 90–240 点云改善；
- 492–589 pointmap 轻微退化约 `0.0020 m`；
- 新序列仍可能保留错检，因此 V4 是后续研究主线，不是身份关联已经解决的结论。

运行：

```bash
zsh streaming_couping/commands_v4_coverage_first.txt
```

第一次运行优先复制已有 `v4_match_additive_union` checkpoint；若不存在才训练。独立保存后
不会被 V5 覆盖：

```text
outputs/streaming_couping_v4_coverage_first/
├── checkpoints/decoupled_dual_branch/checkpoint_best.pt
├── evaluation/
├── final_v4_coverage_first/
├── v4_coverage_first_artifact_manifest.txt
└── v4_coverage_first_reproduction_inputs.sha256
```

## 4. V5 adaptive-best

固定学习结构：

```text
pose feature             = residual_only
rotation update          = bounded_so3 (max 5°)
unknown camera weight    = 0.25
patch attention          = dilated union mask
reference pose policy    = preserve reference, accepted center blend 0.5
```

最终 pointmap 只保留两个候选：

```text
P(0) = P_raw
P(1) = P_learned（参考帧和无效像素回退 raw）
```

对 `P(1)` 运行相同 ray solver，计算非参考帧接受率：

```text
support_ratio = fit_accepted / nonreference_frames

support_ratio >= 0.75  → 使用 P(1)
support_ratio <  0.75  → 使用 P(0)
```

这个 gate 不读取 GT。GT 只在运行结束后计算 ATE、rotation 和 pointmap error。

已记录结果：

| clip | support | 选择 | Raw→V5 ATE | Raw→V5 rotation | Raw→V5 pointmap |
|---|---:|---:|---:|---:|---:|
| 90–240 | 6/6 | learned | 0.36879→0.15932 | 2.72281→1.36900 | 0.15258→0.14045 |
| 492–589 | 2/5 | raw | 0.36079→0.35361 | 2.05885→2.01505 | 0.22446→0.22446 |

因此 V5 在可靠序列使用 learned pointmap，在低支持序列保持 raw pointmap；第二段序列的
位姿仍由 learned rotation 和 ray translation 获得小幅提升。

运行：

```bash
zsh streaming_couping/commands_v5_adaptive_best.txt
```

第一次运行优先迁移已有 `v5_residual_so3_union` checkpoint；若不存在，只训练这一种结构。
命令随后重新计算固定 reference pose、adaptive gate、完整导出、位姿图和数量检查。

```text
outputs/streaming_couping_v5_adaptive_best/
├── checkpoints/decoupled_dual_branch/checkpoint_best.pt
├── reference_pose/
├── adaptive_evaluation/
├── adaptive_upload_summary.csv
├── final_adaptive_pointmap_pose/
├── adaptive_artifact_manifest.txt
└── adaptive_reproduction_inputs.sha256
```

## 5. 输出内容

两个版本的每个 clip 都导出：

```text
<clip>/
├── deployable_native/
│   ├── camera_poses.csv
│   ├── camera_poses.npz
│   ├── full_scene.ply
│   └── instance_*.ply
├── segmentation_masks/
├── geometry_trusted_masks/
├── raw_tracking_masks/
├── pointclouds/
├── comparison_gt_world/
│   ├── full_scene/{ground_truth,streamvggt_raw,ours,overlay}.ply
│   ├── instance_*/
│   ├── camera_poses.csv
│   ├── camera_pose_metrics.csv
│   ├── pointcloud_metrics.csv
│   ├── pose_comparison.png
│   └── pose_comparison.pdf
└── camera_centers_overlay_metric_gt_world.ply
```

`deployable_native` 不读取 GT，使用 StreamVGGT native gauge。`comparison_gt_world` 使用一次
固定 reference-frame Sim(3)，只用于开发期公平比较；raw 和 ours 使用同一个对齐。

## 6. GT 使用边界

推理读取：

```text
RGB / frozen feature cache
SAM3 tracking 与 identity state
冻结 StreamVGGT 输出
训练好的 adapter
ray fit accepted 状态
```

训练时 GT 用于 pose/depth/pointmap supervision；评估时 GT 用于指标和 GT-world 可视化。
adaptive gate、SAM3 tracking、身份注册和最终 native 输出均不读取测试帧 GT。

## 7. 当前结论与下一步

- V4 是实例覆盖率优先、两段序列位姿较好的研究主线；
- V5 是 pointmap 不退化的安全对照；
- 当前只有一个场景的两个序列，不能声称跨场景通用；
- 下一步只研究 V4 的错检拒绝：使用更可靠的多帧空间一致性，在不降低 mask recall 的前提下
  区分外观相似但空间错误的 UNKNOWN；
- 不再恢复已失败的 joint BA、shared-SE(3) 或额外结构消融。
