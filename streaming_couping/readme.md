# streaming_couping

仓库保留两个完整方法和一个独立 camera-fusion 实验：

```text
V4 coverage-first
  appearance-only + additive pose + current-refined ray solver
  研究主线：优先保持实例 mask 覆盖率，后续解决错检

V5 adaptive-best
  residual-only + bounded SO(3) + fixed-reference ray solver
  安全对照：按无 GT ray support 选择 raw/learned pointmap

V6 camera overfit
  Feature Merger + SE(3) correction head
  五帧监督实验：只验证拟合能力和是否真正使用 instance token
```

## 运行

V4：

```bash
zsh streaming_couping/commands_v4_coverage_first.txt
```

V5：

```bash
zsh streaming_couping/commands_v5_adaptive_best.txt
```

V6 五帧 camera 消融：

```bash
zsh streaming_couping/commands_v6_camera_overfit.txt
```

V6 复用冻结 cache，不训练 SAM3/StreamVGGT，不训练 pointmap，也不运行 ray solver。
它从 seed 0 重新训练，并用同一 checkpoint 评估 `normal、instance_off、camera_off、shuffle_time、
wrong_geometry、appearance_only、geometry_only`。只需查看或复制：

```text
outputs/streaming_couping_v6_camera_overfit/v6_summary.csv
```

两条命令都优先迁移已经训练好的对应 checkpoint；没有 checkpoint 时只训练自己的固定结构。
它们共用冻结 feature cache，但 checkpoint、评估和最终导出目录完全分开。

## 保留的入口

```text
configs/v4_coverage_first.yaml
configs/v5_adaptive_best.yaml
configs/v6_camera_overfit.yaml
scripts/run_instance_token_pose.py
scripts/run_v5_reference_pose.py
scripts/run_v5_adaptive.py
scripts/run_v6_camera_overfit.py
scripts/plot_pose_comparison.py
docs/final_joint_pointcloud_pose_method.md
```

V4/V5 最终输出包括 native pointcloud/pose、GT/raw/ours 对比 PLY、三套 mask、位姿 PNG/PDF、
指标 CSV、checkpoint 哈希和 artifact manifest。

方法、结果和 GT 使用边界见
`streaming_couping/docs/final_joint_pointcloud_pose_method.md`。
