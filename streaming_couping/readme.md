# streaming_couping

仓库只保留两个实例引导的 StreamVGGT + SAM3 版本：

```text
V4 coverage-first
  appearance-only + additive pose + current-refined ray solver
  研究主线：优先保持实例 mask 覆盖率，后续解决错检

V5 adaptive-best
  residual-only + bounded SO(3) + fixed-reference ray solver
  安全对照：按无 GT ray support 选择 raw/learned pointmap
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

两条命令都优先迁移已经训练好的对应 checkpoint；没有 checkpoint 时只训练自己的固定结构。
它们共用冻结 feature cache，但 checkpoint、评估和最终导出目录完全分开。

## 保留的入口

```text
configs/v4_coverage_first.yaml
configs/v5_adaptive_best.yaml
scripts/run_instance_token_pose.py
scripts/run_v5_reference_pose.py
scripts/run_v5_adaptive.py
scripts/plot_pose_comparison.py
docs/final_joint_pointcloud_pose_method.md
```

最终输出包括 native pointcloud/pose、GT/raw/ours 对比 PLY、三套 mask、位姿 PNG/PDF、
指标 CSV、checkpoint 哈希和 artifact manifest。

方法、结果和 GT 使用边界见
`streaming_couping/docs/final_joint_pointcloud_pose_method.md`。
