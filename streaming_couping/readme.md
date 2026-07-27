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
  三个完整 SE(3) 对照 + 两个专门化 3DoF head
  camera rotation + persistent-instance camera center
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
一次命令会用相同 seed 和步数训练 `camera_only、instance_only、fusion` 三套 6DoF 对照，再用
fusion checkpoint 评估 `instance_off、camera_off、shuffle_time、wrong_geometry、
appearance_only、geometry_only`。V6.2 另外训练 camera-only rotation head 和 instance-only
camera-center head，二者都只有 3DoF 输出权限，再合法重建 W2C。所有 checkpoint 测试
`210 240` 和 `492 512 520 545 561 589` 时均不重新训练、不运行 solver。第二段已参与 V6.2
设计，只能作为 development evidence；最终结论需要第三段预先锁定的序列。只需复制短表：

```text
outputs/streaming_couping_v6_camera_overfit/v6_cross_clip_summary.csv
```

全部训练与依赖消融仍保存在同目录的 `v6_summary.csv`。
命令还会打印 `v6_frame_diagnostics.csv`：七行 held-out/cross-clip 机制诊断，delta 相对 raw，
负数表示改善。该表只关联推理时可见的实例支持与事后 GT 误差，不用于逐帧选模型或调阈值。

V4/V5 两条命令优先迁移已有 checkpoint；V6 每次从相同 seed 重新训练五个模型。它们共用
冻结 feature cache，但 checkpoint、评估和最终导出目录完全分开。

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
