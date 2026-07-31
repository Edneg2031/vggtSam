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
  三个完整 SE(3) 对照 + 七个专门化 3DoF head
  rotation/center 的 token 来源与坐标参数化 sweep
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

V6 30 帧完整历史压力测试（`90:15:525`）：

```bash
zsh streaming_couping/commands_v6_sam31_long30.txt
```

长序列命令只占三张物理卡：StreamVGGT 的 24 层按连续区间分到两张卡，SAM3.1 单独使用
第三张卡。每层 KV cache 保留完整因果历史，不切块、不重置；DPT/camera/depth/point
输出逐帧卸载到 CPU。默认使用物理卡 `1,2,3`，可在命令前通过
`V6_STREAM_GPU0、V6_STREAM_GPU1、V6_SAM_GPU` 覆盖。该序列复用 frame 90 作为参考帧，
用于验证 30 帧容量和长历史稳定性，不作为未见序列泛化证据。

V6 复用冻结 cache，不训练 SAM3/StreamVGGT，不训练 pointmap，也不运行 ray solver。
一次命令会用相同 seed 和步数训练 `camera_only、instance_only、fusion` 三套 6DoF 对照，再用
fusion checkpoint 评估 `instance_off、camera_off、shuffle_time、wrong_geometry、
appearance_only、geometry_only`。V6.3 另外训练 camera/instance/fusion 三种 rotation head、
三种 camera-frame center head 和 direct-world instance center 对照，再形成 3×3 合法组合。
所有专门 head 都只有 3DoF 输出权限。所有 checkpoint 测试
`210 240` 和 `492 512 520 545 561 589` 时均不重新训练、不运行 solver。第二段已参与 V6.2
设计，只能作为 development evidence；最终结论需要第三段预先锁定的序列。只需复制短表：

```text
outputs/streaming_couping_v6_camera_overfit/v6_component_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_v63_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_v64_aux_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_validation_summary.csv
```

全部训练与依赖消融仍保存在同目录的 `v6_summary.csv`。
命令还会打印 `v6_frame_diagnostics.csv`：七行 held-out/cross-clip 机制诊断，delta 相对 raw，
负数表示改善。该表只关联推理时可见的实例支持与事后 GT 误差，不用于逐帧选模型或调阈值。

正式候选比较使用预先锁定的第三段 `50 60 70 80`，并在训练区间从 90 开始前截止，避免
精确及近邻训练视角泄漏。该段不重新训练，只输出 raw、3DoF local-center 候选 A 和
`λ_rotation=0.1` 候选 B 三行。为与训练段保持同一初始化条件，第三段从参考帧 GT instance
mask 中读取训练使用的 `37/68/54`；哪个在第 50 帧可见就初始化哪个，不要求三者齐全，缺失
ID 保留为空槽。后续帧只要仍有一个可用 persistent instance 就运行；全部失效时两个候选在
该帧严格回退 raw。这里使用 GT reference mask，但 GT pose 仍只参与最终指标计算。

V4/V5 两条命令继续复用原始 SAM3 cache；V6 使用独立的 SAM3.1 cache，并从相同 seed
训练十三个 camera/instance 消融模型。两代 cache、checkpoint 和输出目录完全隔离。

## 保留的入口

```text
configs/v4_coverage_first.yaml
configs/v5_adaptive_best.yaml
configs/v6_camera_overfit.yaml
configs/v6_camera_sweep_base.yaml
configs/v6_sam31_long30_base.yaml
configs/v6_sam31_long30_camera.yaml
configs/recovery_sam31_050_025.yaml
scripts/run_instance_token_pose.py
scripts/run_v5_reference_pose.py
scripts/run_v5_adaptive.py
scripts/run_v6_camera_overfit.py
scripts/plot_pose_comparison.py
docs/final_joint_pointcloud_pose_method.md
```

V4/V5 最终输出包括 native pointcloud/pose、GT/raw/ours 对比 PLY、三套 mask、位姿 PNG/PDF、
指标 CSV、checkpoint 哈希和 artifact manifest。

版本演进、结果对比和当前最终版本总结见
`streaming_couping/docs/sam3_streamvggt_work_summary.md`。

V4/V5/V6 的详细实现与实验约束见
`streaming_couping/docs/final_joint_pointcloud_pose_method.md`。

当前 V6 cache 的实际数据流是：

```text
SAM3.1 raw tracking
  → StreamVGGT 参考物体投影生成 box + positive points
  → adaptive_positive_compete_010
  → SAM3.1 mask 内池化 detector_fpn2
  → 实例几何、instance token、camera/pointmap 实验
```
