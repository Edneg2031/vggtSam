# streaming_couping

当前仓库只保留最终的实例引导点云与相机位姿方法：

```text
StreamVGGT geometry
  -> geometry-aware SAM3 recovery and persistent instance masks
  -> decoupled learned adapter
       patch branch  -> refined world pointmap
       camera branch -> refined camera rotation
  -> angular-Huber point-to-ray solve on tracked instances
  -> refined camera translation
```

最终输出是同一 StreamVGGT native gauge 中的：

```text
refined world pointmap + refined camera pose
```

## 保留的代码

```text
configs/final_joint_pointcloud_pose.yaml  最终配置
scripts/run_instance_token_pose.py        cache/train/eval/ray/export 入口
scripts/plot_pose_comparison.py           GT/raw/ours 位姿图
src/learned_pose/                         最终双分支、损失、ray solver 与导出
src/instance_observations.py              实例 ICP 特征与 tracking cache
src/tracking_recovery.py                  几何筛选后的 SAM3 recovery
docs/final_joint_pointcloud_pose_method.md 方法、结果和实验边界
```

旧的 token-fusion、pose-refinement 和 ray-solver 消融代码已经删除；对应结果保存在最终
方法文档中。

## 运行当前已训练结果

在仓库根目录运行：

```bash
zsh streaming_couping/commands_final_joint_pointcloud_pose.txt
```

它复用已有 feature cache 和最终 checkpoint，依次运行最终 ray translation、三方导出和
位姿图，不重新训练。

## 从头训练

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src:. python -m streaming_couping.scripts.run_instance_token_pose \
  --config streaming_couping/configs/final_joint_pointcloud_pose.yaml \
  --stage all \
  --sam3-device cuda:3 \
  --geometry-device cuda:1 \
  --training-device cuda:1
```

## 测试另一序列

复制最终 YAML，替换 clip 的 `scene_id`、递增 `frame_indices`、第一帧可见的静态
`instance_ids`，并设 `split: test`。为保持真正的跨序列测试，不在新序列上训练：

1. 将现有 `checkpoints/decoupled_dual_branch` 复制到新 `output_dir`；
2. 运行 `--stage cache`；
3. 运行 `--stage ray`；
4. 运行 `--stage export`。

主要结果：

```text
evaluation/ray_pose_compact_summary.csv
final_instance_ray_pose_v3/<clip>/comparison_gt_world/pointcloud_metrics.csv
final_instance_ray_pose_v3/<clip>/comparison_gt_world/camera_pose_metrics.csv
```

场景 `00a231a370` 的五帧示例 `100 200 300 400 500` 已配置好，可直接运行：

```bash
zsh streaming_couping/commands_final_joint_pointcloud_pose_test.txt
```
