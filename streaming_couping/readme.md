# streaming_couping

冻结 SAM3 与 StreamVGGT，研究两者的双向协同：

```text
StreamVGGT geometry
  -> 发现 SAM3 高置信跟错并恢复同一实例
  -> reliable same-ID tracking
  -> persistent static-instance maps
  -> 一个整帧共享 pointmap translation
  -> ray-center camera translation repair
```

## 当前状态

SAM3 恢复阶段已经完成并暂停：

- bed 在帧 119 从原始 IoU 0.0207 恢复到 0.9323；
- same-ID memory 使后续四帧平均 IoU 达到 0.7763；
- cabinet/wardrobe 易例没有被 natural gate 误触发；
- 固定阈值在 `configs/recovery_050_025.yaml`。

相机修复 V1/V2 均未通过完整指标，已记录在
[`docs/instance_pose_refinement.md`](docs/instance_pose_refinement.md)。当前代码运行
第三版：严格 ICP/RMSE gate、temporal conflict filtering、只有验证参与者可写回
object map，以及最多两次的 bounded carry。

## 唯一运行入口

在仓库根目录：

```bash
zsh streaming_couping/commands.txt
```

命令优先复用第一版的 `tracking_cache.npz`；若缓存不存在，会运行一次 SAM3 并在
指定路径生成。reference 后 GT 不参与可部署 gate、ICP、共识、地图更新或 ray
fit。GT 只用于评估和明确命名的 translation oracle。

## 当前代码

```text
scripts/run_instance_pose_refinement_ablation.py  唯一 CLI

src/instance_pose_refinement.py   第三版主流程、消融与 CSV
src/tracking_recovery.py          已验证的自然恢复 + same-ID writeback
src/recovery.py                   geometry gate 与坐标转换
src/instance_point_cloud.py       实例点云与 PLY 导出
src/pose_evaluation.py            ray-center、ATE/RPE、all-pairs 指标
src/pointmap_alignment.py         reference-only Sim(3) 与 GT pointmap
src/backbones/                    冻结 SAM3 / StreamVGGT wrapper
src/aggregation/                  persistent object map 与 revisit geometry
src/bridge/gating.py              geometry-disagreement gate
```

已删除完成阶段的 scheduled probe、七分支 SAM sweep、独立 raw pose/intrinsics
诊断 CLI。其实验结论仍保留在 `docs/`，当前主流程只保留会继续使用的机制。

## 主要输出

```text
tracking_summary.csv
instance_correction_events.csv
instance_icp_diagnostics.csv
instance_ray_fit.csv
instance_pose_summary.csv
instance_pose_frame_metrics.csv
instance_pose_rpe.csv
instance_pose_pair_metrics.csv
instance_pose_pair_summary.csv
instance_pointmap_frame_metrics.csv
instance_pointmap_summary.csv
corrected_pointcloud_summary.csv
metadata.json
instance_<id>/pointclouds/<mode>/*.ply
```

相关文档：

- [`docs/method.md`](docs/method.md)：完整研究方法
- [`docs/instance_pose_refinement.md`](docs/instance_pose_refinement.md)：V1/V2 结论与第三版设计
- [`docs/pose_pointmap_diagnostics.md`](docs/pose_pointmap_diagnostics.md)：已完成的 raw/ray 诊断结果
- [`docs/thread_handoff.md`](docs/thread_handoff.md)：对话接力与当前待办
