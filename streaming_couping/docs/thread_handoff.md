# vggtSam 对话接力上下文

更新时间：2026-07-20。新对话先读取本文件、`method.md`、
`instance_pose_refinement.md`、`commands.txt` 和 Git 工作区。

## 固定研究约束

- 当前实现集中于 `streaming_couping/`；外部旧代码和数据不要随意修改。
- 用户只能在服务器运行 SAM3/StreamVGGT，本机没有 PyTorch/GPU。
- reference 后 GT 不参与可部署 gate、ICP、共识、地图更新或 ray fit。
- 不恢复“每个实例独立修改一次相机”的旧方案。
- 多实例最终只能产生一个整帧共享平移。
- 实例 PLY 导出必须保留。
- 最终尽量只给一条服务器命令。

## 总体研究链路

```text
geometry helps tracking
-> reliable same-ID tracking builds persistent instance maps
-> multiple instances constrain one frame pointmap translation
-> corrected pointmap repairs StreamVGGT camera translation
```

## 已完成：SAM3 geometry recovery

场景 `00a231a370`，实例 `37 68 54`，帧：

```text
90 105 119 130 140 210 240
```

固定配置：

```text
tracker_min_geometry_coverage = 0.50
recovery_min_support_coverage = 0.25
map_update_min_geometry_coverage = 0.50
```

配置文件：`configs/recovery_050_025.yaml`。

bed 的原始 SAM3 在帧 119 score 为 0.9844，但 IoU 仅 0.0207。geometry 选择的
全图文本候选 IoU 为 0.9323；写回原 obj_id memory 后，帧 130/140/210/240 IoU
分别为 0.8674/0.9487/0.6787/0.6105。cabinet/wardrobe 未被 natural gate
误触发。

结论：geometry 能识别高置信跟错，完整候选必须由 SAM3 产生，same-ID memory
writeback 是后续稳定传播的必要因果因素。该阶段暂停，不再做 held-out。

已删除 scheduled probe、七分支、threshold sweep 的执行代码；结果记录仍在
`ablation_plan.md` 与 Git 历史。当前只保留 `tracking_recovery.py` 的自然恢复
主路径和 tracking cache。

## 已完成：raw pose / pointmap 诊断

StreamVGGT raw 指标：

- reference-point Sim(3) ATE 0.3745 m；
- reference-pose + point scale ATE 0.2280 m；
- optimistic trajectory Sim(3) ATE 0.0685 m；
- all-pairs rotation mean 2.39°，translation direction mean 14.56°。

pointmap reference RMSE 0.0642 m；non-reference RMSE 0.1329 m，并随时间增长到
帧 240 的 0.1866 m。

predicted intrinsics 误差约 fx 4.94%、fy 6.40%；主点误差约 1.3 px。

all-point predicted-K/R ray-center 已验证：

- fixed-reference ATE 0.3745 → 0.1759 m；
- RPE translation RMSE 0.1522 → 0.0809 m；
- all-pairs direction mean 14.56° → 11.35°；
- translation@10° 33.3% → 71.4%。

因此后续实例修复以 `ray_only` 为真正 baseline。

独立 raw/ray/intrinsics CLI 已删除；可复用评估算子集中在 `pose_evaluation.py`，
历史结果见 `pose_pointmap_diagnostics.md`。

## 第一版实例修复：负结果

V1：

```text
per-instance translation proposal
-> loose multi-instance consensus
-> one shared frame translation
-> same shared translation updates all participating object maps
-> ray-center pose
```

主要指标：

| mode | ATE | RPE translation RMSE | all-pairs direction mean |
|---|---:|---:|---:|
| `ray_only` | 0.175915 | 0.080938 | 11.345° |
| V1 causal a100 | 0.181835 | 0.090999 | 14.393° |
| V1 causal a025 | 0.176317 | 0.080894 | 11.589° |
| GT translation oracle | 0.128355 | 0.067590 | 9.588° |

V1 a100 ATE 恶化 3.37%，RPE 恶化 12.43%，不能声称 pose 改善。

但 V1 a050 的 non-reference pointmap RMSE 0.132935 → 0.130259，说明实例几何有
弱有效信号；GT oracle 为 0.115600。

失败根因：

1. bed recovery 后 tracking 很好，但跨视角几何表面没有足够 ICP overlap；
2. 140/210 只有 wardrobe proposal，min participants=2 正确拒绝；
3. shared map update 把实例差异写进 causal maps；
4. correspondence ratio 0.15、consensus 0.05 过宽；
5. 130 有修正、140 归零，使 130→140 direction 29.63° → 75.78°。

## 当前实现：第二版

当前唯一代码入口：

```text
scripts/run_instance_pose_refinement_ablation.py
src/instance_pose_refinement.py
```

V2 变化：

1. correspondence ratio 0.05；
2. 显式 ICP RMSE ≤ 0.03 native；
3. final robust-center max residual ≤ 0.02；
4. 每个严格 proposal 用自己的 translation 更新自己的 object map；
5. 相机仍只使用一个 shared translation；
6. 只有恰好一个 proposal、距最近多实例共识 gap ≤ 15、距最近平移 ≤ 0.02
   时允许 short carry，carry 本身不后移该窗口；
7. multi-instance conflict 或长 gap 清空 temporal state。

同次模式：

```text
ray_only
v1_shared_map_a100
v2_strict_shared_map_no_carry_a100
v2_strict_per_instance_no_carry_a100
v2_strict_per_instance_short_carry_a025
v2_strict_per_instance_short_carry_a050
v2_strict_per_instance_short_carry_a100
v2_strict_reference_no_carry_a100
v2_strict_shuffled_short_carry_a100
gt_point_translation_oracle
```

主候选是 `v2_strict_per_instance_short_carry_a100`，尚未经过服务器验证。

## 精简后的代码边界

保留：

```text
tracking_recovery.py       natural recovery + same-ID writeback
recovery.py                geometry gate / mask conversion
instance_point_cloud.py    PLY
instance_pose_refinement.py
pose_evaluation.py
pointmap_alignment.py
backbones/
aggregation/
bridge/gating.py
```

删除：

```text
run_recovery_writeback_ablation.py
recovery_writeback_ablation.py
run_pose_pointmap_diagnostics.py
pose_pointmap_diagnostics.py
instance_map_evaluation.py
```

功能由更小的当前模块覆盖，历史结论保留在文档/Git。

## 下一步

从仓库根目录只运行：

```bash
zsh streaming_couping/commands.txt
```

完成后优先回传：

```text
instance_pose_summary.csv
instance_pose_rpe.csv
instance_pose_pair_summary.csv
instance_pose_pair_metrics.csv
instance_pointmap_summary.csv
instance_pointmap_frame_metrics.csv
instance_correction_events.csv
instance_icp_diagnostics.csv
metadata.json
完整 log
```

重点确认：

- 105 是否被 strict consensus 拒绝；
- 119/130 是否仍形成多实例共识；
- 140 是否只通过 bounded short carry 连续修正；
- 210 是否因长 gap 禁止 carry；
- per-instance map 是否让 240 的 cabinet/wardrobe proposals 更一致；
- V2 是否同时优于 `ray_only` 和 `v1_shared_map_a100`。
