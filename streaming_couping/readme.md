# Streaming Coupling

当前目录保留一条冻结 SAM3 与 StreamVGGT 的对象级 2D–3D 双向闭环。tracking
恢复和实例点云阶段已经完成压力测试；当前实验继续使用可靠实例地图修正
StreamVGGT pointmap translation，再运行已验证的 ray-center camera translation
repair。

```text
SAM3 persistent obj_id / mask memory
                 ↕
 score + geometry coverage + candidate support gate
                 ↕
StreamVGGT persistent 3D object map
```

实验不训练 adapter，也不做 token concat。当前只使用 translation-only
instance ICP，而且多个静态实例必须形成共识后才产生一个整帧共享平移；不会给
每个实例各自修改相机。

## 当前实验

- 场景：ScanNet++ `00a231a370`
- 帧：`90 105 119 130 140 210 240`
- reference：序列位置 0，即帧 90
- 实例：37 cabinet、68 wardrobe、54 bed

37/68 是易例，用于验证闭环不会误伤原始追踪；54 是高置信低 IoU 压力例，用于
验证 geometry-disagreement gate 能否识别“mask 非空但跟错”。

完整恢复实验设计见
[`docs/ablation_plan.md`](docs/ablation_plan.md)。当前点图/位姿诊断设计见
[`docs/pose_pointmap_diagnostics.md`](docs/pose_pointmap_diagnostics.md)，当前
多实例位姿实验见
[`docs/instance_pose_refinement.md`](docs/instance_pose_refinement.md)，服务器
唯一命令见 [`commands.txt`](commands.txt)。

最终验证配置为 `configs/recovery_050_025.yaml`：

```yaml
tracker_min_geometry_coverage: 0.50
recovery_min_support_coverage: 0.25
map_update_min_geometry_coverage: 0.50
```

## 因果数据流

```text
reference GT mask
  -> 初始化原 SAM3 obj_id 与 reference 3D object map
  -> SAM3 原始逐帧跟踪
  -> score 与历史 3D 投影均可靠的 mask 扩充 object map
  -> mask 缺失 / 低分 / 与对齐几何冲突
  -> 当前帧全图文本候选
  -> geometry 只做同类实例选择
  -> 完整候选 mask 写回原 obj_id
  -> 继续传播并评估未来帧
```

reference 后的 GT mask 不进入 natural gate、对象地图更新或几何候选排序。GT
只进入指标和明确命名的 `oracle_candidate` / `oracle_mask` control 分支。

## 一次运行包含的消融

两种事件策略：

- `natural_joint_gate`：实际可部署触发；
- `scheduled_probe`：固定在帧 119 做受控干预，与 natural event 做因果对照。

七个 tracking 分支：

1. `original`
2. `geometry_recovery_no_memory`
3. `reference_geometry_same_id_memory`
4. `geometry_recovery_same_id_memory`
5. `shuffled_geometry_same_id_memory`
6. `oracle_candidate_same_id_memory`
7. `oracle_mask_same_id_memory`（GT-visible-mask writeback control）

其中第 2、4 分支使用完全相同的恢复 mask，唯一变量是是否写入 SAM3 memory；
第 3、4 分支的差异是 object map 是否由可靠历史 tracking masks 扩充；第 5 分支
只打乱非 reference 几何。

每个 tracking 分支还会比较 `all_frames / score_gate / joint_gate` 三种对象地图
写入策略，并提供 GT-mask oracle 与 time-shuffled-mask negative control。

## 恢复实验复现

在仓库根目录执行：

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src:. python -m streaming_couping.scripts.run_recovery_writeback_ablation \
  --config streaming_couping/configs/recovery_050_025.yaml \
  --manifest data/processed/scannetpp_pinhole_2d/manifest.json \
  --scene-id 00a231a370 \
  --instance-ids 37 68 54 \
  --frame-indices 90 105 119 130 140 210 240 \
  --reference-sequence-index 0 \
  --event-policies natural_joint_gate scheduled_probe \
  --probe-sequence-index 2 \
  --sam3-device cuda:3 \
  --geometry-device cuda:1 \
  --output-dir outputs/streaming_couping_threshold_050_025_probe119_37_68_54
```

StreamVGGT 只提取一次；每个实例只跑一次原始 SAM3 tracking；每个后续帧的
global-text candidates 只生成一次并在全部分支复用。memory 分支仍需分别建立
SAM3 session，因为它们的后续 memory 状态不同。

## 阶段性最终结果

bed(54) 是高置信错误追踪压力例：帧 119 的 SAM3 score 为 `0.9844`，但原始 IoU
只有 `0.0207`。natural gate 自动触发后，时序对齐几何选中的完整候选 IoU 为
`0.9323`。

| bed tracking 分支 | cross-view IoU | 后续 4 帧平均 IoU |
|---|---:|---:|
| original | 0.1292 | 0.0002 |
| recovery，无 memory | 0.2812 | 0.0002 |
| recovery + same-ID memory | 0.7986 | 0.7763 |
| shuffled geometry + memory | 0.1292 | 0.0002 |

| bed map 分支 | Chamfer-L1 ↓ | F5 ↑ | F10 ↑ |
|---|---:|---:|---:|
| original | 0.5269 | 0.0359 | 0.0819 |
| recovery，无 memory | 0.2678 | 0.0836 | 0.2167 |
| recovery + same-ID memory | 0.0416 | 0.7152 | 0.9443 |
| GT-mask map oracle | 0.0403 | 0.7396 | 0.9481 |

37/68 的 natural gate 未触发且逐帧结果不变；shuffled geometry 被候选 support
gate 拒绝；natural 与 scheduled 在 bed 上结果相同。完整结果和解释记录在
[`docs/thread_handoff.md`](docs/thread_handoff.md)。

`reference_geometry_same_id_memory` 与完整 reliable-history map 分支结果相同，
所以本例证明的是几何帮助追踪、恢复后的追踪帮助点云；尚未证明历史 tracking
扩图比 reference-only map 更利于恢复。

## 已验证的 ray-center 位姿修复

上一阶段只提取一次 frozen StreamVGGT，输出：

- 固定 reference-point Sim(3) 下的 pose ATE/RPE；
- reference-pose 对齐后的相对漂移；
- 全轨迹 Sim(3) 的乐观 gauge 参照；
- StreamVGGT 官方风格的 all-pairs rotation/translation-direction accuracy；
- 逐帧 paired pointmap RMSE；
- 处理后 GT 与 predicted intrinsics 误差。

raw baseline 已确认 rotation 稳定而局部 translation direction/scale 失败。
当前脚本在同一次推理中增加 pointmap-consistent ray-center 修复：由 point-head
世界点与对应像素射线重估 camera center，固定 rotation，以 `t=-RC` 更新
world-to-camera。它不加载 SAM3、不使用实例 mask，也不修改 pointmap。

服务器结果选择 `ray_predicted_k_all` 为不使用 GT 的主分支：固定
reference-point Sim(3) ATE `0.3745 -> 0.1759 m`，RPE translation RMSE
`0.1522 -> 0.0809 m`。80% trimmed 在所有非 reference 帧均略差，只保留为
消融；同次还运行 GT K/R oracle 和 spatially-shuffled pointmap 负对照。详见
[`docs/pose_pointmap_diagnostics.md`](docs/pose_pointmap_diagnostics.md)。

## 当前多实例 pointmap/pose 消融

当前 `commands.txt` 已切换到一次完整实验：

```text
original + natural-recovered SAM3 tracking（首次运行后缓存）
  -> reference GT prompt 初始化三个独立 static-instance maps
  -> 当前实例点对历史同 ID map 做 translation-only trimmed NN ICP
  -> 至少两个实例的 proposal 在阈值内形成共识
  -> 实例等权 coordinate-wise median 得到一个整帧共享 translation
  -> 对完整 pointmap 只写入一次
  -> predicted-K/R all-point ray-center
  -> 固定 R，以 t=-RC 更新 camera pose
```

无共识时实例修正严格为零，仍保留 `ray_only` 结果。causal map 只接收共识参与者
且 tracker score 可靠的 observation。alpha 消融只缩放整帧 pointmap 写回量；
object map 始终使用完整共享 translation 更新，因此不会混淆地图历史与修正强度。

无 GT 主分支是 `recovered_causal_a100`。`gt_masks_causal_a100` 和
`gt_point_translation_oracle` 只提供评估上限，`shuffled_ids_causal_a100` 是
实例身份负对照。主要看主分支相对 `ray_only` 是否继续降低 pointmap RMSE、
固定 reference-point Sim(3) ATE/RPE，以及能否改善帧 240 和 `210->240`。

## 已完成 tracking 阶段的主要输出

输出根目录：

- `summary.csv`：实例 × event policy × tracking mode 总指标；
- `frame_metrics.csv`：逐帧 IoU、score、恢复前后标记；
- `candidate_screening.csv`：全部后续帧的候选生成/选择上限；
- `candidate_diagnostics.csv`：每个候选的几何覆盖与 GT 诊断；
- `geometry_gate_diagnostics.csv`：逐帧 gate、对象地图更新与几何支持；
- `threshold_sweep.csv`：27 组 gate 阈值的免重跑诊断；
- `map_quality.csv`：对象地图 5/10 cm precision、recall、F-score 与 Chamfer；
- `pointcloud_summary.csv` 和 `pointcloud_frame_metrics.csv`；
- `metadata.json`。

每个 `instance_<id>/<event_policy>/` 下还有对应 CSV、可视化报告和各分支 PLY。

## Map 指标的 GT 使用边界

对象地图质量评估会用 reference 帧全场景对应点拟合一次固定 Sim(3)，随后对所有
帧和分支保持不变。GT pointmap 只计算 5/10 cm surface metrics，不参与：

- StreamVGGT 输出；
- natural gate；
- candidate ranking；
- object map 更新；
- SAM3 memory writeback。

若 manifest 没有 mesh-rasterized GT pointmap，tracking 消融仍会完整运行，
`metadata.json` 会记录 map evaluation 被禁用的原因。

## 已完成 tracking 阶段的判读顺序

1. `candidate_screening.csv` 判断瓶颈在候选生成还是几何选择。
2. scheduled probe 中比较 full memory 与 no-memory 的
   `post_recovery_iou`。
3. 比较 aligned、reference-only 和 shuffled，分别判断 tracking→geometry
   扩图与时序对齐几何是否有效。
4. 检查 natural gate 是否救回 54，同时不伤害 37/68。
5. 在 `map_quality.csv` 比较 `joint_gate` 与 `all_frames` 的 precision/F-score。

本机没有 PyTorch/GPU，只进行静态检查；正式数值需在服务器环境运行。

## 当前代码结构

```text
scripts/run_recovery_writeback_ablation.py  CLI
scripts/run_pose_pointmap_diagnostics.py     已验证 ray-center 诊断 CLI
scripts/run_instance_pose_refinement_ablation.py 当前唯一实验 CLI
src/recovery_writeback_ablation.py          两策略、七分支与汇总
src/instance_pose_refinement.py             多实例共享平移 + ray pose
src/recovery.py                             几何挖掘、联合 gate、可靠地图更新
src/instance_point_cloud.py                 实例 PLY
src/instance_map_evaluation.py              evaluation-only 3D map metrics
src/pose_pointmap_diagnostics.py            raw诊断 + ray-center位姿修复消融
src/backbones/sam3_wrapper.py               tracking、候选与 same-ID 写回
src/backbones/streamvggt_wrapper.py         冻结 StreamVGGT 提取
src/aggregation/                            persistent object map 与投影
src/bridge/gating.py                        tracker/geometry 联合触发
```
