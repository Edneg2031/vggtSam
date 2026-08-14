# StreamVGGT + SAM3 V0：任务与两条路线

## 1. 任务定义

输入是流式 RGB 序列和用户 prompt，输出是：

- 相机轨迹；
- 带 persistent instance ID/prompt 的语义点云；
- 允许物体在未来帧首次出现，不要求第一帧包含所有物体。

当前场景按静态世界处理。StreamVGGT 和 SAM3 保持冻结。目标是在不降低原始
StreamVGGT 点云质量的前提下改善 pose，并进一步研究 SAM memory/mask 能否改善实例几何。

## 2. 已确认事实

- StreamVGGT `point_head` 直接输出以第一帧为参考的 shared-world pointmap；生成 pointmap
  地图不需要用每帧 pose 再变换。
- `depth_head` 输出相机坐标下的 depth；用 depth 建图时必须使用 `K + pose` 反投影。
- 无训练 native-QK retrieval 在当前 30 帧单序列上使 center error 改善 `10.93%`，rotation
  error 改善 `6.15%`。
- `QK pose + raw depth`、QK joint depth 和 QK joint pointmap 都没有改善整体地图。
- 直接 SAM token/memory feature → camera/point head、实例 ICP/voxel object fusion 等旧路线均没有
  得到稳定、可归因的改善，不再原样重复。

## 3. 路线 A：解耦 V0 baseline（保证下限）

```text
full-history KV → raw StreamVGGT point_head → world pointmap
                                              ↓
SAM persistent mask/ID ──────────────→ semantic lifting

retrieved QK KV → StreamVGGT camera_head → improved pose
```

固定输出：

- `pose_output = retrieve_qk_pose`；
- `geometry_output = raw_streamvggt_world_pointmap`；
- `semantic_output = SAM_mask_lifted_raw_world_pointmap`；
- 两个分支均以第一帧为 gauge。

该路线不修改 raw pointmap，因此点云指标不会下降。需要补充一个 pose–pointmap 自投影一致性审计，
确认 QK pose 与 raw world pointmap 共用坐标系时没有明显的逐帧数值失配。

允许的结论：“QK retrieval 改善当前单序列 pose，同时原始语义 pointmap 保持不变。”

不允许的结论：“pose 与 pointmap 已联合优化”或“SAM 已改善几何”。

## 4. 路线 B：SAM-indexed 实例级可微重建（争取增益）

不再对 mask 选中的 raw points 做一次 ICP，而是将 raw pointmap 和 QK pose 作为初值，为每个
persistent instance 建立可优化的 surfel 或 3D Gaussian 几何。

```text
SAM memory → persistent ID + mask history + visibility
                                   ↓
raw pointmap/depth + QK pose → per-instance geometry initialization
                                   ↓
                multi-view RGB + silhouette + depth consistency
                                   ↓
                     refined semantic object geometry
```

SAM 的作用：

- memory bank/registry 提供跨帧同一实例的观测集合；
- mask 提供 silhouette、measurement ownership 和 object/background 边界；
- RGB、depth 和 StreamVGGT geometry 提供真正的几何优化信号；
- SAM hidden feature 不直接作为几何 correspondence。

优化时背景和未成熟实例保持 raw fallback；只有被多帧观测、几何残差和 silhouette 共同支持的
实例才输出 refined geometry。late-birth 实例创建新的 object state，不复用其他 slot。

最小对照：

- raw pointmap；
- 不使用实例的 global reconstruction；
- 正确 SAM persistent ID/mask；
- shuffled ID 和 shifted-mask control。

通过条件：整体地图不退化，成熟实例几何改善，且正确 ID/mask 稳定优于 global、shuffled
和 shifted control。在此之前不修改正式 V0 pointmap。

## 5. 当前决策

- 路线 A 是当前可用 baseline；`commands_v0_incremental_audit.txt`一次完成pose–pointmap一致性审计，
  以及full/recent/3-seed random/QK的等预算上下文对照。
- 一致性审计先输出连续指标，不在看到结果后临时设置通过阈值；上下文对照不读取SAM。
- QK已同时优于recent与3-seed random均值；raw pointmap与QK pose的一致性指标也相对raw pose改善。
- `commands_v0_sam_identity_pose_probe.txt`现用于correct-ID/global/foreground-union/shuffled-ID/
  shifted-mask五分支的等点数、因果、固定方向信息探针；该探针不修改正式V0。
- 只有correct persistent ID同时改善QK的center/rotation并严格优于四个control，才进入迭代优化。
- 路线 B 是后续改善 semantic pointmap 的候选，目前只是方法设计，尚无实验结果。
- 已失败的 joint-QK geometry、SAM-token camera fusion 和 object ICP 仅保留为证伪记录。
