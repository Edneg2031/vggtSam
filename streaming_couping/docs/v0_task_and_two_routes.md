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
- 等预算审计中，QK pose优于recent Top-4和三个random Top-4均值，说明收益不是普通历史截断。
- SAM persistent-ID固定方向pose probe未通过：correct ID的center改善仅`0.0168%`，rotation
  退化`0.2405%`，且未优于shuffled-ID/shifted-mask controls。
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

该路线不修改 raw pointmap。已完成的pose–pointmap自投影审计中，QK pose相对raw pose使
reprojection median从`36.98 px`降至`31.55 px`、relative-depth median从`0.1870`降至`0.1770`，
未观察到额外的坐标系数值失配。

允许的结论：“QK retrieval 改善当前单序列 pose，同时原始语义 pointmap 保持不变。”

不允许的结论：“pose 与 pointmap 已联合优化”或“SAM 已改善几何”。

## 4. 路线 B：SAM-indexed 实例级可微重建（V0之外）

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

## 5. 冻结决策

- 正式V0固定为路线A：QK pose、raw full-history world pointmap、SAM semantic lifting。
- 只保留`commands_v0_baseline.txt`和`commands_v0_semantic_map_ab.txt`两个运行入口。
- SAM mask/persistent ID不进入pose候选；相关probe已结束并删除实现。
- QK joint geometry只作为semantic-map A/B中的失败对照，不进入正式点云输出。
- 路线B不属于V0；若未来启动，应建立新版本并要求correct ID/mask严格优于global、shuffled和shifted controls。
- 已失败的joint-QK geometry、SAM-token camera fusion、object ICP和SAM identity pose probe只保留证据结论，
  不保留可执行入口。
