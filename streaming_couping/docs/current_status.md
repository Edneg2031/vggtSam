# 当前做法与实验状态

更新时间：2026-08-31

## 当前方案

当前默认系统已经从 StreamVGGT 切换到 HorizonStream：

```text
selected RGB frames
    ├─ HorizonStream (horizonstream conda env)
    │    └─ metric depth + confidence + intrinsics + online causal pose
    │            ↓ normalized geometry cache; process exits
    └─ SAM3.1 (3am conda env)
         └─ text masks + persistent instance IDs
                    ↓
        depth backprojection + world-coordinate voxel fusion
                    ↓
        full scene + semantic map + per-instance PLY
```

该系统没有训练。HorizonStream 和 SAM3.1 都使用冻结权重；SAM 不修改 pose/depth，mapping
层也不执行 BA、ICP 或 affine correction。两个模型在独立进程中顺序运行，以隔离依赖并降低
峰值显存。

默认实验仍使用 processed ScanNet++ 场景 `00a231a370`，从 795 帧中选择位置
`90,104,...,776` 共 50 帧，prompt 为 `bed wardrobe chair rug dustbin`。

## 当前代码状态

- HorizonStream 源码已保存在 `externals/horizonstream` 子模块，revision `9602f53`；
- 权重路径为 `/home/bod/86Nas/95_data_bak/FoundationModels/HorizonStream.pt`；
- 新增独立几何生成脚本和 `HorizonStreamGeometryCacheAdapter`；
- cache 固定保存 metric depth、归一化 confidence、`world_to_camera`、intrinsics、处理后 RGB
  和原始图片顺序；
- SAM 使用 cache 中与 depth 完全对齐的处理后 RGB，而不是对原图做不一致的简单 resize；
- cache 与当前选帧、权重、模型配置或源码不一致时拒绝复用；
- StreamVGGT 兼容路径和旧 V0 cache replay 仍保留，但不再是默认路径。

运行入口：

```bash
zsh streaming_couping/commands_run_semantic_map.txt
```

CPU contract/smoke 已通过；真实 HorizonStream 权重与 50 帧 GPU 推理只能在服务器环境验证。
因此当前不能宣称 HorizonStream 已经改善地图，下一份有效结果应同时报告几何可视化、地图完整
场景、实例点云、运行显存以及失败帧。

## 已有 StreamVGGT 证据

单个封存场景的 30 帧 raw V0 结果为：

| 指标 | raw V0 |
|---|---:|
| mean frame IoU | 0.41345 |
| frame IDF1 | 0.50510 |
| pixel IDF1 | 0.78665 |
| voxelIoU@5cm | 0.09964 |
| F-score@5cm | 0.34401 |
| ghost point ratio | 0.25789 |

GT-mask oracle 达到 frame IoU `0.81773`、frame IDF1 `0.89973`、voxelIoU@5cm
`0.18513` 和 F-score@5cm `0.50325`，但 oracle 仅是诊断上界。

4 个 scene-disjoint clip、共 120 帧的 causal affine 诊断没有通过：affine evidence 仅
`1/4` clip 通过，geometry gate 仅 `2/4`；holdout aggregate 的 raw V0
voxelIoU/F5cm 为 `0.00554/0.02582`，self-affine 分支降至 `0.00428/0.01763`。
历史深度 veto 和 temporal point-prompt 分支也出现明显恶化。因此这些优化均不进入当前系统。

这些结果说明现有 StreamVGGT 几何不足以支持继续做局部 affine 修补，也是本次改用
HorizonStream 直接替换几何后端的原因。它们不能用来推断 HorizonStream 的效果。

## 下一次服务器运行应检查

1. 第一阶段是否输出 `backend=horizonstream`、`frames=50` 和合理的 processed size；
2. `horizonstream_geometry.pt` 中 depth 是否有限且为正、相机轨迹是否连续；
3. 第二阶段日志是否为 `geometry_backend=horizonstream`，且没有加载 StreamVGGT checkpoint；
4. `scene_rgb_map.ply` 是否形成同一个坐标系中的场景，而不是明显漂移的帧层；
5. `objects/bed_*.ply`、`objects/rug_*.ply` 等实例是否与完整场景对齐；
6. 若 HorizonStream OOM，先令 `HORIZON_SLIDING_SIZE=1`；SAM 默认 batch 已设为 `1`。

只有完成上述服务器验证后，才能记录 HorizonStream 的实际效果；当前状态是“实现完成，真实
模型结果待跑”，不是性能结论。
