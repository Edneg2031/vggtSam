# SAM3 instance-guided HorizonStream pose refinement

更新时间：2026-09-02

这是一个默认关闭的、免训练的 pose-graph ablation。它不修改 HorizonStream 网络、权重或
KV cache，也不修改 SAM3.1 的 persistent ID 和 mask。它只在两种模型完成推理后，使用
SAM 的长期 instance identity 构造可拒绝的 object-level SE(3) edge，再用优化后的相机
pose 重新调用现有 semantic mapper。

## 数据流

```text
HorizonStream: depth + confidence + intrinsics + online camera pose
SAM3.1:       mask + track score + persistent instance ID
                         │
same instance + large temporal gap
                         │
             mask-constrained RGB patch matching
                         │
        local camera depth + intrinsics -> 3D-3D pairs
                         │
                 RANSAC + weighted SVD
                         │
          HorizonStream sequential edges + object edges
                         │
                   robust pose graph
                         │
       original depth/intrinsics/masks + refined camera pose
                         │
                    semantic map export
```

instance ID 只用于确认两次观察属于同一个候选物体，不会把 object centroid 当作对应点。
匹配位置必须分别落在两个 SAM mask 内；3D 点先保持在各自 camera coordinate，不能先用
当前 world pose 把两边强行对齐。

## 运行

编辑 `streaming_couping/commands_run_semantic_map.txt`：

```zsh
OBJECT_POSE_REFINEMENT=1
FUSION_POLICY="raw"
```

然后执行：

```zsh
zsh streaming_couping/commands_run_semantic_map.txt
```

命令文件默认让 HorizonStream 和 SAM3.1 使用同一个 `horizonstream` Python 环境；如果
服务器上的 SAM 依赖仍是旧的 split environment，可以设置：

```zsh
export STREAMING_COUPING_PYTHON=/path/to/old/sam/environment/bin/python
```

几何推荐先复用已有 `horizonstream_geometry.pt`。refinement 需要 RGB 模式和带
`camera_to_world`/depth/intrinsics 的 geometry provider；旧 V0 cache replay 默认不走这条
路径。

## A0/B0 对照

| 实验 | 设置 | 含义 |
|---|---|---|
| A0 | `OBJECT_POSE_REFINEMENT=0` | HorizonStream online pose + SAM semantic mapping |
| B0 | `OBJECT_POSE_REFINEMENT=1` | A0 的 frozen 输出 + SAM object pose edges + pose graph |

两次运行应使用相同 scene、frame selection、prompt、HorizonStream cache 和 SAM 参数，使用
不同的 `OUTPUT_DIR`。当前仓库没有接入可单独控制的 HorizonStream native loop closure，因而
A1/B1 暂不自动生成；将来接入时应保留 native edge 和 SAM object edge 的 provenance 区分。

## 默认参数与 gate

默认只考虑 track score 至少 `0.50`、mask 至少 32 像素、有效 geometry 点至少 24、时间
间隔至少 10 帧的观察；每个 instance 最多生成 8 个 pair，总候选最多 256 个。默认
`rgb_patch` backend 是无额外模型的局部 RGB 统计/梯度 patch descriptor，也可显式选择
已有的 DINOv3 encoder：

```zsh
OBJECT_POSE_FEATURE_BACKEND="dinov3"
```

每个候选依次经过：

1. mask/track/geometry 质量过滤；
2. mask 内 mutual-nearest feature matching、cosine threshold 和 ratio margin；
3. local-camera 3D correspondence 过滤；
4. weighted RANSAC、Procrustes refinement、inlier ratio 和 RMSE gate；
5. 与 HorizonStream 当前 relative pose 比较。中等 disagreement 标为 `low` 并降权，极端
   disagreement 直接 reject。

pose graph 固定第一帧 gauge，以 HorizonStream 相邻帧 relative pose 作为主体约束，并以
Huber loss 加入 object edge。优化有 per-frame rotation/translation correction bound，防止
单条错误 edge 把轨迹拉崩。没有通过 gate 的 edge 不会进入 graph。

## 输出

开启 refinement 后，`OUTPUT_DIR` 包含：

```text
raw_pose/                  # A0，原始 HorizonStream pose 的地图
object_pose_refined/       # B0，优化 pose 的地图
object_pose_refinement/
  candidate_edges.json
  accepted_edges.json
  rejected_edges.json
  raw_trajectory.txt
  refined_trajectory.txt
  refined_camera_to_world.pt
  pose_refinement_summary.json
pose_refinement_comparison.json
```

如果 `FUSION_POLICY="both"`，会按 policy 增加
`raw_raw_pose`、`raw_object_pose_refined`、`temporal_consensus_raw_pose` 和
`temporal_consensus_object_pose_refined` 四个地图目录；它们仍共享同一次 geometry/SAM
推理。

`pose_refinement_summary.json` 记录 tracked instance 数、候选/接受/拒绝 edge 数、匹配和
inlier 统计、registration RMSE、raw/refined pose change、拒绝原因和 optimizer 状态。
候选生成和优化不读取 GT；GT 只能在独立 evaluation 脚本中使用。

## 解释与限制

这不是 HorizonStream 的内部 loop closure，也不是 feature/token fusion。它是一个 post-
geometry baseline，用于回答“persistent object identity 能否提供额外的 long-range camera
registration constraint”。RGB patch backend 主要用于先验证数据流和 gate 是否成立；如果
accepted edge 很少，应先查看 `rejected_edges.json` 中是匹配不足、深度无效还是 pose
disagreement，而不是直接把阈值放宽。
