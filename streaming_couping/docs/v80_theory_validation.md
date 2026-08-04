# V8 理论验证：坐标审计与 O1/O2 显式位姿求解

V8 matcher-first 实验没有通过三段时间 fold 的 SAM 因果判据。所有 pose head 都能把训练损失
压到接近零，但 SAM+geometry 在 held-out matching 和 pose 上不稳定，long fold 的所有 learned
refinement 还差于 frozen L0。因此本实验不增加 fusion 容量，也不训练任何网络，只回答：

1. 坐标系、尺度和位姿组合是否正确；
2. 给定 GT-world 几何近邻伪对应时，GT 几何能否恢复位姿（O1）；
3. 给定同一对应时，StreamVGGT 预测深度是否足以改善位姿（O2）。

## 一键运行

```bash
zsh streaming_couping/commands_v80_theory_validation.txt
```

它只读取：

```text
outputs/streaming_couping_v74_temporal_scaling/cache/
  00a231a370_90_525_step15_37_68_54.pt
```

不会加载 SAM3.1、StreamVGGT 或 V8 checkpoint，也不占 GPU。cache 缺失时命令会明确退出，
不会隐式重建 backbone 输出。

## 坐标约定

缓存中的 `baseline_world_points` 是 StreamVGGT 共享预测世界坐标，不能直接对当前/历史点做
Kabsch 来求相机位姿，否则正确变换接近单位阵。

本实验使用：

```text
StreamVGGT camera point
  = baseline_depth + predicted intrinsics 反投影

GT camera point
  = target_world_points 经全局 GT W2C 变换
  / reference-frame Sim(3) scale
```

两者因此都使用 StreamVGGT native translation scale。历史 camera point 通过 GT-history 或
L0-history 的 C2W 放入共同 reference gauge，Kabsch 求：

```math
X_{gauge}\approx R_{c2w,t}x_{camera,t}+t_{c2w,t}
```

求得当前 C2W 后取逆得到 W2C。最终 bounded correction 与 target pose 都在 frame 90 gauge、
StreamVGGT native scale 下比较。

## Correspondence 的边界

当前标签是同一 persistent slot 内：

```text
GT-world 3D nearest neighbour
+ 0.10 m 半径
+ mutual nearest neighbour
```

它只是几何邻近伪对应，不保证是同一材质表面点。因此 CSV 重点报告距离、inlier、拟合残差和
最终 pose；不能只用 Top-1 index 声称真实物理对应成立。

Kabsch 权重为 `exp(-d_gt / 0.025m)`，因此半径边缘的伪对应不会与近距离对应拥有相同影响。

历史帧严格复现 V8 causal memory：当前帧先读取上一次写入，再在当前观测后写回。不同实例可以
读取不同历史帧；每个历史 camera point 先通过对应历史 C2W 放到共同 gauge，再一起求当前位姿。

## 三个 Oracle 分支

| 分支 | Correspondence | 几何 | 历史 pose | 含义 |
|---|---|---|---|---|
| O1 | GT-world 伪对应 | GT camera point | GT | solver/convention 上界 |
| O2-GT | GT-world 伪对应 | StreamVGGT depth camera point | GT | 隔离当前预测几何能力 |
| O2-L0 | GT-world 伪对应 | StreamVGGT depth camera point | L0 | 加入历史漂移后的实际能力 |

每个分支同时运行：

- weighted Kabsch；
- deterministic 70% trimmed Kabsch。

求解器支持权重、invalid mask、reflection correction、batch、二阶空间退化检查。平面但非共线
的支持允许求解；共线或点状分布会回退。以下任一条件失败时，最终 pose 逐元素回退 L0：

- correspondence 少于 6；
- 二阶 covariance eigenvalue ratio 过低；
- 点集空间 extent 过小；
- fit RMSE 超限；
- correction rotation/center 超过配置上限；
- 结果非有限。

CSV 同时保存未门控 `proposed_*` 和门控后的 `refined_*`，可以区分“solver 算错”和“候选正确
但被 bounded gate 拒绝”。如果 solver 因空对应、非有限输入或退化几何而没有产生有效 SE(3)，
`proposed_*` 明确写为 `nan`，不会用内部单位变换伪造候选指标；`refined_*` 仍逐元素等于 L0。

## 输出

```text
outputs/streaming_couping_v80_theory_validation/
├── v80_coordinate_audit.csv
├── v80_oracle_pose_pairs.csv
├── v80_oracle_pose_summary.csv
├── v80_theory_decision.md
└── v80_theory_metadata.json
```

主命令会直接打印 summary CSV。需要进一步定位时再复制 pair CSV。

## 判定顺序

```text
坐标审计失败
  → 修 convention，不讨论 SAM

O1 失败
  → 修 solver / correspondence label / pose composition

O1 成功，O2-GT 失败
  → StreamVGGT depth geometry 不足以支持该修正

O2-GT 成功，O2-L0 失败
  → 历史 pose 漂移是主要瓶颈

O2-GT 与 O2-L0 都成功
  → 才进入 O3/O4 predicted correspondence 和 SAM mask/ID/descriptor 分解
```

本实验仍不证明跨场景泛化，也不直接优化单帧 depth/pointmap。
