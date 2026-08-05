# V8 O2.6：预测几何误差分解

## 目的

O2.5 已证明实例支持、坐标变换和显式 Kabsch 求解器能够在 GT camera
geometry 上稳定改善位姿，但 StreamVGGT depth geometry 无法跨 short、medium、
long 三个 fold 通过。本实验不训练模型，也不继续搜索支持超参数，而是隔离：

1. 深度反投影是否正确；
2. predicted intrinsics 是否构成主要误差；
3. predicted depth 是否构成主要误差；
4. 深度误差能否由逐帧 scale 或 affine `a*z+b` 解释；
5. Sim(3)/Umeyama 是否明显优于刚性 SE(3)/Kabsch。

## 固定协议

- 30 帧：90 到 525，每隔 15 帧；
- support：`K=64 / history=2 / mutual / 0.10m`；
- correspondence：GT-world pseudo-match；
- history pose：GT history；
- fold：沿用 short、medium、long，每个 fold 测试四帧；
- 不训练 SAM3.1、StreamVGGT 或 pose head。

## 分支

| Depth | Intrinsics | 含义 |
|---|---|---|
| direct GT camera | 无 | O1 solver/support 回归检查 |
| GT depth | GT | 验证采样和反投影 |
| GT depth | predicted | 隔离 intrinsics |
| predicted | GT | 隔离 depth |
| predicted | predicted | 当前 O2 |
| oracle scale calibrated | GT/predicted | 判断逐帧 scale drift |
| oracle affine calibrated | GT/predicted | 判断 scale + offset |

每个分支同时运行 weighted/trimmed SE(3) Kabsch 和 weighted/trimmed Sim(3)
Umeyama。Sim(3) 的尺度单独输出，不会写入 SE(3) 相机旋转矩阵。

scale/affine 校准读取同帧 GT depth，只是诊断上界，不能作为最终推理方法。

## 一键运行

```bash
zsh streaming_couping/commands_v80_geometry_factorization.txt
```

若 30 帧 cache 已删除，命令会使用 GPU 1、2、3 自动重建；cache 存在且审计通过
时不会加载两个 backbone。

## 复制结果

优先复制：

```bash
cat outputs/streaming_couping_v80_geometry_factorization/v80_o26_decision.csv
cat outputs/streaming_couping_v80_geometry_factorization/v80_o26_medium_diagnosis.csv
cat outputs/streaming_couping_v80_geometry_factorization/v80_o26_decision.md
```

`v80_o26_medium_diagnosis.csv` 会由一键命令直接打印到终端，固定包含 medium
的 420/435/450/465 四帧以及 GT-depth/predicted-K、predicted-depth/GT-K、
当前 O2、affine-depth/GT-K 四个关键分支，无需在服务器上另写筛选命令。

如果需要完整的每个 fold、solver 指标，再复制：

```bash
cat outputs/streaming_couping_v80_geometry_factorization/v80_o26_geometry_factorization.csv
```

需要逐帧定位时再复制：

```bash
cat outputs/streaming_couping_v80_geometry_factorization/v80_o26_depth_intrinsics_diagnostics.csv
```
