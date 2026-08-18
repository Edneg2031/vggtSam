# T0.1：三角化长尾可靠性诊断

T0.1 只读取 T0 已冻结的 `correct_persistent_id` anchors，判断固定的非 GT
几何 gate 能否消除三角化长尾。它不会重新匹配、运行模型、修改 pose、修改
pointmap 或覆盖 V0/T0。

候选阶段使用 T0 artifact 已保存的 mean reprojection error、最大射线夹角、
condition number 和视图数。由于 T0 没有保存逐视图残差，本实验不把 mean
reprojection 冒充 RMS；positive depth 则由 T0 anchor 接受条件保证。

两个 gate 在读取 T0 的 GT-derived score CSV 前写入带签名的冻结 artifact：

```text
primary: positive depth, mean reprojection <= 2 px,
         max ray angle >= 2 deg, condition <= 1000, views >= 2
strict:  primary + views >= 3
```

只有当预先冻结的 gate 同时满足 RMSE、P90、改善比例、至少 30 个 anchor 和
至少 10% 保留率时才判定 GO。

运行：

```bash
zsh streaming_couping/commands_t01_triangulation_reliability.txt
```
