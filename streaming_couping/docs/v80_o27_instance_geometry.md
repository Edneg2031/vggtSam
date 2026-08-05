# V8 O2.7：固定相机内参与 SAM 实例区域深度诊断

## 目的

O2.6 已确认 predicted focal length 逐帧漂移，同时全图 affine depth 在 medium
的 435 帧仍产生错误旋转。本实验只回答两个问题：

1. 固定相机内参是否能用首帧预测或因果中位数稳定；
2. 只在 SAM3.1 实例 mask 内拟合 depth affine，是否能修复 435 帧。

## 固定内容

- GT-world pseudo-correspondence；
- GT history pose；
- `K=64 / history=2 / mutual / 0.10m`；
- weighted/trimmed SE(3) Kabsch；
- short、medium、long 三个既有 fold；
- 不训练、不加载 pose model。

## 消融轴

Depth：

- raw predicted depth；
- full-image GT-oracle affine depth；
- SAM-instance-region GT-oracle affine depth。

Intrinsics：

- current per-frame predicted K；
- frame-90 reference predicted K；
- causal median predicted K；
- GT K oracle。

首帧 K 和因果中位数 K 的选择策略本身可部署，但整条 O2.7 分支仍使用
GT-world pseudo-match 与 GT history，并不是最终推理方法。GT K 和所有 affine
depth 分支还会直接读取 GT，只用于判断能力上界。

## 运行

```bash
zsh streaming_couping/commands_v80_instance_geometry.txt
```

命令会复用 O2.6 的 30 帧 cache；缺失时才使用 GPU 1、2、3 自动重建。

终端会直接打印两张可复制表：

- `v80_o27_decision.csv`：12 个组合的跨 fold 结论；
- `v80_o27_medium_diagnosis.csv`：medium 四帧的六个关键分支。
