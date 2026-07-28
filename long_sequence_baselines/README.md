# Long-sequence external baselines

这个目录只负责会议室长序列的 **HorizonStream / StreamVGGT 原始基线**，不依赖
SAM3，也不修改 `streaming_couping`。两种模型顺序运行，因此最多同时占五张卡。

## 为什么 StreamVGGT 要五卡

StreamVGGT 是逐帧因果推理，但官方实现会保留每一层的全部历史 KV：帧数增加时显存
线性增长。不能用 DDP 把 600 帧拆给五张卡，因为后面的帧依赖前面的状态。这里采用
层间模型并行：24 层固定分为 5/5/5/5/4，每层 KV 永久留在对应 GPU，分区边界只传当前帧
token。历史从不重置，因此保持官方 full-history 语义。

```text
logical cuda:0  patch embed + aggregator 0..4   + depth head
logical cuda:1                aggregator 5..9   + point head
logical cuda:2                aggregator 10..14
logical cuda:3                aggregator 15..19
logical cuda:4                aggregator 20..23 + camera head
```

运行前用 `nvidia-smi` 选择五张空闲物理卡，再编辑命令文件顶部的 `GPU0` 到 `GPU4`。
脚本有安全检查：StreamVGGT 必须恰好只看到五张卡，HorizonStream 必须只看到一张卡，
避免误占共享 GPU。

服务器的 Conda 入口会错误地指向 `/root/anaconda3`，因此命令文件不使用 `conda run`
或 `conda activate`，而是直接调用以下两个真实环境的 Python：

```text
/home/huawei/miniconda3/envs/horizonstream/bin/python
/home/huawei/miniconda3/envs/3am/bin/python
```

这样无论执行脚本前提示符显示 `(horizonstream)` 还是 `(3am)`，两个阶段都会进入正确环境。

## 推荐运行顺序

先在服务器同步代码并初始化 external：

```bash
cd /home/wlh50060092/vggtSam
git pull --ff-only origin main
git submodule update --init --recursive
```

先执行 20 帧 smoke test：

```bash
zsh long_sequence_baselines/commands_meeting_room_smoke.txt
```

如果 HorizonStream smoke 已经成功，只需重新验证五卡 StreamVGGT：

```bash
zsh long_sequence_baselines/commands_streamvggt_5gpu_smoke.txt
```

确认两个 runner 都完成、显存仍有余量后，再运行约 600 帧正式版本：

```bash
zsh long_sequence_baselines/commands_meeting_room_full.txt
```

HorizonStream 使用一张卡和官方 `sliding_size=21`；StreamVGGT 随后使用五张卡。二者
不会同时运行。三卡 smoke 实测推算600帧需要约26–30 GiB/卡；五卡分片后预计降至
约16–19 GiB/卡，适合当前24GB GPU，但正式运行仍需观察 `runtime_per_frame.csv`，因为
StreamVGGT KV 显存还会随帧数增长。

## 输出

正式结果位于：

```text
outputs/long_sequence_baselines/full/
  horizonstream/meeting_room_a02/
  streamvggt/meeting_room_a02/
```

每个方法都输出：

- `poses/abs_pose.txt`：逐帧 world-to-camera 位姿；
- `poses/intri.txt`：内参；
- `plots/trajectory*.png`：无 GT 的预测轨迹图；
- `depth/dpt/*.npy`、`depth/dpt_plasma/*.png`、`depth/conf/*.npy`；
- `images/rgb/*.png`：模型实际看到的预处理图；
- `points/full.ply`：用于观察的点云，最多 200 万点；
- `run_summary.json`：帧数、运行时间和峰值显存。

StreamVGGT 额外保存 `points/full_all.ply`（不按置信度过滤）；`full.ply` 默认保留每帧
置信度前 50% 的点。这样既能看完全公平的原始输出，也能看官方常用置信度过滤后的
可视化。它还保存 `runtime_per_frame.csv`，可检查五卡显存是否按预期线性增长。

## 如何比较

这批数据目前没有 GT，所以不能报告绝对 ATE、深度误差或点云精度。可以在
CloudCompare/MeshLab 分别查看两个 `full.ply`，重点看墙面重影、平面弯曲、长程漂移和
回访位置是否一致；再对照轨迹图、速度和峰值显存。两个模型的输出坐标系和尺度各自
独立，不能直接把两份 PLY 叠加后把坐标差当成误差。后续有 GT 时再做统一 Sim(3)
对齐和定量评估。
