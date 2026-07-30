# Long-sequence external baselines

这个目录只负责会议室长序列的 **HorizonStream / StreamVGGT 原始基线**，不依赖
SAM3，也不修改 `streaming_couping`。两种模型顺序运行，因此最多同时占五张卡。

## 为什么 StreamVGGT 要五卡

StreamVGGT 是逐帧因果推理，但官方实现会保留每一层的全部历史 KV：帧数增加时显存
线性增长。不能用 DDP 把 600 帧拆给五张卡，因为后面的帧依赖前面的状态。这里采用
层间模型并行：24 层固定分为 5/5/5/5/4，每层 KV 永久留在对应 GPU，分区边界只传当前帧
token。历史从不重置，因此保持官方 full-history 语义。

官方缓存保存RoPE前的K，因此每个新帧都会对全部历史K重复执行QK LayerNorm与RoPE，
在280帧时产生约8 GiB显存碎片并OOM。本runner缓存已经完成这两个逐token操作的K；
由于LayerNorm和空间RoPE都不跨token，这与先拼接再计算在数学上等价，却只需处理当前
帧的新K。runner启动时会用三帧小型Attention自动比较两种实现，只有数值一致才继续。

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

确认两个 runner 都完成后，运行前 300 帧的正式公平对比：

```bash
zsh long_sequence_baselines/commands_meeting_room_full.txt
```

也可以分开运行，便于共享GPU调度或单独重试失败的模型：

```bash
zsh long_sequence_baselines/commands_horizonstream_24gb_full.txt
zsh long_sequence_baselines/commands_streamvggt_5gpu_full.txt
```

HorizonStream 使用一张卡；在当前24GB GPU上，`sliding_size=21` 实测需要超过23.69
GiB，因此正式命令固定使用 `sliding_size=10`。StreamVGGT 随后使用五张卡，二者不会
同时运行。StreamVGGT 的 full-history KV 和 attention 临时张量仍随帧数增长，544 帧运行在
约第 410 帧 OOM；因此三个正式命令都固定对自然排序后的同一批前 `300` 张图片推理。
300 帧实测 GPU0 常驻峰值约 `13.53 GiB`，为 attention 临时分配保留了足够余量。

## 输出

300 帧正式结果位于新的独立目录，避免和先前 544 帧 HorizonStream 结果或失败的
StreamVGGT 残留文件混合：

```text
outputs/long_sequence_baselines/frames_300/
  horizonstream/meeting_room_a02/
  streamvggt/meeting_room_a02/
```

每个方法都输出：

- `poses/abs_pose.txt`：逐帧 world-to-camera 位姿；
- `poses/intri.txt`：内参；
- `plots/trajectory*.png`：无 GT 的预测轨迹图；
- `depth/dpt/*.npy`、`depth/dpt_plasma/*.png`、`depth/conf/*.npy`；
- `images/rgb/*.png`：模型实际看到的预处理图；
- `points/depthpose_raw.ply`：统一的 depth + online pose 原始累积，最多 200 万点；
- `points/depthpose_conf.ply`：每帧去掉最低 50% confidence，并保留 1%–99% 深度；
- `points/depthpose_conf_voxel.ply`：相同过滤后做置信度加权 voxel 融合，只保留至少
  两帧观测的 voxel；
- `points/depthpose_summary.json`：实际 voxel 尺寸、点数和平均观测次数；
- `run_summary.json`：帧数、运行时间和峰值显存。

两种方法的 `depthpose_*` 完全使用同一份实现和参数。voxel 尺寸按该模型整段有效深度
中位数的 `1%` 自动确定，因此不会把两个独立尺度的结果强行使用同一个绝对米制阈值。
voxel 中的位置和颜色按归一化 confidence 加权平均，并统计来自多少个不同帧；这不是
简单保留第一个点。

StreamVGGT 额外保存：

- `pointhead_raw.ply`：直接回归的共同坐标 pointmap；
- `pointhead_conf.ply`：point confidence 前 50%；
- `pointhead_conf_voxel.ply`：point-head 结果使用同一加权 voxel 协议；
- `full_all.ply` / `full.ply`：为兼容旧结果，分别保留相同的 point-head raw/conf 内容。

因此比较 `depthpose_*` 可以隔离两种模型的 depth+pose 能力；比较 StreamVGGT 的
`pointhead_*` 与自身 `depthpose_*`，可以判断收益是否来自 world point head。所有结果仍
保留 `runtime_per_frame.csv`，可检查五卡显存是否按预期线性增长。

## 如何比较

这批数据目前没有 GT，所以不能报告绝对 ATE、深度误差或点云精度。可以在
CloudCompare/MeshLab 应先比较两个 `depthpose_conf_voxel.ply`，重点看墙面重影、平面
弯曲、长程漂移和回访位置是否一致；再查看 `depthpose_raw.ply`，确认干净结果不是后处理
掩盖了模型问题。StreamVGGT 的 `pointhead_conf_voxel.ply` 单独表示 point head 能力，不应
直接拿它和 HorizonStream 的 raw depth 点云归因于位姿优劣。两个模型的输出坐标系和尺度
各自独立，不能直接把两份 PLY 叠加后把坐标差当成误差。后续有 GT 时再做统一 Sim(3)
对齐和定量评估。
