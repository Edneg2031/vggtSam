# streaming_couping

当前研究主线是 V8.0：先用 mesh-rasterized GT world pointmap 显式监督 SAM3.1/geometry
correspondence，再冻结匹配矩阵，用 evidence-only pose residual 验证未来帧位姿。V8.1 双
SAM/geometry Value 只作为独立 report-only 消融。

```bash
zsh streaming_couping/commands_v80_supervised_correspondence.txt
```

当前方法的唯一总说明见
[`docs/current_sam31_streamvggt_v80_method.md`](docs/current_sam31_streamvggt_v80_method.md)。

当前 V7.4 的 learned token-to-pose 路线没有通过 SAM 因果判据后，新增了一个不训练任何
pose head 的 V7.5 判别实验。它复用 V7.4 动态实例 cache，将 bounded instance ICP 显式
接回 V5 angular-Huber ray-centre solver，同时比较全图、bbox、同面积随机区域、过期 mask、
错误 ID 和 GT oracle：

```bash
zsh streaming_couping/commands_v75_explicit_pose_causality.txt
```

主结果为
`outputs/streaming_couping_v75_explicit_pose_causality/v75_explicit_pose_causality.csv`；
完整协议见
[`docs/v75_explicit_pose_causality.md`](docs/v75_explicit_pose_causality.md)。V7.5 只解析修正
camera center，rotation 保持 raw；它是 SAM mask/track 的历史因果诊断，不替代当前 V8 方法。

V7.4 使用 `90:15:525` 长序列和 short/medium/long 三个时间前缀 fold，同时比较纯几何、
SAM-off trained control 以及 identity/时间扰动。主结果和动态实例覆盖诊断分别写入：

```text
outputs/streaming_couping_v74_temporal_scaling/v74_temporal_scaling.csv
outputs/streaming_couping_v74_temporal_scaling/v74_dynamic_instance_diagnostics.csv
```

V4–V7.3 仍作为历史方法、消融和复现入口保留。早期三个版本的定位是：

```text
V4 coverage-first
  appearance-only + additive pose + current-refined ray solver
  研究主线：优先保持实例 mask 覆盖率，后续解决错检

V5 adaptive-best
  residual-only + bounded SO(3) + fixed-reference ray solver
  安全对照：按无 GT ray support 选择 raw/learned pointmap

V6 camera overfit
  三个完整 SE(3) 对照 + 七个专门化 3DoF head
  rotation/center 的 token 来源与坐标参数化 sweep
```

## 历史版本运行

V4：

```bash
zsh streaming_couping/commands_v4_coverage_first.txt
```

V5：

```bash
zsh streaming_couping/commands_v5_adaptive_best.txt
```

V6 五帧 camera 消融：

```bash
zsh streaming_couping/commands_v6_camera_overfit.txt
```

V6 30 帧完整历史压力测试（`90:15:525`）：

```bash
zsh streaming_couping/commands_v6_sam31_long30.txt
```

30 帧追踪可视化默认不生成。主命令完成并已有 feature cache 后，只运行一次下面的 CPU
后处理命令：

```bash
zsh streaming_couping/commands_v6_sam31_long30_tracking_visualization_once.txt
```

它不加载 SAM3.1、StreamVGGT 或相机模型，直接从 cache 生成
`RGB | GT | V6 SAM3.1 final tracking` 三联图。`tracking_success_summary.csv`
将非参考、GT 可见且 mask IoU ≥ 0.5 定义为追踪成功；GT 缺席帧的误检单独统计，
不会抬高成功率。重复运行发现 `COMPLETE` 标记时直接复用，不重复渲染。

V7 从轻到重的 fusion 消融拥有独立的 SAM3.1/StreamVGGT cache 和训练输出，不读取
任何 V6 配置或输出目录：

```bash
zsh streaming_couping/commands_v7_fusion_ablation.txt
```

首次运行会用两张卡执行完整历史 StreamVGGT、第三张卡执行 SAM3.1，并构建 train、
temporal holdout、validation、cross-clip 和 long30 所需的四段 cache；之后重复运行会
自动复用。可用 `V7_STREAM_GPU0`、`V7_STREAM_GPU1`、`V7_SAM_GPU` 选择三张物理卡，
或设置 `V7_REBUILD_CACHE=1` 强制从 backbone 重新生成全部 V7 cache。

它依次比较 camera-only、monolithic weighted pooling、当前式 monolithic
cross-attention、identity-Key/geometry-Value 解耦 attention，以及 32 个局部世界几何
token 的分层匹配。所有 train、temporal holdout、validation、cross-clip、long30 指标及
固定 checkpoint 输入扰动都写入一个
`outputs/streaming_couping_v7_fusion_ablation/v7_ablation.csv`。模型选择只读取
temporal holdout 与 validation；cross-clip 和 long30 只报告、不参与选择。CSV 已透视成
每个结构一行，最终只有 raw 加五种结构共六行，便于直接复制。

V7.1 用来回答“instance 是否在 camera 模型之外提供了因果增量”，而不是再次比较谁更
容易拟合五帧：

```bash
zsh streaming_couping/commands_v71_instance_causality.txt
```

它先用长序列早期 `105–255` 训练 L0 camera-only，然后冻结 L0；七种 residual 只用
`270–345` 训练，`360–405` 与独立 validation 用于选择，`420–525` 和 cross-clip
严格只报告。实例结构共享完全相同的 current-observation gate，并同时比较全帧额外
camera 容量、相同 gate 的额外 camera 容量、gate-only、appearance、geometry、解耦
全局 K/V 和 local32。唯一结果是
`outputs/streaming_couping_v71_instance_causality/v71_instance_causality.csv`，
共 raw、冻结 L0 和七种 residual 九行。
只有 instance-content 分支在 development、validation、future 和 cross 全部优于冻结
L0，同时优于额外 camera 容量控制，且 `instance_off` 精确回到冻结 L0，才会标记
`causal_instance_pass=1`。

周一正式占卡前先运行两步 smoke；它先校验/补齐 V7 cache，cache 完整时训练只在
logical `cuda:0` 分配模型：

```bash
zsh streaming_couping/commands_v71_smoke.txt
```

也可以直接运行周一的一键入口，它会先 smoke，成功后才进入可续跑正式实验：

```bash
zsh streaming_couping/commands_v71_monday.txt
```

完整的续跑、产物和故障说明见
`streaming_couping/docs/v71_monday_runbook.md`。

正式命令默认根据带配置签名的 `frozen_l0.pt` 和七个 residual checkpoint 自动续跑，
中断后重复同一命令即可；`V71_FRESH=1` 才会忽略续跑并覆盖训练。除九行主表外还输出：

- `v71_frame_diagnostics.csv`：每帧 raw/L0/refined 误差、有效实例数、可靠性和修正幅度；
- `run_metadata.json`：git commit、dirty 状态、随机种子、PyTorch/CUDA/GPU 和完整配置；
- `v71_result_summary.md`：根据容量控制、门控控制和扰动实验自动生成的结论。

单次结果有希望后再运行多随机种子稳定性实验，默认 seed 为 `0 1 2`：

```bash
zsh streaming_couping/commands_v71_multiseed.txt
```

可通过 `V71_SEEDS="0 1"` 缩减次数；聚合结果写入
`outputs/streaming_couping_v71_multiseed/v71_seed_aggregate.csv`。多 seed 不作为首次
排错命令，避免在明显无效的结构上浪费 GPU。

V7.2 进一步修正了 V7 `local32` 的证据含义：旧 `local32` 只有 mask 内的
StreamVGGT UVD/world geometry，并没有 SAM 的局部 descriptor。V7.2 会在现有 V7 cache
上增量保存 SAM3.1 `detector_fpn2` mask-local feature、UV 和 valid mask，不重新运行
StreamVGGT/追踪，然后对 K=8/16/32 做真正的局部 token 消融：

```bash
zsh streaming_couping/commands_v72_monday.txt
```

正式表同时包含 raw、冻结 L0、两个 camera-capacity control 和六类局部结构，共 22 行：

```text
outputs/streaming_couping_v72_local_token_ablation/v72_local_token_ablation.csv
```

`causal_local_pass=1` 要求 development、validation、future、cross 全部优于 L0，
future/cross 同时优于两个 camera control，并且 `instance_off` 在四个 split 精确回到 L0。
逐帧诊断、缓存审计、多 seed 聚合、论文 CSV/LaTeX、PNG 和打包命令也已保留。完整说明见
`streaming_couping/docs/v72_local_token_runbook.md`，服务器执行检查表见
`streaming_couping/docs/v7_server_experiment_checklist.md`。

V7.3 不再把 SAM3.1 descriptor 直接作为位姿 Value，而只让它决定参考帧到当前帧的局部
对应权重；被传输和送入相机 residual head 的 Value 始终是 StreamVGGT 局部几何。它从
uniform、纯几何、纯 SAM 权重到 SAM+几何权重做 K=8/16 的轻量消融，并自动加载 V7.1
精确 L0、V7.2 camera controls 和同 K 的 V7.2 纯几何强基线：

```bash
V73_GPU=1 zsh streaming_couping/commands_v73_monday.txt
```

主结果仍然只有一张可复制 CSV：

```text
outputs/streaming_couping_v73_correspondence/v73_correspondence_ablation.csv
```

只有 SAM 分支在四个 causal split 全部优于同 K 纯几何、report-only split 优于 camera
capacity controls、`instance_off` 精确回到 L0，并在 SAM-off、uniform、错误 identity 和
时间打乱下全部变差，才标记 `causal_sam_pass=1`。完整方法、命令和字段解释见
`streaming_couping/docs/v73_correspondence_runbook.md`。

单 seed 通过或接近因果判据后，可用
`V73_SEEDS="0 1 2" zsh streaming_couping/commands_v73_multiseed.txt` 检查稳定性；不要在
明显失败的结构上先消耗三倍训练时间。

### 历史 V6 长序列说明

长序列命令只占三张物理卡：StreamVGGT 的 24 层按连续区间分到两张卡，SAM3.1 单独使用
第三张卡。每层 KV cache 保留完整因果历史，不切块、不重置；DPT/camera/depth/point
输出逐帧卸载到 CPU。默认使用物理卡 `1,2,3`，可在命令前通过
`V6_STREAM_GPU0、V6_STREAM_GPU1、V6_SAM_GPU` 覆盖。该序列复用 frame 90 作为参考帧，
用于验证 30 帧容量和长历史稳定性，不作为未见序列泛化证据。

V6 复用冻结 cache，不训练 SAM3/StreamVGGT，不训练 pointmap，也不运行 ray solver。
一次命令会用相同 seed 和步数训练 `camera_only、instance_only、fusion` 三套 6DoF 对照，再用
fusion checkpoint 评估 `instance_off、camera_off、shuffle_time、wrong_geometry、
appearance_only、geometry_only`。V6.3 另外训练 camera/instance/fusion 三种 rotation head、
三种 camera-frame center head 和 direct-world instance center 对照，再形成 3×3 合法组合。
所有专门 head 都只有 3DoF 输出权限。所有 checkpoint 测试
`210 240` 和 `492 512 520 545 561 589` 时均不重新训练、不运行 solver。第二段已参与 V6.2
设计，只能作为 development evidence；最终结论需要第三段预先锁定的序列。只需复制短表：

```text
outputs/streaming_couping_v6_camera_overfit/v6_component_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_v63_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_v64_aux_sweep.csv
outputs/streaming_couping_v6_camera_overfit/v6_validation_summary.csv
```

全部训练与依赖消融仍保存在同目录的 `v6_summary.csv`。
命令还会打印 `v6_frame_diagnostics.csv`：七行 held-out/cross-clip 机制诊断，delta 相对 raw，
负数表示改善。该表只关联推理时可见的实例支持与事后 GT 误差，不用于逐帧选模型或调阈值。

正式候选比较使用预先锁定的第三段 `50 60 70 80`，并在训练区间从 90 开始前截止，避免
精确及近邻训练视角泄漏。该段不重新训练，只输出 raw、3DoF local-center 候选 A 和
`λ_rotation=0.1` 候选 B 三行。为与训练段保持同一初始化条件，第三段从参考帧 GT instance
mask 中读取训练使用的 `37/68/54`；哪个在第 50 帧可见就初始化哪个，不要求三者齐全，缺失
ID 保留为空槽。后续帧只要仍有一个可用 persistent instance 就运行；全部失效时两个候选在
该帧严格回退 raw。这里使用 GT reference mask，但 GT pose 仍只参与最终指标计算。

V4/V5 两条命令继续复用原始 SAM3 cache；V6 使用独立的 SAM3.1 cache，并从相同 seed
训练十三个 camera/instance 消融模型。两代 cache、checkpoint 和输出目录完全隔离。

## 历史 V4/V5/V6 入口

```text
configs/v4_coverage_first.yaml
configs/v5_adaptive_best.yaml
configs/v6_camera_overfit.yaml
configs/v6_camera_sweep_base.yaml
configs/v6_sam31_long30_base.yaml
configs/v6_sam31_long30_camera.yaml
configs/recovery_sam31_050_025.yaml
scripts/run_instance_token_pose.py
scripts/run_v5_reference_pose.py
scripts/run_v5_adaptive.py
scripts/run_v6_camera_overfit.py
scripts/plot_pose_comparison.py
```

V4/V5 最终输出包括 native pointcloud/pose、GT/raw/ours 对比 PLY、三套 mask、位姿 PNG/PDF、
指标 CSV、checkpoint 哈希和 artifact manifest。

旧 V4–V7.5 的命令、配置和版本 runbook 仍可用于复现实验，但不再代表当前架构。当前 V8
的完整数据流、损失与验证协议统一记录在
[`docs/current_sam31_streamvggt_v80_method.md`](docs/current_sam31_streamvggt_v80_method.md)。

历史 V6 cache 的数据流是：

```text
SAM3.1 raw tracking
  → StreamVGGT 参考物体投影生成 box + positive points
  → adaptive_positive_compete_010
  → SAM3.1 mask 内池化 detector_fpn2
  → 实例几何、instance token、camera/pointmap 实验
```
