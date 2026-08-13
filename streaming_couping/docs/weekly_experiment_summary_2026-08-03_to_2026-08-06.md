# StreamVGGT + SAM3.1：V4–V9.8 证据与 V0 理论重启

> 更新：2026-08-13
> 目的：只记录已经证实、已经证伪和下一条路线的理论边界。历史候选实现已删除，结果不因删代码而删除。

## 1. 当前决定

- 正式 baseline 仍叫 **V0**：SAM3.1 做因果多实例发现、mask、永久 ID 和未来 birth；
  StreamVGGT 几何辅助 mask；selected pose 与 raw StreamVGGT 完全相同。
- 不再把 SAM appearance/local/memory token 接到 camera token 或直接 SE(3) head。
- TrackHead-BA、SIFT/Feature-PnP、ALIKED-LightGlue-PnP 的候选代码、命令和配置已经清理。
- E0/E1 已证伪固定 depth 的 edge-DTF 完整 SE(3) 优化：梯度实现正确，但 E0 小扰动恢复向 trust-region
  边界发散，E1 也没有任何 joint/translation 分支三折通过。
- 下一条 pose 路线直接借鉴成熟 RGB-D odometry：**SAM prompted-object exclusion + StreamVGGT depth/K +
  multi-history projective association + point-to-plane ICP/LM**。它作为 G0 独立候选；通过前不能写入 V0。

## 2. 当前证据的数据边界

主要数据只有 ScanNet++ scene `00a231a370` 的一个长 clip：frame 90–525、step 15、30 帧。
未来测试固定为：

| fold | 测试帧 |
|---|---|
| short | 360、375、390、405 |
| medium | 420、435、450、465 |
| long | 480、495、510、525 |

这个 clip 是静态室内扫描。物体在晚帧首次进入视野属于 **late discovery**，不等于物体真实运动。
所以它能验证因果 birth 和静态场景 pose，不能验证“dynamic mask 排除运动物体”是否有效。任何 dynamic-mask
pose 结论必须增加真实动态、带 camera GT 的序列；当前结果也不能称为跨场景泛化。

## 3. 已经证实

1. V7.4/V0 的 forward-only 多实例发现、永久 slot 和未来 birth 已实现；本 clip 有一条 track 在
   frame 150 才出生，不要求第一帧出现全部物体。
2. V4/V5 的完整 learned pipeline 在固定参考实例、同场景 clip 上出现过 aggregate pose 收益，但使用
   reference-frame GT instance 初始化，且没有隔离 SAM token；只能作为旧 E2 结果，不能恢复为新 baseline。
3. GT 3D geometry + Kabsch、GT depth + GT K，以及正确且宽空间分布的 2D correspondence，均能修正
   StreamVGGT pose。这证明 pose **存在可修正空间**，solver 也不是根本不可用。
4. V9.5 中 full-image / instance-background 混合 correspondence 在 0.5–1 px 噪声下通过；V9.6 中
   72×72 grid 的 bilinear K4 能表达连续坐标。宽空间支撑和亚像素坐标是必要条件。
5. SIFT + native StreamVGGT world pointmap + all-causal PnP 在 short fold 有局部 center 收益，说明显式
   2D→3D→pose 链路能启动；但它没有通过全部 fold。

## 4. 已经证伪或应停止重复

- Adapter/pose head 把训练 loss 拟合到零，不证明 SAM token 有用；camera-only、geometry-only、time-only
  或 trained-off 控制同样能拟合。
- pooled token、mask-local `detector_fpn2`、dense detector-FPN、SAM3.1 memory-read feature 均未在三组
  future fold 建立 correspondence/pose 因果收益。V9.7/V9.8 的 future PCK@1 基本为 0。
- history key 从 32 增到 256、mutual/unique assignment、instance-only essential matrix 和多种 robust
  solver 都没有修复 pose。localized-instance support 对亚像素误差过于脆弱。
- predicted depth/K 的三维误差不是单一 scale、shift、affine 或 Sim(3) 可以解决。
- V0 r3 的 direct pose head 只在 short/long 改善，medium center 下降 26.28%，all-fold fail；现已删除。
- TrackHead-BA 的一帧正控制中 visibility/confidence gate 为 0/3072；它没有进入 BA，说明当前 checkpoint/
  预处理契约不可用，不应靠把阈值降到接近零继续调。
- SIFT/Feature-PnP 的预锁主分支只在 short 启动 4/4，medium 1/4，long 0/4；short 仍有 2/4 坏帧。
  all-causal history 提高了 coverage，但没有形成稳定 pose 改善。
- ALIKED-LightGlue 下载/依赖阶段曾中断，没有形成完整 pose 结果；因此删除它是停止该候选，不是声称
  已经科学证伪 LightGlue 本身。

## 5. 四篇论文实际支持什么

### EPO（arXiv:2607.00579）

EPO 确实提供了最直接的新路线：Canny edge、target distance-transform field、foundation-model depth 和
双向 edge reprojection；用 truncated distance 与 Huber loss，不需要显式 feature track。

但论文的成功条件比“固定 depth，只改当前 pose”更强：

- 输入是整组图像的 `K / pose / depth`，先用双向 cycle reprojection 建 connected viewgraph；
- 第一阶段共同优化所有 camera pose 和 focal length；
- 收敛后再放开逐像素 depth scale/shift，与 camera 联合优化；
- 全 viewgraph 的双向损失提供隐式 multi-view regularization；它不是单边、单帧 streaming optimizer。

因此 EPO 证明“edge objective 可用于 3DFM 后处理”，**没有证明固定有偏 StreamVGGT depth 的
current-pose-only 子问题一定成功**。论文也明确列出高频非结构纹理/反射和低连接 viewgraph 两个限制。

### 4D3R（arXiv:2511.05229）

4D3R 先用 motion model 产生 coarse dynamic cue，再把 top-K 点作为 SAM2 prompt 得到精细 dynamic
mask；mask 用于 static point selection，随后执行 masked PnP-RANSAC 和 differentiable dense BA。
DBA 联合更新 pose 与 depth，并使用 optical flow/confidence。它支持“SAM mask 应作为几何 residual 的
选择器”，不支持“仅有 SAM mask + frozen depth edge 就足够”。

### 4DVGGT-D（arXiv:2605.12027）

它不是外部 edge solver，而是从 VGGT attention 得到 dynamic saliency，在第二次 VGGT forward 的早期层
把 dynamic token 的 key 置零，再使用第二次输出的 pose；之后对两次 depth 做 mask/confidence fusion。
论文报告 ATE 改善，但 RTE/RRE 反而恶化，说明“去动态能改善全局基准”不等于逐帧 relative pose
单调改善。它给我们的启发是先稳定 pose、再处理 geometry，以及必须同时报告绝对和相对指标。

### Selfi（arXiv:2512.08930）

Selfi 在 VGGT image tokens 上训练 DPT adapter，用 VGGT depth/pose 生成带 visibility check 的伪
correspondence；下游仍然是 feature matching + dense BA，并在 pose/K 改变后对 depth 做 affine shift。
它在大规模跨场景数据上训练约 150K steps，而不是单 scene adapter。该路线比 SAM token→camera token
更有几何依据，但成本高、仍有 teacher bias；只有 edge 路线的上界成立而真实 edge 失败时才值得进入。

论文：

- [EPO](https://arxiv.org/abs/2607.00579)
- [4D3R](https://arxiv.org/abs/2511.05229)
- [4DVGGT-D](https://arxiv.org/abs/2605.12027)
- [Selfi](https://arxiv.org/abs/2512.08930)

## 6. SAM static mask + edge reprojection 的理论模型

对 source edge pixel `u_s`，使用 source depth 和内参反投影：

```text
X_s = D_s(u_s) K_s^-1 u_s
u_t(ξ) = project(K_t, T_t(ξ) T_s^-1 X_s)
```

令 `Φ_t^static` 为 target **静态 edge** 的 distance-transform field，基本 residual 是：

```text
E_s→t(ξ) = Σ w(u_s) · Huber(min(Φ_t^static(u_t(ξ)), λ))
E_pair = E_s→t + E_t→s
```

其中 `w` 必须同时包含：source static gate、target static gate、StreamVGGT depth/point confidence、正深度、
图内、双向 depth/visibility consistency。还需要 raw-pose prior、bounded SE(3) update、coarse-to-fine 和
固定 gauge。

它可能改善 pose 的原因不是 SAM 提供了 pose 信息，而是：

1. edge distance field 在 edge normal 方向提供连续、可微、亚像素 image-space gradient；
2. 多视图重投影把该梯度通过投影 Jacobian 传给 SE(3)；
3. SAM 若正确排除 independently moving edge，会删除违反静态场景模型的系统残差。

它无法保证成功的原因是：

- distance transform 是 nearest-edge many-to-one，没有 edge identity；重复边、平行线和密集纹理可让错误
  pose 也得到低 loss；
- edge 只沿法线约束，长直线存在 aperture/退化方向；camera center 通常比 rotation 更难稳定；
- depth、K、pose 在投影里耦合。固定错误 depth 时，optimizer 可能用错误 translation 吸收 depth bias；
- 几何轮廓正是常见 edge，也是 depth 最容易在前/背景间混合的位置；mask 边界需要单独的
  erosion/dilation band 和 depth-consistency gate，不能默认轮廓 depth 可信；
- 遮挡、出视野、illumination edge、反射和动态阴影会产生错误梯度；
- raw pose 已较准时，最近错误 edge 可能比真实 edge 更近，目标高度非凸；
- current-pose-only 固定历史会把历史 pose/depth 的误差当成当前 pose 的误差，弱于 EPO 的全 viewgraph
  联合优化。

结论：**方案有明确几何机制，值得做 feasibility；但原始“固定一切、只改当前 SE(3)”不能直接升级成
baseline，应先验证 pose-only 是否具有上界，并准备一个低维 depth affine 对照。**

## 7. 当前 SAM mask 的真实含义

当前配置只 prompt `bed` 和 `wardrobe`。SAM3.1 会发现这些概念的多个实例并支持晚 birth，但它不是
class-agnostic motion segmentation。并且家具的 semantic mask 不等于 dynamic mask：床可以静止，未 prompt
的人可以运动。

当前 `static_score` 来自 StreamVGGT geometry registration/consensus，也依赖 raw pose/pointmap；它可作为
部署 cue，却不是独立运动真值。新实验中的 `sam_static_edge` 应严格定义为：

```text
全图 edge
− 已发现且被 geometry/temporal gate 判为 dynamic 或 identity-unknown 的 SAM mask 内 edge
```

不能只采样“静态实例内部 edge”，否则会重现 V9 已证伪的局部支撑问题。新物体首次发现前仍会污染
edge residual，这是部署限制，必须逐帧记录 `discovered_dynamic_area` 和 `unknown_area`。

## 8. 当前代码能否承载该实验

V0 cache 已经包含 `image_paths/stream_images`、raw pose encoding、predicted depth、depth confidence、native
world pointmap、point confidence，以及 output/StreamVGGT 两套坐标下的 SAM/trusted/associated masks。内参可由
raw pose encoding 用现有 StreamVGGT decoder 取出，因此 E0 不需要重新缓存任何 SAM token。

工程上仍要先补三项审计：

1. edge、depth、mask、K 必须全部落在同一个 processed pixel-center convention；
2. source/target 双向投影在 raw pose 下必须通过正深度、图内和 depth-cycle sanity check；
3. cache 中虽然为评测保留了 `target_*` GT 字段，optimizer 模块必须接收不含这些字段的 deployable view，
   全部分支写完后再由独立 scorer 读取 GT，防止无意泄漏。

## 9. 推荐的验证顺序

### E0：先回答 edge objective 是否有 pose 信息

在当前 ScanNet++ clip 上做离线、因果 prefix 诊断，不写入 V0：

1. `raw`：不优化；
2. `all_edge`：所有 edge；
3. `sam_static_edge`：只排除部署时可知的 dynamic/unknown SAM 区域；
4. `shuffled_static_mask`：把同帧 dynamic mask 做确定性平移后再取补集，保持面积和形状；
5. 诊断上界（不算部署方法）：GT depth、GT pose 小扰动恢复、GT static mask。

四个正式分支必须锁定相同 edge 数、viewgraph、pair/window、optimizer、update bound 和初始化。
`shuffled_static_mask` 不能随机打散每个像素，否则面积/边界统计不公平。

必须额外比较两种优化变量：

- `pose_only`：固定 K/depth，只优化 causal window pose；
- `pose_plus_depth_affine`：每帧仅增加 `aD+b`，而不是直接放开逐像素 depth。

如果 GT-pose perturbation 都不能被 edge loss拉回，是实现/目标问题；如果 GT depth 能过而 predicted depth
不能过，瓶颈是 geometry；如果 affine 能过而 pose-only 不能过，说明原方案自由度不足；如果 all-edge
能过而 SAM/static/shuffle无差异，只能称 EPO 帮助 pose，不能称 SAM 帮助。

### E1：再验证真正的 dynamic-mask 因果收益

当前静态 ScanNet++ clip 不够。至少增加多个有真实 moving object、camera GT 的序列，例如 MPI Sintel、
Bonn RGB-D Dynamic 或合适的 DyCheck 序列。需要先把 dynamic discovery 从 `[bed, wardrobe]` 升级为覆盖
测试场景运动物体的 detector/motion cue；否则 `sam_static_edge` 控制没有语义。

通过条件在看结果前锁定：

- causal prefix check 通过，GT 只用于最后评分；
- 每个 fold 的 optimizer 都 active，不能把 exact-raw fallback 计成改善；
- `all_edge` 相对 raw 的 center/rotation 分别报告，不用 composite loss 掩盖坏项；
- `sam_static_edge` 在相同 edge 数下稳定优于 `all_edge` 和 area-matched shuffle；
- 报告 mean/median、每帧 worse count、ATE/RTE/RRE 和 edge residual；residual 下降本身不是 pose 证据；
- 当前 scene 成功只能算 sanity check；跨多个动态 scene 后才可升级 V0 pose。

### E2：何时转向 Selfi-style feature

只有在以下组合出现时才进入：GT/部署 mask + edge upper bound 能改善 pose，但真实 RGB edge 因重复纹理、
遮挡或光照不稳定而失败。届时训练的是跨场景 VGGT geometric-localization feature，并用 visibility、
matching 和 BA 验证；不再训练 SAM descriptor 或 direct pose head。

## 10. 最终可行性结论

- **方向可行，原方案需修正后再实现。** EPO-style residual 有明确几何 Jacobian，也避开了已经失败的
  SAM-token correspondence 学习。
- SAM 的合理角色是剔除违反 static-world assumption 的 residual，不是提供 camera pose token。
- 最大风险不是优化资源，而是 mask 定义、predicted depth bias、nearest-edge ambiguity 和 viewgraph 连接度。
- 第一版不应声称“复现 EPO”，而应称 `causal masked edge pose feasibility`；它是 EPO 的受限子问题。
- 在 E0 上界完成前不写正式 edge optimizer 到 V0；V0 继续保持 raw StreamVGGT pose，避免候选失败污染
  已经可用的因果 tracking baseline。

## 11. E0/E1 实测结论与 G0 成熟路线（2026-08-13）

E0-r2 的可微 edge-DTF 铯路通过梯度检查，但 144 个小扰动试验中没有一帧通过 pose basin gate；平均
rotation/translation recovery fraction 分别为 `-1.974/-1.998`。E1 在固定 `0.25° / 0.25% scene-scale`
候选上确认负梯度 100% 降低 edge loss、正梯度 100% 增加 edge loss，说明实现方向可信；但 joint 和
translation 三折均失败。`sam_object_excluded_edge` 的 rotation-only 在 11/12 帧改善，不足以部署完整
SE(3)，且 prompted bed/wardrobe mask 不等于完整 dynamic truth。

因此停止 edge-DTF 调参。G0 改为 KinectFusion/ElasticFusion/BundleFusion 一类 RGB-D odometry 的标准几何
因子：当前 depth 反投影、向 `[1,2,4]` 个历史帧投影、在历史 vertex/normal map 上重关联，使用 robust
point-to-plane residual，coarse-to-fine LM、raw-pose prior、condition/trust-region 和固定能量下降接受规则。
SAM 只排除 prompted tracked-object 区域，并与 full-image、shifted-mask 控制等量比较；不读取 appearance
token、不训练模型。候选生成 API 不接收 GT，GT 只在所有候选固化后评分。V0 selected pose 仍是 raw。

G0 的实现入口是：

```bash
zsh streaming_couping/commands_g0_static_projective_icp.txt
```

当前单个静态 ScanNet++ clip 只能回答 projective ICP 是否能修正 pose，不能证明 SAM 动态排除有效；真正的
SAM-specific claim 仍需在含运动物体且有 camera GT 的多个序列上，使 SAM exclusion 同时优于 full-image
和同形状 shifted-mask control。
