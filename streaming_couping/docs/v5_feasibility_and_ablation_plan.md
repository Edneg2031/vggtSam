# V5 可行性分析与分阶段消融计划

## 1. 目的

这份计划不把外部建议当作待办清单。目标是用尽量少的结构改动回答三个问题：

1. 位姿提升是否来自真实的跨帧实例几何，而不是单场景外观记忆？
2. analytic translation solver 的收益是否来自独立历史锚点，而不是当前 pointmap 与当前相机的内部自洽？
3. learned pointmap 是否真的改善几何；若不能稳定泛化，是否应退回“raw 几何 + refined pose”？

当前 V4 和所有已有 checkpoint 均保留为对照。任何新路径都必须支持：

```text
module_off → 与缓存的 raw StreamVGGT 严格相等
无有效实例 → 与 raw StreamVGGT 严格相等
solver 失败 → 回退到输入 pose
mask 外几何 → 与 raw StreamVGGT 严格相等
```

训练时间不作为限制。凡是改变了 token 输入、identity policy、rotation composition、
spatial attention 或 pointmap 输出参数化的实验，都必须使用相同训练预算独立重训。只有
纯解析后处理（例如在同一 pose/pointmap 预测上切换 solver source）共用 checkpoint，
目的是控制变量，而不是节省训练时间。

## 2. 当前数据流和张量

```text
appearance              [B,S,K,2*C_sam]
geometry                [B,S,K,20]
quality                 [B,S,K,3]
identity state          [B,S,K]
persistent token        [B,S,K,512]

camera hidden           [B,S,D]
camera query            [B,S,1,D]
DPT token level         [B,S,N,D]
instance masks          [B,S,K,H,W]
world pointmap          [B,S,H,W,3]
```

`S` 是输入视角顺序，`K` 是 persistent instance 数量。帧号可以不单调，但所有 memory
更新都必须按 `S` 的顺序因果执行。

## 3. 建议逐项可行性

| 方向 | 当前事实 | 主要冲突或风险 | 决策 |
|---|---|---|---|
| historical-anchor solver | 当前 solver 使用本帧 refined pointmap 和本帧 ray；问题真实存在 | 当前缓存没有 Track Head 的显式 2D tracks | 实现基于 ICP 对应的历史 3D anchor 接口；旧 solver 保留 |
| pose projection residual | pose tokenizer 当前把 geometry 清零，只保留 appearance | 直接重新加入完整 point cloud 会把错误几何带回相机，并增加过拟合 | 只加入低维 mask/projected-map/ray residual |
| SO(3) rotation residual | 当前在 9 维 `[T,quat,FoV]` encoding 上做差并相加 | 最终矩阵虽合法，但 correction 不具备群结构且会同时改变 T/FoV | 新模型只组合受限 SO(3) rotation；T 由 solver 决定，K 暂时冻结 |
| mask-local patch attention | V4 已在 mask union 内更新，并在输出端严格恢复 mask 外 raw 值 | union 内的 cabinet patch 仍可查询 wardrobe token | 实现为可切换的 `[B,S,Q,K]` pairwise mask bias，并与 union 版分别重训 |
| depth residual | 当前四层 DPT token 同时进入 frozen depth head 和 point head | 不能简单宣称现有 adapter 只预测 depth；需要新 residual head 和 native-scale 处理 | 暂不实现；先使用 raw pointmap 在 refined pose 下重新放置 |
| weighted shared K | 当前 ray solver 使用参考帧预测 K | 与 pose/FoV 同时优化会增加 gauge ambiguity | 仅作为 solver 后续无训练消融，不进入首轮 |
| appearance prototypes | 单 EMA 无法完整表示多视角 | 当前只有很少帧和三个实例，prototype 会增加容量和超参数 | 暂缓；先用 memory-off 和 wrong-ID memory 对照验证 EMA 是否真的有用 |
| provisional recovery | recovery 会立即写回 SAM3 memory | 实现需要延迟确认、重跑 tracking，代价大；object map 已有严格几何门控 | 暂缓；先记录 recovery 来源，并用三态门控隔离其错误输出 |
| learned sample gate | 当前 attention 有 scalar layer gate，但已有 identity/confidence/spatial gate | 再加 MLP gate 容易在单场景中学成视角分类器 | 首轮使用确定性 reliability，不新增 gate MLP |

## 4. 必须先解决的结构冲突

### 4.1 UNKNOWN 权重不能直接乘 token

草稿中的：

```text
unknown_token = 0.25 * token
```

发生在 cross-attention 的 `instance_norm` 之前。LayerNorm 会消除绝大部分整体幅值差异，因此
这不能可靠表达 0.25 权重。

正确做法是输出显式：

```text
instance_reliability [B,S,K]
MATCH    = 1.0
UNKNOWN  = configured low weight
MISMATCH = 0.0
```

并将它用于 attention logit bias 和/或 attention output reliability。geometry 和 ray 分支仍只
允许 `MATCH`。

### 4.2 普通 token shuffle 不是有效负对照

camera cross-attention 把实例视为无序集合。只在 `K` 维上同时置换 token 和 valid flag，
理论上输出不变，因此不能证明模型是否使用 object identity。

有效对照应为：

```text
memory_id_shuffle:
  保持当前实例观测不变，只把历史 memory 分配给错误 object ID

spatial_token_shuffle:
  保持实例 mask 不变，只把与该 mask 绑定的 token 换成其他实例 token
```

前者用于 camera/persistent memory，后者只在实现 per-instance spatial attention 后用于
pointmap。

### 4.3 mask-local 已部分存在

当前 V4 已有两层保护：

1. patch residual 只在可信实例 mask union 对应的 patch 上保留；
2. DPT 解码后，mask union 外的 depth/pointmap 被逐像素替换为 raw StreamVGGT。

因此首轮不需要 background/no-op token。真正缺少的是 mask union 内的
`patch-instance` 对应关系。若后续实现，mask bias 应为：

```text
spatial_weight [B,S,Q,K]
```

而不是当前的：

```text
union_gate [B,S,Q]
```

对于没有任何实例权重的 patch，直接跳过 attention，避免全 `-inf` softmax。

### 4.4 historical anchor 在没有 Track Head 时的最小实现

首版不假装拥有独立的 learned point tracks，而是明确命名为：

```text
historical_anchor_icp_correspondence
```

流程：

```text
当前 mask 像素 u_i
  → 当前 raw pointmap 点 Y_i
  → bounded ICP 与历史 object map 建立最近邻
  → 得到历史点 X_j 与当前像素 u_i
  → 使用 X_j、u_i、K、refined R 解相机中心 C
```

当前 pointmap 只用于匹配，最终 point-to-ray 方程使用历史世界点 `X_j`。它比当前 solver
独立，但仍不是完全独立的 Track Head 对应，因此日志必须写清 correspondence source。

### 4.5 pointcloud 输出必须区分两种提升

首轮输出三份点云：

```text
raw_direct_pointmap
raw_geometry_refined_pose
learned_direct_pointmap
```

`raw_geometry_refined_pose` 将 raw world pointmap 先转回 raw camera-local 坐标，再用最终
pose 放回世界坐标。它只反映 pose 改善，不允许网络自由改变物体形状。

在 learned pointmap 未通过泛化门槛前，deployable 默认应使用
`raw_geometry_refined_pose`，learned pointmap 仅作为实验输出。

## 5. 分阶段消融

所有实验固定：

```text
训练序列：90 105 119 130 140
同序列 holdout：210 240
无重叠测试：492 512 520 545 561 589
实例：37 68 54
随机种子、epochs、loss 权重和 checkpoint 选择规则保持一致
```

### Stage A：三态门控本身

这一阶段不判断 pose/pointmap，只判断身份安全性。

| ID | 设置 | 训练 | 目的 |
|---|---|---:|---|
| A0 | V4 binary identity gate | 否 | 现有基线 |
| A1 | tri-state，仅 MATCH 进入所有分支 | 是 | 分离“状态定义变化”对训练的影响 |
| A2 | tri-state，UNKNOWN 低权重进入 camera | 是 | 验证遮挡观测是否对 pose 有帮助 |

必须报告：

```text
每帧 MATCH / UNKNOWN / MISMATCH / ABSENT 数量
每个实例状态、point count、fitness、RMSE、reason
camera 使用数
geometry/ray 使用数
memory update 数
```

通过条件：

- 561 的错误大 mask 仍为 `MISMATCH`；
- 原序列因少点、局部可见或低 fitness 的观测主要变成 `UNKNOWN`，不是 `MATCH`；
- `UNKNOWN` 不更新 geometry memory，不进入 learned pointmap 和 ray。

### Stage B：solver source，共用同一上游 checkpoint

先解决最关键的内部自洽问题。solver 不参与当前 adapter 的训练 loss，因此同一轮比较必须
固定上游 pose/pointmap checkpoint，只切换 solver source；如果同时更换 checkpoint，就无法
判断收益来自 solver 还是上游网络。

| ID | Rotation | 3D source | Solver |
|---|---|---|---|
| B0 | raw | current raw pointmap | current pointmap solver |
| B1 | learned V4 | current raw pointmap | current pointmap solver |
| B2 | learned V4 | current refined pointmap | 当前 V4 solver |
| B3 | raw | historical object anchors | historical-anchor solver |
| B4 | learned V4 | historical object anchors | historical-anchor solver |

必须同时报告：

```text
ATE
translation error mean
translation RPE
pair translation direction error
internal point-to-ray RMSE
accepted ratio
condition number
center shift
correspondence count
anchor age
anchor source frame IDs
```

判定规则：

- 不能只看 internal RMSE；
- B4 必须在 holdout 和 492–589 上都比 B2 更安全，才替换当前 solver；
- 若 B4 只降低 internal RMSE 但 GT translation 变差，historical correspondence 无效；
- 任一帧不满足点数、条件数或 residual 阈值时必须回退输入 center。

### Stage C：pose feature 与 SO(3)，每组独立重训

不能把旧 checkpoint 直接换成 SO(3) 组合后声称公平；旧 adapter 是按 additive pose
encoding 训练的。以下每组使用同样训练预算重新训练：

| ID | Pose feature | Rotation composition | Memory |
|---|---|---|---|
| C0 | appearance only | additive encoding | EMA，当前 V4 |
| C1 | appearance only | bounded SO(3) | EMA |
| C2 | projection/ray residual only | bounded SO(3) | geometry history |
| C3 | appearance + residual | bounded SO(3) | EMA |
| C4 | C3 checkpoint | bounded SO(3) | inference-time memory off |
| C5 | C3 checkpoint | bounded SO(3) | inference-time wrong-ID memory |

C0–C3 分别独立重训。C4/C5 必须复用 C3 checkpoint，作为“已训练模型是否真的依赖
memory”的负对照；如果把它们重新训练，网络可能学会补偿，回答的是另一个问题。
`memory off` 不删除输入维度，而是把 history 和 current-history 部分置零。
`wrong-ID memory` 只置换历史 memory，不能置换完整 token 集合。可额外训练一个
memory-off capacity baseline，但不能替代 C4。

必须报告：

```text
rotation error
rotation correction magnitude
ATE / translation metrics
projection centroid residual before/after
coverage / IoU before/after
effective instances
per-state reliability
```

判定规则：

- C1 对比 C0：只回答 SO(3) 组合是否更安全；
- C2 对比 C1：只回答显式几何残差是否优于外观记忆；
- C3 对比 C2：只回答 appearance 是否提供额外身份可靠性；
- C4/C5 若与 C3 几乎相同，说明 persistent memory 没有被有效使用；
- 492–589 的 rotation 不得再次出现“总体 ATE 改善但 rotation 退化”而没有明确回退。

### Stage D：pointcloud 输出策略

先不增加 depth residual 网络。D2 和 D3 的 attention 结构不同，必须分别独立重训；D0 和
D1 没有可训练参数。

| ID | Pointcloud | Spatial policy |
|---|---|---|
| D0 | raw direct pointmap | none |
| D1 | raw geometry + final refined pose | none |
| D2 | 当前 learned direct pointmap，独立重训 | trusted mask union + exact outside fallback |
| D3 | learned direct pointmap，独立重训 | per-instance mask attention + exact outside fallback |

D3 与 D2 都实现并分别重训，因为 D3 正是在检验 D2 的退化是否来自实例之间的空间串扰。
但它们都只是候选；若 D3 在无重叠序列仍不能稳定优于 D1，则停止 learned pointmap 路线，
deployable 使用 D1。

新增无 GT drift 指标：

```text
inside_mask_pointmap_drift
boundary_pointmap_drift
outside_mask_pointmap_drift
```

并继续报告 GT：

```text
full_scene pointmap error
per-instance pointmap error
background pointmap error
```

通过条件：

- `outside_mask_pointmap_drift == 0`，允许浮点比较容差 `1e-7`；
- full scene 不得退化超过 1%；
- 任一实例不得退化超过 10%，否则该帧/实例必须回退 D1；
- D3 的 spatial-token shuffle 必须显著恶化实例区域，否则 per-instance attention 没有被使用。

### Stage E：暂缓项

只有前四阶段得到稳定结论后才考虑：

```text
weighted_clip_median intrinsics
depth residual + reliability
multi-prototype appearance memory
provisional recovery state machine
leave-one-object-out training
```

## 6. 配置和输出命名

建议把研究开关限定为以下几类，避免重新形成大量 mode：

```yaml
identity:
  policy: binary | tri_state
  unknown_camera_weight: 0.25

pose:
  feature_mode: appearance_only | residual_only | appearance_and_residual
  rotation_mode: additive_encoding | bounded_so3
  max_rotation_update_degrees: 5.0
  memory_ablation: normal | off | wrong_id

solver:
  mode: current_raw | current_refined | historical_anchor

geometry:
  output_mode: raw_direct | raw_reposed | learned_direct
  spatial_mode: union | per_instance
```

每个阶段分别输出：

```text
evaluation/identity_state_summary.csv
evaluation/solver_source_ablation.csv
evaluation/pose_feature_ablation.csv
evaluation/pointcloud_output_ablation.csv
evaluation/frame_split_audit.csv
```

不要把所有组合做笛卡尔积。训练便宜并不意味着可以混淆变量：Stage A 的三个 identity
policy 独立重训；Stage B 在选中的 A checkpoint 上固定上游、只比较 solver；Stage C 的
pose 结构独立重训并固定 Stage B 选出的 solver；Stage D 再固定选中的 pose。这样每个结论
仍然只对应一个结构变化。

## 7. GT 和公平性

GT 只允许用于：

```text
adapter 训练监督
checkpoint 选择
最终评估
```

以下步骤不得读取 GT：

```text
identity state
memory update
projection residual feature
historical correspondence
solver accept/reject
pointcloud output selection
```

训练和测试配置在运行前应共同执行 split audit：

```text
train frame IDs
test frame IDs
intersection
minimum numeric frame distance
```

当前 `492–589` 与 `90–140` 无重叠，但仍需要把检查写入日志，避免以后更换序列时误用。

## 8. 当前停止条件

如果出现以下任一情况，不继续增加网络容量：

1. historical-anchor solver 在无重叠序列不优于 current solver；
2. residual-only pose 与 appearance-only 没有差异；
3. wrong-ID memory 不影响结果；
4. learned pointmap 在无重叠序列继续明显伤害两个以上实例；
5. internal residual 改善而 GT pose/pointmap 指标退化。

此时最可信的最终系统应收束为：

```text
SAM3 + 三态几何身份安全层
projection-residual bounded SO(3) rotation（仅在通过消融时）
historical-anchor translation（仅在通过消融时）
raw StreamVGGT geometry reposed by final pose
```

## 9. 已实现的一次运行入口

V5 首轮消融已收束为 6 个独立训练的结构变体：

```text
v4_match_additive_union
v5_unknown_additive_union
v5_unknown_so3_union
v5_residual_so3_union
v5_combined_so3_union
v5_combined_so3_per_instance
```

同一结构 checkpoint 上同时比较 `current_raw`、`current_refined` 和
`historical_anchor` solver。只有最后一个逐实例候选额外执行 `memory_off`、
`wrong_id_memory` 和 `spatial_token_shuffle`，这些负对照不重新训练。

服务器只需运行：

```bash
zsh streaming_couping/commands_v5_ablation_suite.txt
```

脚本会缓存两个序列、完成全部训练和评估，并在最后打印：

```text
outputs/streaming_couping_v5_ablation/v5_upload_summary.csv
```

该表每个结构仅保留 holdout/test 各一行，同时包含 pose、direct learned
pointmap、raw geometry reposed by learned/analytic pose、三态数量、solver 接受数和
严格 module-off 等价性。完整诊断仍保存在各 variant 目录，但首轮分析只需复制这一张表。
