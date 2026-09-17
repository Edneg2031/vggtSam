# 交接文档：SAM 物体锚点修正 HorizonStream 位姿漂移

2026-09-17 · 实习工作交接

本文是**交接文档**，说明项目当前状态、运行方式、已知限制与后续建议。方法细节见 [`streaming_couping/docs/method.md`](streaming_couping/docs/method.md)；实验过程与证据见 [`streaming_couping/experiments/experiments.md`](streaming_couping/experiments/experiments.md)。三份文档合并自原 16 份，原文见 git 历史。

---

## 1 项目概述

### 1.1 问题定义

HorizonStream 是流式几何基础模型，逐帧输出 metric depth、depth confidence、intrinsics 与相机位姿。其位姿由运行时累积器（`online_motion_averaging`）在线因果产生，**误差会累积且不自纠正** —— 模型本身没有任何跨帧的景物级约束。

本课题验证：**SAM3.1 的 persistent object tracks 能否作为跨帧稳定的物体锚点，检测并修正 HorizonStream 的累计位姿漂移，并把修正反馈进后续帧。**

约束：不训练任何模型；不修改 HorizonStream 的 backbone / KV / GLA cache；不引入 DINO 或任何 learned score；GT 仅在所有反馈决策冻结之后加载，仅用于评测。

### 1.2 方法概要

对每个 (帧, 实例)，以 mask ∩ 有效深度取最多 256 个相机系点，与同一实例的参考云做最近点对齐，解一个 6DoF 增量；各物体的 `log(ΔT)` 经加权 Huber IRLS 取加权中位数得到每帧修正；修正通过门控后，作为**绝对目标**写入累积器。整体为**两遍式**：提案基于 raw 几何一次算完，修正之后仅重放累积器。

完整的坐标约定、可靠性分数、门控规则与判据定义见 `streaming_couping/docs/method.md`。

### 1.3 交付物清单

| 交付物 | 位置 |
|---|---|
| 方法文档 | `docs/method.md` |
| 实验文档（结果、消融、已排除方向、早期实验） | `experiments/experiments.md` |
| 本文档（仓库根） | `HANDOVER.md` |
| 主执行链路与读取工具 | `streaming_couping/commands_*.txt` |
| 实现 | `streaming_couping/src/`、`streaming_couping/scripts/` |
| 单元测试 | `streaming_couping/tests/` |
| 环境探查脚本 | `streaming_couping/scripts/report_environment.py` |

---

## 2 当前状态

### 2.1 已验证结论

主结果：`robust_semantic` + v1 prompt 集，**两个帧窗均通过全部七条事前判据**。

| 帧窗 | raw direct ATE | d_ATE | d_sim3 | 判定 |
|---|---:|---:|---:|---|
| 100 帧 | 0.1166 m | **+14.25%** | **+5.35%** | GO |
| 150 帧 | 0.1095 m | **+15.4 ~ 15.6%** | 正 | GO |

150 帧的区间来自**两次独立的 stage-1 运行**（+15.60% 与 +15.36%），其差值即跨运行误差，见 §6.2。点云指标同步改善（accuracy −29%、ghost −44%、F5cm +12%），且与轨迹判据独立地给出同一排序。

**证据、消融与逐项数据见 `streaming_couping/experiments/experiments.md` §1–§5。**

### 2.2 代码与测试状态

| 项 | 数量 |
|---|---:|
| 命令入口 | 9 |
| `scripts/*.py` | 29 |
| `src/**/*.py` | 70 |
| 测试文件 | 28 |
| 文档 | 3 |

测试结果：**231 通过，1 失败，3 跳过**。

失败项为 `test_instance_point_consistency.py::test_instance_consistency_is_causal_and_rejects_far_points`（`assert 4 == 3`）。该失败**在本次代码精简之前即存在**，与本课题的主链路无关，未修复。

### 2.3 外部依赖

以下为实测结果（`scripts/report_environment.py`，2026-09-17）。

**模型权重**

| 用途 | 路径 | 大小 | 状态 |
|---|---|---:|---|
| HorizonStream（Stage 1） | `/home/bod/86Nas/95_data_bak/FoundationModels/HorizonStream.pt` | 4.5 GiB | OK |
| SAM3.1（Stage 2a） | `/home/bod/86Nas/95_data_bak/FoundationModels/sam3.1/sam3.1_multiplex.pt` | 3.3 GiB | OK |
| StreamVGGT | `/home/bod/86Nas/95_data_bak/FoundationModels/StreamVGGT/checkpoints.pth` | 4.7 GiB | 存在，但**属旧线，主链路不使用** |

**数据集**

| 用途 | 路径 | 大小 | 状态 |
|---|---|---:|---|
| ScanNet++ manifest | `data/processed/scannetpp_pinhole_2d/manifest.json` | 4.9 MiB | OK |
| 存储根 | `/data184/open_source/vggtSam` | — | OK |

**已知配置遗留**：`configs/recovery_dynamic_instance.yaml` 中仍保留 StreamVGGT 条目（`device: cuda:0`）。该条目由旧线使用，当前主链路不读取；后续清理时可一并移除。

---

## 3 运行说明

### 3.1 环境要求

**硬件（实测）**

| 项 | 值 |
|---|---|
| GPU | 8 × NVIDIA GeForce RTX 3090（23.7 GiB 各） |
| CUDA（驱动侧） | 12.8 |
| CPU | 48 核 |
| 内存 | 377.6 GiB |
| 存储根可用空间 | 144.7 GiB / 1130.2 GiB |
| 仓库所在盘可用空间 | 19.6 GiB / 2382.3 GiB |

仓库盘剩余空间偏紧（19.6 GiB），而单次运行的产物（几何缓存、诊断、点云）量级为 GiB 级，长期迭代时需留意。

**解释器（两套，版本不同，不可互换）**

| 环境变量 | 路径 | Python | PyTorch | CUDA | 用途 |
|---|---|---|---|---|---|
| `HORIZONSTREAM_PYTHON` | `/home/huawei/miniconda3/envs/horizonstream/bin/python` | 3.11.14 | 2.8.0+cu128 | 12.8 | Stage 1（几何） |
| `STREAMING_COUPING_PYTHON` | `/home/huawei/miniconda3/envs/3am/bin/python` | 3.11.15 | 2.5.1+cu118 | 11.8 | Stage 2a/2b/3（SAM、分析、评测） |

两套环境的 PyTorch 与 CUDA 版本不同，因此**不可交叉调用**：Stage 1 必须在前者下运行，其余阶段必须在后者下运行。命令文件已按此分工，无需手工切换。

**依赖包（实测）**：两套解释器均无缺失。所需包按用途分为三组：几何与分析（`torch`、`numpy`、`PIL`）、SAM3 运行时（`iopath`、`ftfy`、`regex`、`huggingface_hub`、`timm`、`einops`、`pycocotools`）、绘图（`matplotlib`）。

> `matplotlib` 为近期新增（用于生成对比图）。缺失时判定与指标表仍会产出，仅图片生成失败。

**设备分配**：Stage 1 使用 `HORIZONSTREAM_DEVICE`（默认 `cuda:0`）；Stage 2a 使用 recovery config 中的 `sam3.device`（当前为 `cuda:2`）。机器上无 SLURM，进程直接占用物理卡，因此并行运行多组实验时需自行通过 `CUDA_VISIBLE_DEVICES` 隔离。

**环境探查**：

```bash
python -m streaming_couping.scripts.report_environment
```

### 3.2 运行入口

**全仓库只有一个运行入口**：

```bash
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

该入口内部依次调用 `commands_run_scannet_object_pose_feedback_100f.txt` 与 `commands_evaluate_scannet_object_pose_loss_object_only.txt`，三者构成完整链路，**缺一不可**（精简过程中已逐项确认）。

一轮运行即产出全部结论性内容：判定与判据表、逐代对照表、两张对比图、候选账本摘要，以及完整日志（`sweep.log` / `analysis.log`）。

**不再保留独立的读取入口**。各类分析脚本仍然存在，可直接调用：

| 需要什么 | 调用方式 |
|---|---|
| 重读判定与逐变体指标 | `python -m streaming_couping.scripts.summarize_feedback_branches --run-dir <run>` |
| 生成两张对比图 | `python -m streaming_couping.scripts.plot_pose_comparison --run-dir <run> --out <png>`<br>`python -m streaming_couping.scripts.plot_object_cloud_comparison ...` |
| 物体点云指标 | `python -m streaming_couping.scripts.evaluate_pose_feedback_object_map ...` |
| 候选账本 | `python -m streaming_couping.scripts.show_sam3_candidate_ledger --run-dir <run>` |
| 逐代逐类别对照 | `python -m streaming_couping.scripts.compare_feedback_categories ...` |
| 闭环实验 | `python -m streaming_couping.scripts.run_object_pose_loss_reestimate ...` |
| 环境探查 | `python -m streaming_couping.scripts.report_environment` |
| 单元测试 | `python -m pytest streaming_couping/tests/ -q` |

各脚本的参数见其 `--help`，或将已删除的同名 `.txt` 从 git 历史取出作为完整调用示例。

**默认配置为单分支、单次重复**，即产出 §2.1 主结果的那一轮。以下两项为显式开启，各自需要额外的 GPU 开销：

| 环境变量 | 作用 |
|---|---|
| `OBJECT_POSE_FEEDBACK_BRANCHES=baseline,fresh,...` | 运行提案侧消融分支 |
| `OBJECT_POSE_FEEDBACK_SAM_REPEATS=3` | 重复运行以测量噪声底 |

### 3.3 配置项

| 配置 | 位置 | 说明 |
|---|---|---|
| 帧窗 | `object_pose_feedback_env.zsh` 的 `FRAME_COUNT` / `FRAME_START` / `FRAME_STRIDE` | 帧窗同时决定 run 目录名前缀，不同帧窗的产物互不干扰 |
| 代数（generation） | 主链路的 `GENERATION` | 一代即一个 prompt 集；词表由 `GENERATION_PROMPTS` 按标签查出，标签与词表不会不一致 |
| prompt 集 | 同上 `GENERATION_PROMPTS` | 临时词表可用 `OBJECT_POSE_FEEDBACK_PROMPTS` 覆盖 |
| 反馈阈值 | `ObjectPoseFeedbackConfig` 与主链路命令行参数 | 约 40 项，均可在命令行覆盖 |

帧窗由环境变量覆盖，无需修改文件（该覆盖对运行与读取均有效）：

```bash
OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt
```

若只需查看某个**已存在**的历史 run 而不重跑，直接对该 run 目录调用 §3.2 的分析脚本，例如 `summarize_feedback_branches --run-dir <run>`。

### 3.4 产出物

每轮在 `<run>.baseline/` 下产出：

| 产物 | 内容 |
|---|---|
| `object_pose_feedback/summary.json` | 各共识变体的判定、判据与指标 |
| `object_pose_feedback/object_proposals.csv` | 每条提案的特征、拒绝原因与 GT 误差 |
| `object_pose_feedback/attribution.json` | 归因分析结果 |
| `object_pose_feedback/poses.pt` | raw / GT / 各变体轨迹 |
| `object_pose_feedback/pose_comparison.png` | 轨迹俯视与逐帧平移/旋转误差（GT / raw / 修正后） |
| `object_pose_feedback/object_cloud_comparison.png` | 逐物体点云，同一批点以三条轨迹分别置入世界 |
| `object_pose_refinement/feedback_diagnostics.pt` | 观测点云、参考云与提案（供 CPU 重放） |
| `gt_evaluation/` | 逐实例点云指标 |
| `sweep.log` / `analysis.log` | 完整运行日志（标准输出仅保留结论性内容） |

---

## 4 代码结构

### 4.1 目录组织

```
streaming_couping/
  commands_*.txt               9 个入口
  object_pose_feedback_env.zsh 帧窗与 run 目录名的唯一定义处
  src/semantic_mapping/        主链路：几何适配、SAM 适配、提案、共识、门控、重放
  src/{aggregation,backbones,bridge,learned_pose,solvers}/
  scripts/                     29 个可执行模块
  tests/                       28 个测试文件
  configs/                     运行配置
  docs/method.md               方法文档
  experiments/experiments.md   实验文档
(仓库根) HANDOVER.md            交接文档（本文件）
```

### 4.2 主执行链路

`commands_run_scannet_object_pose_feedback_branches.txt` → `commands_run_scannet_object_pose_feedback_100f.txt` → `commands_evaluate_scannet_object_pose_loss_object_only.txt`

**三者缺一不可**：删除其中任何一个，主入口立即失效。精简过程中已就此逐项确认。

主链路依次执行：Stage 1 几何 → Stage 2a 语义与提案 → GT-mask oracle 对照（CPU）→ 重放确定性门 → 各变体的 CPU 重放 → Stage 2b 分析 → 对照表 → 两张图 → 候选账本。

### 4.3 关键模块职责

| 模块 | 职责 |
|---|---|
| `src/semantic_mapping/adapters.py` | 几何与分割适配；**含物体筛选的跟踪层规则**（名额、判重） |
| `src/semantic_mapping/object_pose_loss_refinement.py` | 逐实例 6DoF 提案求解（ICP 式交替 + Adam） |
| `src/semantic_mapping/object_pose_feedback.py` | 可靠性分数、跨物体共识、门控、判据 |
| `src/semantic_mapping/pipeline.py` | 四阶段编排 |
| `scripts/run_object_pose_feedback.py` | Stage 2b：共识、门控、累积器注入重放 |
| `scripts/run_object_pose_loss_replay.py` | 由诊断重跑 refiner（CPU，用于分支消融） |
| `scripts/run_object_pose_loss_reestimate.py` | 闭环实验：以外部轨迹为基座重解提案 |

---

## 5 已排除的技术方向

以下方向均已实测，**不建议重复尝试**。各方向的检验方式与完整数据见 `streaming_couping/experiments/experiments.md` §6。

| 方向 | 结论 |
|---|---|
| 改进分割质量 | GT mask 替换 SAM 后提案误差**反而更差**（0.1010 对 0.0892），瓶颈不在 mask |
| 更新参考集 | 将 anchor 年龄由 37.5 降至 8、相关系数由 0.79 降至 0.17，提案误差**无变化**（0.0893→0.0891）；该相关系混淆所致 |
| 旋转与平移解耦 | 可全部通过判据，但跨配置摆动于 +3% 与 +19% 之间，属**不稳定配置**，不宜作为主线 |
| 仅修正平移 | 显著更差，sim3 转负；旋转与平移在联合求解中耦合 |
| 扩充 prompt | 单次增词即可挤占全局名额并触发判重，导致结果由 +15.36% 跌至 −11.87% |
| 闭环 / 流式 | 提案残差降 18.0%，轨迹反而差 3.6%，全部变体 NO_GO |
| 更换优化目标 | 目标函数对共模误差免疫（`P`、`Q` 同源），无法感知其贡献的绝对位姿误差 |

---

## 6 已知限制与风险

### 6.1 结论的适用范围

| 项 | 状态 |
|---|---|
| 场景数 | **1**（`00a231a370`） |
| 帧窗 | 2 个，为**同一场景的复核**，非第二个场景 |
| 协议角色 | 开发窗口；判据与阈值即在窗口上确定，**无 held-out 证据** |
| prompt 集 | 方法的一部分，非自由参数 |
| 点云绝对数值 | 不可引用（预测云稀疏约 100 倍），仅 raw 与修正的对比有效 |

### 6.2 复现风险

**1. 不得删除 `outputs/` 目录。** 两次独立 stage-1 运行的 d_ATE 相差约 **0.24 个百分点**（实测 +15.60% → +15.36%），大于多个分支之间的差异。主链路会将新一代的几何缓存**自上一代复制**，使两代共享同一次 stage-1、从而可比；删除该目录即失去这一条件，且对照一旦丢失便无法事后恢复。

**2. 噪声底存在两种，不可混用。** 共享同一次几何缓存时为 **1.8e-05**；跨独立 stage-1 运行时为 **2.4e-03**，后者约为前者的 130 倍。判断"差异是否显著"时须使用与比较方式相对应的那一个。

**3. 150 帧的数值来自两次不同运行。** 引用时须注明是哪一次（`streaming_couping/experiments/experiments.md` 各处已标注）。

### 6.3 工程注意事项

**1. prompt 列表不可加。** 增词可能挤掉已有词，机制有两条：跨 prompt 判重（按**出生帧** 排序，重叠者丢弃后到者）与全局 **16 条**名额（所有词共享）。增词是摊薄名额，而非增加检测器。

**2. "修正点云"存在两种不同机制，不可混淆。**

| | (a) 逐实例修正点云 | (b) 位姿反馈（主线） |
|---|---|---|
| 相机位姿 | 保持 raw | 被修正 |
| 修正对象 | 每个实例自身的点云 | **置点所用的位姿** |

(b) 中不存在逐点修正：同一批相机系点由不同轨迹置入世界。以 (a) 的逐实例结果推断其在 (b) 共识中的权重是错误的。

**3. 判定不得依据 ICP loss。** 判据仅取位姿指标（七条，见 `streaming_couping/docs/method.md` §7）。loss 下降不代表位姿改善 —— 已有实例显示 loss 降 58% 而 GT 指标持平。

**4. 静态导入分析在本仓库不可靠。** 精简过程中该手段出现三次误判（漏解析相对导入、正则捕获错误标识符、将仍在使用的模块判为不可达）。判断某一文件属于旧线还是当前线，应依据其 **docstring**。

---

## 7 未决问题

| 问题 | 状态 |
|---|---|
| 缺少第二个场景 | 现有结论全部来自单一场景 |
| `anchor_rho` 由 0.79 降至 0.09 | 未解释。该量在 100 帧上为最强预测因子，窗口变长后理应保持 |
| 缺少可用的物体预筛选判据 | 两个为"物体质量"设计的分数均反预测，硬拒筛选零区分度；有效的两个判据均需先形成共识 |
| 跟踪层规则未变更 | 16 条名额与按出生帧判重为已知机制，但未修改；这是控制"哪些物体参与"的可改之处 |
| 单次增词实验的归因未分离 | 丢失 `wardrobe` 与新增 `cabinet` 同时发生，无法区分二者对结果的影响 |

---

## 8 后续工作建议

按优先级排列：

1. **补充第二个场景。** 当前结论的场景数为 1，且位于开发窗口。这是将结论由"在该窗口上成立"提升为"该方法成立"的唯一途径；其余工作均为次要。

2. **查清 `anchor_rho` 的消失。** 该量在 100 帧上是最强预测因子（`fresh` 分支即为此设计），在 150 帧上却归零。可能是测量问题，也可能指向尚未发现的机制。

3. **分离单次增词实验的归因。** 需构造"新词进入而原词不丢"的条件（提高 `--max-objects` 或调整判重排序），以确定结果劣化源于丢失原词还是引入新词。

4. **建立物体预筛选判据。** 现有判据分三类且生效时机不同（详见 `streaming_couping/docs/method.md` §5）；若要继续提升，可改进的是跟踪层规则，而非物体质量打分。

---

## 附录 A 文档索引

| 文档 | 内容 |
|---|---|
| [`streaming_couping/docs/method.md`](streaming_couping/docs/method.md) | 方法：系统结构、逐步流程、坐标约定、六层筛选、七条判据、两种噪声底 |
| [`streaming_couping/experiments/experiments.md`](streaming_couping/experiments/experiments.md) | 实验：主结果、点云传导、环路验证、消融、已排除方向、早期实验、复现命令 |
| 本文档 | 交接：状态、环境、运行、代码结构、限制与风险、后续建议 |

每轮运行产出的两张图位于 `<run>.baseline/object_pose_feedback/`，说明见 §3.4。

---

## 附录 B 命令速查

```bash
# 运行（GPU，唯一入口）
zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 读取历史帧窗
OBJECT_POSE_FEEDBACK_FRAME_COUNT=100 zsh streaming_couping/commands_run_scannet_object_pose_feedback_branches.txt

# 完整测试套件
python -m pytest streaming_couping/tests/ -q
```

分析脚本的调用方式见 §3.2。
