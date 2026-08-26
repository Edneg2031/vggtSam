# 当前方法与实验结果


## 1. 研究目标

本项目面向室内场景的 3D Gaussian Splatting 联合重建与分割，核心问题是：

> 在共享高斯几何的前提下，解耦 RGB、语义和几何的可见性与梯度，避免有噪声的
> 深度或语义伪标签破坏外观，同时让重建得到的高斯适合二维和三维分割。

当前包含两条语义路线：

- **封闭集语义**：DINOv2 + Mask2Former，固定为 21 类，用于 FourFloor、
  RobotLab 和 ScanNet++ 等室内数据。
- **开放词汇语义**：采用 GLS 风格的 SAM + OpenCLIP + DEVA，用于 LERF-OVS
  自由文本查询。

## 2. 整体流程

```text
RGB 图像 + COLMAP 相机与稀疏点云
        |
        +-- Metric3D 逆深度 -- 使用 SfM 点进行逐帧尺度校正
        |
        +-- 封闭集：DINOv2 + Mask2Former --> 21 维类别 soft logits
        |
        `-- 开放词汇：
              SAM 区域 --> OpenCLIP 512 维 --> 场景自编码器 --> 16 维语言监督
              DEVA 跨视图跟踪 ----------------------------> 实例 ID

gsplat 联合训练
        |
        +-- RGB 分支：SH + RGB opacity --> L1 + SSIM --> PPISP
        +-- 语义分支：语义特征 + 语义 opacity
        +-- 几何分支：尺度校正后的逆深度 + 深度法线 + E-rank
        `-- 采样与拓扑：共视图像对 batch + 标准增密和剪枝

输出
        +-- RGB/PPISP 渲染与重建指标
        +-- 语义/语言渲染与二维 mIoU、mBIoU
        `-- 语义 PLY、三维类别/实例提取和包围框
```

### 2.1 共享表示与解耦

每个高斯包含标准 3DGS 的几何与外观参数：

```text
中心位置、尺度、旋转、SH、rgb_opacity
```

语义分支额外增加：

```text
semantic_feature、semantic_opacity
```

两种 opacity 分别用于独立的 alpha blending：

- RGB 渲染使用 `rgb_opacity`；
- 语义或语言渲染使用 `semantic_opacity`；
- 语义损失不能直接修改 RGB opacity 或 RGB 颜色；
- 两条分支仍共享高斯位置和增密拓扑，因此几何是重建与分割之间的交互接口。

这是当前方法的主要语义/外观解耦。作为对照，GLS 控制组在 RGB 和语言渲染中
共用 RGB opacity。

### 2.2 损失函数

整体目标函数为：

$$
\mathcal{L}=\mathcal{L}_{rgb}+\mathcal{L}_{semantic}
+\lambda_d\mathcal{L}_{depth}+\lambda_n\mathcal{L}_{normal}
+\lambda_e\mathcal{L}_{erank}+\lambda_p\mathcal{L}_{PPISP}.
$$

- `L_rgb`：L1 + SSIM。
- 封闭集 `L_semantic`：逐像素平均的 soft KL + hard cross-entropy。
- 开放词汇 `L_semantic`：SAM 区域 OpenCLIP 特征 L1 + `0.3` 倍 DEVA 实例 CE。
- `L_depth`：渲染逆深度与 Metric3D 逆深度之间的损失，Metric3D 深度预先经过
  鲁棒的 SfM 尺度校正。
- `L_normal`：渲染深度法线的一致性损失。
- `L_erank`：抑制针状的高各向异性高斯。
- `L_PPISP`：相机相关的光度校正及其正则项。

当前开放词汇配置中，语言损失会更新语言特征和语义 opacity；向高斯中心回传
`0.05` 倍梯度，并停止向尺度和旋转回传。与完整的语言到几何反向传播相比，该
设置更加稳定。

### 2.3 可选研究模块

- **B6 证据门控**：只有当语义不一致和深度残差共同表明监督不可靠时，才降低
  对应区域的几何监督。
- **SGAD 切向偏移**：通过有界的一维源相机射线偏移，将几何锚点与 RGB 外观
  渲染位置解耦。
- **FDS-GS Random Drop**：带梯度 warmup 条件的后期高斯随机丢弃。
- **Guided-MVS**：可选的外部深度预处理模块。

这些模块是围绕稳定基线设计的消融实验，并不会默认同时启用。

## 3. 主要实验结果

### 3.1 FourFloor 封闭集重建与分割

| 方法 | Raw PSNR | PPISP PSNR | mIoU | 高斯数量 |
| --- | ---: | ---: | ---: | ---: |
| 稳定基线 B0 | 22.751 | **25.024** | 0.7022 | 2,370,961 |
| B6 证据门控 | **22.787** | 24.929 | **0.7060** | 2,401,687 |

B6 在原始重建质量基本不变的情况下，将 mIoU 提高了 `0.38` 个百分点。该提升
真实但较小，因此 B0 仍作为稳定参照，B6 作为语义引导几何的主要消融。

### 3.2 ScanNet++ 全分辨率重建

实验使用场景 `b20a261fdf` 和 `8b5caf3398`，分辨率为 `1920 x 1440`，batch
size 为 2，训练 15k 个优化步。下表为两个场景的平均值。

| 方法 | PSNR | SSIM | LPIPS |
| --- | ---: | ---: | ---: |
| 基线，Raw | 20.946 | 0.8959 | 0.2270 |
| 基线，PPISP | **22.617** | 0.9020 | 0.2151 |
| SGAD tangent，Raw | 20.927 | 0.8957 | 0.2308 |
| SGAD tangent，PPISP | 22.565 | **0.9029** | 0.2186 |
| GLS 论文 | 20.690 | 0.8496 | **0.1709** |

基线的 PSNR 和 SSIM 高于 GLS 论文报告值，其中 PPISP 对 PSNR 的改善明显；GLS
报告的 LPIPS 更好。SGAD tangent 在 `b20a261fdf` 上有所提升，但在
`8b5caf3398` 上使多数指标下降，因此目前将其作为几何/外观解耦消融，而不是
默认方法。

该 GLS 对比仍属于参考对比，因为 LPIPS backbone、初始化和渲染实现尚未确认
完全一致。

### 3.3 LERF-OVS 开放词汇分割

当前评估器使用 polygon JSON GT、GLS hardest-negative OpenCLIP relevancy、固定
阈值 `0.5`，并对所有“标注帧-查询对象”组合的指标取平均。

| 场景 | 本项目解耦语言分支 mIoU | GLS 论文 mIoU | 本地 GLS 复现 |
| --- | ---: | ---: | ---: |
| Figurines | 0.4992 | 0.4973 | - |
| Ramen | 0.2106 | 0.3121 | 约 0.20 |
| Teatime | 0.4973 | 0.5801 | 约 0.30 |
| Waldo Kitchen | 0.3353 | 0.3215 | 约 0.15 |
| 四场景平均 | 0.3856 | 0.4278 | - |

结果说明：

- 表中本项目结果来自已有的 30k checkpoint，并统一使用固定阈值 `0.5`。
- Ramen 使用修正后的 GLS `default` SAM 层时，mIoU 可从 `0.2106` 提高到
  `0.2181`。
- 本地 GLS 复现值是当前记录的近似结果。只有在评估帧、查询集合、阈值和聚合
  方式完全相同时，才能与本项目结果直接比较。
- `gls-exact` 共享 RGB opacity 控制组在 Ramen 上得到 `0.2070`，与本地 GLS
  复现的约 `0.20` 接近。
- 当前解耦分支在 Ramen 上与本地 GLS 接近，在 Teatime 和 Waldo Kitchen 上明显
  更高；相对于 GLS 论文表格，主要短板仍是 Ramen 的小物体分割。

Ramen 中最明显的失败类别是 `corn`、`onion segments`、`kamaboko` 和 `spoon`
等小型或细长物体。区域均衡采样能够改善部分小类别，但会降低其他类别，整体
mIoU 没有提高，因此只保留为消融。

## 4. 当前结论

1. **独立语义 opacity 有效。** GLS 风格的共享 opacity 控制组没有超过解耦分支，
   而且 Ramen 的定位准确率明显更低。
2. **几何是当前剩余的耦合通道。** 不受限制的语义几何梯度会破坏高斯尺度与
   形状；弱化的中心位置梯度是目前更稳定的折中。
3. **语义尚未参与增密统计。** 当前 GLS 风格语言特征通过独立的 N-D pass 渲染。
   其梯度可以更新高斯参数，但不会进入 RGB rasterization 对应的增密统计，因此
   可能限制语义边界处的新高斯生成。
4. **GLS 论文值与本地复现值差异明显。** 最终论文对比必须使用同一个评估器，
   并明确记录评估帧、查询数量、阈值、LPIPS backbone 和指标聚合方式。

## 5. 当前论文主线

当前论文主线可以概括为：

> 基于任务特定 opacity 和受控梯度路由的外观、几何与语义联合高斯重建。


