# Decoupled instance-guided pose and geometry V2

## 目的

V1证明了两个方向，但也暴露了冲突：`camera_sam_only`改善GT pose，却不能直接移动
frozen world pointmap；`all_token_fusion`改善直接pointmap/depth，却弱于SAM-only pose。
V2因此不再让同一组可学习token同时承担两个任务。

## 分支

Pose分支使用appearance-only causal instance token，仅更新CameraHead读取的camera hidden。
Geometry分支拥有独立tokenizer和attention，仅更新每个DPT层`patch_start_idx:`之后的
image patch tokens；camera/register前缀逐元素保持baseline。

`decoupled_dual_branch`同时运行两条分支：

```text
SAM appearance -> pose tokenizer -> camera cross-attention -> CameraHead
SAM + geometry -> geometry tokenizer -> patch cross-attention -> Depth/PointHead
```

两条分支不共享learned encoder、attention、gate或zero projection。SAM3与StreamVGGT
仍全部冻结，VGGT token仍不会写入SAM3。

## 模式

| mode | 作用 |
|---|---|
| `camera_sam_only` | V1最佳pose control |
| `patch_sam_only` | appearance-only geometry control |
| `patch_sam_geometry_strict` | track/geometry/static三重硬门控 |
| `patch_sam_geometry_tracker_gate` | 仅track硬门控，geometry/static作为特征 |
| `decoupled_dual_branch` | SAM pose + tracker-gated patch geometry主方法 |
| `all_token_fusion` | 共享all-token负干扰对照 |

每个checkpoint使用aligned/module-off/zero-appearance/zero-geometry/shuffle-ID/
shuffle-time推理消融。双分支额外使用`pose_branch_off`和`geometry_branch_off`；其它模式
自动跳过这两个不适用的控制。

## 损失

Pose模式使用原有camera、relative rotation、translation direction、instance rigid与
centroid损失。Patch模式使用fixed-reference pointmap、scale-invariant depth和新增的
fixed-reference depth loss。新增项的scale只由frozen baseline reference depth拟合一次，
用于约束V1 all-token中观察到的depth尺度漂移。

双分支的总损失为pose与geometry损失之和，但梯度分别进入独立参数。当前版本不加入
pose-pointmap consistency loss，避免在两个分支分别稳定前再次强行耦合。

Pointmap/depth评估分别输出`full_scene`、`tracked_instance_union`和`background`区域。
这用于检查实例引导是否只改善目标区域，以及background退化是否超过可接受范围。

## 运行

从仓库根目录：

```bash
zsh streaming_couping/commands_instance_token_pose_decoupled_v2.txt
```

配置复用`outputs/streaming_couping_instance_token_pose/cache`。完整cache存在时不会重新运行
SAM3或StreamVGGT feature extraction；会训练6个adapter mode并完成全部评估。当前配置仍是
同一个训练clip，只能用于结构诊断，不能声称跨场景泛化。
