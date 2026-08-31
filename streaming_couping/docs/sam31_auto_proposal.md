# SAM3.1 class-agnostic visual-point discovery

This branch tests whether the low prompt-SAM tracking IoU is primarily caused
by the annotation-assisted noun list. It does not change StreamVGGT geometry,
map fusion, or any trained parameter.

## Candidate generation

For fixed causal discovery frames, the branch samples a deterministic grid of
positive visual points. Points already covered by a retained track are skipped.
SAM3.1 creates a new object ID from each remaining point, then the birth-frame
mask is filtered by:

- minimum and maximum mask area;
- the prompt point being inside the returned mask;
- mask IoU NMS;
- intersection-over-smaller containment suppression.

Accepted objects receive permanent slots and are propagated forward. No text
prompt, class label, future mask, or GT field is used for proposal generation.
The slot capacity remains 16 to match the frozen downstream cache contract.

## Evaluation

The command evaluates three branches:

- `prompt_sam31_online_forward`;
- `sam31_auto_visual_points`;
- `gt_mask_oracle`.

Prompt-SAM and auto-SAM receive separate Hungarian assignments after their
artifacts are frozen. Reusing prompt-SAM slot assignment for auto-SAM would be
invalid because the two detectors discover different objects in different
slot orders. The oracle keeps the prompt-SAM assignment and remains an upper
bound under the same prompt slot coverage.

The evaluator writes two GT scopes:

- `prompt_scope` preserves the old comparison against objects named by the
  configured noun list;
- `all_instance_scope` includes every positive instance visible in the clip.

The latter is the relevant open-world diagnostic for the class-agnostic branch.
Its oracle is built from the auto branch's own frozen assignment, while the
prompt branch remains label-compatible with its configured noun list. The
all-instance numbers are reported as diagnostics and are not silently turned
into the historical GO/NO-GO gate.

This distinction matters: a high score in `prompt_scope` does not prove that
the visual-point grid discovers objects outside the configured category set.

该实验命令已归档；本文仅保留 auto-proposal 诊断的设置、结果和结论。

Changing the grid, discovery stride, area limits, or duplicate thresholds
requires `--overwrite` in the generation step; existing artifacts are checked
against the stored policy and are not silently mixed with a new run.

The auto branch currently carries only the generic semantic label `object`.
It diagnoses class-agnostic mask discovery and ownership; semantic category
classification must be evaluated separately if this branch is retained.
