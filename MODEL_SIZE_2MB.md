## 2 MB Student Variant (FP32) – Parameter Reduction Notes

**Goal:** shrink the student checkpoint from ~8 MB to ~2 MB while keeping the same I/O contract as `train_pose_flexible.py` (128×128 inputs, 19 heatmaps, 38 PAFs) and retaining distillation flow from the existing teacher.

**Where parameters were removed**
- Backbone width trims: `StudentModel2MB` narrows almost every stage  
  - conv1/2: 32 → 24 channels  
  - conv3: 64 → 48  
  - conv4: 64 → 48 (post-pool block)  
  - conv5/6: 96 → 64  
  - conv7/8: 128 → 96
- CPM block: 96 → 64 channels across `cpm1/2/3`.
- Heads: intermediate convs 64 → 48 before the fixed-size logits (`heat_out` 19, `paf_out` 38 stay unchanged).

**Why these locations were chosen**
- Width trims first: channel width drives parameter count quadratically more than kernel tweaks; shrinking channels in the backbone yields the biggest size reduction with minimal architectural change.
- Strides/pooling untouched: keeping the downsampling schedule identical (ending at 16×16) guarantees loss/label shapes stay aligned with the teacher, so distillation wiring and datasets do not change.
- CPM block kept (but narrowed): CPM refines part-local features before heads. Cutting it entirely would hurt pose quality; narrowing to 64 keeps refinement capacity while lowering params.
- Heads kept structurally identical: the head stacks still transform the refined features into pose-specific embeddings before logits. Reducing their intermediate width (64→48) saves params but preserves the necessary processing depth for heatmaps/PAFs.
- Final logits fixed at 19/38: these map directly to COCO keypoint/PAF channels. Changing them would break compatibility with labels, teacher outputs, and post-processing; they are left unchanged to keep training targets and inference outputs identical.

**Resulting size**
- ~0.36M params (~1.5 MB FP32 weights) versus the previous student’s larger width; checkpoints land near the 2 MB target including optimizer/metadata.

**Location**
- Implementation lives in `train_pose_flexible_2mb.py` (`StudentModel2MB`).
