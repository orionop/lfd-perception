# Failure modes — `lfdws_t001` (carrot pick / press / drop)

Honest log of what does NOT work in the current pipeline, on the single
trial Mark sent on 2026-05-19. Maintained so we can hand this to him in the
next sync and not pretend the pipeline is more polished than it is.

## A. Tracking / segmentation issues observed in the carrot trial

### A1. Carrot mask shrinks during the press
- At grasp (`f250`) the carrot mask is ~38 000 px.
- During the press (`f370`) it stabilises around ~23 000 px.
- After release (`f469`) it collapses to ~5 500 px.
- **Cause:** partial occlusion. The carrot is behind the gripper/cup or rolls
  out of frame. SAM 2 keeps a small mask alive but the area drops.
- **Implication for downstream:** absolute mask area is not a reliable
  "presence" signal; need bbox / centroid instead. Or seed a second prompt
  after release.

### A2. Cup mask is tiny before the press
- Cup centroid only stabilises once the gripper is near it.
- Before frame ~300, the cup mask is < 1 500 px.
- **Cause:** the cup is partly out of frame in the early phase (Mark
  flagged the camera is off-centre; see B1).
- **Implication:** the cup propagation backward from the press is the
  correct seed point — we cannot reliably find the cup before contact occurs.

### A3. DADO-style label-free proposers fail on this scene
- DINOv2 attention is uniformly low / noisy on these frames — the CLS-to-patch
  saliency does not isolate the carrot or the cup.
- Depth-Anything depth is clean and useful for foreground separation, but
  multiplying by attention leaves a noisy proposal map.
- **Cause:** the workspace is small, low-contrast, with metal table textures
  that DINO finds equally salient. Attention foundation models trained on
  internet imagery are not tuned for this in-lab top-down view.
- **Implication:** validates that proprioceptive cueing is load-bearing,
  not decorative — vision-only label-free proposal does not work here.

## B. Data limitations (from Mark's emails)

### B1. Camera is off-centre
- Mark flagged this in the 2nd of the 19 May emails. Visible: many frames
  have the cup near the right edge or partially clipped.
- **Implication:** the cup mask quality before the press is dominated by
  this, not by SAM. Will be fixed in the next measurement on Mark's side.

### B2. No depth available
- Mark offered depth on request, currently absent. Pipeline runs on RGB
  only.
- **Implication:** any depth-aware proposer (DADO, RGB-D SAM extensions)
  needs the depth stream to be turned on. Worth requesting for the next bag.

## C. Pipeline / engineering limitations

### C1. Seed points are hand-picked image fractions
- `ROLE_SEEDS` in `identify_objects.py` uses hard-coded image fractions
  (0.70, 0.30) for the grasped object and (0.55, 0.65) for the contact
  receiver. These were tuned on `lfdws_t001` and will likely fail on a new
  scene with a different camera pose.
- **Fix:** project the end-effector pose into the image as the prompt.
  Blocked on ZED-to-base extrinsics + intrinsics + URDF — not in the data
  Mark has shared so far. **Requested in next email.**

### C2. Only forward propagation for the grasped object
- The grasped-object seed is at the grasp event (~frame 250). Frames 0–249
  (the reach phase) have no carrot mask. For the contact-receiver we
  propagate backward + forward from the press, so the cup is masked
  throughout — but the carrot is not.
- **Implication:** "what's on the table" before any interaction is unknown
  to the system. Acceptable for downstream LfD (which cares about
  interaction phases), but flag for completeness.

### C3. Object identity not preserved across episodes
- Within one demo, obj_id=1 is the carrot and obj_id=2 is the cup.
- Across demos, there is no mechanism to say "the same physical carrot was
  used in trial 001 and trial 005". If Mark needs that, we'd need
  visual-feature matching across episodes (e.g. DINOv2 nearest-neighbour
  on bbox crops) — not currently implemented.

### C4. Force-direction overlay is uncalibrated
- `force_overlay.py` fits a base->image linear map from the carrot mask
  centroids over time, rather than using real camera intrinsics and
  extrinsics. The arrow direction is a visual sanity check, not a
  geometric measurement.
- **Fix:** real ZED intrinsics + base-to-camera transform from Mark.

### C5. Hand-tuned event thresholds
- Gripper width threshold = midpoint of (open, closed). Force-contact event
  = peak of baseline-subtracted magnitude restricted to the held window.
  These worked here; they will fail on demos with no force contact, or
  with multiple pick-and-place cycles in a single bag.
- **Fix:** generalise to multi-event detection (find local extrema, not just
  the global one) when more diverse demos arrive.

## D. Things we cannot evaluate from this trial alone

- Generalisation to other tasks (only one task recorded so far).
- Generalisation to other objects (one carrot, one cup, one plate).
- Generalisation to other camera poses (camera was fixed, off-centre).
- Robustness to lighting (lab lighting only).
- Quantitative mask quality (no hand-labelled ground-truth masks yet — we
  evaluate by eyeballing).

## Recommended asks for Mark (next message)

1. ZED intrinsics + ZED-to-robot-base extrinsics — fixes C1, C4
2. Franka URDF or EE/tool frame definition — fixes C1
3. Depth stream enabled in next bag — unblocks B2, depth-aware proposers
4. Recentred camera in next measurement — addresses A2, B1
5. 2–3 more trials with varied objects/positions — addresses D and unblocks
   step 4 of the writeup roadmap
