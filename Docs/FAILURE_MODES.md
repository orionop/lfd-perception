# Failure modes — `lfdws_t001` (carrot pick / press / drop)

Honest log of what does NOT work in the current pipeline, on the single
trial currently available. Maintained so we can surface it in the next sync
and not pretend the pipeline is more polished than it is.

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
- **Cause:** the cup is partly out of frame in the early phase (the lab
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

## B. Data limitations

### B1. Camera is off-centre
- Many frames have the cup near the right edge or partially clipped.
- **Implication:** the cup mask quality before the press is dominated by
  this, not by SAM. To be addressed in the next measurement.

### B2. No depth available
- Depth is available on request, currently absent. Pipeline runs on RGB
  only.
- **Implication:** any depth-aware proposer (DADO, RGB-D SAM extensions)
  needs the depth stream to be turned on. Worth requesting for the next bag.

## C. Pipeline / engineering limitations

### C1. Seed points: hand-tuned defaults, vision-only auto-seed fallback
- The hard-coded image fractions (e.g. `(0.70, 0.30)` for the grasped
  object) were tuned on `lfdws_t001` and would fail on a new scene with a
  different camera pose.
- **Partial fix in place:** `Code/auto_seed.py` runs SAM's automatic mask
  generator on each event frame and writes a seed point to
  `figures/identify/auto_seeds.csv`. Both `Code/propagate_demo.py` and
  `Code/propagate_cup.py` now consume that CSV when present and fall back
  to the hard-coded defaults otherwise. Cup seeds come out clean. The
  carrot seed in this trial lands on the gripper finger (514, 54), so the
  early carrot mask is degraded — see A4.
- **Proper fix (scaffolded, awaiting data):** `Code/project_ee.py` projects
  the `current_pose` point into the image via `calibration.yaml`. It runs
  DRY (base-frame geometry only, no drawing) until the ZED intrinsics +
  `base→camera` extrinsics are filled in. The Franka Research 3 arm URDF
  is vendored (`Data/fr3.urdf`); `current_pose` is the TCP under default
  config, so no extra offset is needed for EE projection. **Blocked only on
  the lab's camera calibration** (in progress — extrinsics + depth promised).

### A4. Auto-seed carrot drift
- On `lfdws_t001` the vision-only auto-seed for the grasped role lands on
  the gripper finger rather than on the carrot body. SAM 2 then
  propagates the gripper finger for the first ~50 frames and only drifts
  onto the carrot once the gripper moves.
- **Cause:** at the grasp event the gripper occupies the
  upper-centre region the role prior favours, and is more "object-like"
  to DINOv2 than the partially-occluded carrot tip.
- **Fix:** end-effector projection (see C1). Until then, tightening the
  role prior to "below the gripper opening" would help.

### C2. Only forward propagation for the grasped object
- The grasped-object seed is at the grasp event (~frame 250). Frames 0–249
  (the reach phase) have no carrot mask. For the contact-receiver we
  propagate backward + forward from the press, so the cup is masked
  throughout — but the carrot is not.
- **Implication:** "what's on the table" before any interaction is unknown
  to the system. Acceptable for downstream LfD (which cares about
  interaction phases), but flag for completeness.

### C3. Object identity across episodes
- Within one demo, obj_id=1 is the carrot and obj_id=2 is the cup.
- **Partial fix in place:** `Code/object_identity.py` embeds per-frame
  mask crops with frozen DINOv2 and clusters them. Crops are now
  tightened to the actual mask (background blacked out inside the crop)
  to avoid table pixels dominating the embedding. At distance threshold
  0.6 it returns 3 clusters on this trial (close to the true count of
  two; the over-merging is driven by the gripper-finger drift in A4).
- Across demos still needs the same script run on the merged
  sidecar; not yet exercised against multiple bags.

### C4. Force-direction overlay is uncalibrated
- `force_overlay.py` fits a base->image linear map from the carrot mask
  centroids over time, rather than using real camera intrinsics and
  extrinsics. The arrow direction is a visual sanity check, not a
  geometric measurement.
- **Fix (scaffolded):** `Code/project_ee.py` does the real geometric
  wrench-line projection (Bicchi 1990 contact ray, transformed
  base->camera and projected through K). It needs three calibration
  blocks filled in `calibration.yaml`: camera intrinsics, `base→camera`
  extrinsics, and `bota_frame→base` (the F/T-sensor mount on the wrist).
  The mount transform is **not derivable from any URDF** — it is physical
  hardware geometry and must come from the lab. Until then `project_ee.py`
  runs DRY and `force_overlay.py` remains the uncalibrated stand-in.

### C5. Hand-tuned event thresholds — multi-event detector in place
- Gripper width threshold = midpoint of (open, closed). Force-contact
  event = peak of baseline-subtracted magnitude restricted to the held
  window. The original single-event detector returned exactly one of each;
  it would silently lose information on multi-cycle bags.
- **Fix in place:** `Code/multi_event.py` returns *all* threshold
  crossings + *all* local force-magnitude peaks above baseline and groups
  them into interaction cycles. On `lfdws_t001` it finds one cycle with
  one grasp, one release, and four force peaks (vs. the single global peak
  the original detector returned). The single-event detector in
  `analyze_demo.py` remains for the writeup's "first version" results;
  downstream code that needs multi-cycle handling should consume
  `figures/identify/events_multi.json`.

## D. Things we cannot evaluate from this trial alone

- Generalisation to other tasks (only one task recorded so far).
- Generalisation to other objects (one carrot, one cup, one plate).
- Generalisation to other camera poses (camera was fixed, off-centre).
- Robustness to lighting (lab lighting only).
- Quantitative mask quality (no hand-labelled ground-truth masks yet — we
  evaluate by eyeballing).

## Recommended asks for the next sync

1. ZED intrinsics (K + distortion) + `base→camera` extrinsics — fills the
   first two blocks of `calibration.yaml`, takes `project_ee.py` LIVE,
   fixes C1 and C4.
2. `bota_frame→base` F/T-sensor mount transform — the only piece not in any
   URDF; needed for the wrench-line projection (C4 / the research direction).
3. Confirm whether `current_pose` is the fingertip TCP or the flange (default
   FR3+Hand = TCP; the script handles either via `calibration.yaml`).
4. Depth stream enabled in next bag — unblocks B2, depth-aware proposers.
5. Recentred camera in next measurement — addresses A2, B1.
6. 2–3 more trials with varied objects/positions — addresses D and unblocks
   step 4 of the writeup roadmap.

Resolved since last sync: Franka Research 3 arm URDF obtained
(`Data/fr3.urdf`); EE/tool frame no longer an open ask (item 2 of the
previous list).
